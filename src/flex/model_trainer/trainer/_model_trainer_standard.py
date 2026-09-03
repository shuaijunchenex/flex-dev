import contextlib
from typing import Any, Dict, Optional
import torch.nn as nn
import torch
import numpy as np
import random
import math
from ...ml_utils.tqdm_utils import pbar, tqdm_write
from flex.model_trainer.model_trainer_args import ModelTrainerArgs

from ..model_trainer import ModelTrainer
from ...ml_algorithms import ModelExtractor
from ...ml_utils import console
from ...ml_utils.model_utils import ModelUtils
from ...ml_utils.metric_calculator import MetricCalculator
from ...ml_utils.training_utils import TrainingUtils

class ModelTrainer_Standard(ModelTrainer):
    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")

        self.device = trainer_args.device or ModelUtils.accelerator_device()
        self.model: nn.Module = trainer_args.model
        trainer_args.device = self.device  # keep ta.device in sync
        self._epoch_idx: int = 0

        # AMP: prefer BF16 on capable CUDA (A10G, L4, A100…); fallback to FP32
        self.amp_enabled: bool = bool(trainer_args.get("use_amp", False))
        self._amp_dtype: Optional[torch.dtype] = (
            TrainingUtils.resolve_amp_dtype(self.device) if self.amp_enabled else None
        )
        if self.amp_enabled and self._amp_dtype is None:
            console.debug("[AMP] BF16 not supported on this device — running in FP32.")
        elif self.amp_enabled:
            console.debug(f"[AMP] Enabled with dtype={self._amp_dtype}")
        # BF16 does not need GradScaler; only FP16 does (not used here)
        self._scaler = None

        # Wrap with DataParallel when multiple CUDA GPUs are available
        self.model = ModelUtils.wrap_data_parallel(self.model, self.device)
        trainer_args.model = self.model
        self.metrics = MetricCalculator()
        return

    def set_model(self, model: nn.Module):
        self.trainer_args.model = model
        if str(next(model.parameters()).device) != self.trainer_args.device:
            self.trainer_args.model = model.to(self.trainer_args.device)
        self.model = ModelUtils.wrap_data_parallel(self.trainer_args.model, self.device)
        self.trainer_args.model = self.model
        return self

    def train_step(self) -> Dict[str, Any]:
        ta = self.trainer_args
        if ta.optimizer is None:
            raise ValueError("Trainer optimizer is None.")
        if ta.model is None:
            raise ValueError("Trainer model is None.")
        if ta.loss_func is None:
            raise ValueError("Trainer loss function is None.")
        if ta.train_loader is None:
            raise ValueError("Trainer train_loader is None.")

        train_dl = ta.train_loader.data_loader
        if not hasattr(train_dl, "__iter__"):
            raise TypeError(f"train_loader must be an iterable DataLoader, got {type(train_dl).__name__}")

        total_epochs = getattr(ta, "total_epochs", getattr(ta, "epochs", None))

        ta.model.to(self.device)
        ta.model.train()

        self.metrics.reset()

        loop = pbar(
            train_dl,
            desc=f"Training (epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''})",
            leave=False, ncols=120, mininterval=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        for inputs, labels in loop:
            inputs = TrainingUtils.to_device(inputs, self.device)
            labels = TrainingUtils.to_device(labels, self.device).long()

            # Validate label values before loss computation
            min_label = labels.min().item()
            max_label = labels.max().item()
            num_classes = None
            if ta.loss_func is not None and hasattr(ta.loss_func, 'weight') and ta.loss_func.weight is not None:
                num_classes = ta.loss_func.weight.size(0)
            elif hasattr(ta.model, 'config') and hasattr(ta.model.config, 'num_labels'):
                num_classes = ta.model.config.num_labels
            
            # Diagnose label range issue
            if num_classes is not None and (min_label < 0 or max_label >= num_classes):
                from ...ml_utils import console
                unique_labels = torch.unique(labels).cpu().numpy().tolist()
                console.error(
                    f"[Label Range Error] min={min_label}, max={max_label}, num_classes={num_classes}, "
                    f"unique_labels={unique_labels}. In extreme non-IID, each client should have all label types."
                )
                raise ValueError(
                    f"Invalid label values: min={min_label}, max={max_label}, but num_classes={num_classes}"
                )

            try:
                batch_size = int(inputs.size(0))
            except Exception:
                batch_size = int(labels.size(0))

            ta.optimizer.zero_grad(set_to_none=True)

            with TrainingUtils.make_autocast(self.device, self.amp_enabled, self._amp_dtype):
                outputs = ta.model(inputs)
                loss = ta.loss_func(outputs, labels)

            loss.backward()
            ta.optimizer.step()

            loss_scalar = float(loss.item())
            self.metrics.update(loss_scalar, batch_size)

            loop.set_postfix(
                batch=self.metrics.total_batch,
                loss=f"{loss_scalar:.4f}",
                avg_loss=f"{self.metrics.avg_loss:.4f}",
                avg_loss_keras=f"{self.metrics.keras_loss:.4f}",
                lr=ta.optimizer.param_groups[0]["lr"]
            )

        tqdm_write(
            f"[Epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''} Finished] "
            f"avg_loss={self.metrics.avg_loss:.6f} | keras_loss={self.metrics.keras_loss:.6f} | "
            f"batches={self.metrics.total_batch} | samples={self.metrics.total_samples} | device={ta.device}"
        )

        return self.metrics.get_stats()

    def train(self, epochs: int) -> Any:
        self.trainer_args.total_epochs = epochs
        self._epoch_idx = 0

        ModelUtils.model_training_info(self.model, self.trainer_args.optimizer)
        before_state = {k: v.detach().clone() for k, v in self._unwrap(self.trainer_args.model).state_dict().items()}
        before_weight_l2 = self._state_dict_l2_norm(before_state)

        train_stats: Dict[str, Any] = {
            "train_loss_sum": 0.0,
            "train_loss_power_two_sum": 0.0,
            "epoch_loss": [],
            "keras_train_loss_sum": 0.0,
            "keras_train_loss_power_two_sum": 0.0,
            "keras_epoch_loss": [],
            "num_batches_sum": 0,
            "num_samples_sum": 0,
            "initial_loss": 0.0,
        }

        for _ in range(epochs):
            self._epoch_idx += 1

            step_out = self.train_step() 
            avg_loss = float(step_out["avg_loss"])
            keras_loss = float(step_out["keras_loss"])

            num_batches = int(step_out.get("num_batches", 0))
            num_samples = int(step_out.get("num_samples", 0))

            train_stats["train_loss_sum"] += avg_loss
            train_stats["train_loss_power_two_sum"] += avg_loss ** 2
            train_stats["epoch_loss"].append(avg_loss)

            train_stats["keras_train_loss_sum"] += keras_loss
            train_stats["keras_train_loss_power_two_sum"] += keras_loss ** 2
            train_stats["keras_epoch_loss"].append(keras_loss)

            train_stats["num_batches_sum"] += num_batches
            train_stats["num_samples_sum"] += num_samples

        self._epoch_idx = 0

        train_stats["avg_loss"] = train_stats["train_loss_sum"] / max(epochs, 1)
        train_stats["keras_avg_loss"] = train_stats["keras_train_loss_sum"] / max(epochs, 1)

        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])
        train_stats["keras_sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["keras_train_loss_power_two_sum"])

        after_state = self._unwrap(self.trainer_args.model).state_dict()
        after_weight_l2 = self._state_dict_l2_norm(after_state)
        train_stats["weight_l2_before"] = before_weight_l2
        train_stats["weight_l2_after"] = after_weight_l2
        train_stats["weight_l2_delta"] = self._state_dict_l2_distance(before_state, after_state)
        train_stats["weight_l2_delta_keras"] = self._state_dict_l2_distance_layerwise(before_state, after_state)

        return self._unwrap(self.trainer_args.model).state_dict(), train_stats

    def observe(self, epochs=5) -> Any:
        self.trainer_args.total_epochs = epochs
        train_stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum": 0}

        for _ in range(epochs):
            step_out = self.train_step()
            train_loss = float(step_out["avg_loss"])
            train_stats["train_loss_sum"] += train_loss
            train_stats["train_loss_power_two_sum"] += train_loss ** 2
            train_stats["epoch_loss"].append(train_loss)

        train_stats["avg_loss"] = train_stats["train_loss_sum"] / epochs
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])
        return self._unwrap(self.trainer_args.model).state_dict(), train_stats