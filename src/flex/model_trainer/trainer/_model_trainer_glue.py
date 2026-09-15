import math
import contextlib
from typing import Any, Dict
import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler
from flex.ml_utils.model_utils import ModelUtils
from flex.model_trainer import ModelTrainer, ModelTrainerArgs
from flex.ml_utils.metric_calculator import MetricCalculator
from flex.ml_utils.training_utils import TrainingUtils
from flex.ml_utils.model_ewma import ModelEWMA


class ModelTrainer_GLUE(ModelTrainer):
    """
    Generic GLUE-style text classification trainer (TinyBERT / RoBERTa, etc.).
    - Optional AMP via TrainingUtils.make_autocast
    - Optional EMA via ModelEWMA
    - Otherwise mirrors ModelTrainer_Standard interface
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")

        self.device = ModelUtils.accelerator_device()
        self.model: nn.Module = trainer_args.model

        ta = self.trainer_args
        self.amp_enabled: bool = bool(getattr(ta, "amp_enabled", False))

        self.use_grad_scaler: bool = bool(getattr(ta, "use_grad_scaler", True))
        self._scaler = None
        if self.amp_enabled and torch.cuda.is_available() and self.use_grad_scaler:
            self._scaler = GradScaler(enabled=True)

        ema_decay = getattr(ta, "ema_decay", None)
        self._ema = None
        if isinstance(ema_decay, (float, int)) and 0.0 < float(ema_decay) < 1.0:
            self._ema = ModelEWMA(self.model, decay=float(ema_decay), device=self.device)

        self.metrics = MetricCalculator()
        self._epoch_idx = 0

        # ensure model on target device
        if str(next(self.model.parameters()).device) != str(self.trainer_args.device):
            self.model = self.model.to(self.trainer_args.device)
            self.trainer_args.model = self.model
        self.model = ModelUtils.wrap_data_parallel(self.model, self.device)
        self.trainer_args.model = self.model

    def set_model(self, model: nn.Module):
        self.trainer_args.model = model
        if str(next(model.parameters()).device) != self.trainer_args.device:
            self.trainer_args.model = model.to(self.trainer_args.device)
        self.model = ModelUtils.wrap_data_parallel(self.trainer_args.model, self.device)
        self.trainer_args.model = self.model
        return self

    def _ctx_model_train(self):
        self.trainer_args.model.train()
        return contextlib.nullcontext()

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
        self.metrics.reset()

        from ...ml_utils.tqdm_utils import pbar
        loop = pbar(
            train_dl,
            desc=f"Training (epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''})",
            leave=False, ncols=120, mininterval=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        with self._ctx_model_train():
            for inputs, labels in loop:
                if hasattr(inputs, "to"):
                    inputs = inputs.to(ta.device)
                elif isinstance(inputs, dict):
                    inputs = {
                        key: (
                            TrainingUtils.to_device(value, self.device)
                            if hasattr(value, "to")
                            else value
                        )
                        for key, value in inputs.items()
                    }
                labels = labels.to(ta.device)

                # Ensure labels are long type for classification
                if labels.dtype != torch.long:
                    labels = labels.long()

                try:
                    batch_size = int(labels.size(0))
                except Exception:
                    batch_size = 1

                ta.optimizer.zero_grad(set_to_none=True)

                with TrainingUtils.make_autocast(device=self.device, enabled=self.amp_enabled):
                    outputs = ta.model(inputs)
                    # Handle case where model outputs logits (batch_size, num_classes)
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        outputs = outputs['logits']
                    loss = ta.loss_func(outputs, labels)

                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                    self._scaler.step(ta.optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    ta.optimizer.step()

                # NEW: Optional Scheduler step per batch
                scheduler = getattr(ta, "scheduler", None)
                if scheduler is not None:
                    scheduler.step()

                if self._ema is not None:
                    self._ema.update(ta.model)

                loss_scalar = float(loss.detach().item())
                self.metrics.update(loss_scalar, batch_size)

                loop.set_postfix(
                    batch=self.metrics.total_batch,
                    loss=f"{loss_scalar:.4f}",
                    avg_loss=f"{self.metrics.avg_loss:.4f}",
                    avg_loss_keras=f"{self.metrics.keras_loss:.4f}",
                    lr=ta.optimizer.param_groups[0]["lr"]
                )

        from ...ml_utils.tqdm_utils import tqdm_write
        tqdm_write(
            f"[Epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''} Finished] "
            f"avg_loss={self.metrics.avg_loss:.6f} | keras_loss={self.metrics.keras_loss:.6f} | "
            f"batches={self.metrics.total_batch} | samples={self.metrics.total_samples} | device={ta.device}"
        )
        return self.metrics.get_stats()

    def train(self, epochs, is_return_wbab=False) -> Any:
        self.trainer_args.total_epochs = epochs
        self._epoch_idx = 0

        ModelUtils.model_training_info(self.model, self.trainer_args.optimizer)
        # Snapshot the pre-training weights on CPU. The L2 helpers move tensors
        # to CPU internally, so keeping this clone off-GPU avoids a full-model
        # (~356M param) transient allocation on the GPU each round — which
        # otherwise raises peak usage and fragments memory across many clients.
        before_state = {k: v.detach().cpu() for k, v in self._unwrap(self.trainer_args.model).state_dict().items()}
        before_weight_l2 = self._state_dict_l2_norm(before_state)

        stats: Dict[str, Any] = {
            "train_loss_sum": 0.0,
            "train_loss_power_two_sum": 0.0,
            "epoch_loss": [],
            "keras_train_loss_sum": 0.0,
            "keras_train_loss_power_two_sum": 0.0,
            "keras_epoch_loss": [],
            "num_batches_sum": 0,
            "num_samples_sum": 0,
        }

        for _ in range(epochs):
            self._epoch_idx += 1
            step_out = self.train_step()
            avg_loss = float(step_out["avg_loss"])
            keras_loss = float(step_out["keras_loss"])

            num_batches = int(step_out.get("num_batches", 0))
            num_samples = int(step_out.get("num_samples", 0))

            stats["train_loss_sum"] += avg_loss
            stats["train_loss_power_two_sum"] += avg_loss ** 2
            stats["epoch_loss"].append(avg_loss)

            stats["keras_train_loss_sum"] += keras_loss
            stats["keras_train_loss_power_two_sum"] += keras_loss ** 2
            stats["keras_epoch_loss"].append(keras_loss)

            stats["num_batches_sum"] += num_batches
            stats["num_samples_sum"] += num_samples

        self._epoch_idx = 0
        stats["avg_loss"] = stats["train_loss_sum"] / max(epochs, 1)
        stats["keras_avg_loss"] = stats["keras_train_loss_sum"] / max(epochs, 1)

        stats["sqrt_train_loss_power_two_sum"] = math.sqrt(stats["train_loss_power_two_sum"])
        stats["keras_sqrt_train_loss_power_two_sum"] = math.sqrt(stats["keras_train_loss_power_two_sum"])

        after_state = self._unwrap(self.trainer_args.model).state_dict()
        after_weight_l2 = self._state_dict_l2_norm(after_state)
        stats["weight_l2_before"] = before_weight_l2
        stats["weight_l2_after"] = after_weight_l2
        stats["weight_l2_delta"] = self._state_dict_l2_distance(before_state, after_state)

        return self._unwrap(self.trainer_args.model).state_dict(), stats

    def observe(self, epochs=5) -> Any:
        self.trainer_args.total_epochs = epochs
        stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum": 0}

        for _ in range(epochs):
            step_out = self.train_step()
            loss = float(step_out["avg_loss"])
            stats["train_loss_sum"] += loss
            stats["train_loss_power_two_sum"] += loss ** 2
            stats["epoch_loss"].append(loss)

        stats["avg_loss"] = stats["train_loss_sum"] / max(epochs, 1)
        stats["sqrt_train_loss_power_two_sum"] = math.sqrt(stats["train_loss_power_two_sum"])
        return self._unwrap(self.trainer_args.model).state_dict(), stats
