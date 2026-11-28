from __future__ import annotations

import math
from typing import Any, Optional, Dict

import torch
import torch.nn as nn
import numpy as np

from flex.model_trainer.model_trainer_args import ModelTrainerArgs
from flex.ml_utils import console
from flex.ml_utils.model_ewma import ModelEWMA
from flex.ml_utils.model_utils import ModelUtils
from flex.model_trainer import ModelTrainer
from flex.ml_utils.metric_calculator import MetricCalculator


class ModelTrainer_ComplexCV(ModelTrainer):
    """
    Complex CV Model Trainer with advanced optimizations:
      - AMP (Automatic Mixed Precision)
      - Gradient Accumulation
      - Gradient Clipping
      - Scheduler step per batch
      - EMA (Exponential Moving Average)
      - Mixup Data Augmentation
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")

        self.device = ModelUtils.accelerator_device()
        self.model: nn.Module = ModelUtils.wrap_data_parallel(trainer_args.model, self.device)

        self._epoch_idx: int = 0
        self._use_amp = self._check_use_amp()
        self._scaler = self._make_scaler()

        self._ewma: Optional[ModelEWMA] = None
        ema_decay = getattr(trainer_args, "ema_decay", 0)
        if ema_decay and ema_decay > 0:
            self._ewma = ModelEWMA(self.model, decay=ema_decay, device=self.device)

        self.metrics = MetricCalculator()
        return

    def set_model(self, model: nn.Module):
        self.trainer_args.model = model
        if str(next(model.parameters()).device) != self.trainer_args.device:
            self.trainer_args.model = model.to(self.trainer_args.device)
        self.model = ModelUtils.wrap_data_parallel(self.trainer_args.model, self.device)
        self.trainer_args.model = self.model
        
        if self._ewma is not None:
            ema_decay = getattr(self.trainer_args, "ema_decay", 0.999)
            self._ewma = ModelEWMA(self.model, decay=ema_decay, device=self.device)
        return self

    def train_step(self) -> Dict[str, Any]:
        ta = self.trainer_args
        if ta.optimizer is None: raise ValueError("Trainer optimizer is None.")
        if ta.model is None: raise ValueError("Trainer model is None.")
        if ta.loss_func is None: raise ValueError("Trainer loss function is None.")
        if ta.train_loader is None: raise ValueError("Trainer train_loader is None.")

        train_dl = ta.train_loader.data_loader
        if not hasattr(train_dl, "__iter__"):
            raise TypeError(f"train_loader must be an iterable, got {type(train_dl).__name__}")

        total_epochs = getattr(ta, "total_epochs", getattr(ta, "epochs", None))

        ta.model.to(self.device)
        ta.model.train()
        ta.optimizer.zero_grad(set_to_none=True)

        self.metrics.reset()

        # Configs
        grad_accum_steps: int = int(getattr(ta, "grad_accum_steps", 1))
        clip_grad_norm: float = float(getattr(ta, "clip_grad_norm", 0.0))
        mixup_alpha: float = float(getattr(ta, "mixup_alpha", 0.0))

        from ...ml_utils.tqdm_utils import pbar, tqdm_write
        loop = pbar(
            train_dl,
            desc=f"Training (epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''})",
            leave=False, ncols=120, mininterval=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for inputs, labels in loop:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            batch_size = inputs.size(0)

            # Mixup augmentation
            do_mixup = mixup_alpha > 0 and self.model.training
            if do_mixup:
                inputs, labels_a, labels_b, lam = self._mixup_data(inputs, labels, mixup_alpha)

            with self._autocast_context():
                outputs = ta.model(inputs)
                if do_mixup:
                    loss = self._mixup_criterion(ta.loss_func, outputs, labels_a, labels_b, lam)
                else:
                    loss = ta.loss_func(outputs, labels)
                
                loss_scaled = loss / max(1, grad_accum_steps)

            # Backward
            if self._scaler is not None:
                self._scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Optimizer Step
            if (self.metrics.total_batch + 1) % grad_accum_steps == 0:
                if clip_grad_norm > 0:
                    if self._scaler is not None:
                        self._scaler.unscale_(ta.optimizer)
                    nn.utils.clip_grad_norm_(ta.model.parameters(), max_norm=clip_grad_norm)

                if self._scaler is not None:
                    self._scaler.step(ta.optimizer)
                    self._scaler.update()
                else:
                    ta.optimizer.step()

                ta.optimizer.zero_grad(set_to_none=True)

                # Scheduler step
                if ta.get("scheduler") is not None:
                    ta.scheduler.step()

            # EMA Update
            if self._ewma is not None:
                self._ewma.update(self.model)

            loss_scalar = float(loss.item())
            self.metrics.update(loss_scalar, batch_size)

            loop.set_postfix(
                batch=self.metrics.total_batch,
                loss=f"{loss_scalar:.4f}",
                avg_loss=f"{self.metrics.avg_loss:.4f}",
                lr=ta.optimizer.param_groups[0]["lr"]
            )

        tqdm_write(
            f"[Epoch {self._epoch_idx} Finished] avg_loss={self.metrics.avg_loss:.6f} | device={self.device}"
        )
        return self.metrics.get_stats()

    def train(self, epochs: int) -> Any:
        self.trainer_args.total_epochs = epochs
        self._epoch_idx = 0

        ModelUtils.model_training_info(self.model, self.trainer_args.optimizer)
        before_state = {k: v.detach().clone() for k, v in self._unwrap(self.model).state_dict().items()}
        before_weight_l2 = self._state_dict_l2_norm(before_state)

        train_stats = {"train_loss_sum": 0.0, "epoch_loss": [], "train_loss_power_two_sum": 0.0}

        for _ in range(epochs):
            self._epoch_idx += 1
            step_stats = self.train_step()
            avg_loss = float(step_stats["avg_loss"])
            train_stats["train_loss_sum"] += avg_loss
            train_stats["train_loss_power_two_sum"] += avg_loss ** 2
            train_stats["epoch_loss"].append(avg_loss)

        self._epoch_idx = 0
        train_stats["avg_loss"] = train_stats["train_loss_sum"] / max(epochs, 1)
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])

        after_state = self._unwrap(self.model).state_dict()
        after_weight_l2 = self._state_dict_l2_norm(after_state)
        train_stats["weight_l2_before"] = before_weight_l2
        train_stats["weight_l2_after"] = after_weight_l2
        train_stats["weight_l2_delta"] = self._state_dict_l2_distance(before_state, after_state)

        return self._unwrap(self.model).state_dict(), train_stats

    def observe(self, epochs=5) -> Any:
        return self.train(epochs)

    # --- Internal Helpers ---

    def _check_use_amp(self) -> bool:
        ta = self.trainer_args
        want_amp = bool(ta.get("use_amp", ta.get("amp", False)))
        if not want_amp: return False
        return self.device.type in ["cuda", "mps"]

    def _make_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        if self.device.type == "cuda" and self._check_use_amp():
            return torch.cuda.amp.GradScaler(enabled=True)
        return None

    def _autocast_context(self):
        if not self._use_amp:
            return _NullCtx()
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=torch.float16)
        if self.device.type == "mps":
            return torch.autocast(device_type="mps", dtype=torch.float16)
        return _NullCtx()

    def _mixup_data(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False
