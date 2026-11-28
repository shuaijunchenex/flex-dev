"""
ModelTrainer_PipelineTest
--------------------------
A minimal trainer that processes ONLY ONE sample (one batch of size 1) per
training step.  It is intentionally designed to be as fast as possible so the
entire FL pipeline (data loading → training → aggregation → logging) can be
verified end-to-end without waiting for full dataset training.

Usage
-----
Set ``trainer_type: pipeline_test`` in your YAML, e.g.:

    trainer:
      trainer_type: pipeline_test
      device: cpu          # or mps / cuda
      save_path: ./.training_results/
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn

from ...ml_utils.tqdm_utils import pbar

from flex.model_trainer.model_trainer_args import ModelTrainerArgs
from ..model_trainer import ModelTrainer
from ...ml_utils import console
from ...ml_utils.model_utils import ModelUtils
from ...ml_utils.metric_calculator import MetricCalculator


class ModelTrainer_PipelineTest(ModelTrainer):
    """
    Pipeline-test trainer: runs exactly **one sample** (the very first batch
    from the DataLoader, truncated to a single item) per training step.

    All return values are structurally identical to ``ModelTrainer_Standard``
    so the rest of the FL framework sees no difference.
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")

        self.device = trainer_args.device or ModelUtils.accelerator_device()
        self.model: nn.Module = trainer_args.model
        trainer_args.device = self.device
        self.metrics = MetricCalculator()

    # ------------------------------------------------------------------
    def set_model(self, model: nn.Module):
        self.trainer_args.model = model
        if str(next(model.parameters()).device) != self.trainer_args.device:
            self.trainer_args.model = model.to(self.trainer_args.device)
        self.model = self.trainer_args.model
        return self

    # ------------------------------------------------------------------
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
            raise TypeError(
                f"train_loader must be an iterable DataLoader, got {type(train_dl).__name__}"
            )

        total_epochs = getattr(ta, "total_epochs", getattr(ta, "epochs", None))

        ta.model.to(self.device)
        ta.model.train()
        self.metrics.reset()

        console.info(
            f"[PipelineTest] Epoch {self._epoch_idx}"
            f"{'/' + str(total_epochs) if total_epochs else ''} — "
            "training on ONE sample only (pipeline verification mode)."
        )

        # ---- grab ONLY the very first batch, truncated to 1 sample ----
        first_batch = next(iter(train_dl))
        inputs, labels = first_batch
        inputs = inputs[:1].to(self.device)   # keep only 1 sample
        labels = labels[:1].to(self.device).long()

        ta.optimizer.zero_grad()
        outputs = ta.model(inputs)
        loss = ta.loss_func(outputs, labels)
        loss.backward()
        ta.optimizer.step()

        loss_scalar = float(loss.item())
        self.metrics.update(loss_scalar, batch_size=1)

        console.info(
            f"[PipelineTest] Epoch {self._epoch_idx} done — "
            f"loss={loss_scalar:.4f} | device={self.device}"
        )

        return self.metrics.get_stats()

    # ------------------------------------------------------------------
    def train(self, epochs: int) -> Any:
        self.trainer_args.total_epochs = epochs
        self._epoch_idx = 0

        ModelUtils.model_training_info(self.model, self.trainer_args.optimizer)
        before_state = {
            k: v.detach().clone()
            for k, v in self._unwrap(self.trainer_args.model).state_dict().items()
        }
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
        }

        for _ in range(epochs):
            self._epoch_idx += 1
            step_out = self.train_step()

            avg_loss   = float(step_out["avg_loss"])
            keras_loss = float(step_out["keras_loss"])
            num_batches = int(step_out.get("num_batches", 0))
            num_samples = int(step_out.get("num_samples", 0))

            train_stats["train_loss_sum"]             += avg_loss
            train_stats["train_loss_power_two_sum"]   += avg_loss ** 2
            train_stats["epoch_loss"].append(avg_loss)

            train_stats["keras_train_loss_sum"]             += keras_loss
            train_stats["keras_train_loss_power_two_sum"]   += keras_loss ** 2
            train_stats["keras_epoch_loss"].append(keras_loss)

            train_stats["num_batches_sum"] += num_batches
            train_stats["num_samples_sum"] += num_samples

        self._epoch_idx = 0

        train_stats["avg_loss"]        = train_stats["train_loss_sum"] / max(epochs, 1)
        train_stats["keras_avg_loss"]  = train_stats["keras_train_loss_sum"] / max(epochs, 1)
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(
            train_stats["train_loss_power_two_sum"]
        )
        train_stats["keras_sqrt_train_loss_power_two_sum"] = math.sqrt(
            train_stats["keras_train_loss_power_two_sum"]
        )

        after_state = self._unwrap(self.trainer_args.model).state_dict()
        after_weight_l2 = self._state_dict_l2_norm(after_state)
        train_stats["weight_l2_before"]       = before_weight_l2
        train_stats["weight_l2_after"]        = after_weight_l2
        train_stats["weight_l2_delta"]        = self._state_dict_l2_distance(before_state, after_state)
        train_stats["weight_l2_delta_keras"]  = self._state_dict_l2_distance_layerwise(before_state, after_state)

        return self._unwrap(self.trainer_args.model).state_dict(), train_stats

    # ------------------------------------------------------------------
    def observe(self, epochs: int = 5) -> Any:
        self.trainer_args.total_epochs = epochs
        train_stats = {
            "train_loss_sum": 0.0,
            "epoch_loss": [],
            "train_loss_power_two_sum": 0.0,
        }
        for _ in range(epochs):
            step_out = self.train_step()
            train_loss = float(step_out["avg_loss"])
            train_stats["train_loss_sum"]           += train_loss
            train_stats["train_loss_power_two_sum"] += train_loss ** 2
            train_stats["epoch_loss"].append(train_loss)

        train_stats["avg_loss"] = train_stats["train_loss_sum"] / epochs
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(
            train_stats["train_loss_power_two_sum"]
        )
        return self._unwrap(self.trainer_args.model).state_dict(), train_stats
