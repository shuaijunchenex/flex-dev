"""
Legacy standard trainer with the pre-training loss pass.

This preserves the July-6 RBLA reproduction behavior without putting the
extra DataLoader pass back into the normal standard trainer.
"""

from typing import Any, Dict
import math

import torch

from ...ml_utils.model_utils import ModelUtils
from ._model_trainer_standard_legacy import ModelTrainer_StandardLegacy


class ModelTrainer_StandardLegacyPrepass(ModelTrainer_StandardLegacy):
    """Pre-July-7 standard trainer plus initial-loss prepass."""

    def _eval_initial_loss(self) -> float:
        ta = self.trainer_args
        model = ta.model
        model.eval()
        total_loss = 0.0
        total_samples = 0
        device = ta.device or self.device

        with torch.no_grad():
            for inputs, labels in ta.train_loader.data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                outputs = model(inputs)
                loss = ta.loss_func(outputs, labels)
                batch_size = int(inputs.size(0))
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size

        model.train()
        return total_loss / max(total_samples, 1)

    def train(self, epochs: int) -> Any:
        self.trainer_args.total_epochs = epochs
        self._epoch_idx = 0

        ModelUtils.model_training_info(self.model, self.trainer_args.optimizer)
        before_state = {
            k: v.detach().clone()
            for k, v in self._unwrap(self.trainer_args.model).state_dict().items()
        }
        before_weight_l2 = self._state_dict_l2_norm(before_state)
        initial_loss = self._eval_initial_loss()

        train_stats: Dict[str, Any] = {
            "train_loss_sum": 0.0,
            "train_loss_power_two_sum": 0.0,
            "epoch_loss": [],
            "keras_train_loss_sum": 0.0,
            "keras_train_loss_power_two_sum": 0.0,
            "keras_epoch_loss": [],
            "num_batches_sum": 0,
            "num_samples_sum": 0,
            "initial_loss": initial_loss,
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
        train_stats["keras_sqrt_train_loss_power_two_sum"] = math.sqrt(
            train_stats["keras_train_loss_power_two_sum"]
        )

        after_state = self._unwrap(self.trainer_args.model).state_dict()
        after_weight_l2 = self._state_dict_l2_norm(after_state)
        train_stats["weight_l2_before"] = before_weight_l2
        train_stats["weight_l2_after"] = after_weight_l2
        train_stats["weight_l2_delta"] = self._state_dict_l2_distance(before_state, after_state)
        train_stats["weight_l2_delta_keras"] = self._state_dict_l2_distance_layerwise(
            before_state, after_state
        )

        return self._unwrap(self.trainer_args.model).state_dict(), train_stats
