"""Standard MNIST trainer plus isolated Strong-A proximal regularization."""
from __future__ import annotations

from typing import Any, Dict

import torch

from flex.ml_algorithms.rbla_problem import StrongAConfig, StrongAProximalLoss
from flex.ml_utils.training_utils import TrainingUtils

from .._model_trainer_standard import ModelTrainer_Standard


class ModelTrainer_RBLAStrongA(ModelTrainer_Standard):
    def __init__(self, trainer_args):
        super().__init__(trainer_args)
        self._strong_a_anchors: Dict[str, torch.Tensor] = {}
        self._strong_a_loss = StrongAProximalLoss()
        self._strong_a_task_sum = 0.0
        self._strong_a_reg_sum = 0.0
        self._strong_a_batches = 0

    def set_strong_a_context(self, anchors: Dict[str, torch.Tensor], config: dict) -> None:
        parsed = StrongAConfig.from_dict(config)
        self._strong_a_loss = StrongAProximalLoss(parsed)
        self._strong_a_anchors = {
            key: value.detach().to(self.device)
            for key, value in anchors.items()
        }

    def clear_strong_a_context(self) -> None:
        self._strong_a_anchors = {}

    def train_step(self) -> Dict[str, Any]:
        from flex.ml_utils.tqdm_utils import pbar

        ta = self.trainer_args
        if ta.optimizer is None or ta.model is None or ta.loss_func is None or ta.train_loader is None:
            raise ValueError("Strong-A trainer is missing model, optimizer, loss, or train loader")

        train_dl = ta.train_loader.data_loader
        total_epochs = getattr(ta, "total_epochs", getattr(ta, "epochs", None))
        ta.model.to(self.device)
        ta.model.train()
        self.metrics.reset()

        loop = pbar(
            train_dl,
            desc=f"RBLA Strong-A (epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''})",
            leave=False,
            ncols=120,
            mininterval=0.1,
        )
        for inputs, labels in loop:
            inputs = TrainingUtils.to_device(inputs, self.device)
            labels = TrainingUtils.to_device(labels, self.device).long()
            batch_size = int(labels.size(0))
            ta.optimizer.zero_grad(set_to_none=True)

            with TrainingUtils.make_autocast(self.device, self.amp_enabled, self._amp_dtype):
                outputs = ta.model(inputs)
                task_loss = ta.loss_func(outputs, labels)
                parameter_map = dict(self._unwrap(ta.model).named_parameters())
                a_loss = self._strong_a_loss(parameter_map, self._strong_a_anchors)
                loss = task_loss + self._strong_a_loss.config.lambda_a * a_loss

            loss.backward()
            ta.optimizer.step()

            loss_value = float(loss.item())
            task_value = float(task_loss.item())
            a_value = float(a_loss.item())
            self.metrics.update(loss_value, batch_size)
            self._strong_a_task_sum += task_value
            self._strong_a_reg_sum += a_value
            self._strong_a_batches += 1
            loop.set_postfix(
                loss=f"{loss_value:.4f}",
                task=f"{task_value:.4f}",
                a_prox=f"{a_value:.6f}",
            )

        return self.metrics.get_stats()

    def train(self, epochs: int) -> Any:
        self._strong_a_task_sum = 0.0
        self._strong_a_reg_sum = 0.0
        self._strong_a_batches = 0
        weights, stats = super().train(epochs)
        denominator = max(self._strong_a_batches, 1)
        stats["strong_a_task_loss"] = self._strong_a_task_sum / denominator
        stats["strong_a_prox_loss"] = self._strong_a_reg_sum / denominator
        stats["strong_a_lambda"] = self._strong_a_loss.config.lambda_a
        return weights, stats
