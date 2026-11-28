from __future__ import annotations

import torch
import torch.nn as nn

from flex.model_trainer.model_trainer_args import ModelTrainerArgs

from ..model_trainer import ModelTrainer


class ModelTrainer_SplitClient(ModelTrainer):
    """
    Split learning client trainer, runs the front model one batch at a time.

    forward() returns the activation to send to the server, backward() applies the
    cut gradient the server sends back and steps the optimizer.
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")

        self.model: nn.Module = trainer_args.model
        self._activation: torch.Tensor | None = None  # waiting for its cut gradient

    @property
    def device(self) -> str:
        return self.trainer_args.device

    def set_model(self, model: nn.Module):
        self.trainer_args.model = model
        self.model = model
        return self

    def forward(self, inputs: torch.Tensor, need_cut_grad: bool = True) -> torch.Tensor:
        """
        Forward one batch and return the detached activation.
        With need_cut_grad False the front is frozen: no graph is built and backward() is not needed.
        """
        model = self.model.to(self.device)
        model.train()
        inputs = inputs.to(self.device)

        if not need_cut_grad:
            with torch.no_grad():
                return model(inputs)

        self._activation = model(inputs)
        return self._activation.detach()

    def backward(self, cut_grad: torch.Tensor) -> None:
        """
        Backward the cut gradient of the last forward() batch and step the optimizer.
        """
        optimizer = self.trainer_args.optimizer
        if optimizer is None:
            raise ValueError("Optimizer is required for backward.")
        if self._activation is None:
            raise RuntimeError("backward called without a preceding forward.")

        optimizer.zero_grad()
        self._activation.backward(cut_grad.to(self.device))
        optimizer.step()
        optimizer.zero_grad()
        self._activation = None

    def train_step(self) -> float:
        raise NotImplementedError("Use forward and backward in split learning.")

    def train(self, epochs, is_return_wbab=False):
        raise NotImplementedError("Use forward and backward in split learning.")
