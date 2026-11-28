from __future__ import annotations

import torch
import torch.nn as nn

from flex.model_trainer.model_trainer_args import ModelTrainerArgs

from ..model_trainer import ModelTrainer


class ModelTrainer_SplitServer(ModelTrainer):
    """
    Split learning server trainer, runs the rear model on the activation sent by a client.

    The rear is trained when an optimizer is set, otherwise it is frozen and only the
    cut gradient is computed. step() is forward() followed by backward().
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Server Model is None.")
        if trainer_args.loss_func is None:
            raise ValueError("Server loss function is None.")

        self.model: nn.Module = trainer_args.model

    @property
    def device(self) -> str:
        return self.trainer_args.device

    def set_model(self, model: nn.Module):
        self.trainer_args.model = model
        self.model = model
        return self

    def forward(self, activation: torch.Tensor, labels: torch.Tensor, need_cut_grad: bool = True):
        """
        Forward the activation through the rear model.
        Return (cut, loss, stats) to pass to backward(). A graph is built only when the cut
        gradient is needed or the rear is trained.
        """
        update_rear = self.trainer_args.optimizer is not None
        model = self.model.to(self.device)
        model.train(update_rear)

        cut = activation.to(self.device).detach().requires_grad_(need_cut_grad)
        labels = labels.to(self.device)

        with torch.set_grad_enabled(need_cut_grad or update_rear):
            outputs = model(cut)
            loss = self.trainer_args.loss_func(outputs, labels)

        stats = {"correct": int(outputs.argmax(1).eq(labels).sum().item()), "n_examples": len(labels)}
        return cut, loss, stats

    def backward(self, cut: torch.Tensor, loss: torch.Tensor) -> torch.Tensor | None:
        """
        Backward the loss and step the optimizer if the rear is trained.
        Return the cut gradient for the client, or None when the client needs none.
        """
        optimizer = self.trainer_args.optimizer

        if optimizer is None:
            return torch.autograd.grad(loss, cut)[0].detach() if cut.requires_grad else None

        optimizer.zero_grad()
        loss.backward()
        cut_grad = cut.grad.detach() if cut.requires_grad else None
        optimizer.step()
        optimizer.zero_grad()
        return cut_grad

    def step(self, activation: torch.Tensor, labels: torch.Tensor, need_cut_grad: bool = True):
        """
        One server step, return (cut_grad, loss value, stats).
        """
        cut, loss, stats = self.forward(activation, labels, need_cut_grad)
        cut_grad = self.backward(cut, loss)
        return cut_grad, loss.item(), stats

    def train_step(self) -> float:
        raise NotImplementedError("Use forward, backward and step in split learning.")

    def train(self, epochs, is_return_wbab=False):
        raise NotImplementedError("Use forward, backward and step in split learning.")
