from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
from ...ml_utils.tqdm_utils import pbar

from flex.model_trainer.model_trainer_args import ModelTrainerArgs

from ..model_trainer import ModelTrainer
from ...ml_utils.model_utils import ModelUtils
from ...ml_utils import console

class ModelTrainer_SlServer(ModelTrainer):
	"""Split-learning server trainer that updates the rear model segment."""

	def __init__(self, trainer_args: ModelTrainerArgs):
		super().__init__(trainer_args)

		if trainer_args.model is None:
			raise ValueError("Server model is None.")
		if trainer_args.optimizer is None:
			raise ValueError("Server optimizer is None.")
		if trainer_args.loss_func is None:
			raise ValueError("Server loss function is None.")

		self.device = trainer_args.device or ModelUtils.accelerator_device()
		self.model: nn.Module = trainer_args.model.to(self.device)
		self.debug = getattr(trainer_args, "debug", False)

	def set_model(self, model: nn.Module):
		self.trainer_args.model = model
		if str(next(model.parameters()).device) != self.trainer_args.device:
			self.trainer_args.model = model.to(self.trainer_args.device)
		self.model = self.trainer_args.model
		return self

	@staticmethod
	def _accumulate_metrics(accumulator: Dict[str, float], metrics: Dict[str, float]) -> None:
		for key, value in metrics.items():
			if isinstance(value, (int, float)):
				accumulator[key] = accumulator.get(key, 0.0) + float(value)

	@staticmethod
	def _average_metrics(accumulator: Dict[str, float], denominator: int) -> Dict[str, float]:
		if denominator <= 0:
			return {key: 0.0 for key in accumulator}
		return {key: value / denominator for key, value in accumulator.items()}

	def forward_only(
		self,
		smashed_data: torch.Tensor,
		labels: torch.Tensor,
		training: bool = True,
	) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
		ta = self.trainer_args
		server_input = smashed_data.to(self.device).detach().requires_grad_(True)
		labels = labels.to(self.device)

		ta.model.to(self.device)
		ta.model.train() if training else ta.model.eval()

		outputs = ta.model(server_input)
		loss = ta.loss_func(outputs, labels)

		metrics: Dict[str, float] = {"loss": float(loss.item())}
		if outputs.ndim > 1 and outputs.shape[-1] > 1:
			predictions = outputs.argmax(dim=1)
			accuracy = (predictions == labels).float().mean().item()
			metrics["accuracy"] = float(accuracy)

		return server_input, loss, metrics

	def backward_only(
		self,
		server_input: torch.Tensor,
		loss: torch.Tensor,
		training: bool = True,
	) -> Tuple[torch.Tensor, float]:
		ta = self.trainer_args
		optimizer = ta.optimizer
		if optimizer is None:
			raise ValueError("Optimizer is required for backward_only.")

		optimizer.zero_grad()
		loss.backward()
		activation_grad = server_input.grad.detach()

		if hasattr(self, "debug") and self.debug:
			console.debug(
				f"[SL-Server] loss={float(loss.item()):.4f}, grad_norm={float(activation_grad.norm().item()):.4f}")

		if training:
			optimizer.step()
		optimizer.zero_grad()

		activation_grad = activation_grad.to(server_input.device)
		return activation_grad, float(loss.item())

	def train_step(self) -> Tuple[float, Dict[str, float]]:
		ta = self.trainer_args
		train_dl = ta.train_loader.data_loader if hasattr(ta.train_loader, "data_loader") else ta.train_loader
		if not hasattr(train_dl, "__iter__"):
			raise TypeError(f"train_loader must be iterable, got {type(train_dl).__name__}")

		running_loss = 0.0
		total_batch = 0
		metrics_accumulator: Dict[str, float] = {}

		loop = pbar(
			train_dl,
			desc="SL Server Training",
			leave=False,
			ncols=120,
			mininterval=0.1,
			bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
		)

		for smashed_data, labels in loop:
			total_batch += 1
			server_input, loss, metrics = self.forward_only(smashed_data, labels, True)
			activation_grad, loss_value = self.backward_only(server_input, loss, True)
			running_loss += float(loss_value)
			self._accumulate_metrics(metrics_accumulator, metrics)
			loop.set_postfix(batch=total_batch, loss=f"{loss_value:.4f}")

		avg_loss = running_loss / max(total_batch, 1)
		return avg_loss, self._average_metrics(metrics_accumulator, total_batch)

	def train(self, epochs: int) -> Any:
		self.trainer_args.total_epochs = epochs

		train_stats = {
			"train_loss_sum": 0.0,
			"epoch_loss": [],
			"train_loss_power_two_sum": 0.0,
			"epoch_metrics": [],
		}

		for _ in range(epochs):
			epoch_loss, metrics = self.train_step()
			train_stats["train_loss_sum"] += epoch_loss
			train_stats["train_loss_power_two_sum"] += epoch_loss ** 2
			train_stats["epoch_loss"].append(epoch_loss)
			train_stats["epoch_metrics"].append(metrics)

		train_stats["avg_loss"] = train_stats["train_loss_sum"] / max(epochs, 1)
		train_stats["sqrt_train_loss_power_two_sum"] = train_stats["train_loss_power_two_sum"] ** 0.5

		return self.trainer_args.model.state_dict(), train_stats

	def train_step_from_remote(self, smashed_data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, float, Dict[str, float]]:
		server_input, loss, metrics = self.forward_only(smashed_data, labels, True)
		activation_grad, loss_value = self.backward_only(server_input, loss, True)
		return activation_grad, loss_value, metrics

	def observe(self, epochs: int = 1) -> Any:
		self.trainer_args.total_epochs = epochs

		train_dl = self.trainer_args.train_loader.data_loader if hasattr(self.trainer_args.train_loader, "data_loader") else self.trainer_args.train_loader
		if not hasattr(train_dl, "__iter__"):
			raise TypeError(f"train_loader must be iterable, got {type(train_dl).__name__}")

		running_loss = 0.0
		total_batch = 0
		metrics_accumulator: Dict[str, float] = {}

		for smashed_data, labels in train_dl:
			total_batch += 1
			server_input, loss, metrics = self.forward_only(smashed_data, labels, False)
			activation_grad, loss_value = self.backward_only(server_input, loss, False)
			running_loss += float(loss_value)
			self._accumulate_metrics(metrics_accumulator, metrics)

		avg_loss = running_loss / max(total_batch, 1)
		stats = {
			"avg_loss": avg_loss,
			"metrics": self._average_metrics(metrics_accumulator, total_batch),
		}
		return self.trainer_args.model.state_dict(), stats

