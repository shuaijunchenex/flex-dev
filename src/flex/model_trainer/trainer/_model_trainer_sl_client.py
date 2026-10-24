from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from ...ml_utils.tqdm_utils import pbar

from flex.model_trainer.model_trainer_args import ModelTrainerArgs

from ..model_trainer import ModelTrainer
from ...ml_utils.model_utils import ModelUtils


ActivationGrad = torch.Tensor
ServerLoss = float
ServerMetrics = Dict[str, float]
ServerStepFn = Callable[[torch.Tensor, torch.Tensor, bool], Tuple[ActivationGrad, ServerLoss, ServerMetrics]]


class ModelTrainer_SlClient(ModelTrainer):
	"""Split-learning client trainer handling the front model segment."""

	def __init__(self, trainer_args: ModelTrainerArgs):
		super().__init__(trainer_args)

		if trainer_args.model is None:
			raise ValueError("Training model is None.")
		if trainer_args.optimizer is None:
			raise ValueError("Training optimizer is None.")

		self.device = trainer_args.device or ModelUtils.accelerator_device()
		self.model: nn.Module = trainer_args.model.to(self.device)
		self._server_step: Optional[ServerStepFn] = None
		self._cached_activations: List[torch.Tensor] = []

	def set_model(self, model: nn.Module):
		self.trainer_args.model = model
		if str(next(model.parameters()).device) != self.trainer_args.device:
			self.trainer_args.model = model.to(self.trainer_args.device)
		self.model = self.trainer_args.model
		return self

	def attach_server_step(self, server_step: ServerStepFn) -> None:
		"""Registers the callable used to execute the server-side step."""

		self._server_step = server_step

	def _ensure_server_step(self) -> ServerStepFn:
		if self._server_step is None:
			raise RuntimeError("Server step callable is not attached to SL client trainer.")
		return self._server_step

	@staticmethod
	def _accumulate_metrics(accumulator: Dict[str, float], metrics: ServerMetrics) -> None:
		for key, value in metrics.items():
			if isinstance(value, (int, float)):
				accumulator[key] = accumulator.get(key, 0.0) + float(value)

	@staticmethod
	def _average_metrics(accumulator: Dict[str, float], denominator: int) -> Dict[str, float]:
		if denominator <= 0:
			return {key: 0.0 for key in accumulator}
		return {key: value / denominator for key, value in accumulator.items()}

	def forward_only(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
		ta = self.trainer_args
		train_dl = ta.train_loader.data_loader if hasattr(ta.train_loader, "data_loader") else ta.train_loader
		if not hasattr(train_dl, "__iter__"):
			raise TypeError(f"train_loader must be iterable, got {type(train_dl).__name__}")

		ta.model.to(self.device)
		ta.model.train()

		self._cached_activations = []
		forward_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []

		loop = pbar(
			train_dl,
			desc="SL Client Forward",
			leave=False,
			ncols=120,
			mininterval=0.1,
			bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
		)

		for inputs, labels in loop:
			inputs = inputs.to(self.device)
			smashed_data = ta.model(inputs) # activation
			smashed_data = smashed_data.to(self.device)
			smashed_data.retain_grad()
			self._cached_activations.append(smashed_data)
			forward_outputs.append((smashed_data.detach(), labels))

		return forward_outputs

	def forward_batch(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Forward a single batch and cache the activation for later backward.

		This enables interleaved per-batch forward/backward (true mini-batch SL),
		avoiding the stale-activation incoherence of the bulk forward_only approach.
		The caller must call backward_batch() with the server activation gradient
		immediately after the server backward step for this batch.
		"""
		ta = self.trainer_args
		ta.model.to(self.device)
		ta.model.train()

		inputs = inputs.to(self.device)
		smashed_data = ta.model(inputs)
		smashed_data = smashed_data.to(self.device)
		smashed_data.retain_grad()
		# Overwrite cache with single-item list; interleaved callers process one at a time.
		self._cached_activations = [smashed_data]
		return smashed_data.detach()

	def backward_batch(self, activation_grad: torch.Tensor) -> float:
		"""Backward a single batch using the server-provided activation gradient.

		Must be called immediately after forward_batch() for the same batch.
		Returns the gradient norm for that batch.
		"""
		optimizer = self.trainer_args.optimizer
		if optimizer is None:
			raise ValueError("Optimizer is required for backward_batch.")
		if not self._cached_activations:
			raise RuntimeError("backward_batch called without a preceding forward_batch.")

		act = self._cached_activations[0]
		if not isinstance(activation_grad, torch.Tensor):
			raise TypeError("activation_grad must be a torch.Tensor.")

		optimizer.zero_grad()
		act.backward(activation_grad.to(self.device))
		grad_norm = float(activation_grad.norm().item())
		optimizer.step()
		optimizer.zero_grad()

		self._cached_activations = []
		return grad_norm

	def backward_only(self, activation_grads: Iterable[torch.Tensor]) -> Dict[str, float]:
		optimizer = self.trainer_args.optimizer
		if optimizer is None:
			raise ValueError("Optimizer is required for backward_only.")

		grads = list(activation_grads)
		if len(grads) != len(self._cached_activations):
			raise ValueError("Activation gradients count does not match cached activations.")

		optimizer.zero_grad()
		grad_norm_sum = 0.0

		for act, grad in zip(self._cached_activations, grads):
			if not isinstance(grad, torch.Tensor):
				raise TypeError("Each activation gradient must be a torch.Tensor.")
			act.backward(grad.to(self.device))
			grad_norm_sum += float(grad.norm().item())
			optimizer.step()
			optimizer.zero_grad()

		self._cached_activations.clear()
		batch_count = max(len(grads), 1)
		return {"avg_grad_norm": grad_norm_sum / batch_count}

	def train(self, epochs: int) -> Any:
		raise NotImplementedError("Use forward_only and backward_only in split-learning workflow.")

	def train_step(self) -> float:
		raise NotImplementedError("Use forward_only/backward_only in split-learning workflow.")

	def observe(self, epochs: int = 1) -> Any:
		raise NotImplementedError("Use forward_only for client forward pass in SL observation.")

