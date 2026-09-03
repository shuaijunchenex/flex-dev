from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch

from ..fed_aggregator_abc import AbstractFedAggregator, FedAggregatorArgs
from ....ml_utils import console


class SflAggregator_SflAvg(AbstractFedAggregator):
	"""Aggregates client gradients via simple arithmetic mean for split learning."""

	def __init__(self, args: FedAggregatorArgs | None = None):
		super().__init__(args)
		self._aggregation_method = "sflavg"

	def _before_aggregation(self) -> None:
		if not self._aggregation_data_dict:
			raise ValueError("SFL aggregation requires at least one client update.")

	def _do_aggregation(self) -> None:
		sample_gradient: Dict[str, torch.Tensor] = self._aggregation_data_dict[0][0]
		aggregated_gradients = OrderedDict()

		for key, tensor in sample_gradient.items():
			aggregated_gradients[key] = torch.zeros_like(tensor, device=self._device)

		client_count = len(self._aggregation_data_dict)
		console.debug(f"[SFL-Avg] Aggregating gradients from {client_count} clients...")

		for gradient_dict, _ in self._aggregation_data_dict:
			for key, tensor in gradient_dict.items():
				aggregated_gradients[key] += tensor.to(self._device)

		divisor = max(client_count, 1)
		for key in aggregated_gradients:
			aggregated_gradients[key] /= divisor

		self._aggregated_weight = aggregated_gradients

	def _after_aggregation(self) -> None:
		first_key = next(iter(self._aggregated_weight))
		console.debug(
			f"[SFL-Avg] Aggregation done. First gradient mean = {self._aggregated_weight[first_key].mean():.6f}"
		)

