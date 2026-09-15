from __future__ import annotations

import copy
import os
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from flex.fed_strategy.server_strategy_impl._close_optimal_server import (
    CloseOptimalServerStrategy,
)
from flex.fed_strategy.strategy_args import StrategyArgs
from flex.fl_algorithms.aggregation.fed_aggregator_args import FedAggregatorArgs
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from flex.ml_algorithms.lora.impl.lora_ms import MSLoRALinear


class _ToyLoRAModel(nn.Module):
    def __init__(self, rank: int, alpha: int):
        super().__init__()
        self.layer = MSLoRALinear(
            3,
            3,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            merge_weights=False,
        )
        self.lora_config = {
            "suffix_A": "lora_A",
            "suffix_B": "lora_B",
            "sp_suffix": "sp_aggregated",
        }


class _Client:
    def __init__(self, node_id: str, model: nn.Module):
        self.node_id = node_id
        self.node_var = SimpleNamespace(
            model=model,
            model_weight=copy.deepcopy(model.state_dict()),
            cache_weight=None,
        )

    def receive_weight(self, weight):
        self.node_var.cache_weight = weight

    def set_local_weight(self):
        self.node_var.model_weight = self.node_var.cache_weight


class _Evaluator:
    def __init__(self):
        self.weight = None

    def update_model(self, weight):
        self.weight = weight


def _truncated_svd(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * singular[:rank].unsqueeze(0)) @ vh[:rank, :]


class CloseOptimalServerScalingTest(unittest.TestCase):
    def test_effective_delta_is_preserved_for_server_and_clients(self):
        client_1 = _Client("client_1", _ToyLoRAModel(rank=1, alpha=4))
        client_2 = _Client("client_2", _ToyLoRAModel(rank=2, alpha=4))
        server_model = _ToyLoRAModel(rank=2, alpha=6)

        with torch.no_grad():
            client_1.node_var.model.layer.weight.zero_()
            client_1.node_var.model.layer.lora_A.copy_(
                torch.tensor([[1.0, 2.0, 0.0]])
            )
            client_1.node_var.model.layer.lora_B.copy_(
                torch.tensor([[1.0], [0.0], [2.0]])
            )

            client_2.node_var.model.layer.weight.zero_()
            client_2.node_var.model.layer.lora_A.copy_(
                torch.tensor([[0.0, 1.0, 1.0], [2.0, 0.0, 1.0]])
            )
            client_2.node_var.model.layer.lora_B.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
            )

        state_1 = copy.deepcopy(client_1.node_var.model.state_dict())
        state_2 = copy.deepcopy(client_2.node_var.model.state_dict())
        original_b_1 = state_1["layer.lora_B"].clone()
        original_b_2 = state_2["layer.lora_B"].clone()

        updates = [
            {
                "updated_weights": state_1,
                "train_record": {"node_id": "client_1", "data_sample_num": 1},
            },
            {
                "updated_weights": state_2,
                "train_record": {"node_id": "client_2", "data_sample_num": 1},
            },
        ]
        aggregator = FedAggregatorFactory.create_aggregator(
            FedAggregatorArgs(
                {"method": "close_optimal", "device": "cpu", "lambda": 1.0}
            )
        )
        node_var = SimpleNamespace(
            model=server_model,
            inference_model=server_model,
            aggregation_method=aggregator,
            client_updates=updates,
            aggregated_weight=None,
            model_weight=copy.deepcopy(server_model.state_dict()),
            model_evaluator=_Evaluator(),
        )
        server = SimpleNamespace(
            node_var=node_var,
            client_nodes=[client_1, client_2],
        )
        args = StrategyArgs(
            {
                "role": "server",
                "strategy_name": "close_optimal",
                "maintain_lora_scale_ratio": True,
            }
        )
        strategy = CloseOptimalServerStrategy(args, server)

        expected = 0.5 * (
            client_1.node_var.model.layer.scaling
            * (state_1["layer.lora_B"] @ state_1["layer.lora_A"])
            + client_2.node_var.model.layer.scaling
            * (state_2["layer.lora_B"] @ state_2["layer.lora_A"])
        )

        strategy.aggregation()
        torch.testing.assert_close(
            node_var.aggregated_weight["layer.sp_aggregated"], expected
        )
        torch.testing.assert_close(state_1["layer.lora_B"], original_b_1)
        torch.testing.assert_close(state_2["layer.lora_B"], original_b_2)

        strategy.apply_weight()
        server_effective = server_model.layer.scaling * (
            node_var.model_evaluator.weight["layer.lora_B"]
            @ node_var.model_evaluator.weight["layer.lora_A"]
        )
        torch.testing.assert_close(server_effective, _truncated_svd(expected, 2))

        strategy.broadcast()
        for client, rank in ((client_1, 1), (client_2, 2)):
            local_weight = client.node_var.model_weight
            local_effective = client.node_var.model.layer.scaling * (
                local_weight["layer.lora_B"] @ local_weight["layer.lora_A"]
            )
            torch.testing.assert_close(
                local_effective,
                _truncated_svd(expected, rank),
                rtol=1e-5,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
