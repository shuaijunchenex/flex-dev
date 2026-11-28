"""FedEx-LoRA numerical identities and the standard round lifecycle."""

from __future__ import annotations

import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from flex.fed_strategy.client_strategy_impl._fedex_lora_client import (
    FedExLoRAClientTrainingStrategy,
)
from flex.fed_strategy.server_strategy_impl._fedex_lora_server import FedExLoRAServerStrategy
from flex.fl_algorithms.aggregation.fed_aggregator_args import FedAggregatorArgs
from flex.fl_algorithms.aggregation.methods._fed_aggregator_fedex_lora import (
    FedAggregator_FedExLoRA,
)
from flex.fl_algorithms.aggregation.methods._fed_aggregator_flora import FedAggregator_Flora
from flex.ml_algorithms.lora.fedex_lora import FedExBroadcast
from flex.ml_algorithms.lora.impl.lora_ms import MSLoRALinear


def _model(rank=1, alpha=3):
    model = nn.Sequential(
        MSLoRALinear(2, 3, r=rank, lora_alpha=alpha, merge_weights=False, bias=False),
        nn.Linear(3, 2),
    )
    return model.double()


class _Client:
    def __init__(self, node_id, model):
        self.node_id = node_id
        self.node_var = SimpleNamespace(
            model=model, model_weight=model.state_dict(), data_sample_num=3
        )
        self.strategy = FedExLoRAClientTrainingStrategy({}, self)

    def receive_weight(self, weight):
        self.strategy.receive_weight(weight)

    def set_local_weight(self):
        self.strategy.set_local_weight()


class _Evaluator:
    def __init__(self, model):
        self.model = model

    def update_model(self, state):
        self.model.load_state_dict(state, strict=True)


class _Server:
    def __init__(self, model, weighting_mode):
        args = FedAggregatorArgs({"weighting_mode": weighting_mode})
        self.node_var = SimpleNamespace(
            model=model,
            model_weight=model.state_dict(),
            model_evaluator=_Evaluator(copy.deepcopy(model)),
            aggregation_method=FedAggregator_FedExLoRA(args),
            training_logger=SimpleNamespace(begin=lambda _: None),
            client_updates=[],
        )

    def set_client_nodes(self, clients):
        self.client_nodes = clients


class FedExLoRATest(unittest.TestCase):
    def setUp(self):
        self.a = [
            torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            torch.tensor([[3.0, -1.0]], dtype=torch.float64),
        ]
        self.b = [
            torch.tensor([[1.0], [2.0], [1.0]], dtype=torch.float64),
            torch.tensor([[2.0], [0.0], [-1.0]], dtype=torch.float64),
        ]

    def _setup_round(self, weighting_mode, *, alpha=3):
        server = _Server(_model(alpha=alpha), weighting_mode)
        clients = [_Client("client1", _model(alpha=alpha)), _Client("client2", _model(alpha=alpha))]
        strategy = FedExLoRAServerStrategy({}, server)
        strategy.prepare({}, clients)
        strategy.broadcast()
        return server, clients, strategy

    def _uploads(self, clients, counts):
        uploads = []
        for client, a, b, count in zip(clients, self.a, self.b, counts):
            state = copy.deepcopy(client.node_var.model_weight)
            state["0.lora_A"] = a
            state["0.lora_B"] = b
            state["1.weight"].zero_()
            uploads.append({
                "updated_weights": state,
                "train_record": {
                    "node_id": client.node_id,
                    "base_version": client.strategy._version,
                    "data_sample_num": count,
                },
            })
        return uploads

    def test_scaled_exact_aggregation_and_round_commit(self):
        for weighting_mode, probabilities in (("uniform", (0.5, 0.5)), ("samples", (0.25, 0.75))):
            with self.subTest(weighting_mode=weighting_mode):
                server, clients, strategy = self._setup_round(weighting_mode)
                expected_a = sum(p * a for p, a in zip(probabilities, self.a))
                expected_b = sum(p * b for p, b in zip(probabilities, self.b))
                effective_mean = sum(3 * p * (b @ a) for p, a, b in zip(probabilities, self.a, self.b))
                initial = copy.deepcopy(server.node_var.model_weight)
                for _ in range(2):
                    previous = server.node_var.model_weight
                    updates = self._uploads(clients, (1, 3))
                    strategy.receive_client_updates(updates)
                    strategy.aggregation()
                    strategy.apply_weight()
                    committed = server.node_var.model_weight
                    self.assertIs(server.node_var.aggregated_weight, committed)
                    torch.testing.assert_close(committed["0.lora_A"], expected_a)
                    torch.testing.assert_close(committed["0.lora_B"], expected_b)
                    torch.testing.assert_close(
                        committed["0.weight"] + 3 * committed["0.lora_B"] @ committed["0.lora_A"],
                        previous["0.weight"] + effective_mean,
                    )
                    torch.testing.assert_close(committed["1.weight"], initial["1.weight"])
                    strategy.apply_weight()
                    self.assertIs(server.node_var.model_weight, committed)
                    with self.assertRaisesRegex(ValueError, "Stale"):
                        strategy.receive_client_updates(updates)
                    strategy.broadcast()
                    for client in clients:
                        torch.testing.assert_close(client.node_var.model_weight, committed)
                        self.assertIsNot(client.node_var.model_weight["0.weight"], committed["0.weight"])
                    torch.testing.assert_close(server.node_var.model_evaluator.model.state_dict(), committed)

    def test_single_client_has_zero_residual(self):
        result = FedAggregator_FedExLoRA.compute_fedex_layer(self.a[:1], self.b[:1], [5], scaling=3.0)
        torch.testing.assert_close(result.residual, torch.zeros_like(result.residual))

    def test_same_uploads_match_flora_at_unit_scale(self):
        server, clients, strategy = self._setup_round("samples", alpha=1)
        uploads = self._uploads(clients, (1, 3))
        previous = server.node_var.model_weight["0.weight"]
        flora_update = FedAggregator_Flora().aggregate(uploads)["0.sp_aggregated"]

        strategy.receive_client_updates(uploads)
        strategy.aggregation()
        strategy.apply_weight()
        committed = server.node_var.model_weight
        effective = committed["0.weight"] + committed["0.lora_B"] @ committed["0.lora_A"]
        torch.testing.assert_close(effective, previous + flora_update, rtol=1e-6, atol=1e-6)

    def test_zero_sample_client_does_not_affect_uniform_mean(self):
        server, clients, strategy = self._setup_round("uniform")
        before = server.node_var.model_weight
        strategy.receive_client_updates(self._uploads(clients, (0, 0)))
        strategy.aggregation()
        strategy.apply_weight()
        self.assertIs(server.node_var.model_weight, before)
        self.assertEqual(strategy._version, 0)

        strategy.receive_client_updates(self._uploads(clients, (0, 3)))
        strategy.aggregation()
        strategy.apply_weight()
        after = server.node_var.model_weight
        torch.testing.assert_close(after["0.weight"], before["0.weight"])
        torch.testing.assert_close(after["0.lora_A"], self.a[1])
        torch.testing.assert_close(after["0.lora_B"], self.b[1])

    def test_profile_must_match_rank_and_scale(self):
        for model in (_model(rank=2), _model(alpha=7)):
            with self.subTest(rank=model[0].r, scale=model[0].scaling):
                strategy = FedExLoRAServerStrategy({}, _Server(_model(), "uniform"))
                with self.assertRaisesRegex(ValueError, "common rank and effective scaling"):
                    strategy.prepare({}, [_Client("client1", model)])

    def test_client_rejects_older_model_version(self):
        server, clients, _ = self._setup_round("uniform")
        state = server.node_var.model_weight
        clients[0].receive_weight(FedExBroadcast(2, state))
        clients[0].set_local_weight()
        with self.assertRaisesRegex(ValueError, "older"):
            clients[0].receive_weight(FedExBroadcast(1, state))

    def test_client_uploads_only_adapters_and_ignores_same_version(self):
        server, clients, strategy = self._setup_round("uniform")
        client = clients[0]
        trained_state = copy.deepcopy(client.node_var.model_weight)
        trained_state["0.lora_B"] = self.b[0]

        def train():
            client.node_var.model_weight = trained_state
            return trained_state, {"avg_loss": 1.0}

        with patch.object(client.strategy, "local_training_step", side_effect=train):
            upload, record = client.strategy.run_local_training()
        self.assertEqual(set(upload), {"0.lora_A", "0.lora_B"})
        self.assertNotIn("updated_weights", record)
        self.assertEqual(record["base_version"], 0)
        self.assertEqual(record["data_sample_num"], 3)
        self.assertEqual(record["train_record"], {"avg_loss": 1.0})

        strategy.broadcast()
        self.assertIs(client.node_var.model_weight, trained_state)
        client.receive_weight(FedExBroadcast(1, server.node_var.model_weight))
        client.set_local_weight()
        torch.testing.assert_close(client.node_var.model_weight, server.node_var.model_weight)


if __name__ == "__main__":
    unittest.main()
