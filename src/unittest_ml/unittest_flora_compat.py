"""Focused compatibility tests for FLoRA aggregation and backbone merging."""

from __future__ import annotations

import os
import unittest
from collections import OrderedDict

import torch
import torch.nn as nn

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from flex.fl_algorithms.aggregation.methods._fed_aggregator_flora import (  # noqa: E402
    FedAggregator_Flora,
)
from flex.fl_algorithms.aggregation.fed_aggregator_args import (  # noqa: E402
    FedAggregatorArgs,
)
from flex.ml_algorithms.lora.lora_utils import LoRAUtils  # noqa: E402
from flex.ml_algorithms.lora.impl.lora_ms import MSLoRALinear  # noqa: E402
from flex.fed_strategy.client_strategy_impl._flora_client import (  # noqa: E402
    FloraClientTrainingStrategy,
)


class TestFloraFreshLoRAInitialization(unittest.TestCase):
    def test_zero_initial_delta_has_nonzero_first_gradient(self):
        torch.manual_seed(7)
        model = nn.Sequential(
            OrderedDict(
                {
                    "adapter": MSLoRALinear(
                        in_features=4,
                        out_features=3,
                        r=2,
                        lora_alpha=2,
                        merge_weights=False,
                    )
                }
            )
        )

        both_zero = OrderedDict(
            (key, torch.zeros_like(value) if "lora_" in key else value.clone())
            for key, value in model.state_dict().items()
        )
        fresh_state = FloraClientTrainingStrategy._reset_lora_state(
            model,
            both_zero,
        )
        model.load_state_dict(fresh_state, strict=True)

        adapter = model.adapter
        self.assertGreater(torch.linalg.vector_norm(adapter.lora_A).item(), 0.0)
        self.assertEqual(torch.count_nonzero(adapter.lora_B).item(), 0)
        self.assertEqual(
            torch.count_nonzero(adapter.lora_B @ adapter.lora_A).item(),
            0,
        )

        loss = model(torch.randn(5, 4)).sum()
        loss.backward()

        self.assertIsNotNone(adapter.lora_B.grad)
        self.assertGreater(torch.linalg.vector_norm(adapter.lora_B.grad).item(), 0.0)


class TestFloraBackboneMerge(unittest.TestCase):
    def test_conv_delta_is_reshaped_to_kernel_layout(self):
        backbone_weight = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
        flat_delta = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10.0

        merged = LoRAUtils.merge_flora_delta_to_backbone(
            {"conv.sp_aggregated": flat_delta},
            {"conv.weight": backbone_weight},
        )

        expected = backbone_weight + flat_delta.reshape_as(backbone_weight)
        self.assertEqual(tuple(merged["conv.weight"].shape), (2, 3, 2, 2))
        torch.testing.assert_close(merged["conv.weight"], expected)

    def test_embedding_delta_is_transposed_to_weight_layout(self):
        backbone_weight = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        product_delta = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 10.0

        merged = LoRAUtils.merge_flora_delta_to_backbone(
            {"embedding.sp_aggregated": product_delta},
            {"embedding.weight": backbone_weight},
        )

        expected = backbone_weight + product_delta.transpose(0, 1)
        self.assertEqual(tuple(merged["embedding.weight"].shape), (5, 3))
        torch.testing.assert_close(merged["embedding.weight"], expected)

    def test_linear_delta_keeps_native_layout(self):
        backbone_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        product_delta = torch.full((2, 3), 0.5)

        merged = LoRAUtils.merge_flora_delta_to_backbone(
            {"linear.sp_aggregated": product_delta},
            {"linear.weight": backbone_weight},
        )

        torch.testing.assert_close(
            merged["linear.weight"],
            backbone_weight + product_delta,
        )

    def test_incompatible_delta_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Cannot align FLoRA update"):
            LoRAUtils.merge_flora_delta_to_backbone(
                {"broken.sp_aggregated": torch.zeros(2, 5)},
                {"broken.weight": torch.zeros(2, 3, 2, 2)},
            )


class TestFloraNonLoraAggregation(unittest.TestCase):
    def test_integer_buffer_is_copied_while_float_tensor_is_averaged(self):
        first = OrderedDict(
            {
                "layer.weight": torch.full((2, 3), 1.0),
                "layer.lora_A": torch.ones(1, 3),
                "layer.lora_B": torch.ones(2, 1),
                "bn.num_batches_tracked": torch.tensor(7, dtype=torch.int64),
                "feature_mask": torch.tensor([True, False]),
            }
        )
        second = OrderedDict(
            {
                "layer.weight": torch.full((2, 3), 5.0),
                "layer.lora_A": torch.full((1, 3), 2.0),
                "layer.lora_B": torch.full((2, 1), 3.0),
                "bn.num_batches_tracked": torch.tensor(19, dtype=torch.int64),
                "feature_mask": torch.tensor([False, True]),
            }
        )

        aggregator = FedAggregator_Flora(
            FedAggregatorArgs({"method": "flora", "device": "cpu"})
        )
        aggregated = aggregator._aggregate_state_dicts(
            [first, second],
            weights=[1.0, 3.0],
        )

        torch.testing.assert_close(
            aggregated["layer.weight"],
            torch.full((2, 3), 4.0),
        )
        self.assertEqual(aggregated["bn.num_batches_tracked"].dtype, torch.int64)
        self.assertEqual(aggregated["bn.num_batches_tracked"].item(), 7)
        self.assertTrue(torch.equal(aggregated["feature_mask"], first["feature_mask"]))


if __name__ == "__main__":
    unittest.main()
