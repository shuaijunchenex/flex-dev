"""Small mathematical checks; no pretrained weights or datasets required."""

from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flex.fl_algorithms.aggregation.fed_aggregator_args import FedAggregatorArgs
from flex.fl_algorithms.aggregation.methods._fed_aggregator_florist import FedAggregator_Florist


def make_aggregator(threshold=0.9):
    return FedAggregator_Florist(FedAggregatorArgs({"device": "cpu", "threshold": threshold}))


def client(state, count):
    return {"updated_weights": state, "train_record": {"data_sample_num": count}}


def random_clients(ranks=(1, 2, 3), counts=(1, 3, 6), dims=(9, 8), dtype=torch.float64):
    generator = torch.Generator().manual_seed(617)
    out_dim, in_dim = dims
    return [client({
        "proj.lora_A": torch.randn(rank, in_dim, generator=generator, dtype=dtype),
        "proj.lora_B": torch.randn(out_dim, rank, generator=generator, dtype=dtype),
        "head.weight": torch.randn(2, out_dim, generator=generator, dtype=dtype),
    }, count) for rank, count in zip(ranks, counts)]


def dense_update(clients):
    total = sum(c["train_record"]["data_sample_num"] for c in clients)
    return sum((c["train_record"]["data_sample_num"] / total)
               * (c["updated_weights"]["proj.lora_B"] @ c["updated_weights"]["proj.lora_A"])
               for c in clients)


def product(state):
    return state["proj.lora_B"] @ state["proj.lora_A"]


class TestFloristAggregation(unittest.TestCase):
    def test_heterogeneous_weighted_update_and_head(self):
        clients = random_clients()
        original = deepcopy(clients)
        aggregator = make_aggregator(1)
        result = aggregator.aggregate(clients)
        torch.testing.assert_close(product(result), dense_update(clients), atol=1e-12, rtol=1e-12)
        expected_head = sum(c["updated_weights"]["head.weight"] * w
                            for c, w in zip(clients, (0.1, 0.3, 0.6)))
        torch.testing.assert_close(result["head.weight"], expected_head)
        self.assertEqual(result["proj.lora_A"].shape, (6, 8))
        self.assertEqual(result["proj.lora_A"].dtype, torch.float64)
        for old, current in zip(original, clients):
            for key, value in old["updated_weights"].items():
                torch.testing.assert_close(value, current["updated_weights"][key], rtol=0, atol=0)
        stats = aggregator.last_stats
        self.assertEqual(stats["layers"]["proj"]["input_stack_rank"], 6)
        self.assertEqual(stats["layers"]["proj"]["global_rank"], 6)
        self.assertEqual(stats["layers"]["proj"]["relative_truncation_error"], 0)
        self.assertEqual(stats["download_bytes"], sum(t.numel() * t.element_size() for t in result.values()))

    def test_threshold_uses_squared_singular_values_and_minimum_rank(self):
        state = {"proj.lora_A": torch.eye(3, dtype=torch.float64),
                 "proj.lora_B": torch.diag(torch.tensor([3., 2., 1.], dtype=torch.float64))}
        for tau, rank in ((0.6, 1), (0.9, 2), (1., 3)):
            with self.subTest(tau=tau):
                aggregator = make_aggregator(tau)
                result = aggregator.aggregate([client(state, 1)])
                self.assertEqual(result["proj.lora_A"].shape[0], rank)
                expected = state["proj.lora_B"].clone()
                expected[rank:, rank:] = 0
                torch.testing.assert_close(product(result), expected)
                error = torch.linalg.norm(expected - state["proj.lora_B"]) / torch.linalg.norm(state["proj.lora_B"])
                self.assertAlmostEqual(aggregator.last_stats["layers"]["proj"]["relative_truncation_error"], error.item())

    def test_matches_independent_dense_truncated_svd(self):
        clients = random_clients()
        dense = dense_update(clients)
        u, singular, vh = torch.linalg.svd(dense, full_matrices=False)
        energy = singular.square()
        rank = int(torch.searchsorted(energy.cumsum(0) / energy.sum(), 0.8).item()) + 1
        result = make_aggregator(0.8).aggregate(clients)
        expected = (u[:, :rank] * singular[:rank]) @ vh[:rank]
        torch.testing.assert_close(product(result), expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(result["proj.lora_A"].shape[0], rank)

    def test_threshold_boundary_and_independent_layer_ranks(self):
        state = {
            "proj.lora_A": torch.eye(2, dtype=torch.float64),
            "proj.lora_B": torch.diag(torch.tensor([2., 1.], dtype=torch.float64)),
            "other.lora_A": torch.eye(2, dtype=torch.float64),
            "other.lora_B": torch.eye(2, dtype=torch.float64),
        }
        for threshold, expected_rank in ((0.8, 1), (math.nextafter(0.8, 1), 2)):
            result = make_aggregator(threshold).aggregate([client(state, 1)])
            self.assertEqual(result["proj.lora_A"].shape[0], expected_rank)
            self.assertEqual(result["other.lora_A"].shape[0], 2)

    def test_tau_one_keeps_thin_dimensions_of_rank_deficient_update(self):
        state = {"proj.lora_A": torch.eye(3, dtype=torch.float64),
                 "proj.lora_B": torch.diag(torch.tensor([3., 0., 0.], dtype=torch.float64))}
        result = make_aggregator(1).aggregate([client(state, 1)])
        self.assertEqual(result["proj.lora_A"].shape[0], 3)
        torch.testing.assert_close(product(result), state["proj.lora_B"])

    def test_three_thin_svds_without_dense_update_decomposition(self):
        clients = random_clients()
        shapes, original_svd = [], torch.linalg.svd
        def record(value, **kwargs):
            self.assertFalse(kwargs["full_matrices"])
            shapes.append(tuple(value.shape))
            return original_svd(value, **kwargs)
        with patch("torch.linalg.svd", side_effect=record):
            make_aggregator().aggregate(clients)
        self.assertEqual(shapes, [(9, 6), (6, 8), (6, 6)])

    def test_zero_update_has_explicit_rank_one_factors(self):
        clients = random_clients()
        for c in clients:
            c["updated_weights"]["proj.lora_B"].zero_()
        for tau in (0.9, 1.0):
            aggregator = make_aggregator(tau)
            result = aggregator.aggregate(clients)
            self.assertEqual(tuple(result["proj.lora_A"].shape), (1, 8))
            self.assertEqual(tuple(result["proj.lora_B"].shape), (9, 1))
            self.assertEqual(torch.count_nonzero(result["proj.lora_A"]).item(), 0)
            self.assertEqual(torch.count_nonzero(result["proj.lora_B"]).item(), 0)
            self.assertEqual(aggregator.last_stats["layers"]["proj"]["retained_energy"], 1)

    def test_client_order_does_not_change_effective_update(self):
        clients = random_clients()
        expected = product(make_aggregator(0.85).aggregate(clients))
        actual = product(make_aggregator(0.85).aggregate(clients[::-1]))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_stack_rank_exceeding_matrix_dimensions(self):
        clients = random_clients(ranks=(3, 4, 5), dims=(4, 5))
        result = make_aggregator(1).aggregate(clients)
        self.assertEqual(result["proj.lora_A"].shape[0], 4)
        torch.testing.assert_close(product(result), dense_update(clients), atol=1e-12, rtol=1e-12)

    def test_fp32_and_half_inputs_use_fp32_aggregation(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            clients = random_clients(dtype=dtype)
            result = make_aggregator(1).aggregate(clients)
            fp32_clients = deepcopy(clients)
            for c in fp32_clients:
                c["updated_weights"] = {k: v.float() for k, v in c["updated_weights"].items()}
            self.assertEqual(result["proj.lora_A"].dtype, torch.float32)
            torch.testing.assert_close(product(result), dense_update(fp32_clients), atol=1e-5, rtol=1e-5)

    def test_zero_weight_client_is_allowed(self):
        clients = random_clients(counts=(0, 0, 5))
        result = make_aggregator(1).aggregate(clients)
        torch.testing.assert_close(product(result), dense_update(clients), atol=1e-12, rtol=1e-12)

    def test_invalid_thresholds(self):
        for tau in (0, -0.1, 1.1, float("nan"), float("inf"), True, "0.9"):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                make_aggregator(tau)

    def test_invalid_counts_and_payloads(self):
        for count in (-1, float("nan"), float("inf"), True, "1"):
            clients = random_clients()
            clients[0]["train_record"]["data_sample_num"] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                make_aggregator().aggregate(clients)
        for invalid in ([], {}, [{}], random_clients(counts=(0, 0, 0))):
            with self.assertRaises(ValueError):
                make_aggregator().aggregate(invalid)

    def test_invalid_keys_pairs_shapes_and_types(self):
        def remove_partner(clients):
            for c in clients:
                del c["updated_weights"]["proj.lora_B"]
        def mismatch_keys(clients):
            del clients[0]["updated_weights"]["head.weight"]
        def mismatch_rank(clients):
            clients[0]["updated_weights"]["proj.lora_B"] = torch.ones(9, 2)
        def mismatch_dimensions(clients):
            clients[0]["updated_weights"]["proj.lora_A"] = torch.ones(1, 7)
        def bad_head_shape(clients):
            clients[0]["updated_weights"]["head.weight"] = torch.ones(3, 9)
        def bad_dtype(clients):
            clients[0]["updated_weights"]["proj.lora_A"] = torch.ones(1, 8, dtype=torch.int64)
        def empty_rank(clients):
            clients[0]["updated_weights"]["proj.lora_A"] = torch.ones(0, 8)
        def bad_dimensions(clients):
            clients[0]["updated_weights"]["proj.lora_A"] = torch.ones(1, 8, 1)
        for mutate in (remove_partner, mismatch_keys, mismatch_rank, mismatch_dimensions,
                       bad_head_shape, bad_dtype, empty_rank, bad_dimensions):
            clients = random_clients()
            mutate(clients)
            with self.subTest(case=mutate.__name__), self.assertRaises(ValueError):
                make_aggregator().aggregate(clients)


if __name__ == "__main__":
    unittest.main()
