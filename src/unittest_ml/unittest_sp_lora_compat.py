"""
Unit test: SP/Close-Optimal aggregation compatibility with LoRA models.

Verifies that:
  1. LoRA models (MLP, CNN, TinyBERT) can be created via NNModelFactory.
  2. SP aggregation produces correct sp_aggregated keys without losing non-LoRA params.
  3. svd_split_global_weight reconstructs a state_dict compatible with
     load_state_dict(strict=True).
  4. Close-Optimal aggregation follows the same compatible path.
  5. The new _lora_config model attribute is populated correctly.

Run:
    cd unittest_ml
    python unittest_sp_lora_compat.py
"""

from __future__ import annotations

import copy
import os
import unittest

import torch
import torch.nn as nn

# ------------------------------------------------------------------
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------------------------------------------

from flex.ml_models.nn_model_factory import NNModelFactory
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from flex.fl_algorithms.aggregation.fed_aggregator_args import FedAggregatorArgs
from flex.ml_algorithms.lora.lora_utils import LoRAUtils


# ======================================================================
# Helpers
# ======================================================================

def _perturb_state_dict(sd: dict, scale: float = 0.01) -> dict:
    """Add small Gaussian noise to floating-point trainable params."""
    new_sd = {}
    for k, v in sd.items():
        if v.is_floating_point() and v.requires_grad:
            new_sd[k] = v + torch.randn_like(v) * scale
        else:
            new_sd[k] = v.clone().detach()
    return new_sd


def _simulate_sp_aggregation(state_dicts: list[dict], weights: list[float] | None = None):
    """Run SP aggregation on a list of client state_dicts."""
    if weights is None:
        weights = [1.0] * len(state_dicts)

    # Build the format expected by the SP aggregator
    client_data = [
        {"updated_weights": sd, "train_record": {"data_sample_num": w}}
        for sd, w in zip(state_dicts, weights)
    ]

    agg_args = FedAggregatorArgs({"method": "sp", "device": "cpu"})
    aggregator = FedAggregatorFactory.create_aggregator(agg_args)
    return aggregator.aggregate(client_data)


def _simulate_close_optimal_aggregation(state_dicts: list[dict], weights: list[float] | None = None):
    """Run Close-Optimal aggregation on a list of client state_dicts."""
    if weights is None:
        weights = [1.0] * len(state_dicts)

    client_data = [
        {"updated_weights": sd, "train_record": {"data_sample_num": w}}
        for sd, w in zip(state_dicts, weights)
    ]

    agg_args = FedAggregatorArgs({"method": "close_optimal", "device": "cpu", "lambda": 0.5})
    aggregator = FedAggregatorFactory.create_aggregator(agg_args)
    return aggregator.aggregate(client_data)


def _round_trip_and_verify(
    test_case: unittest.TestCase,
    model: nn.Module,
    aggregator_name: str,
    aggregation_fn,
):
    """
    Full round-trip test:
      1. Snapshot original state_dict
      2. Generate 3 perturbed client state_dicts
      3. Aggregate
      4. Reconstruct via svd_split_global_weight
      5. load_state_dict(strict=True) must succeed
      6. Verify key counts match
    """
    original_sd = copy.deepcopy(model.state_dict())
    original_keys = set(original_sd.keys())

    # Generate 3 client state_dicts with different perturbations
    torch.manual_seed(42)
    client_sds = [_perturb_state_dict(original_sd, scale=0.01 * (i + 1)) for i in range(3)]
    client_weights = [10.0, 20.0, 30.0]

    # --- Aggregate ---
    aggregated = aggregation_fn(client_sds, client_weights)
    agg_keys = set(aggregated.keys())

    # sp_aggregated keys must exist for each LoRA prefix
    sp_keys = {k for k in agg_keys if k.endswith(".sp_aggregated")}
    test_case.assertGreater(
        len(sp_keys), 0,
        f"[{aggregator_name}] Expected at least one .sp_aggregated key in aggregated output"
    )

    # Original lora_A / lora_B keys must NOT be in aggregated output (SP design)
    lora_keys_in_agg = {k for k in agg_keys if k.endswith(".lora_A") or k.endswith(".lora_B")}
    test_case.assertEqual(
        len(lora_keys_in_agg), 0,
        f"[{aggregator_name}] Aggregated output should NOT contain lora_A/lora_B keys"
    )

    # --- Reconstruct ---
    rank_dict = LoRAUtils.get_lora_ranks(model)
    lora_cfg = getattr(model, "lora_config", None) or {}
    reconstructed = LoRAUtils.svd_split_global_weight(
        aggregated,
        rank_dict,
        lora_suffix_A=lora_cfg.get("suffix_A", "lora_A"),
        lora_suffix_B=lora_cfg.get("suffix_B", "lora_B"),
        sp_suffix=lora_cfg.get("sp_suffix", "sp_aggregated"),
    )
    recon_keys = set(reconstructed.keys())

    # All original keys must be present in the reconstructed dict
    missing = original_keys - recon_keys
    test_case.assertEqual(
        len(missing), 0,
        f"[{aggregator_name}] Missing keys after reconstruction: {sorted(missing)}"
    )

    # No sp_aggregated keys should remain
    sp_in_recon = {k for k in recon_keys if k.endswith(".sp_aggregated")}
    test_case.assertEqual(
        len(sp_in_recon), 0,
        f"[{aggregator_name}] Reconstructed dict should NOT contain sp_aggregated keys"
    )

    # --- strict load ---
    model_copy = copy.deepcopy(model)
    try:
        model_copy.load_state_dict(reconstructed, strict=True)
    except RuntimeError as e:
        test_case.fail(
            f"[{aggregator_name}] load_state_dict(strict=True) failed: {e}"
        )

    # --- Verify key count matches ---
    test_case.assertEqual(
        len(original_keys), len(recon_keys),
        f"[{aggregator_name}] Key count mismatch: "
        f"original={len(original_keys)}, reconstructed={len(recon_keys)}"
    )

    return True


# ======================================================================
# Test class
# ======================================================================

class TestSPLoRACompatibility(unittest.TestCase):
    """Test SP aggregation round-trip for all LoRA model types."""

    # ------------------------------------------------------------------
    # Simple LoRA MLP
    # ------------------------------------------------------------------
    def test_simple_lora_mlp_sp(self):
        """Simple LoRA MLP → SP aggregate → reconstruct → strict load."""
        args = NNModelFactory.create_args({
            "nn_model": {
                "name": "simple_lora_mlp",
                "lora_rank": 4,
                "lora_scaling": 0.5,
                "rank_ratio": 1,
                "use_bias": True,
            }
        })
        model = NNModelFactory.create(args)

        # Verify lora_config is set
        self.assertIsNotNone(model.lora_config, "simple_lora_mlp should have lora_config")
        self.assertFalse(
            model.lora_config.get("has_non_lora_params", True),
            "simple_lora_mlp has_no_non_lora_params should be True (all LoRA-wrapped)"
        )

        _round_trip_and_verify(self, model, "SP/simple_lora_mlp", _simulate_sp_aggregation)
        print("  ✅ simple_lora_mlp  SP round-trip passed")

    def test_simple_lora_mlp_close_optimal(self):
        """Simple LoRA MLP → Close-Optimal aggregate → reconstruct → strict load."""
        args = NNModelFactory.create_args({
            "nn_model": {
                "name": "simple_lora_mlp",
                "lora_rank": 4,
                "lora_scaling": 0.5,
                "rank_ratio": 1,
                "use_bias": True,
            }
        })
        model = NNModelFactory.create(args)
        _round_trip_and_verify(self, model, "CloseOpt/simple_lora_mlp", _simulate_close_optimal_aggregation)
        print("  ✅ simple_lora_mlp  Close-Optimal round-trip passed")

    # ------------------------------------------------------------------
    # CIFAR LoRA CNN (has BatchNorm → non-LoRA params)
    # ------------------------------------------------------------------
    def test_cifar_lora_cnn_sp(self):
        """CIFAR LoRA CNN (with BatchNorm) → SP aggregate → reconstruct → strict load."""
        args = NNModelFactory.create_args({
            "nn_model": {
                "name": "cifar_lora_cnn",
                "lora_rank_conv": 4,
                "lora_rank_fc": 4,
                "lora_alpha_conv": 8,
                "lora_alpha_fc": 8,
                "dropout": 0.0,
                "merge_weights": False,
                "use_bias": True,
            }
        })
        model = NNModelFactory.create(args)

        self.assertIsNotNone(model.lora_config, "cifar_lora_cnn should have lora_config")
        self.assertTrue(
            model.lora_config.get("has_non_lora_params", False),
            "cifar_lora_cnn has BatchNorm → has_non_lora_params should be True"
        )

        # Additional check: verify BatchNorm keys exist in original state_dict
        sd_keys = set(model.state_dict().keys())
        bn_keys = {k for k in sd_keys if "bn" in k.lower()}
        self.assertGreater(len(bn_keys), 0, "CIFAR LoRA CNN should have BatchNorm keys")

        _round_trip_and_verify(self, model, "SP/cifar_lora_cnn", _simulate_sp_aggregation)
        print("  ✅ cifar_lora_cnn   SP round-trip passed (BatchNorm preserved)")

    def test_cifar_lora_cnn_close_optimal(self):
        """CIFAR LoRA CNN → Close-Optimal aggregate → reconstruct → strict load."""
        args = NNModelFactory.create_args({
            "nn_model": {
                "name": "cifar_lora_cnn",
                "lora_rank_conv": 4,
                "lora_rank_fc": 4,
                "lora_alpha_conv": 8,
                "lora_alpha_fc": 8,
                "dropout": 0.0,
                "merge_weights": False,
                "use_bias": True,
            }
        })
        model = NNModelFactory.create(args)
        _round_trip_and_verify(self, model, "CloseOpt/cifar_lora_cnn", _simulate_close_optimal_aggregation)
        print("  ✅ cifar_lora_cnn   Close-Optimal round-trip passed (BatchNorm preserved)")

    # ------------------------------------------------------------------
    # TinyBERT LoRA (has LayerNorm + position/token_type embeddings)
    # ------------------------------------------------------------------
    def test_tinybert_lora_sp(self):
        """TinyBERT LoRA → SP aggregate → reconstruct → strict load."""
        try:
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("transformers not installed — skipping TinyBERT LoRA SP test")

        args = NNModelFactory.create_args({
            "nn_model": {
                "name": "tiny_bert_lora",
                "pretrained_model": "prajjwal1/bert-tiny",
                "num_classes": 2,
                "lora_r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "merge_weights": False,
                "lora_embedding": True,
                "pad_id": 0,
            }
        })
        model = NNModelFactory.create(args)

        self.assertIsNotNone(model.lora_config, "tiny_bert_lora should have lora_config")
        self.assertTrue(
            model.lora_config.get("has_non_lora_params", False),
            "TinyBERT has LayerNorm + position/token_type embeddings → has_non_lora_params=True"
        )

        # Verify non-LoRA keys exist in state_dict
        sd_keys = set(model.model.state_dict().keys())
        ln_keys = {k for k in sd_keys if "LayerNorm" in k}
        pos_keys = {k for k in sd_keys if "position_embeddings" in k}
        self.assertGreater(len(ln_keys), 0, "TinyBERT should have LayerNorm keys")
        self.assertGreater(len(pos_keys), 0, "TinyBERT should have position_embeddings keys")

        _round_trip_and_verify(self, model, "SP/tiny_bert_lora", _simulate_sp_aggregation)
        print("  ✅ tiny_bert_lora    SP round-trip passed (LayerNorm + embeddings preserved)")

    def test_tinybert_lora_close_optimal(self):
        """TinyBERT LoRA → Close-Optimal aggregate → reconstruct → strict load."""
        try:
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("transformers not installed — skipping TinyBERT LoRA CloseOpt test")

        args = NNModelFactory.create_args({
            "nn_model": {
                "name": "tiny_bert_lora",
                "pretrained_model": "prajjwal1/bert-tiny",
                "num_classes": 2,
                "lora_r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "merge_weights": False,
                "lora_embedding": True,
                "pad_id": 0,
            }
        })
        model = NNModelFactory.create(args)
        _round_trip_and_verify(self, model, "CloseOpt/tiny_bert_lora", _simulate_close_optimal_aggregation)
        print("  ✅ tiny_bert_lora    Close-Optimal round-trip passed (LayerNorm + embeddings preserved)")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    unittest.main()
