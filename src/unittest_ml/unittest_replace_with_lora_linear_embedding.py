"""
Unit test: LoRAUtils.replace_with_lora_linear_embedding

Verifies that:
  1. nn.Linear layers targeted by name are correctly replaced with MSLoRALinear.
  2. Non-targeted nn.Linear layers remain unchanged.
  3. nn.Embedding layers are replaced with MSEmbedding when replace_embedding=True.
  4. Pretrained weights & biases are correctly copied.
  5. Forward pass produces identical logits before/after replacement (merged mode).
  6. LoRA A/B parameters are trainable while base weights are frozen.
  7. target_module_names=None replaces ALL Linear layers.
  8. target_module_names=["query","value"] only replaces attention Q/V projections.

Uses RoBERTa-large from HuggingFace (cached locally under hf_models/ if available).

Run:
    cd unittest_ml
    python unittest_lora_utils.py
"""

from __future__ import annotations

import copy
import os
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# ------------------------------------------------------------------
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------------------------------------------

from flex.ml_algorithms.lora.lora_utils import LoRAUtils
from flex.ml_algorithms.lora.impl.lora_ms import MSLoRALinear, MSEmbedding


# ======================================================================
# Helpers
# ======================================================================

def _resolve_local_model_path(model_name: str = "roberta-large") -> tuple[str, bool]:
    """
    Resolve model source: prefer local hf_models cache, fall back to HF Hub.
    Returns (model_source, is_local).
    """
    # hf_models is at <project_root>/../hf_models/  (sibling of flex-src)
    # unittest_ml is at <project_root>/src/unittest_ml/
    project_root = Path(__file__).resolve().parents[1]  # flex-src
    local_dir = project_root.parent / "hf_models" / model_name

    has_safetensors = (local_dir / "model.safetensors").exists()
    has_bin = (local_dir / "pytorch_model.bin").exists()
    has_config = (local_dir / "config.json").exists()
    is_local = (has_safetensors or has_bin) and has_config

    model_source = str(local_dir) if is_local else model_name
    return model_source, is_local


def _count_layers(module: nn.Module):
    """Count different layer types in a module tree.

    Uses ``type(m) is ...`` rather than ``isinstance`` because
    MSLoRALinear <: nn.Linear and MSEmbedding <: nn.Embedding.
    """
    n_linear = sum(1 for m in module.modules() if type(m) is nn.Linear)
    n_mslora = sum(1 for m in module.modules() if isinstance(m, MSLoRALinear))
    n_embed = sum(1 for m in module.modules() if type(m) is nn.Embedding)
    n_msembed = sum(1 for m in module.modules() if isinstance(m, MSEmbedding))
    return n_linear, n_mslora, n_embed, n_msembed


# ======================================================================
# Test class
# ======================================================================

class TestReplaceWithLoRA(unittest.TestCase):
    """Test LoRAUtils.replace_with_lora_linear_embedding on RoBERTa-large."""

    @classmethod
    def setUpClass(cls):
        """Load RoBERTa-large once for all tests."""
        from transformers import AutoModelForSequenceClassification, AutoConfig

        model_source, is_local = _resolve_local_model_path("roberta-large")

        config = AutoConfig.from_pretrained(
            model_source,
            num_labels=2,
            local_files_only=is_local,
        )

        cls.model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            config=config,
            local_files_only=is_local,
        )
        cls.model.eval()

        # Snapshot the original state for comparison
        cls.original_state = {k: v.clone() for k, v in cls.model.state_dict().items()}

        # Count original layers
        cls.orig_linear, _, cls.orig_embed, _ = _count_layers(cls.model)

    # ------------------------------------------------------------------
    # 1. target_module_names=None → replace ALL Linear layers
    # ------------------------------------------------------------------
    def test_replace_all_linear(self):
        """When target_module_names=None, every nn.Linear → MSLoRALinear."""
        model = copy.deepcopy(self.__class__.model)

        LoRAUtils.replace_with_lora_linear_embedding(
            model, lora_r=8, lora_alpha=16, lora_dropout=0.1,
            merge_weights=False, target_module_names=None, replace_embedding=False,
        )

        n_linear, n_mslora, n_embed, n_msembed = _count_layers(model)

        self.assertEqual(n_linear, 0,
                         "All nn.Linear should be replaced (target_module_names=None)")
        self.assertEqual(n_mslora, self.__class__.orig_linear,
                         "MSLoRALinear count should match original nn.Linear count")
        self.assertEqual(n_embed, self.__class__.orig_embed,
                         "nn.Embedding should NOT be replaced (replace_embedding=False)")
        self.assertEqual(n_msembed, 0,
                         "No MSEmbedding should exist (replace_embedding=False)")

    # ------------------------------------------------------------------
    # 2. target_module_names=["query", "value"] → only Q/V projections
    # ------------------------------------------------------------------
    def test_replace_query_value_only(self):
        """Only attention query/value projections are replaced."""
        model = copy.deepcopy(self.__class__.model)

        LoRAUtils.replace_with_lora_linear_embedding(
            model, lora_r=8, lora_alpha=16, lora_dropout=0.1,
            merge_weights=False,
            target_module_names=["query", "value"],
            replace_embedding=False,
        )

        n_linear, n_mslora, _, _ = _count_layers(model)

        # After replacement, there should be fewer nn.Linear and more MSLoRALinear
        self.assertGreater(n_linear, 0,
                           "Some nn.Linear (e.g. classifier, key, dense) should remain")
        self.assertGreater(n_mslora, 0,
                           "At least some MSLoRALinear should exist (query/value)")

        # Verify that only query/value layers became MSLoRALinear
        for name, module in model.named_modules():
            if isinstance(module, MSLoRALinear):
                name_lower = name.split(".")[-1].lower()
                self.assertTrue(
                    "query" in name_lower or "value" in name_lower,
                    f"MSLoRALinear at '{name}' is not a query/value projection"
                )

    # ------------------------------------------------------------------
    # 3. replace_embedding=True
    # ------------------------------------------------------------------
    def test_replace_embedding(self):
        """When replace_embedding=True, nn.Embedding → MSEmbedding."""
        model = copy.deepcopy(self.__class__.model)

        LoRAUtils.replace_with_lora_linear_embedding(
            model, lora_r=8, lora_alpha=16, lora_dropout=0.1,
            merge_weights=False, target_module_names=["query", "value"],
            replace_embedding=True,
        )

        n_linear, n_mslora, n_embed, n_msembed = _count_layers(model)

        self.assertEqual(n_embed, 0,
                         "All nn.Embedding should be replaced (replace_embedding=True)")
        self.assertEqual(n_msembed, self.__class__.orig_embed,
                         "MSEmbedding count should match original nn.Embedding count")

    # ------------------------------------------------------------------
    # 4. Pretrained weights are correctly copied (merge_weights=True)
    # ------------------------------------------------------------------
    def test_weight_copy_and_forward_equivalence(self):
        """
        With merge_weights=True and lora_r>0, after calling .eval(),
        the forward pass should produce nearly identical logits
        (LoRA B is zero-initialized, so ΔW=0 before training).
        """
        model = copy.deepcopy(self.__class__.model)

        LoRAUtils.replace_with_lora_linear_embedding(
            model, lora_r=8, lora_alpha=16, lora_dropout=0.1,
            merge_weights=True, target_module_names=["query", "value"],
            replace_embedding=False,
        )
        model.eval()  # triggers merge

        # Create a dummy input
        torch.manual_seed(42)
        dummy_input = torch.randint(0, 50265, (2, 16))  # RoBERTa vocab size ~50265
        attention_mask = torch.ones_like(dummy_input)

        with torch.no_grad():
            orig_out = self.__class__.model(dummy_input, attention_mask=attention_mask)
            lora_out = model(dummy_input, attention_mask=attention_mask)

        # Logits should be very close (B=0 → ΔW=0)
        max_diff = (orig_out.logits - lora_out.logits).abs().max().item()
        self.assertLess(max_diff, 1e-4,
                        f"Logits differ after LoRA replacement: max_diff={max_diff:.6f}")

        # Verify base weights are frozen
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                self.assertTrue(param.requires_grad,
                                f"LoRA param '{name}' should be trainable")
            elif "weight" in name or "bias" in name:
                # Non-LoRA weights may still be trainable (classifier, LayerNorm etc.)
                pass

        # Specifically verify LoRA-injected Linear base weights are frozen
        for name, module in model.named_modules():
            if isinstance(module, MSLoRALinear):
                self.assertFalse(module.weight.requires_grad,
                                 f"MSLoRALinear base weight '{name}.weight' should be frozen")

    # ------------------------------------------------------------------
    # 5. LoRA A/B are trainable, base weight frozen
    # ------------------------------------------------------------------
    def test_lora_trainable_base_frozen(self):
        """After replacement, lora_A/lora_B are trainable, base weights frozen."""
        model = copy.deepcopy(self.__class__.model)

        LoRAUtils.replace_with_lora_linear_embedding(
            model, lora_r=8, lora_alpha=16, lora_dropout=0.1,
            merge_weights=False, target_module_names=None,
            replace_embedding=True,  # also wrap embeddings so they don't inflate trainable count
        )

        trainable_params = 0
        frozen_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()

        total = trainable_params + frozen_params
        trainable_ratio = trainable_params / total

        # With full LoRA wrapping (including embeddings), only a small fraction
        # of parameters should be trainable (LoRA A/B matrices).
        # RoBERTa-large: ~355M total → LoRA r=8 across all layers ≈ ~2M trainable.
        self.assertLess(trainable_ratio, 0.20,
                        f"Trainable params should be <20% of total, got {trainable_ratio:.4f}")
        self.assertGreater(trainable_params, 0,
                           "At least some LoRA params should be trainable")

        # Verify all lora_A/lora_B are trainable
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                self.assertTrue(param.requires_grad,
                                f"LoRA param '{name}' must be trainable")

    # ------------------------------------------------------------------
    # 6. Bias is correctly copied
    # ------------------------------------------------------------------
    def test_bias_copy(self):
        """Biases from original nn.Linear are copied to MSLoRALinear."""
        model = copy.deepcopy(self.__class__.model)

        # Collect original biases before replacement
        original_biases = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                original_biases[name] = module.bias.data.clone()

        LoRAUtils.replace_with_lora_linear_embedding(
            model, lora_r=8, lora_alpha=16, lora_dropout=0.1,
            merge_weights=False, target_module_names=None,
            replace_embedding=False,
        )

        for name, module in model.named_modules():
            if isinstance(module, MSLoRALinear):
                if name in original_biases:
                    self.assertIsNotNone(module.bias,
                                         f"MSLoRALinear '{name}' should have bias")
                    torch.testing.assert_close(
                        module.bias.data, original_biases[name],
                        msg=f"Bias mismatch for '{name}'"
                    )

    # ------------------------------------------------------------------
    # 7. Custom lora_r is respected
    # ------------------------------------------------------------------
    def test_custom_lora_rank(self):
        """The lora_A/lora_B matrices have the specified rank."""
        custom_r = 12
        model = copy.deepcopy(self.__class__.model)

        LoRAUtils.replace_with_lora_linear_embedding(
            model, lora_r=custom_r, lora_alpha=24, lora_dropout=0.05,
            merge_weights=False, target_module_names=None,
            replace_embedding=False,
        )

        for name, module in model.named_modules():
            if isinstance(module, MSLoRALinear):
                self.assertEqual(module.lora_A.shape[0], custom_r,
                                 f"lora_A rank mismatch at '{name}': "
                                 f"expected {custom_r}, got {module.lora_A.shape[0]}")
                self.assertEqual(module.lora_B.shape[1], custom_r,
                                 f"lora_B rank mismatch at '{name}': "
                                 f"expected {custom_r}, got {module.lora_B.shape[1]}")

    # ------------------------------------------------------------------
    # 8. lora_alpha / lora_dropout / merge_weights are propagated
    # ------------------------------------------------------------------
    def test_lora_config_propagation(self):
        """Verify lora_alpha, lora_dropout, merge_weights are correctly set."""
        model = copy.deepcopy(self.__class__.model)

        LoRAUtils.replace_with_lora_linear_embedding(
            model, lora_r=4, lora_alpha=32, lora_dropout=0.2,
            merge_weights=False, target_module_names=["query"],
            replace_embedding=False,
        )

        for name, module in model.named_modules():
            if isinstance(module, MSLoRALinear):
                self.assertEqual(module.lora_alpha, 32,
                                 f"lora_alpha mismatch at '{name}'")
                self.assertEqual(module.lora_dropout.p, 0.2,
                                 f"lora_dropout mismatch at '{name}'")
                self.assertFalse(module.merge_weights,
                                 f"merge_weights should be False at '{name}'")
                self.assertEqual(module.scaling, 32.0 / 4.0,
                                 f"scaling mismatch at '{name}': "
                                 f"expected {32.0/4.0}, got {module.scaling}")


if __name__ == "__main__":
    unittest.main()
