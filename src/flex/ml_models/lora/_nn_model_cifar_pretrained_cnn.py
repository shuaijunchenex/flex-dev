"""CIFAR backbones compatible with the chenyaofo/pytorch-cifar-models checkpoints."""
from __future__ import annotations

import hashlib
import math
from abc import abstractmethod
from pathlib import Path

import torch
from torch import nn

from ..nn_model import NNModel
from ..nn_model_args import NNModelArgs
from ...ml_algorithms.lora import MSLoRAConv2d


@torch.no_grad()
def _adapt_convolution(layer, name, rank, alpha, reference_rank, reference_scale, seed):
    kernel = layer.kernel_size[0]
    adapted = MSLoRAConv2d(
        layer.in_channels, layer.out_channels, kernel, r=rank, lora_alpha=alpha,
        stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
        groups=layer.groups, padding_mode=layer.padding_mode,
        bias=layer.bias is not None, merge_weights=False,
    )
    adapted.weight.copy_(layer.weight)
    if layer.bias is not None:
        adapted.bias.copy_(layer.bias)
        adapted.bias.requires_grad_(False)

    # Generate the same maximum-width A for a named layer in both experimental arms.
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))
    reference_A = adapted.weight.new_empty(reference_rank * kernel, layer.in_channels * kernel)
    nn.init.kaiming_uniform_(reference_A, a=math.sqrt(5), generator=generator)
    # Express the reference initialization in this layer's alpha/r coordinates.
    adapted.lora_A.copy_(reference_A[:rank * kernel] * math.sqrt(reference_scale / adapted.scaling))
    return adapted


class NNModel_CifarPretrainedLoRA(NNModel):
    """Frozen CIFAR feature extractor with trainable convolution factors and linear head."""

    classifier_name: str

    @staticmethod
    @abstractmethod
    def _build_backbone() -> nn.Module:
        """Build the original 10-class network for strict checkpoint loading."""

    @staticmethod
    @abstractmethod
    def _is_lora_target(name: str, layer: nn.Conv2d) -> bool:
        """Select the convolutions that receive trainable LoRA factors."""

    def create_model(self, args: NNModelArgs):
        base_rank = int(args.get("lora_rank_conv", 8))
        ratio = float(args.rank_ratio)
        alpha = float(args.get("lora_alpha_conv", 8))
        if base_rank < 1 or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("lora_rank_conv and rank_ratio must be positive")
        rank = round(base_rank * ratio)
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Effective rank and lora_alpha_conv must be positive")
        reference_rank = max(8, base_rank, rank)
        return self._create_pretrained(args, lambda name: rank, lambda r: alpha, reference_rank)

    def _create_pretrained(self, args, rank_for_layer, alpha_for_rank, reference_rank):
        super().create_model(args)
        checkpoint_path = args.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError(f"{args.model_type} requires a local checkpoint_path")
        if args.get("merge_weights", False):
            raise ValueError("The federated CIFAR adapters require merge_weights=False")
        seed = int(args.get("initialization_seed", 42))
        reference_scale = alpha_for_rank(reference_rank) / reference_rank

        # Model construction should not advance the RNG used for minibatches or augmentation.
        with torch.random.fork_rng(devices=[]):
            self.backbone = self._build_backbone()
            checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(checkpoint, strict=True)
            self.backbone.requires_grad_(False)
            classifier = self.backbone.get_submodule(self.classifier_name)
            feature_dim = next(m.in_features for m in classifier.modules() if isinstance(m, nn.Linear))
            setattr(self.backbone, self.classifier_name, nn.Identity())
            torch.random.default_generator.manual_seed(seed)
            self.head = nn.Linear(feature_dim, args.num_classes)

            for name, conv in list(self.backbone.named_modules()):
                if not isinstance(conv, nn.Conv2d) or not self._is_lora_target(name, conv):
                    continue
                rank = rank_for_layer(name)
                if rank == 0:
                    continue
                parent_name, _, child_name = name.rpartition(".")
                parent = self.backbone.get_submodule(parent_name)
                adapted = _adapt_convolution(
                    conv, name, rank, alpha_for_rank(rank), reference_rank, reference_scale, seed,
                )
                setattr(parent, child_name, adapted)

        self._lora_config = {
            "suffix_A": "lora_A", "suffix_B": "lora_B",
            "sp_suffix": "sp_aggregated", "has_non_lora_params": True,
        }
        return self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, x):
        return self.head(self.backbone(x))
