"""Fixed layer1 cut of the pretrained CIFAR ResNet20."""
from collections import OrderedDict
import math

from torch import nn

from flex.ml_models.lora._nn_model_cifar_resnet20_lora import NNModel_CifarResNet20LoRA
from flex.ml_models.nn_model_args import NNModelArgs


class NNModel_SplitResNet20(NNModel_CifarResNet20LoRA):
    _front_stages = ("conv1", "bn1", "relu", "layer1")

    def create_model(self, args: NNModelArgs):
        self.part = args.get("part", "full")
        if self.part not in ("front", "rear", "full") or args.get("cut_after", "layer1") != "layer1":
            raise ValueError("Expected front/rear/full ResNet20 with cut_after=layer1")
        front_rank = int(args.get("front_rank", 0))
        rear_rank = int(args.get("rear_rank", 8))
        reference_rank = int(args.get("reference_rank", max(8, front_rank, rear_rank)))
        if min(front_rank, rear_rank) < 0 or reference_rank < max(1, front_rank, rear_rank):
            raise ValueError("Expected nonnegative ranks within a positive reference_rank")

        scope = args.get("rear_scope", "adapter")
        if scope not in ("adapter", "dense", "frozen"):
            raise ValueError(f"Unknown rear_scope: {scope}")
        scale_mode = args.get("scale_mode", "constant_alpha_over_rank")
        if scale_mode == "constant_alpha_over_rank":
            scale = float(args.get("alpha_over_rank", 1.0))
        elif scale_mode == "fixed_alpha":
            scale = float(args.get("lora_alpha_conv", 8))
        else:
            raise ValueError(f"Unknown scale_mode: {scale_mode}")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("LoRA scale must be positive and finite")

        def alpha_for_rank(rank):
            return scale * rank if scale_mode == "constant_alpha_over_rank" else scale

        def rank_for_layer(name):
            if name.startswith("layer1."):
                return front_rank if self.part != "rear" else 0
            return rear_rank if self.part != "front" and scope == "adapter" else 0

        self._create_pretrained(args, rank_for_layer, alpha_for_rank, reference_rank)
        stages = OrderedDict()
        for name, module in self.backbone.named_children():
            side = "front" if name in self._front_stages else "rear"
            if self.part in ("full", side):
                if side == "rear" and scope == "dense":
                    module.requires_grad_(True)
                stages[name] = module
        self.backbone = nn.Sequential(stages)
        if self.part == "front":
            del self.head
        else:
            self.head.requires_grad_(args.get("train_head", True))
        return self.train(self.training)

    def adapter_keys(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.named_parameters()
                     if name.endswith((".lora_A", ".lora_B")))

    def forward(self, x):
        features = self.backbone(x)
        return features if self.part == "front" else self.head(features)
