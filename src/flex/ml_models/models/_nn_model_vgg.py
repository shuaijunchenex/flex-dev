from __future__ import annotations
from typing import Any

import torch.nn as nn

from .. import AbstractNNModel, NNModelArgs, NNModel


# ---------------------------------------------------------------------------
# VGG9
# ---------------------------------------------------------------------------
# Architecture (9 weight layers = 6 Conv + 3 FC):
#   Block-1:  Conv(3→64,3,p1) + BN + ReLU  ×2  + MaxPool(2,2)
#   Block-2:  Conv(64→128,3,p1) + BN + ReLU  ×2  + MaxPool(2,2)
#   Block-3:  Conv(128→256,3,p1) + BN + ReLU  ×2  + MaxPool(2,2)
#   AdaptiveAvgPool → (4×4) so that FC input is always 256×4×4 = 4096
#   FC: 4096 → 512 → 512 → num_classes
#
# Works for: CIFAR-10/100 (32×32), CINIC-10 (32×32), Tiny-ImageNet (64×64)
# ---------------------------------------------------------------------------
class NNModel_VGG9(NNModel):
    """VGG-9: compact VGG variant commonly used in FL benchmarks."""

    def __init__(self):
        super().__init__()

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        num_classes: int = getattr(args, "num_classes", 10) or 10
        dropout: float   = getattr(args, "dropout",     0.5) or 0.5

        self._features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Normalise spatial dims → 4×4 regardless of input resolution.
        self._pool = nn.AdaptiveAvgPool2d((4, 4))

        self._classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 512),         nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        return self  # Note: return self

    def forward(self, x) -> Any:
        x = self._features(x)
        x = self._pool(x)
        x = x.view(x.size(0), -1)
        x = self._classifier(x)
        return x


# ---------------------------------------------------------------------------
# VGG16
# ---------------------------------------------------------------------------
# Architecture (16 weight layers = 13 Conv + 3 FC) — standard VGG-16 with
# BatchNorm, adapted for smaller inputs via AdaptiveAvgPool2d.
#   Block-1:  Conv(3→64)  ×2 + MaxPool
#   Block-2:  Conv(64→128) ×2 + MaxPool
#   Block-3:  Conv(128→256) ×3 + MaxPool
#   Block-4:  Conv(256→512) ×3 + MaxPool
#   Block-5:  Conv(512→512) ×3 + MaxPool
#   AdaptiveAvgPool → (7×7) so FC input is always 512×7×7 = 25088
#   FC: 25088 → 4096 → 4096 → num_classes
#
# Works for: CIFAR (32×32), CINIC-10 (32×32), Tiny-ImageNet (64×64),
#            ImageNet (224×224)
# ---------------------------------------------------------------------------
class NNModel_VGG16(NNModel):
    """VGG-16 with BatchNorm, resolution-agnostic via AdaptiveAvgPool2d."""

    def __init__(self):
        super().__init__()

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        num_classes: int = getattr(args, "num_classes", 10) or 10
        dropout: float   = getattr(args, "dropout",     0.5) or 0.5

        def _block(in_ch, out_ch, n_conv):
            layers = []
            for i in range(n_conv):
                layers += [
                    nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return layers

        self._features = nn.Sequential(
            *_block(3,   64,  2),   # Block-1
            *_block(64,  128, 2),   # Block-2
            *_block(128, 256, 3),   # Block-3
            *_block(256, 512, 3),   # Block-4
            *_block(512, 512, 3),   # Block-5
        )
        # Normalise to 7×7 for any input resolution.
        self._pool = nn.AdaptiveAvgPool2d((7, 7))

        self._classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, 4096),         nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )
        return self  # Note: return self

    def forward(self, x) -> Any:
        x = self._features(x)
        x = self._pool(x)
        x = x.view(x.size(0), -1)
        x = self._classifier(x)
        return x
