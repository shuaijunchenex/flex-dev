"""Linear FLoRG adapter with one trainable Gram factor."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GramLoRALinear(nn.Module):
    """Apply W0 + L C.T C R with fixed semi-orthogonal L and R."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        *,
        generator: torch.Generator | None = None,
        factor_init_std: float = 0.02,
    ):
        super().__init__()
        width = min(base.in_features, base.out_features)
        if not 1 <= rank <= width:
            raise ValueError("Gram adapter rank must be between 1 and the smaller feature width")
        if not math.isfinite(factor_init_std) or factor_init_std <= 0:
            raise ValueError("Gram factor initialization standard deviation must be positive and finite")
        self.base = base.requires_grad_(False)
        self.rank = rank

        work_dtype = torch.float64 if base.weight.dtype == torch.float64 else torch.float32
        left = torch.linalg.qr(
            torch.randn(base.out_features, width, generator=generator, dtype=work_dtype, device="cpu")
        ).Q
        right = torch.linalg.qr(
            torch.randn(base.in_features, width, generator=generator, dtype=work_dtype, device="cpu")
        ).Q.T
        self.register_buffer("basis_left", left.contiguous().to(base.weight))
        self.register_buffer("basis_right", right.contiguous().to(base.weight))
        factor = torch.randn(rank, width, generator=generator, dtype=work_dtype, device="cpu")
        initial = (factor * factor_init_std).to(base.weight)
        if not torch.isfinite(initial).all().item() or not torch.count_nonzero(initial).item():
            raise ValueError("Gram factor initialization must be finite and nonzero")
        self.gram_factor = nn.Parameter(initial)

    def effective_update(self) -> torch.Tensor:
        return (self.basis_left @ self.gram_factor.T) @ (self.gram_factor @ self.basis_right)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = F.linear(inputs, self.basis_right)
        reduced = F.linear(projected, self.gram_factor)
        expanded = F.linear(reduced, self.gram_factor.T)
        return self.base(inputs) + F.linear(expanded, self.basis_left)
