"""
Semantic-Anchored Rank Alignment (rblasa) -- alignment loss module.

Provides slot-level cosine-alignment and subspace-level projection-alignment
losses with round-dependent decay scheduling and B-space warm-up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class rblasaConfig:
    """Hyperparameters for rblasa alignment regularization."""

    # ── Slot-level alignment ──
    lambda_slot_0: float = 0.1       # initial slot alignment weight
    lambda_slot_min: float = 0.01    # floor for slot alignment weight
    slot_weight_type: str = "1/s"    # "1/s", "uniform", or "linear"

    # ── Subspace-level alignment ──
    lambda_sub_0: float = 0.1        # initial subspace alignment weight
    lambda_sub_min: float = 0.01     # floor for subspace alignment weight

    # ── Decay schedule (shared β for slot + sub) ──
    beta: float = 0.01               # exponential decay rate  λ(t)=λ₀·exp(-β·t)

    # -- Warm-up (B starts at zero, QR would fail) --
    warmup_rounds: int = 5           # only A-space alignment before this round

    # ── Rank expansion ──
    enable_rank_expansion: bool = False  # if True, support dynamic rank increase

    @classmethod
    def from_dict(cls, d: dict) -> "rblasaConfig":
        """Build config from a flat dict (e.g. YAML section)."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# Alignment loss
# ---------------------------------------------------------------------------

class rblasaAlignmentLoss(nn.Module):
    """
    Slot-level + subspace-level alignment regularisation for heterogeneous
    LoRA federated training.

    Intended usage (per client, per round)::

        rblasa = rblasaAlignmentLoss(config)
        lambda_slot, lambda_sub = rblasa.get_lambdas(round_idx)
        slot_loss = rblasa.compute_slot_loss(B_local, A_local, B_global, A_global)
        sub_loss  = rblasa.compute_subspace_loss(B_local, B_global, r_i, round_idx)
        total_reg = lambda_slot * slot_loss + lambda_sub * sub_loss

    The total training loss is:  L_task + total_reg.
    """

    def __init__(self, config: rblasaConfig | None = None):
        super().__init__()
        self.cfg = config or rblasaConfig()

    # ------------------------------------------------------------------
    # Lambda schedule
    # ------------------------------------------------------------------
    def get_lambdas(self, round_idx: int) -> Tuple[float, float]:
        """Return (λ_slot, λ_sub) for the given round index."""
        t = float(round_idx)
        lam_s = max(self.cfg.lambda_slot_min,
                     self.cfg.lambda_slot_0 * math.exp(-self.cfg.beta * t))
        lam_b = max(self.cfg.lambda_sub_min,
                     self.cfg.lambda_sub_0 * math.exp(-self.cfg.beta * t))
        return lam_s, lam_b

    # ------------------------------------------------------------------
    # Slot importance weights
    # ------------------------------------------------------------------
    def get_slot_weights(self, r_i: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Return omega_s for s = 0 .. r_i-1.  Lower slots get higher weight."""
        if self.cfg.slot_weight_type == "1/s":
            w = torch.tensor([1.0 / max(s + 1, 1) for s in range(r_i)], device=device)
        elif self.cfg.slot_weight_type == "uniform":
            w = torch.ones(r_i, device=device) / float(r_i)
        elif self.cfg.slot_weight_type == "linear":
            w = torch.linspace(1.0, 1.0 / max(r_i, 1), r_i, device=device)
        else:
            raise ValueError(f"Unknown slot_weight_type: {self.cfg.slot_weight_type}")
        return w / w.sum()  # normalise so total weight = 1

    # ------------------------------------------------------------------
    # Slot-level cosine alignment
    # ------------------------------------------------------------------
    def compute_slot_loss(
        self,
        B_local: torch.Tensor,    # [d_out, r_i]
        A_local: torch.Tensor,    # [r_i, d_in]
        B_global: torch.Tensor,   # [d_out, r_i]
        A_global: torch.Tensor,   # [r_i, d_in]
    ) -> torch.Tensor:
        """
        L_slot = Σ_s ω_s · [(1 - cos(b_i_s, b_g_s)) + (1 - cos(a_i_s, a_g_s))]

        B columns (b_s) and A rows (a_s) are treated as rank-slot vectors.
        """
        r_i = A_local.shape[0]
        w = self.get_slot_weights(r_i, device=A_local.device)  # [r_i]

        # ── B-column cosine distance ──
        b_loc = F.normalize(B_local, dim=0)    # [d_out, r_i]
        b_glo = F.normalize(B_global, dim=0)   # [d_out, r_i]
        b_cos = (b_loc * b_glo).sum(dim=0)     # [r_i]
        loss_b = ((1.0 - b_cos) * w).sum()

        # ── A-row cosine distance ──
        a_loc = F.normalize(A_local, dim=1)    # [r_i, d_in]
        a_glo = F.normalize(A_global, dim=1)   # [r_i, d_in]
        a_cos = (a_loc * a_glo).sum(dim=1)     # [r_i]
        loss_a = ((1.0 - a_cos) * w).sum()

        return loss_b + loss_a

    # ------------------------------------------------------------------
    # Subspace-level projection alignment (B-space)
    # ------------------------------------------------------------------
    def compute_subspace_loss(
        self,
        B_local: torch.Tensor,    # [d_out, r_i]
        B_global: torch.Tensor,   # [d_out, r_i]
        r_i: int,
        round_idx: int,
    ) -> torch.Tensor:
        """
        L_sub = || P_{B_i} - P_{B_g}^{(r_i)} ||_F^2

        where P_X = Q_X Q_X^T, Q_X = orthonormal basis from QR(X).

        During warm-up (round_idx < warmup_rounds) returns 0 because
        B is initialised to 0 and QR would fail.
        """
        if round_idx < self.cfg.warmup_rounds:
            return torch.tensor(0.0, device=B_local.device)

        # QR decomposition -> orthonormal basis
        Q_loc, _ = torch.linalg.qr(B_local)          # [d_out, r_i]
        Q_glo, _ = torch.linalg.qr(B_global[:, :r_i])  # [d_out, r_i]

        P_loc = Q_loc @ Q_loc.T    # [d_out, d_out]
        P_glo = Q_glo @ Q_glo.T    # [d_out, d_out]

        return torch.norm(P_loc - P_glo, p="fro") ** 2

    # ------------------------------------------------------------------
    # Combined forward (for convenience)
    # ------------------------------------------------------------------
    def forward(
        self,
        B_local: torch.Tensor,
        A_local: torch.Tensor,
        B_global: torch.Tensor,
        A_global: torch.Tensor,
        r_i: int,
        round_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_reg, slot_loss, sub_loss).
        """
        lam_slot, lam_sub = self.get_lambdas(round_idx)
        slot = self.compute_slot_loss(B_local, A_local, B_global, A_global)
        sub = self.compute_subspace_loss(B_local, B_global, r_i, round_idx)
        total = lam_slot * slot + lam_sub * sub
        return total, slot.detach(), sub.detach()


# ---------------------------------------------------------------------------
# Utility: collect all LoRA (A, B) pairs from a state_dict / model
# ---------------------------------------------------------------------------

def collect_lora_pairs(
    state_dict: Dict[str, torch.Tensor],
    lora_suffixes: Tuple[str, str] = ("lora_A", "lora_B"),
) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    """
    Return a list of (prefix, lora_A_tensor, lora_B_tensor) for every
    LoRA layer found in *state_dict*.

    ``prefix`` is the layer name without the ``.lora_A``/``.lora_B`` suffix.
    """
    suffix_A, suffix_B = lora_suffixes
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if suffix_A in key:
            prefix = key.replace(f".{suffix_A}", "")
            pairs.setdefault(prefix, {})["A"] = tensor
        elif suffix_B in key:
            prefix = key.replace(f".{suffix_B}", "")
            pairs.setdefault(prefix, {})["B"] = tensor
    result = []
    for prefix, d in pairs.items():
        if "A" in d and "B" in d:
            result.append((prefix, d["A"], d["B"]))
    return result
