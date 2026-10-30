"""
Semantic-Anchored Rank Alignment (SARA) — alignment loss module.

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
class SARAConfig:
    """Hyperparameters for SARA alignment regularization."""

    # ── Slot-level alignment ──
    lambda_slot_0: float = 0.1       # initial slot alignment weight
    lambda_slot_min: float = 0.01    # floor for slot alignment weight
    slot_weight_type: str = "1/s"    # "1/s", "uniform", "linear", "share_degree", "coverage_aware", or "rank_decay"
    rank_ratio_list: List[float] = field(default_factory=list)
    share_degree_power: float = 1.0
    lambda_tail: float = 0.1
    coverage_eta: float = 1.0

    # ── Rank-decay slot lambdas (slot_weight_type = "rank_decay") ──
    rank_decay_lambda_max: float = 0.1   # λ for slot 0 (strongest alignment)
    rank_decay_lambda_min: float = 0.01  # λ for slot R-1 (weakest alignment)
    rank_decay_gamma: float = 1.0        # shape: 1.0=linear, >1=faster drop, <1=smoother

    # ── Subspace-level alignment ──
    lambda_sub_0: float = 0.1        # initial subspace alignment weight
    lambda_sub_min: float = 0.01     # floor for subspace alignment weight

    # ── Decay schedule (shared β for slot + sub) ──
    beta: float = 0.01               # exponential decay rate  λ(t)=λ₀·exp(-β·t)

    # ── Warm-up (B starts at zero → QR would fail) ──
    warmup_rounds: int = 5           # only A-space alignment before this round

    # ── Rank expansion ──
    enable_rank_expansion: bool = False  # if True, support dynamic rank increase

    # ── Adaptive SARA rank-level weighting ──
    rank_weight_min: float = 0.5
    rank_weight_max: float = 1.5
    rank_weight_gamma: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "SARAConfig":
        """Build config from a flat dict (e.g. YAML section)."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# Alignment loss
# ---------------------------------------------------------------------------

def _safe_qr(x: torch.Tensor) -> torch.Tensor:
    """QR decomposition with automatic MPS→CPU fallback.

    ``torch.linalg.qr`` is not implemented on MPS.  When the input tensor
    resides on an MPS device, it is temporarily moved to CPU, decomposed,
    and the resulting Q is moved back to the original device.
    """
    dev = x.device
    if dev.type == "mps":
        Q, _ = torch.linalg.qr(x.to("cpu"))
        return Q.to(dev)
    Q, _ = torch.linalg.qr(x)
    return Q

class SARAAlignmentLoss(nn.Module):
    """
    Slot-level + subspace-level alignment regularisation for heterogeneous
    LoRA federated training.

    Intended usage (per client, per round)::

        sara = SARAAlignmentLoss(config)
        lambda_slot, lambda_sub = sara.get_lambdas(round_idx)
        slot_loss = sara.compute_slot_loss(B_local, A_local, B_global, A_global)
        sub_loss  = sara.compute_subspace_loss(B_local, B_global, r_i, round_idx)
        total_reg = lambda_slot * slot_loss + lambda_sub * sub_loss

    The total training loss is:  L_task + total_reg.
    """

    def __init__(self, config: SARAConfig | None = None):
        super().__init__()
        self.cfg = config or SARAConfig()

    # ------------------------------------------------------------------
    # Lambda schedule
    # ------------------------------------------------------------------
    def get_lambdas(self, round_idx: int) -> Tuple[float, float]:
        """Return (λ_slot, λ_sub) for the given round index."""
        t = float(round_idx)
        if self.cfg.slot_weight_type in {"coverage_aware", "coverage-aware", "rank_decay"}:
            lam_s = 1.0
        else:
            lam_s = max(self.cfg.lambda_slot_min,
                         self.cfg.lambda_slot_0 * math.exp(-self.cfg.beta * t))
        lam_b = max(self.cfg.lambda_sub_min,
                     self.cfg.lambda_sub_0 * math.exp(-self.cfg.beta * t))
        return lam_s, lam_b

    def _coverage_scores(self, r_i: int, device: torch.device, global_rank: int | None = None) -> torch.Tensor:
        R = int(global_rank or r_i)
        ratios = list(self.cfg.rank_ratio_list or [])
        if not ratios:
            return torch.ones(r_i, dtype=torch.float32, device=device)
        client_ranks = [
            max(1, min(R, int(round(R * float(ratio)))))
            for ratio in ratios
        ]
        counts = [
            sum(1 for rank in client_ranks if rank >= slot)
            for slot in range(1, r_i + 1)
        ]
        return torch.tensor(counts, dtype=torch.float32, device=device) / float(len(client_ranks))

    # ------------------------------------------------------------------
    # Slot importance weights
    # ------------------------------------------------------------------
    def get_slot_weights(self, r_i: int, device: torch.device = torch.device("cpu"), global_rank: int | None = None) -> torch.Tensor:
        """Return ω_s for s = 0 … r_i-1.  Lower slots get higher weight."""
        if self.cfg.slot_weight_type == "1/s":
            w = torch.tensor([1.0 / max(s + 1, 1) for s in range(r_i)], device=device)
        elif self.cfg.slot_weight_type == "uniform":
            w = torch.ones(r_i, device=device) / float(r_i)
        elif self.cfg.slot_weight_type == "linear":
            w = torch.linspace(1.0, 1.0 / max(r_i, 1), r_i, device=device)
        elif self.cfg.slot_weight_type in {"share_degree", "share"}:
            R = int(global_rank or r_i)
            ratios = list(self.cfg.rank_ratio_list or [])
            if ratios:
                client_ranks = [
                    max(1, min(R, int(round(R * float(ratio)))))
                    for ratio in ratios
                ]
                counts = [
                    sum(1 for rank in client_ranks if rank >= slot)
                    for slot in range(1, r_i + 1)
                ]
                w = torch.tensor(counts, dtype=torch.float32, device=device)
                power = float(self.cfg.share_degree_power)
                if power != 1.0:
                    w = w.clamp_min(1.0).pow(power)
            else:
                w = torch.tensor([1.0 / max(s + 1, 1) for s in range(r_i)], device=device)
        else:
            raise ValueError(f"Unknown slot_weight_type: {self.cfg.slot_weight_type}")
        return w / w.sum().clamp_min(torch.finfo(w.dtype).eps)

    def get_slot_lambdas(
        self,
        r_i: int,
        round_idx: int,
        device: torch.device = torch.device("cpu"),
        global_rank: int | None = None,
    ) -> torch.Tensor:
        """Return coverage-aware lambda_s,t for each local rank slot."""
        lambda_warm = self.cfg.lambda_slot_0 * math.exp(-self.cfg.beta * float(round_idx))
        q = self._coverage_scores(r_i, device=device, global_rank=global_rank).clamp(0.0, 1.0)
        tail = self.cfg.lambda_tail * torch.pow(1.0 - q, float(self.cfg.coverage_eta))
        return lambda_warm + tail

    def get_slot_lambdas_rank_decay(
        self,
        r_i: int,
        device: torch.device = torch.device("cpu"),
        global_rank: int | None = None,
    ) -> torch.Tensor:
        r"""Return per-slot lambdas using rank-position decay.

        .. math::

            \lambda_k = \lambda_{\min}
                      + (\lambda_{\max} - \lambda_{\min})
                      \cdot \left(1 - \frac{k}{R-1}\right)^\gamma

        where :math:`k` is 0-based and :math:`R` = *global_rank*.

        - Slot 0 receives :math:`\approx \lambda_{\max}` (strongest anchor).
        - Slot :math:`R-1` receives :math:`\approx \lambda_{\min}` (weakest).
        - *gamma* controls the decay shape.
        """
        R = int(global_rank or r_i)
        if R <= 1:
            return torch.full((r_i,), self.cfg.rank_decay_lambda_max, device=device)

        k = torch.arange(r_i, device=device, dtype=torch.float32)
        rank_pos = (k / max(R - 1, 1)).clamp(0.0, 1.0)
        decay = (1.0 - rank_pos) ** self.cfg.rank_decay_gamma
        return self.cfg.rank_decay_lambda_min + (
            self.cfg.rank_decay_lambda_max - self.cfg.rank_decay_lambda_min
        ) * decay

    # ------------------------------------------------------------------
    # Slot-level cosine alignment
    # ------------------------------------------------------------------
    def compute_slot_loss(
        self,
        B_local: torch.Tensor,    # [d_out, r_i]
        A_local: torch.Tensor,    # [r_i, d_in]
        B_global: torch.Tensor,   # [d_out, r_i]
        A_global: torch.Tensor,   # [r_i, d_in]
        global_rank: int | None = None,
        round_idx: int | None = None,
    ) -> torch.Tensor:
        """
        L_slot = Σ_s ω_s · [(1 - cos(b_i_s, b_g_s)) + (1 - cos(a_i_s, a_g_s))]

        B columns (b_s) and A rows (a_s) are treated as rank-slot vectors.
        """
        r_i = A_local.shape[0]

        # ── B-column cosine distance ──
        b_loc = F.normalize(B_local, dim=0)    # [d_out, r_i]
        b_glo = F.normalize(B_global, dim=0)   # [d_out, r_i]
        b_cos = (b_loc * b_glo).sum(dim=0)     # [r_i]

        # ── A-row cosine distance ──
        a_loc = F.normalize(A_local, dim=1)    # [r_i, d_in]
        a_glo = F.normalize(A_global, dim=1)   # [r_i, d_in]
        a_cos = (a_loc * a_glo).sum(dim=1)     # [r_i]

        #per_slot_loss = 2.0 - torch.abs(b_cos + a_cos)
        per_slot_loss = (1.0 - b_cos) + (1.0 - a_cos)

        if self.cfg.slot_weight_type in {"coverage_aware", "coverage-aware"}:
            slot_lambdas = self.get_slot_lambdas(
                r_i,
                int(round_idx or 0),
                device=A_local.device,
                global_rank=global_rank,
            )
            return (per_slot_loss * slot_lambdas).sum()

        if self.cfg.slot_weight_type == "rank_decay":
            slot_lambdas = self.get_slot_lambdas_rank_decay(
                r_i,
                device=A_local.device,
                global_rank=global_rank,
            )
            return (per_slot_loss * slot_lambdas).sum()

        w = self.get_slot_weights(r_i, device=A_local.device, global_rank=global_rank)  # [r_i]
        return (per_slot_loss * w).sum()

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
        L_sub = || P_{B_i} − P_{B_g}^{(r_i)} ||_F²

        where P_X = Q_X Q_X^T, Q_X = orthonormal basis from QR(X).

        During warm-up (round_idx < warmup_rounds) returns 0 because
        B is initialised to 0 and QR would fail.
        """
        if round_idx < self.cfg.warmup_rounds:
            return torch.tensor(0.0, device=B_local.device)

        # QR decomposition → orthonormal basis
        # (torch.linalg.qr is not implemented on MPS; fall back to CPU)
        Q_loc = _safe_qr(B_local)
        Q_glo = _safe_qr(B_global[:, :r_i])

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
        global_rank: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_reg, slot_loss, sub_loss).
        """
        lam_slot, lam_sub = self.get_lambdas(round_idx)
        slot = self.compute_slot_loss(
            B_local, A_local, B_global, A_global, global_rank=global_rank, round_idx=round_idx)
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
