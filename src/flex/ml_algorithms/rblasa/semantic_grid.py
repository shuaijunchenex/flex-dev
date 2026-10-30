"""
Semantic Grid utilities for RBLA-SASG (Rank-Based LoRA Aggregation with
Semantic Anchoring and Semantic Grid).
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Semantic-grid mapping
# ---------------------------------------------------------------------------

class LinearSemanticGrid:
    """Uniform (evenly-spaced) semantic slot mapping.

    Spreads ``r_i`` slots evenly across the global grid ``[1, R_max]``,
    with the first slot always anchored at 1 and the last at ``R_max``.
    This is the original RBLA-SASG fixed mapping.

    Parameters
    ----------
    max_rank : int
        Global maximum rank :math:`R_{\max}`.
    """

    def __init__(self, max_rank: int) -> None:
        if max_rank < 1:
            raise ValueError(f"max_rank must be ≥ 1, got {max_rank}")
        self._R_max = max_rank

    @property
    def max_rank(self) -> int:
        return self._R_max

    def get_slot_mapping(self, local_rank: int) -> List[int]:
        """Compute slot set :math:`\Phi_i` by uniform spacing.

        Args:
            local_rank: Client's local LoRA rank :math:`r_i` (≥ 1).

        Returns:
            1‑based slot indices, length ``local_rank``, sorted ascending.
        """
        r_i = local_rank
        R = self._R_max

        if r_i <= 0 or r_i > R:
            raise ValueError(f"local_rank {r_i} out of range [1, {R}]")
        if R == 1:
            return [1]
        if r_i == 1:
            return [1]

        slots: List[int] = []
        for k in range(1, r_i + 1):
            s = 1 + int(round((k - 1) * (R - 1) / (r_i - 1)))
            slots.append(s)
        return slots


def semantic_grid_mapping(
    local_rank: int,
    max_rank: int,
) -> List[int]:
    """Convenience wrapper — delegates to :class:`LinearSemanticGrid`.

    .. deprecated::
        Prefer :class:`LinearSemanticGrid` or :class:`AdaptiveSemanticGrid`
        for new code.
    """
    return LinearSemanticGrid(max_rank).get_slot_mapping(local_rank)


def semantic_slot_set(local_rank: int, max_rank: int) -> Set[int]:
    """Return the set of global semantic slots for a client (convenience)."""
    return set(semantic_grid_mapping(local_rank, max_rank))


def inverse_slot_index(slots: List[int], global_slot: int) -> int:
    """
    Given a client's ordered slot list and a global slot number,
    return the 0-based local rank index.

    Raises ValueError if *global_slot* is not in *slots*.
    """
    try:
        return slots.index(global_slot)
    except ValueError:
        raise ValueError(f"global_slot {global_slot} not in client slot set {slots}")


# ---------------------------------------------------------------------------
# Semantic Anchoring loss
# ---------------------------------------------------------------------------

def semantic_anchoring_loss(
    A_local: torch.Tensor,          # [r_i, d_in]
    B_local: torch.Tensor,          # [d_out, r_i]
    A_global: torch.Tensor,         # [R_max, d_in]
    B_global: torch.Tensor,         # [d_out, R_max]
    phi: List[int],                 # semantic grid mapping
    omega: List[float] | None = None,  # per-slot importance weights
) -> torch.Tensor:
    """
    Compute semantic anchoring loss (vectorized).

    L_SA = Σ_k ω_{φ(k)} [ (1 - cos(B_i[:,k], B_g[:, φ(k)]))
                         + (1 - cos(A_i[k,:], A_g[φ(k), :])) ]

    Args:
        A_local:   Client's A matrix [r_i, d_in].
        B_local:   Client's B matrix [d_out, r_i].
        A_global:  Global A matrix [R_max, d_in].
        B_global:  Global B matrix [d_out, R_max].
        phi:       Semantic grid slots for this client (1-indexed).
        omega:     Per-slot importance weights (len = R_max, 1-indexed).
                   Defaults to uniform weight 1.0.

    Returns:
        Scalar loss tensor.
    """
    if omega is None:
        omega = [1.0] * (B_global.shape[1] + 1)  # +1 for 1-indexed access

    device = A_local.device
    dtype = A_local.dtype

    # 0-based indices into global matrices
    idx = torch.tensor([s - 1 for s in phi], device=device)

    # B anchoring: cosine between corresponding columns
    #   B_local: [d_out, r_i], B_global: [d_out, R_max]
    #   Gather B_global columns → [d_out, r_i]
    B_g = B_global[:, idx]                     # [d_out, r_i]
    cos_b = F.cosine_similarity(B_local.T, B_g.T, dim=1)  # [r_i]

    # A anchoring: cosine between corresponding rows
    #   A_local: [r_i, d_in], A_global: [R_max, d_in]
    A_g = A_global[idx, :]                     # [r_i, d_in]
    cos_a = F.cosine_similarity(A_local, A_g, dim=1)  # [r_i]

    # Per-slot weights
    w = torch.tensor(
        [omega[s] if s < len(omega) else 1.0 for s in phi],
        device=device, dtype=dtype,
    )  # [r_i]

    per_slot_loss = 2.0 - cos_b - cos_a
    loss = (w * per_slot_loss).sum() / w.sum().clamp_min(torch.finfo(dtype).eps)
    return loss


def make_slot_importance_weights(
    max_rank: int,
    alpha: float = 0.0,
) -> List[float]:
    """
    Build semantic-slot importance weights ω_s.

    - alpha = 0.0 → uniform weight 1.0 for all slots.
    - alpha > 0.0 → stronger anchoring for early (shared-core) slots:
        ω_s = 1 / (1 + α * (s - 1))

    Returns:
        List[float] of length max_rank + 1 (1-indexed, index 0 unused).
    """
    weights = [0.0]  # index 0 placeholder
    for s in range(1, max_rank + 1):
        if alpha <= 0.0:
            weights.append(1.0)
        else:
            weights.append(1.0 / (1.0 + alpha * (s - 1)))
    return weights


def alignment_lambda(
    t: int,
    lambda_0: float = 1.0,
    lambda_min: float = 0.0,
    beta: float = 0.0,
) -> float:
    """
    Decay schedule for semantic anchoring coefficient λ_t.

    λ_t = max(λ_min, λ_0 * exp(-β * t))

    Args:
        t:           Current communication round (0-indexed).
        lambda_0:    Initial alignment strength.
        lambda_min:  Minimum alignment strength.
        beta:        Decay rate.

    Returns:
        Scalar λ_t.
    """
    return max(lambda_min, lambda_0 * math.exp(-beta * t))


# ---------------------------------------------------------------------------
# Slot-based weight slice helpers
# ---------------------------------------------------------------------------

def slice_global_for_client(
    A_global: torch.Tensor,       # [R_max, d_in]
    B_global: torch.Tensor,       # [d_out, R_max]
    phi: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract assigned semantic slots from global factors for a client.

    Returns (A_slice [r_i, d_in], B_slice [d_out, r_i]).
    """
    indices = [s - 1 for s in phi]  # 0-based
    A_slice = A_global[indices, :].clone()
    B_slice = B_global[:, indices].clone()
    return A_slice, B_slice


def scatter_client_to_global(
    A_global: torch.Tensor,       # [R_max, d_in] (mutated in-place)
    B_global: torch.Tensor,       # [d_out, R_max] (mutated in-place)
    A_local: torch.Tensor,        # [r_i, d_in]
    B_local: torch.Tensor,        # [d_out, r_i]
    phi: List[int],
) -> None:
    """
    Scatter client LoRA factors into global factor slots (in-place).
    """
    for k, s in enumerate(phi):
        idx = s - 1
        A_global[idx, :] = A_local[k, :].to(device=A_global.device, dtype=A_global.dtype)
        B_global[:, idx] = B_local[:, k].to(device=B_global.device, dtype=B_global.dtype)


# ======================================================================
# Adaptive Semantic Grid
# ======================================================================

class AdaptiveSemanticGrid:
    r"""Dynamic semantic slot selection for RBLA-SASG.

    Replaces the fixed uniform semantic-grid mapping with a two-phase
    strategy:

    **Warm-up phase** (:math:`t < T_w`)
        Uses the original uniform :func:`semantic_grid_mapping`.

    **Adaptive phase** (:math:`t \ge T_w`)
        Ranks intermediate semantic slots by their smoothed update
        strength :math:`\bar{E}_s` and selects the top-:math:`(r_i-2)`
        intermediate slots, while always preserving the endpoint anchors
        :math:`s=1` and :math:`s=R_{\max}` for clients with rank ≥ 2.

    Slot update strength is defined as the Frobenius norm of the
    rank‑1 LoRA update:

    .. math::
        E_s = \|B_g[:,s]\|_2 \cdot \|A_g[s,:]\|_2
            = \|\Delta W_s\|_F.

    An exponential moving average is maintained per slot:

    .. math::
        \bar{E}_s^{(t)} = \rho\,\bar{E}_s^{(t-1)} + (1-\rho)\,E_s^{(t)}.

    Parameters
    ----------
    max_rank : int
        Global maximum rank :math:`R_{\max}`.
    warmup_rounds : int
        Number of warm‑up rounds :math:`T_w` (default 3).
    rho : float
        EMA momentum :math:`\rho` (default 0.9).
    eps : float
        Numerical epsilon (default 1e-12).
    """

    def __init__(
        self,
        max_rank: int,
        warmup_rounds: int = 3,
        rho: float = 0.9,
        eps: float = 1e-12,
    ) -> None:
        if max_rank < 1:
            raise ValueError(f"max_rank must be ≥ 1, got {max_rank}")
        self._R_max = max_rank
        self._T_w = warmup_rounds
        self._rho = rho
        self._eps = eps

        # EMA slot strengths  Ē_s  (1-indexed, index 0 unused)
        self._ema: List[float] = [0.0] * (max_rank + 1)
        self._ema_initialized: bool = False
        self._round_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def max_rank(self) -> int:
        return self._R_max

    @property
    def warmup_rounds(self) -> int:
        return self._T_w

    @property
    def round_counter(self) -> int:
        return self._round_counter

    @property
    def ema_strengths(self) -> List[float]:
        """Return a copy of the current EMA strengths (1-indexed)."""
        return list(self._ema)

    def is_warmup(self) -> bool:
        """Return ``True`` during the warm‑up phase."""
        return self._round_counter < self._T_w

    # ------------------------------------------------------------------
    # Slot mapping
    # ------------------------------------------------------------------

    def get_slot_mapping(self, local_rank: int) -> List[int]:
        """Compute semantic slot set :math:`\Phi_i` for a client.

        During warm‑up, delegates to :func:`semantic_grid_mapping`.
        After warm‑up, selects intermediate slots by largest EMA strength.

        Args:
            local_rank: Client's local LoRA rank :math:`r_i` (≥ 1).

        Returns:
            1‑based slot indices, length ``local_rank``, sorted ascending.
        """
        r_i = local_rank
        R = self._R_max

        if r_i <= 0 or r_i > R:
            raise ValueError(f"local_rank {r_i} out of range [1, {R}]")
        if R == 1:
            return [1]

        # ── Warm‑up: uniform mapping ──
        if self.is_warmup() or not self._ema_initialized:
            return semantic_grid_mapping(r_i, R)

        # ── Adaptive mapping ──
        if r_i == 1:
            return [1]
        if r_i == 2:
            return [1, R]

        # r_i > 2: endpoints + top-(r_i-2) intermediate slots
        selected: Set[int] = {1, R}
        middle_slots = list(range(2, R))  # [2, 3, …, R-1]
        k = r_i - 2  # how many intermediate slots to pick

        if k >= len(middle_slots):
            selected.update(middle_slots)
        else:
            # Pick top-k intermediate slots by EMA strength
            scored = [(self._ema[s], s) for s in middle_slots]
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, s in scored[:k]:
                selected.add(s)

        return sorted(selected)

    # ------------------------------------------------------------------
    # Update EMA from global factors
    # ------------------------------------------------------------------

    def update_strengths(
        self,
        A_global: torch.Tensor,   # [R_max, d_in]
        B_global: torch.Tensor,   # [d_out, R_max]
    ) -> None:
        r"""Compute per‑slot update strengths and update the EMA.

        .. math::
            E_s = \|B_g[:,s]\|_2 \cdot \|A_g[s,:]\|_2

        Should be called **once per communication round** after
        aggregation, before the next broadcast.

        Args:
            A_global: Global :math:`A_g` matrix ``[R_max, d_in]``.
            B_global: Global :math:`B_g` matrix ``[d_out, R_max]``.
        """
        self._round_counter += 1

        with torch.no_grad():
            b_norms = B_global.norm(p=2, dim=0)   # [R_max]
            a_norms = A_global.norm(p=2, dim=1)   # [R_max]
            E = b_norms * a_norms                  # [R_max]

        for s in range(1, self._R_max + 1):
            val = float(E[s - 1].item()) + self._eps
            if not self._ema_initialized:
                self._ema[s] = val
            else:
                self._ema[s] = self._rho * self._ema[s] + (1.0 - self._rho) * val

        self._ema_initialized = True

    def update_strengths_per_prefix(
        self,
        A_dict: Dict[str, torch.Tensor],   # prefix → [R_prefix, d_in]
        B_dict: Dict[str, torch.Tensor],   # prefix → [d_out, R_prefix]
    ) -> Dict[str, List[float]]:
        r"""Convenience method for per‑prefix LoRA layers.

        Updates a separate EMA for each prefix (keyed by prefix name).
        Returns a dict ``{prefix: ema_list}`` where each list is
        1‑indexed (index 0 = 0.0).

        .. note::
            This method creates and caches per‑prefix
            :class:`AdaptiveSemanticGrid` instances internally.
        """
        results: Dict[str, List[float]] = {}
        for prefix in A_dict:
            if prefix not in B_dict:
                continue
            A = A_dict[prefix]
            B = B_dict[prefix]
            R_p = A.shape[0]
            # Lazily create / retrieve per‑prefix grid
            cache_attr = f"_grid_{prefix.replace('.', '_')}"
            grid = getattr(self, cache_attr, None)
            if grid is None or grid.max_rank != R_p:
                grid = AdaptiveSemanticGrid(
                    max_rank=R_p,
                    warmup_rounds=self._T_w,
                    rho=self._rho,
                    eps=self._eps,
                )
                setattr(self, cache_attr, grid)
            grid.update_strengths(A, B)
            results[prefix] = list(grid._ema)
        return results

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset round counter and EMA state."""
        self._round_counter = 0
        self._ema = [0.0] * (self._R_max + 1)
        self._ema_initialized = False
