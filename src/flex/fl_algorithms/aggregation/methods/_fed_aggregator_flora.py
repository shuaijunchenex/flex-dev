import torch
from collections import OrderedDict, defaultdict
from typing import Dict, List

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console


class FedAggregator_Flora(AbstractFedAggregator):
    """FLoRA (Wang et al., NeurIPS 2024) — paper-faithful stacking aggregation.

    **Aggregation logic (noise‑free stacking):**

        1. Vertically stack  √w_i · A_i  → A_stack  (Σr_i × in_dim)
        2. Horizontally stack √w_i · B_i → B_stack  (out_dim × Σr_i)
        3. Exact product:

               ΔW = B_stack @ A_stack = Σ_i w_i · B_i @ A_i

    Unlike a naive FedAvg of A and B separately, the stacking product is
    **exact** — it never introduces the ``B̄ @ Ā`` cross-term noise, and it is
    completely independent of each client's individual LoRA rank, so
    heterogeneous ranks are handled natively.

    **Output contract:**

    The aggregator emits:
    - Non‑LoRA keys (LayerNorm, embeddings, bias, …) via standard
      data‑volume‑weighted FedAvg.
    - For every LoRA layer: ``{prefix}.sp_aggregated`` = ΔW (the pure
      stacking product, **not** including the base weight).

    **NO** ``lora_A`` / ``lora_B`` keys are emitted.  The downstream
    ``FloraServerStrategy`` merges ΔW into the frozen backbone
    (W ← W + ΔW) and broadcasts the **full merged W** to all clients.
    Clients then freeze the new W and train fresh A/B from zero each round.

    This differs from the SP‑style pathway where the backbone is permanently
    frozen and the broadcast payload consists of SVD‑factorised LoRA
    components.  In FLoRA the backbone **evolves** round‑by‑round through
    the merge step.

    Rank handling
    -------------
    The paper performs **no rank truncation** during aggregation; the merged
    update is full-rank and noise-free.  The merge step (W ← W + ΔW)
    happens in the server strategy, not here, so the aggregator itself
    remains purely a stacking engine.
    """

    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        self._aggregation_method = "flora"
        self._lora_suffix_A = "lora_A"
        self._lora_suffix_B = "lora_B"
        self._sp_suffix = "sp_aggregated"
        return

    def _before_aggregation(self) -> None:
        return

    def _do_aggregation(self) -> None:
        pairs = self._aggregation_data_dict
        if not isinstance(pairs, list) or len(pairs) == 0:
            raise ValueError("[Flora] Expected list of (state_dict, data_volume).")

        state_dicts = [sd for sd, _vol in pairs]
        weights     = [float(vol) for _sd, vol in pairs]

        console.debug(f"\n[Flora] Aggregating {len(state_dicts)} clients (paper-faithful stacking)...")
        total_data_vol = sum(weights)
        for i, (_sd, vol) in enumerate(pairs):
            console.debug(f"  Client {i}: {vol} samples ({vol / total_data_vol * 100:.1f}%)")

        dev = self._device
        sds_on_device = [{k: v.to(dev) for k, v in sd.items()} for sd in state_dicts]

        ordered = self._aggregate_state_dicts(sds_on_device, weights)
        self._aggregated_weight = ordered

        first_param_name = next(iter(ordered.keys()))
        console.debug(f"[Flora] Aggregated first param mean: {ordered[first_param_name].float().mean():.6f}")

    def _after_aggregation(self) -> None:
        console.debug("[Flora] Aggregation completed.")

    # ---------- key helpers ----------
    @staticmethod
    def _suffix_of(key: str) -> str:
        """Return the last dotted component (e.g. 'layer.lora_A' -> 'lora_A')."""
        return key.rsplit(".", 1)[-1]

    @staticmethod
    def _prefix_of(key: str) -> str:
        """Return the layer prefix before the last dot (shared by lora_A/lora_B)."""
        return key.rsplit(".", 1)[0]

    def _is_lora_A(self, key: str) -> bool:
        return self._suffix_of(key) == self._lora_suffix_A

    def _is_lora_B(self, key: str) -> bool:
        return self._suffix_of(key) == self._lora_suffix_B

    # ---------- core ----------
    def _aggregate_state_dicts(
        self,
        state_dicts: List[Dict[str, torch.Tensor]],
        weights: List[float],
    ) -> "OrderedDict[str, torch.Tensor]":
        """Produce a state dict with FedAvg base params plus, for every LoRA
        layer, a single ``{prefix}.sp_aggregated`` full-rank ΔW entry.

        The output contains **no** ``lora_A`` / ``lora_B`` tensors; rank
        projection is performed downstream by ``svd_split_global_weight`` at
        each node's own rank.
        """
        # ── Paper notation ───────────────────────────────────────────────
        #   N            : number of participating clients         (= len(state_dicts))
        #   n_i          : #samples of client i                    (= weights[i] before norm)
        #   p_i = n_i/Σn : data-volume aggregation weight, Σ p_i = 1   (FLoRA Eq. 4)
        #   (A_i, B_i)   : client i's LoRA factors, rank r_i
        # Normalise raw sample counts → p_i so the merged update is a proper
        # data-weighted average (FLoRA weights every client by p_i).
        tw = float(sum(weights))
        weights = [w / tw for w in weights] if tw > 0 else [1.0 / len(weights)] * len(weights)

        keys = list(state_dicts[0].keys())

        # ── Step 1 — Non-LoRA / frozen-base params ───────────────────────
        # The paper only aggregates the LoRA adapters; the frozen backbone W_0
        # is shared and identical across clients.  For any param that is NOT a
        # LoRA factor (LayerNorm, embeddings, biases, the base `.weight`, …) we
        # take the standard data-weighted average  Σ_i p_i · θ_i.
        base_agg: Dict[str, torch.Tensor] = {}
        for key in keys:
            if self._is_lora_A(key) or self._is_lora_B(key):
                continue
            values = [sd[key] for sd in state_dicts]
            # Integer/bool buffers (for example BatchNorm's
            # ``num_batches_tracked``) do not admit a meaningful weighted
            # average. Casting normalized weights to an integer dtype would
            # also turn every weight below one into zero. Preserve the first
            # client's shared-backbone value, matching the other aggregators.
            if not (torch.is_floating_point(values[0]) or torch.is_complex(values[0])):
                base_agg[key] = values[0].clone()
                continue
            stacked = torch.stack(values, dim=0)                                  # (N, *shape)
            view_shape = (len(weights),) + (1,) * (stacked.dim() - 1)             # broadcast p_i
            w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(*view_shape)
            base_agg[key] = (stacked * w).sum(dim=0)                              # Σ_i p_i · θ_i

        # ── Step 2 — Collect each client's (A_i, B_i) per LoRA layer ─────
        # Group the N clients' LoRA factors by the layer prefix they belong to,
        # so that for every adapter location we hold the lists {A_i} and {B_i}
        # that the stacking operator (Step 3) consumes.
        A_by_prefix: Dict[str, List[torch.Tensor]] = defaultdict(list)
        B_by_prefix: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for key in keys:
            if self._is_lora_A(key):
                A_by_prefix[self._prefix_of(key)] = [sd[key] for sd in state_dicts]   # {A_1 … A_N}
            elif self._is_lora_B(key):
                B_by_prefix[self._prefix_of(key)] = [sd[key] for sd in state_dicts]   # {B_1 … B_N}

        # ── Step 3 — FLoRA stacking aggregation (the core of the paper) ──
        # For each adapter location, stack-merge {A_i},{B_i} into the exact,
        # noise-free update ΔW = Σ_i p_i · B_i A_i, and emit it as the full
        # `out×in` matrix `{prefix}.sp_aggregated`.  No rank truncation here —
        # downstream nodes project ΔW to their own rank at load time.
        ordered: "OrderedDict[str, torch.Tensor]" = OrderedDict(
            (k, base_agg[k]) for k in keys
            if not (self._is_lora_A(k) or self._is_lora_B(k))
        )
        for prefix in A_by_prefix:
            dW = self._flora_delta(A_by_prefix[prefix], B_by_prefix[prefix], weights)  # ΔW = Σ p_i B_i A_i
            ordered[f"{prefix}.{self._sp_suffix}"] = dW

        return ordered

    @staticmethod
    def _flora_delta(
        A_list: List[torch.Tensor],
        B_list: List[torch.Tensor],
        weights: List[float],
    ) -> torch.Tensor:
        """FLoRA stacking — exact, noise-free, rank-agnostic merged update.

        -  A_stack = cat( √w_i · A_i )  along dim=0  →  (Σr_i, in_dim)
        -  B_stack = cat( √w_i · B_i )  along dim=1  →  (out_dim, Σr_i)
        -  ΔW      = B_stack @ A_stack  =  Σ_i w_i · B_i @ A_i  →  (out_dim, in_dim)

        Computed in fp32 for numerical stability, then cast back to the LoRA
        tensors' dtype.  No SVD / truncation is performed here — the result is
        the full-rank update, exactly as the paper prescribes.
        """
        # ── (a) Per-client weight √p_i ───────────────────────────────────
        # Split each data-volume weight p_i into √p_i and apply it to BOTH the
        # A and B side.  Because the two halves recombine multiplicatively in
        # step (d) as (√p_i·B_i)(√p_i·A_i) = p_i·B_iA_i, this injects the exact
        # data weighting p_i into the stacked product without ever averaging A
        # or B on their own (which would create the cross-term noise FLoRA
        # avoids).
        sqrt_weights = [w ** 0.5 for w in weights]                                   # {√p_1 … √p_N}

        # ── (b) Vertically stack the A factors  (FLoRA "stacking" of A) ───
        #   A_stack = [ √p_1·A_1 ; √p_2·A_2 ; … ; √p_N·A_N ]   along rows.
        #   Shapes : A_i is (r_i, in_dim)  →  A_stack is (Σ_i r_i, in_dim).
        #   The heterogeneous ranks r_i simply concatenate — no padding,
        #   no truncation, so every client's full update is preserved.
        A_stack = torch.cat([sw * a for sw, a in zip(sqrt_weights, A_list)], dim=0)  # (Σr_i, in_dim)

        # ── (c) Horizontally stack the B factors (FLoRA "stacking" of B) ──
        #   B_stack = [ √p_1·B_1 , √p_2·B_2 , … , √p_N·B_N ]   along columns.
        #   Shapes : B_i is (out_dim, r_i)  →  B_stack is (out_dim, Σ_i r_i).
        B_stack = torch.cat([sw * b for sw, b in zip(sqrt_weights, B_list)], dim=1)  # (out_dim, Σr_i)

        # ── (d) Single product = exact noise-free merged update ──────────
        #   ΔW = B_stack · A_stack
        #      = Σ_i (√p_i·B_i)(√p_i·A_i) = Σ_i p_i · B_i A_i.
        #   This is FLoRA's key identity: stacking-then-multiplying yields the
        #   *exact* weighted sum of per-client products B_iA_i, unlike naive
        #   FedAvg of A,B which yields (Σp_iB_i)(Σp_iA_i) ≠ Σp_iB_iA_i.
        #   Done in fp32 for numerical stability, then cast back to LoRA dtype.
        dW = B_stack.to(torch.float32) @ A_stack.to(torch.float32)                   # (out_dim, in_dim)
        return dW.to(A_list[0].dtype)
