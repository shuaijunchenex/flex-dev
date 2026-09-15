import torch.nn as nn
import torch
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
from .impl.lora_linear import LoRALinear

class LoRAUtils:
    @staticmethod
    def set_lora_mode_for_model(model: nn.Module, mode: str) -> None:
        """
        Set LoRA mode for all LoRALinear modules inside the model.

            Args:
                model: Target nn.Module that may contain LoRALinear submodules.
                mode:  Mode string understood by LoRALinear.set_lora_mode (e.g., "train", "freeze", "merge", etc.).
            """
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.set_lora_mode(mode)

    @staticmethod
    def get_lora_ranks(
        model: nn.Module,
        suffix_A: str = "lora_A",
        suffix_B: str = "lora_B",
    ) -> Dict[str, int]:
        """
        Scan model parameters and infer the LoRA rank per layer prefix.

        Conventions:
        - A-parameter is named "<prefix>.<suffix_A>" with shape [r, in].
        - B-parameter is named "<prefix>.<suffix_B>" with shape [out, r].
        Typically r = A.shape[0] = B.shape[1].

        Args:
            model:     Model that holds LoRA parameters.
            suffix_A:  Suffix for LoRA A matrix parameter name (default: "lora_A").
            suffix_B:  Suffix for LoRA B matrix parameter name (default: "lora_B").

        Returns:
            Mapping {layer_prefix: rank}. If only one side exists, its dimension is used.
            If both sides exist but ranks disagree, a ValueError is raised.
        """
        ranks: Dict[str, int] = {}
        lora_A_params: Dict[str, torch.Tensor] = {}
        lora_B_params: Dict[str, torch.Tensor] = {}

        # Collect A/B params keyed by prefix
        for name, param in model.named_parameters():
            if name.endswith(suffix_A):
                prefix = name[: -(len(suffix_A) + 1)]  # strip ".lora_A"
                lora_A_params[prefix] = param
            elif name.endswith(suffix_B):
                prefix = name[: -(len(suffix_B) + 1)]  # strip ".lora_B"
                lora_B_params[prefix] = param

        # Infer rank per prefix
        all_prefixes = set(lora_A_params.keys()) | set(lora_B_params.keys())
        for prefix in all_prefixes:
            r: Optional[int] = None
            if prefix in lora_A_params:
                r = int(lora_A_params[prefix].shape[0])  # A: [r, in]
            if prefix in lora_B_params:
                r_B = int(lora_B_params[prefix].shape[1])  # B: [out, r]
                r = r if r is not None else r_B
                # If both exist, they must agree (typical LoRA constraint)
                if r != r_B:
                    raise ValueError(
                        f"Inconsistent LoRA rank for '{prefix}': A={r}, B={r_B}"
                    )
            ranks[prefix] = int(r) if r is not None else 0

        return ranks

    @staticmethod
    def svd_split(
        weight: torch.Tensor,
        r: int,
        method: str = "sqrt",                 # "sqrt":  B = U * sqrt(S),  A = sqrt(S) * V^T
                                            # "full":  B = U * S,        A = V^T
        upcast_min_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Low-rank factorization by truncated SVD that returns LoRA-style (A, B).

        Accepts:
            - 2D weight: [out, in]
            - 4D conv weight: [out_c, in_c, kH, kW] (internally flattened to [out, in])

        Returns:
            (A, B) where:
            - A: [r, in]
            - B: [out, r]
            For conv weights, A/B are returned in 2D; callers should reshape ΔW back
            to [out_c, in_c, kH, kW] during forward reconstruction if needed.

        Notes:
            - Half/bfloat16 inputs are upcast to 'upcast_min_dtype' for numerical stability.
            - The effective rank rr is clamped to [1, min(out, in)].
        """
        if weight.dim() == 2:
            out_dim, in_dim = weight.shape
            W2d = weight
            reshape_back = None
        elif weight.dim() == 4:
            out_c, in_c, kh, kw = weight.shape
            W2d = weight.reshape(out_c, in_c * kh * kw)
            out_dim, in_dim = W2d.shape
            # A/B are 2D; users reconstruct ΔW and reshape to (out_c, in_c, kh, kw) externally.
            reshape_back = (out_c, in_c, kh, kw)
        else:
            raise ValueError(f"svd_split only supports 2D/4D tensors, got {weight.dim()}D")

        rr = max(1, min(int(r), out_dim, in_dim))

        # Upcast for stability if needed
        orig_dtype = W2d.dtype
        work = W2d if W2d.dtype not in (torch.float16, torch.bfloat16) else W2d.to(upcast_min_dtype)

        # U: [out, k], S: [k], Vh: [k, in]; k = min(out, in)
        U, S, Vh = torch.linalg.svd(work, full_matrices=False)
        U_r  = U[:, :rr]
        S_r  = S[:rr]
        Vh_r = Vh[:rr, :]

        if method == "sqrt":
            S_sqrt = torch.sqrt(torch.clamp(S_r, min=0))
            B = U_r * S_sqrt.unsqueeze(0)      # [out, r]
            A = S_sqrt.unsqueeze(1) * Vh_r     # [r, in]
        elif method == "full":
            B = U_r * S_r.unsqueeze(0)         # [out, r]
            A = Vh_r                           # [r, in]
        else:
            raise ValueError(f"Unknown method: {method}")

        # Cast back to original dtype if upcasted
        if B.dtype != orig_dtype:
            B = B.to(orig_dtype)
            A = A.to(orig_dtype)

        return A, B

    @staticmethod
    def svd_split_global_weight(
        global_weight: Dict[str, torch.Tensor],
        rank_dict: Dict[str, int],
        *,
        lora_suffix_A: str = "lora_A",
        lora_suffix_B: str = "lora_B",
        sp_suffix: str = "sp_aggregated",
        svd_method: str = "sqrt",
    ) -> "OrderedDict[str, torch.Tensor]":
        """
        Decompose aggregated weights (e.g., '<prefix>.sp_aggregated') into LoRA A/B
        with target ranks provided by rank_dict.

        Output order per layer:
            <prefix>.weight  -> (optional) <prefix>.bias -> <prefix>.lora_A -> <prefix>.lora_B

        Behavior:
            - Keys ending with f".{sp_suffix}" are SVD-decomposed into A/B matrices.
            - All other keys (non-LoRA params like LayerNorm, position embeddings) are
              passed through unchanged, preserving them for strict state_dict loading.
            - For each sp_aggregated key, look up target rank from rank_dict[prefix].
            - Perform SVD with effective rank eff_r = min(target_r, out, in).
            - If target_r > eff_r, right-/down-pad with zeros to match target_r.

        Args:
            global_weight: Mapping of parameter names to tensors, including '<prefix>.sp_aggregated'.
            rank_dict:     Mapping {prefix: target_rank}. Must contain all prefixes to be split.
            lora_suffix_A / lora_suffix_B: Output suffixes for A/B parameters.
            sp_suffix:     Suffix that marks aggregated base+delta (default: 'sp_aggregated').
            svd_method:    'sqrt' or 'full' (see svd_split).

        Returns:
            OrderedDict of tensors in a stable, layer-grouped order.
        """
        # Decomposing at rank_dict equals "factorize at rank_dict, then slice at
        # rank_dict" (the slice is a no-op), so reuse that single pipeline instead
        # of duplicating the per-layer SVD / passthrough scaffolding.
        factored = LoRAUtils.svd_factorize_global_weight(
            global_weight,
            rank_dict,
            sp_suffix=sp_suffix,
            svd_method=svd_method,
        )
        return LoRAUtils.materialize_lora_from_factors(
            factored,
            rank_dict,
            lora_suffix_A=lora_suffix_A,
            lora_suffix_B=lora_suffix_B,
        )

    # ------------------------------------------------------------------
    # Shared utility: zero-pad LoRA A/B to a target rank
    # ------------------------------------------------------------------

    @staticmethod
    def pad_lora_to_rank(
        A: torch.Tensor,
        B: torch.Tensor,
        target_rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero-pad *A* and *B* to *target_rank* when the current rank is lower.

        ``A`` is assumed to have shape ``(r, in_dim)`` and is padded along dim 0.
        ``B`` is assumed to have shape ``(out_dim, r)`` and is padded along dim 1.

        This is a no-op when ``A.shape[0] >= target_rank``.

        Used by :meth:`svd_split_global_weight` (SP / Flora strategies) to
        ensure decomposed LoRA ranks always match the model's expected
        architecture.
        """
        r = A.shape[0]
        if r >= target_rank:
            return A, B

        A_pad = A.new_zeros((target_rank, A.shape[1]))
        B_pad = B.new_zeros((B.shape[0], target_rank))
        A_pad[:r, :] = A
        B_pad[:, :r] = B
        return A_pad, B_pad

    # ------------------------------------------------------------------
    # Shared layer-dict assembly helpers (used by the SVD / factor pipelines)
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_weight_bias(
        out: "OrderedDict[str, torch.Tensor]",
        src: Dict[str, torch.Tensor],
        prefix: str,
        *,
        weight_fallback: Optional[torch.Tensor] = None,
    ) -> None:
        """Copy '<prefix>.weight' (+ optional '<prefix>.bias') from *src* into *out*.

        When *weight_fallback* is given it is used if '<prefix>.weight' is absent
        (the ΔW proxy in SP aggregation); when None, the weight is emitted only if
        present in *src*.
        """
        w_key = f"{prefix}.weight"
        if weight_fallback is not None:
            out[w_key] = src.get(w_key, weight_fallback)
        elif w_key in src:
            out[w_key] = src[w_key]

        b_key = f"{prefix}.bias"
        if b_key in src:
            out[b_key] = src[b_key]

    @staticmethod
    def _passthrough_remaining(
        out: "OrderedDict[str, torch.Tensor]",
        src: Dict[str, torch.Tensor],
        skip_suffixes: Tuple[str, ...],
    ) -> "OrderedDict[str, torch.Tensor]":
        """Copy keys from *src* into *out* unless already present or ending with a skip suffix.

        Preserves non-LoRA params (LayerNorm, embeddings, …) for strict loading.
        """
        for k, v in src.items():
            if any(k.endswith(s) for s in skip_suffixes):
                continue
            if k in out:
                continue
            out[k] = v.clone().detach() if torch.is_tensor(v) else v
        return out

    # ------------------------------------------------------------------
    # One-time SVD factorization cache (performance optimization)
    # ------------------------------------------------------------------
    #
    # Instead of every consumer (server evaluator + each client) running a fresh
    # SVD on the same full ΔW (``{prefix}.sp_aggregated``), the server SVD-factorizes
    # every layer **once** at the largest rank any consumer will request
    # (``rank_cap`` == the server rank).  The sqrt-factors are cached and then each
    # node recovers its own rank-r LoRA A/B by *slicing* the cached factors.
    #
    # Why slicing is exact: ``svd_split`` (sqrt method) returns the *top* singular
    # components in descending order, so slicing the rank-``r_cap`` factors to the
    # first ``r`` rows/cols equals the rank-``r`` truncated SVD exactly, for any
    # ``r <= r_cap``.

    @staticmethod
    def has_factored_keys(
        weight: Dict[str, torch.Tensor],
        factor_suffix_A: str = "sp_A_cap",
    ) -> bool:
        """Return ``True`` if *weight* holds cached SVD factors (``<prefix>.sp_A_cap``)."""
        return any(k.endswith(f".{factor_suffix_A}") for k in weight)

    @staticmethod
    def svd_factorize_global_weight(
        global_weight: Dict[str, torch.Tensor],
        rank_cap,                                  # int OR Dict[str, int]: per-layer cap (server rank)
        *,
        sp_suffix: str = "sp_aggregated",
        factor_suffix_A: str = "sp_A_cap",
        factor_suffix_B: str = "sp_B_cap",
        svd_method: str = "sqrt",
    ) -> "OrderedDict[str, torch.Tensor]":
        """
        Perform the SVD of every ``<prefix>.sp_aggregated`` (= full ΔW) **once**, at the
        largest rank any consumer will ever request (``rank_cap``), and return the
        cacheable sqrt-factors so that downstream nodes can recover their own
        rank-r LoRA A/B by *slicing* instead of re-running SVD.

        Output (mirrors :meth:`svd_split_global_weight` layout, but in factor form)::

            <prefix>.weight    (base weight, or ΔW proxy when no base exists)
            <prefix>.bias      (optional)
            <prefix>.sp_A_cap  (= A_cap, shape [eff_cap, in])
            <prefix>.sp_B_cap  (= B_cap, shape [out, eff_cap])

        All non-``sp_aggregated`` keys are passed through unchanged.

        Args:
            global_weight: Aggregated weights containing ``<prefix>.sp_aggregated`` = ΔW.
            rank_cap:      Per-prefix cap (dict) or a single int applied to all layers.
                           Must be >= every consumer's per-layer rank so slicing is lossless.
            sp_suffix:     Marks the aggregated full ΔW (default ``sp_aggregated``).
            factor_suffix_A / factor_suffix_B: Output suffixes for the cached factors.
            svd_method:    ``sqrt`` (sliceable) or ``full`` (see :meth:`svd_split`).

        Returns:
            OrderedDict of base/bias/passthrough tensors plus the cached factors.
        """
        def _cap_for(prefix: str) -> int:
            if isinstance(rank_cap, dict):
                if prefix not in rank_cap:
                    raise KeyError(f"rank_cap is missing the cap rank for layer '{prefix}'")
                return int(rank_cap[prefix])
            return int(rank_cap)

        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        for k, W in global_weight.items():
            if not k.endswith(f".{sp_suffix}"):
                continue

            prefix = k[: -len(sp_suffix) - 1]  # strip ".sp_aggregated"

            # Base weight (ΔW proxy when no '<prefix>.weight' exists) + optional bias.
            LoRAUtils._emit_weight_bias(out, global_weight, prefix, weight_fallback=W)

            # The single SVD: factorize at the cap rank; factors stay sliceable.
            eff_cap = max(1, min(_cap_for(prefix), W.shape[0], W.shape[1]))
            A_cap, B_cap = LoRAUtils.svd_split(W, eff_cap, method=svd_method)
            out[f"{prefix}.{factor_suffix_A}"] = A_cap
            out[f"{prefix}.{factor_suffix_B}"] = B_cap

        # Passthrough every remaining non-sp_aggregated key.
        return LoRAUtils._passthrough_remaining(out, global_weight, (f".{sp_suffix}",))

    @staticmethod
    def materialize_lora_from_factors(
        factored_weight: Dict[str, torch.Tensor],
        rank_dict: Dict[str, int],
        *,
        factor_suffix_A: str = "sp_A_cap",
        factor_suffix_B: str = "sp_B_cap",
        lora_suffix_A: str = "lora_A",
        lora_suffix_B: str = "lora_B",
    ) -> "OrderedDict[str, torch.Tensor]":
        """
        Project cached sqrt-factors (``<prefix>.sp_A_cap`` / ``<prefix>.sp_B_cap``) down
        to each layer's own LoRA rank by **slicing** (zero SVD), producing the same
        ``<prefix>.lora_A`` / ``<prefix>.lora_B`` output as :meth:`svd_split_global_weight`.

        For target rank ``r <= r_cap``::

            A = A_cap[:r, :]      B = B_cap[:, :r]

        which equals the rank-r sqrt-SVD **exactly**.  When ``r > r_cap`` (should not
        happen when ranks are capped at the server rank) the result is zero-padded as
        a defensive fallback.

        Output order per layer matches :meth:`svd_split_global_weight`::

            <prefix>.weight -> (optional) <prefix>.bias -> <prefix>.lora_A -> <prefix>.lora_B

        Non-factor keys are passed through unchanged.

        Args:
            factored_weight: Output of :meth:`svd_factorize_global_weight`.
            rank_dict:       Mapping ``{prefix: target_rank}`` for this consumer.
            factor_suffix_A / factor_suffix_B: Suffixes of the cached factors.
            lora_suffix_A / lora_suffix_B:     Output LoRA A/B suffixes.

        Returns:
            OrderedDict of tensors in a stable, layer-grouped order.
        """
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        for k, A_cap in factored_weight.items():
            if not k.endswith(f".{factor_suffix_A}"):
                continue

            prefix = k[: -len(factor_suffix_A) - 1]  # strip ".sp_A_cap"
            b_factor_key = f"{prefix}.{factor_suffix_B}"
            if b_factor_key not in factored_weight:
                raise KeyError(f"Missing '{b_factor_key}' for layer '{prefix}'")
            B_cap = factored_weight[b_factor_key]

            if prefix not in rank_dict:
                raise KeyError(f"rank_dict is missing the rank for layer '{prefix}'")
            target_r = int(rank_dict[prefix])

            # Base weight + optional bias passthrough (resolved at factorize time).
            LoRAUtils._emit_weight_bias(out, factored_weight, prefix)

            # Slice the cached factors to the target rank (exact, no SVD).
            eff_r = max(1, min(target_r, int(A_cap.shape[0])))
            A, B = A_cap[:eff_r, :], B_cap[:, :eff_r]
            if target_r > eff_r:
                A, B = LoRAUtils.pad_lora_to_rank(A, B, target_r)

            out[f"{prefix}.{lora_suffix_A}"] = A
            out[f"{prefix}.{lora_suffix_B}"] = B

        # Passthrough remaining non-factor keys (drop the consumed cap factors).
        return LoRAUtils._passthrough_remaining(
            out, factored_weight, (f".{factor_suffix_A}", f".{factor_suffix_B}")
        )

    @staticmethod
    def cache_svd_factored_matrix(
        node_var,
        global_weight: Dict[str, torch.Tensor],
        rank_cap,
        *,
        sp_suffix: str = "sp_aggregated",
        factor_suffix_A: str = "sp_A_cap",
        factor_suffix_B: str = "sp_B_cap",
        svd_method: str = "sqrt",
        attr_name: str = "sp_factored_weight",
    ) -> "OrderedDict[str, torch.Tensor]":
        """
        Compute the one-time SVD factorization of the aggregated weights and cache
        the sliceable factors on ``node_var``.

        This is the single SVD performed per round: the full ΔW of every LoRA layer
        is factorized once to ``rank_cap`` (the maximum rank any consumer will ask
        for, i.e. the server rank).  The cached factor dict is stored as a dedicated
        node attribute ``node_var.<attr_name>`` (default ``sp_factored_weight``) and
        also returned.  Downstream consumers (server evaluator + every client) then
        obtain their own rank-r LoRA A/B by slicing these factors via
        :meth:`materialize_lora_from_factors` — no further SVD is needed.

        Args:
            node_var:      The FedNodeVars instance to cache the factors on.
            global_weight: Aggregated weights containing ``<prefix>.sp_aggregated``.
            rank_cap:      Per-prefix cap (dict) or a single int (server rank).
            attr_name:     Attribute name to store the cache under on ``node_var``.

        Returns:
            The cached factor OrderedDict.
        """
        factored = LoRAUtils.svd_factorize_global_weight(
            global_weight,
            rank_cap,
            sp_suffix=sp_suffix,
            factor_suffix_A=factor_suffix_A,
            factor_suffix_B=factor_suffix_B,
            svd_method=svd_method,
        )
        # Declare / set the dedicated cache attribute on the node variables.
        setattr(node_var, attr_name, factored)
        return factored

    @staticmethod
    def convert_lora_for_sp_inference(
        base_state_dict: dict,
        lora_template_state_dict: dict,
        suffix_a: str = "lora_A",
        suffix_b: str = "lora_B",
        remove_key: str = "sp_aggregated",
        clone_base: bool = True,
        overwrite_existing: bool = True,
    ) -> OrderedDict:
        """
        Convert a base state_dict for SP-style inference:
        1) Replace each `{prefix}.weight` with `{prefix}.sp_aggregated` when available.
        2) Remove all keys whose last dotted component is `remove_key` (default: 'sp_aggregated').
        3) Ensure LoRA A/B keys (matching the template) exist in the output, initialized to zeros.
           - A/B zeros are dtype/device-aligned to `{prefix}.weight` if it exists; otherwise aligned to template.

        Args:
            base_state_dict: The model state_dict containing base weights and possibly `{prefix}.sp_aggregated`.
            lora_template_state_dict: A state_dict that indicates which LoRA A/B keys (shapes) should exist.
            suffix_a / suffix_b: LoRA suffixes to detect (e.g., 'lora_A'/'lora_B' or 'lora_down'/'lora_up').
            remove_key: Keys whose last dotted component equals this will be removed (default: 'sp_aggregated').
            clone_base: If True, tensors are cloned/detached; otherwise references are kept.
            overwrite_existing: If True, existing A/B in base will be overwritten with zeros.

        Returns:
            OrderedDict: The converted state_dict ready for SP inference.
        """

        # ---------- Helpers ----------
        def last_component(k: str) -> str:
            """Return the last dotted component of a key."""
            return k.rsplit(".", 1)[-1]

        def key_prefix(k: str) -> str:
            """Return the prefix before the last dot; if no dot, return ''."""
            return k.rsplit(".", 1)[0] if "." in k else ""

        def is_exact_remove_key(k: str) -> bool:
            """Remove only if the *last* component equals `remove_key`."""
            return last_component(k) == remove_key

        def is_lora_key(k: str) -> bool:
            """Check if key ends with LoRA A/B suffix."""
            return k.endswith(suffix_a) or k.endswith(suffix_b)

        def split_lora_prefix(k: str) -> Tuple[Optional[str], Optional[str]]:
            """Return (prefix, suffix) if `k` is a LoRA A/B key, else (None, None)."""
            if k.endswith(suffix_a):
                return k[: -len(suffix_a)].rstrip("."), suffix_a
            if k.endswith(suffix_b):
                return k[: -len(suffix_b)].rstrip("."), suffix_b
            return None, None

        # ---------- Stage 0: Index `{prefix}.sp_aggregated` before we drop them ----------
        # We only use entries whose LAST component equals `remove_key` to map prefixes.
        sp_map = {}
        for k, v in base_state_dict.items():
            if is_exact_remove_key(k) and torch.is_tensor(v):
                pref = key_prefix(k)  # '{prefix}.sp_aggregated' -> '{prefix}'
                sp_map[pref] = v

        # ---------- Stage 1: Copy base and simultaneously DROP all '*.sp_aggregated' ----------
        # We drop any key whose last dotted component == remove_key.
        new_sd = OrderedDict()
        for k, v in base_state_dict.items():
            if is_exact_remove_key(k):
                continue  # strip SP cache
            if clone_base and torch.is_tensor(v):
                new_sd[k] = v.detach().clone()
            else:
                new_sd[k] = v

        # ---------- Stage 2: Overwrite `{prefix}.weight` with `{prefix}.sp_aggregated` when present ----------
        # For each prefix found in sp_map, replace the base weight if possible.
        for pref, sp_tensor in sp_map.items():
            weight_key = f"{pref}.weight"
            if weight_key in new_sd and torch.is_tensor(new_sd[weight_key]):
                # Align sp tensor to existing weight dtype/device (safer if base dtype differs from template)
                tgt = new_sd[weight_key]
                sp_aligned = sp_tensor.to(dtype=tgt.dtype, device=tgt.device)
                new_sd[weight_key] = sp_aligned.detach().clone() if clone_base else sp_aligned
            else:
                # If there is no base weight, still install it (use sp dtype/device as-is).
                new_sd[weight_key] = sp_tensor.detach().clone() if clone_base else sp_tensor

        # ---------- Stage 3: Ensure LoRA A/B keys exist (zeros), shapes from template ----------
        added = 0
        for k_tmpl, v_tmpl in lora_template_state_dict.items():
            if not is_lora_key(k_tmpl):
                continue

            pref, sfx = split_lora_prefix(k_tmpl)
            if pref is None:
                continue

            # Prefer aligning zeros to the now-final `{prefix}.weight` if present,
            # otherwise align to the template tensor dtype/device.
            ref = new_sd.get(f"{pref}.weight", v_tmpl)
            if not torch.is_tensor(ref):
                ref = v_tmpl

            zero_like = torch.zeros_like(v_tmpl, dtype=ref.dtype, device=ref.device)
            if overwrite_existing or (k_tmpl not in new_sd):
                new_sd[k_tmpl] = zero_like
                added += 1

        if added == 0:
            # If the template has no LoRA keys, inform the caller (same behavior as before).
            raise ValueError(
                f"No LoRA parameters ending with '{suffix_a}' or '{suffix_b}' were found in the template. "
                f"(All '{remove_key}' caches have been removed, and weights were updated from them where present.)"
            )

        return new_sd

    @staticmethod
    def _natural_list(s: str):
        import re
        """把字符串拆成 [文本/数字] 列表，支持 layers.10 vs layers.2 的自然排序。"""
        parts = []
        for token in s.split("."):
            # 再把 token 中的数字段拆开
            parts.extend(int(t) if t.isdigit() else t for t in re.split(r'(\d+)', token) if t != "")
        return parts

    @staticmethod
    def sort_state_dict_by_suffix(
        state_dict: dict,
        suffix_weight: str = "weight",
        suffix_bias: str   = "bias",
        suffix_a: str      = "lora_A",
        suffix_b: str      = "lora_B",
    ) -> OrderedDict:
        """
        返回一个按层前缀自然排序、且同层内按 [weight, bias, lora_A, lora_B, 其他] 排序的新 OrderedDict。
        """
        prio_map = {
            suffix_weight: 0,
            suffix_bias:   1,
            suffix_a:      2,
            suffix_b:      3,
        }

        def split_prefix_suffix(k: str):
            if "." in k:
                prefix, suf = k.rsplit(".", 1)
            else:
                prefix, suf = k, ""
            return prefix, suf

        def sort_key(k: str):
            prefix, suf = split_prefix_suffix(k)
            prio = prio_map.get(suf, 99)
            return (LoRAUtils._natural_list(prefix), prio, LoRAUtils._natural_list(suf))

        items = sorted(state_dict.items(), key=lambda kv: sort_key(kv[0]))
        return OrderedDict(items)

    @staticmethod
    def replace_with_lora_linear_embedding(
        module: nn.Module,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        merge_weights: bool = True,
        target_module_names: Optional[List[str]] = None,
        replace_embedding: bool = False,
    ) -> None:
        """
        Walk through *module* and replace every nn.Linear → MSLoRALinear,
        and optionally every nn.Embedding → MSEmbedding, copying pretrained weights.

        Args:
            module:              Root module to transform in-place.
            lora_r:              LoRA rank.
            lora_alpha:          LoRA alpha scaling.
            lora_dropout:        Dropout rate for LoRA paths.
            merge_weights:       Whether to merge LoRA into base weight during eval.
            target_module_names: If given, only replace Linear layers whose attribute
                                 name contains one of these substrings (case-insensitive).
                                 Example: ["query", "value"] to only replace attention Q/V.
                                 None means replace ALL Linear layers.
            replace_embedding:   If True, also replace nn.Embedding with MSEmbedding.
        """
        from .impl.lora_ms import MSLoRALinear, MSEmbedding

        def _name_matches(attr_name: str) -> bool:
            if target_module_names is None:
                return True
            attr_lower = attr_name.lower()
            return any(t.lower() in attr_lower for t in target_module_names)

        replacements: list[tuple[nn.Module, str, nn.Module]] = []

        for parent in module.modules():
            for attr_name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and _name_matches(attr_name):
                    new_layer = MSLoRALinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        merge_weights=merge_weights,
                        bias=(child.bias is not None),
                    )
                    with torch.no_grad():
                        new_layer.weight.copy_(child.weight.data)
                        if child.bias is not None:
                            new_layer.bias.copy_(child.bias.data)
                    replacements.append((parent, attr_name, new_layer))

                elif isinstance(child, nn.Embedding) and replace_embedding:
                    new_emb = MSEmbedding(
                        num_embeddings=child.num_embeddings,
                        embedding_dim=child.embedding_dim,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        merge_weights=merge_weights,
                        padding_idx=child.padding_idx,
                        max_norm=child.max_norm,
                        norm_type=child.norm_type,
                        scale_grad_by_freq=child.scale_grad_by_freq,
                        sparse=child.sparse,
                    )
                    with torch.no_grad():
                        new_emb.weight.copy_(child.weight.data)
                    replacements.append((parent, attr_name, new_emb))

        for parent, attr_name, new_module in replacements:
            setattr(parent, attr_name, new_module)

    @staticmethod
    def replace_weight_and_bias(
        sd1: dict,
        sd2: dict,
        *,
        suffixes=("weight", "bias"),
        cast_to_target: bool = True,   # 将 sd2 的张量转成 sd1 对应张量的 dtype/device
        strict_shape: bool = True,     # 形状不一致时抛错；设为 False 则跳过该键
        clone: bool = True             # 返回张量是否 clone().detach()
    ) -> OrderedDict:
        """
        返回: new_sd = sd1 的拷贝，其中 *.weight / *.bias 被 sd2 的对应键替换（若存在）。
        """
        new_sd = OrderedDict()
        for k, v1 in sd1.items():
            # 只匹配最后一段后缀（避免误匹配 running_mean 等）
            tail = k.rsplit(".", 1)[-1]
            should_replace = tail in suffixes and (k in sd2) \
                            and torch.is_tensor(v1) and torch.is_tensor(sd2[k])
            if should_replace:
                v2 = sd2[k]
                if strict_shape and v1.shape != v2.shape:
                    raise ValueError(f"Shape mismatch on '{k}': {v1.shape} vs {v2.shape}")
                if (not strict_shape) and (v1.shape != v2.shape):
                    # 形状不一致且允许跳过
                    new_sd[k] = v1.clone().detach() if (clone and torch.is_tensor(v1)) else v1
                    continue
                if cast_to_target:
                    v2 = v2.to(dtype=v1.dtype, device=v1.device)
                new_sd[k] = v2.clone().detach() if clone else v2
            else:
                new_sd[k] = v1.clone().detach() if (clone and torch.is_tensor(v1)) else v1
        return new_sd

    # ------------------------------------------------------------------
    # FLoRA merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_flora_delta_to_backbone(
        delta: "torch.Tensor",
        backbone_weight: "torch.Tensor",
        *,
        weight_key: str,
    ) -> "torch.Tensor":
        """Fit a 2-D FLoRA product to its backbone weight layout.

        Linear layers already use the same ``[out, in]`` layout as ``B @ A``.
        ``MSEmbedding`` applies the transposed product, while ``MSLoRAConv2d``
        flattens its 4-D kernel into the two LoRA factors. Resolve those two
        layouts explicitly and reject every other mismatch instead of relying
        on PyTorch broadcasting.
        """
        target_shape = tuple(backbone_weight.shape)
        delta_shape = tuple(delta.shape)

        if delta_shape == target_shape:
            return delta

        # MSEmbedding: B @ A is [embedding_dim, num_embeddings], whereas the
        # backbone weight is [num_embeddings, embedding_dim].
        if delta.ndim == 2 and backbone_weight.ndim == 2:
            transposed = delta.transpose(0, 1)
            if tuple(transposed.shape) == target_shape:
                return transposed

        # MSLoRAConv2d uses the same flattened element order as
        # ``(B @ A).view(self.weight.shape)`` in its forward implementation.
        if (
            delta.ndim == 2
            and backbone_weight.ndim == 4
            and delta.numel() == backbone_weight.numel()
        ):
            return delta.reshape(target_shape)

        raise ValueError(
            "Cannot align FLoRA update for "
            f"'{weight_key}': delta shape {delta_shape}, "
            f"backbone shape {target_shape}"
        )

    @staticmethod
    def merge_flora_delta_to_backbone(
        aggregated: Dict[str, "torch.Tensor"],
        backbone: Dict[str, "torch.Tensor"],
        *,
        sp_suffix: str = "sp_aggregated",
    ) -> "OrderedDict[str, torch.Tensor]":
        """Merge FLoRA stacking :math:`\\Delta W` into the frozen backbone.

        For every ``{prefix}.sp_aggregated`` key in *aggregated*, compute

        .. math::

            W_{\\text{new}} = W_{\\text{old}} + \\Delta W_{\\text{flora}}

        where :math:`W_{\\text{old}}` comes from *backbone* and
        :math:`\\Delta W` is the stacking product emitted by
        ``FedAggregator_Flora``.

        Non‑LoRA parameters (LayerNorm, embeddings, bias, …) are passed
        through from *aggregated* as weighted FedAvg results.

        Args:
            aggregated: Aggregator output containing
                ``{prefix}.sp_aggregated`` entries.
            backbone:   The current frozen backbone state dict
                (``W_old``).
            sp_suffix:  Suffix marking ΔW keys (default:
                ``"sp_aggregated"``).

        Returns:
            Merged state dict with updated ``.weight`` entries.
        """
        merged: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        sp_prefixes: set[str] = set()

        for key, val in aggregated.items():
            if key.endswith(f".{sp_suffix}"):
                # ---- LoRA layer: W_new = W_old + ΔW ----
                prefix = key.rsplit(".", 1)[0]
                sp_prefixes.add(prefix)
                weight_key = f"{prefix}.weight"

                if weight_key in backbone:
                    aligned_delta = LoRAUtils._align_flora_delta_to_backbone(
                        val,
                        backbone[weight_key],
                        weight_key=weight_key,
                    )
                    w_old = backbone[weight_key].to(
                        dtype=torch.float32, device=aligned_delta.device
                    )
                    merged[weight_key] = (
                        w_old + aligned_delta.to(torch.float32)
                    ).to(dtype=backbone[weight_key].dtype)
                else:
                    # Fallback (should not happen): use ΔW as-is
                    merged[weight_key] = val
            else:
                # Non‑LoRA parameter (FedAvg result)
                merged[key] = val

        # Preserve backbone keys that were NOT covered by the aggregator
        # (e.g. layers without a LoRA adapter).
        for key, val in backbone.items():
            if key in merged:
                continue
            prefix = key.rsplit(".", 1)[0] if "." in key else ""
            if prefix not in sp_prefixes:
                merged[key] = val

        return merged

    @staticmethod
    def build_eval_weight_with_merged_backbone(
        merged_backbone: Dict[str, "torch.Tensor"],
        model: "nn.Module",
        *,
        suffix_a: str = "lora_A",
        suffix_b: str = "lora_B",
    ) -> "OrderedDict[str, torch.Tensor]":
        """Build an evaluation‑ready state dict from a merged FLoRA backbone.

        Because FLoRA already absorbed :math:`\\Delta W` into *W*, the
        evaluator should use the merged *W* and **zero‑init** LoRA A/B
        so that the forward pass computes :math:`Wx` (not :math:`Wx + BAx`).

        Args:
            merged_backbone: The state dict produced by
                :meth:`merge_flora_delta_to_backbone`.
            model:           The model whose ``state_dict`` keys define the
                expected parameter set (used for shape/dtype reference).
            suffix_a:        LoRA A suffix (default ``"lora_A"``).
            suffix_b:        LoRA B suffix (default ``"lora_B"``).

        Returns:
            State dict ready for ``model.load_state_dict(strict=True)``.
        """
        model_state = model.state_dict()
        eval_weight: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        for key in model_state.keys():
            if key in merged_backbone:
                eval_weight[key] = merged_backbone[key].clone().detach()
            elif key.endswith(f".{suffix_a}") or key.endswith(f".{suffix_b}"):
                eval_weight[key] = torch.zeros_like(model_state[key])
            else:
                eval_weight[key] = model_state[key].clone().detach()

        return eval_weight
