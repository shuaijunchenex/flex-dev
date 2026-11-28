"""Compact canonicalization of LoRA factor pairs.

The production implementation in this module only factorizes a compact core
whose dimensions are at most ``R x R``.  It never materializes the dense
``d_out x d_in`` LoRA update.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any, Mapping
import warnings

import torch


@dataclass(frozen=True)
class CanonicalizationResult:
    """Canonical LoRA factors and compact-factor diagnostics."""

    lora_A: torch.Tensor
    lora_B: torch.Tensor
    singular_values: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class StateDictCanonicalizationResult:
    """Result of canonicalizing every complete LoRA pair in a state dict."""

    state_dict: Mapping[str, Any]
    singular_values: dict[str, torch.Tensor]
    diagnostics: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CanonicalizationConfig:
    """Round scheduling and numerical options for server canonicalization."""

    enabled: bool = False
    start_round: int = 0
    interval: int = 1
    deterministic_sign: bool = True
    svd_fallback: bool = True
    log_diagnostics: bool = True
    eps: float = 1e-12
    compute_dtype: torch.dtype | None = None
    ordering: str = "singular_value"
    activation_chunk_size: int | None = 4096
    activation_fallback: bool = True
    overcomplete_policy: str = "error"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CanonicalizationConfig":
        config = value or {}

        # The project's DictPath.get() returns None for a missing key even when
        # a default argument is supplied, so normalize missing values here.
        def get_value(key: str, default: Any) -> Any:
            found = config.get(key, default)
            return default if found is None else found

        start_round = int(get_value("start_round", 0))
        interval = int(get_value("interval", 1))
        eps = float(get_value("eps", 1e-12))
        if start_round < 0:
            raise ValueError("canonicalization.start_round must be non-negative")
        if interval <= 0:
            raise ValueError("canonicalization.interval must be positive")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("canonicalization.eps must be finite and positive")

        ordering = str(get_value("ordering", "singular_value")).strip().lower()
        if ordering not in {"singular_value", "activation_aware"}:
            raise ValueError(
                "canonicalization.ordering must be 'singular_value' or 'activation_aware'"
            )

        overcomplete_policy = str(
            get_value("overcomplete_policy", "error")
        ).strip().lower()
        if overcomplete_policy not in {"error", "zero_pad"}:
            raise ValueError(
                "canonicalization.overcomplete_policy must be 'error' or 'zero_pad'"
            )

        chunk_value = get_value("activation_chunk_size", 4096)
        activation_chunk_size = int(chunk_value) if chunk_value is not None else None
        if activation_chunk_size is not None and activation_chunk_size <= 0:
            raise ValueError("canonicalization.activation_chunk_size must be positive or null")

        dtype_value = get_value("compute_dtype", None)
        if dtype_value is None:
            compute_dtype = None
        elif isinstance(dtype_value, torch.dtype):
            compute_dtype = dtype_value
        else:
            dtype_name = str(dtype_value).lower().removeprefix("torch.")
            dtype_by_name = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float64": torch.float64,
                "fp64": torch.float64,
            }
            if dtype_name not in dtype_by_name:
                raise ValueError(
                    "canonicalization.compute_dtype must be float32/fp32 or float64/fp64"
                )
            compute_dtype = dtype_by_name[dtype_name]

        if compute_dtype not in (None, torch.float32, torch.float64):
            raise ValueError("canonicalization.compute_dtype must be torch.float32 or torch.float64")
        return cls(
            enabled=bool(get_value("enabled", False)),
            start_round=start_round,
            interval=interval,
            deterministic_sign=bool(get_value("deterministic_sign", True)),
            svd_fallback=bool(get_value("svd_fallback", True)),
            log_diagnostics=bool(get_value("log_diagnostics", True)),
            eps=eps,
            compute_dtype=compute_dtype,
            ordering=ordering,
            activation_chunk_size=activation_chunk_size,
            activation_fallback=bool(get_value("activation_fallback", True)),
            overcomplete_policy=overcomplete_policy,
        )

    def should_run(self, round_index: int) -> bool:
        """Return whether canonicalization is scheduled for a zero-based round."""

        return (
            self.enabled
            and round_index >= self.start_round
            and (round_index - self.start_round) % self.interval == 0
        )


def _context_suffix(context: str | None) -> str:
    return f" for {context}" if context else ""


def _check_finite(tensor: torch.Tensor, name: str, context: str | None) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"{name} contains NaN or Inf{_context_suffix(context)}")


def _select_compute_dtype(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    compute_dtype: torch.dtype | None,
) -> torch.dtype:
    if compute_dtype is not None:
        if compute_dtype not in (torch.float32, torch.float64):
            raise ValueError("compute_dtype must be torch.float32 or torch.float64")
        return compute_dtype
    if lora_A.dtype == torch.float64 or lora_B.dtype == torch.float64:
        return torch.float64
    return torch.float32


def _compact_svd(
    core: torch.Tensor,
    *,
    svd_fallback: bool,
    context: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        return torch.linalg.svd(core, full_matrices=False)
    except RuntimeError as first_error:
        if not svd_fallback:
            raise RuntimeError(
                f"compact SVD failed{_context_suffix(context)} and fallback is disabled"
            ) from first_error
        try:
            # One deliberately bounded retry: the core is only R x R.
            cpu_core = core.detach().to(device="cpu", dtype=torch.float64)
            u, singular_values, vh = torch.linalg.svd(cpu_core, full_matrices=False)
            return (
                u.to(device=core.device, dtype=core.dtype),
                singular_values.to(device=core.device, dtype=core.dtype),
                vh.to(device=core.device, dtype=core.dtype),
            )
        except RuntimeError as fallback_error:
            raise RuntimeError(
                f"compact SVD failed{_context_suffix(context)}; CPU float64 fallback also failed"
            ) from fallback_error


def _activation_fallback_diagnostics(
    singular_values: torch.Tensor,
    reason: str,
) -> dict[str, Any]:
    """Describe a requested activation-aware ordering that used SVD order instead."""

    rank = int(singular_values.numel())
    return {
        "ordering_requested": "activation_aware",
        "ordering_applied": "singular_value",
        "activation_available": False,
        "activation_sample_count": 0,
        "original_singular_values": singular_values.tolist(),
        "activation_importance": [],
        "functional_scores": [],
        "ordering_indices": list(range(rank)),
        "cumulative_functional_energy": [],
        "functional_error_curve": [],
        "functional_energy_near_zero": False,
        "fallback_reason": reason,
    }


def _apply_activation_aware_ordering(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    singular_values: torch.Tensor,
    activation_input: torch.Tensor | None,
    *,
    activation_chunk_size: int | None,
    activation_fallback: bool,
    eps: float,
    context: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Jointly reorder balanced SVD factors by their activation response energy.

    For row-major activations ``X=[m,d_in]`` and the canonical SVD atoms
    ``C_s = sigma_s u_s v_s.T``, orthogonality of the output directions gives

    ``||X @ (DeltaW - sum_{s in S} C_s).T||_F^2``
    ``= sum_{s not in S} sigma_s^2 ||X @ v_s||_2^2``.

    Selecting the largest scores is therefore optimal among the existing SVD
    rank-one atoms.  It is not a globally optimal activation-weighted
    approximation over all possible rank-r matrices.  No dense update or input
    covariance is materialized here.
    """

    def fail_or_fallback(
        reason: str,
        error_type: type[Exception] = ValueError,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        message = f"activation-aware ordering failed{_context_suffix(context)}: {reason}"
        if not activation_fallback:
            raise error_type(message)
        return (
            lora_A,
            lora_B,
            singular_values,
            _activation_fallback_diagnostics(singular_values, reason),
        )

    if activation_input is None:
        return fail_or_fallback("activation_input is None")
    if not isinstance(activation_input, torch.Tensor):
        return fail_or_fallback("activation_input must be a torch.Tensor", TypeError)
    if activation_input.ndim < 1:
        return fail_or_fallback("activation_input must have at least one dimension")
    if int(activation_input.shape[-1]) != int(lora_A.shape[1]):
        return fail_or_fallback(
            "activation_input last dimension "
            f"{int(activation_input.shape[-1])} does not match d_in={int(lora_A.shape[1])}"
        )
    if not torch.is_floating_point(activation_input):
        return fail_or_fallback("activation_input must have floating-point dtype", TypeError)

    sample_count = int(activation_input.numel() // max(int(lora_A.shape[1]), 1))
    if sample_count == 0:
        return fail_or_fallback("activation_input contains zero samples")
    if not bool(torch.isfinite(activation_input).all().item()):
        return fail_or_fallback("activation_input contains NaN or Inf", FloatingPointError)

    if activation_chunk_size is not None and activation_chunk_size <= 0:
        raise ValueError("activation_chunk_size must be positive or None")
    chunk_size = sample_count if activation_chunk_size is None else activation_chunk_size
    activation_flat = activation_input.reshape(sample_count, int(lora_A.shape[1]))
    projected_square_sum = singular_values.new_zeros(singular_values.shape)
    factor_transpose = lora_A.transpose(0, 1)
    for start in range(0, sample_count, chunk_size):
        activation_chunk = activation_flat[start : start + chunk_size].to(
            device=lora_A.device,
            dtype=lora_A.dtype,
        )
        projected = activation_chunk @ factor_transpose
        projected_square_sum.add_(projected.square().sum(dim=0))

    projected_energy = projected_square_sum / float(sample_count)
    functional_scores = singular_values * projected_energy
    if not bool(torch.isfinite(functional_scores).all().item()):
        return fail_or_fallback(
            "activation projection produced NaN or Inf functional scores",
            FloatingPointError,
        )

    # The incoming singular values are already descending. A stable score sort
    # therefore breaks ties by larger singular value and then original index.
    order = torch.argsort(functional_scores, descending=True, stable=True)
    ordered_A = lora_A[order, :]
    ordered_B = lora_B[:, order]
    ordered_singular_values = singular_values[order]
    ordered_scores = functional_scores[order]

    activation_importance = torch.zeros_like(projected_energy)
    positive_singular_values = singular_values > 0
    activation_importance[positive_singular_values] = (
        projected_energy[positive_singular_values]
        / singular_values[positive_singular_values]
    )
    total_energy = ordered_scores.sum()
    energy_near_zero = bool((total_energy <= eps).item())
    retained_energy = ordered_scores.cumsum(dim=0)
    functional_error = (
        (total_energy - retained_energy).clamp_min(0.0) / (total_energy + eps)
    ).clamp(0.0, 1.0)
    if energy_near_zero:
        cumulative_energy = torch.zeros_like(ordered_scores)
    else:
        cumulative_energy = (retained_energy / total_energy).clamp(0.0, 1.0)

    diagnostics: dict[str, Any] = {
        "ordering_requested": "activation_aware",
        "ordering_applied": "activation_aware",
        "activation_available": True,
        "activation_sample_count": sample_count,
        "original_singular_values": singular_values.tolist(),
        "activation_importance": activation_importance.tolist(),
        "functional_scores": functional_scores.tolist(),
        "ordering_indices": order.tolist(),
        "cumulative_functional_energy": cumulative_energy.tolist(),
        "functional_error_curve": functional_error.tolist(),
        "functional_energy_near_zero": energy_near_zero,
        "fallback_reason": None,
    }
    return ordered_A, ordered_B, ordered_singular_values, diagnostics


@torch.no_grad()
def canonicalize_lora_factor_pair(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
    deterministic_sign: bool = True,
    svd_fallback: bool = True,
    eps: float = 1e-12,
    context: str | None = None,
    ordering: str = "singular_value",
    activation_input: torch.Tensor | None = None,
    activation_chunk_size: int | None = None,
    activation_fallback: bool = True,
    overcomplete_policy: str = "error",
) -> CanonicalizationResult:
    """Canonicalize ``A=[R,d_in]`` and ``B=[d_out,R]`` through an ``R x R`` core.

    Distinct singular directions are invariant to invertible factor gauges.  A
    deterministic pivot rule also removes each vector's +/- ambiguity.  A
    repeated-singular-value subspace still has unavoidable orthogonal rotation
    freedom; sign fixing does not make a basis inside that subspace unique.
    """

    if not isinstance(lora_A, torch.Tensor) or not isinstance(lora_B, torch.Tensor):
        raise TypeError(f"LoRA A and B must be tensors{_context_suffix(context)}")
    if lora_A.ndim != 2 or lora_B.ndim != 2:
        raise ValueError(
            f"LoRA factors must be 2D, got A{tuple(lora_A.shape)} and B{tuple(lora_B.shape)}"
            f"{_context_suffix(context)}"
        )
    if not torch.is_floating_point(lora_A) or not torch.is_floating_point(lora_B):
        raise TypeError(f"LoRA factors must have floating-point dtype{_context_suffix(context)}")
    if lora_A.device != lora_B.device:
        raise ValueError(
            f"LoRA A and B must share a device, got {lora_A.device} and {lora_B.device}"
            f"{_context_suffix(context)}"
        )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    normalized_ordering = str(ordering).strip().lower()
    if normalized_ordering not in {"singular_value", "activation_aware"}:
        raise ValueError("ordering must be 'singular_value' or 'activation_aware'")
    if activation_chunk_size is not None and activation_chunk_size <= 0:
        raise ValueError("activation_chunk_size must be positive or None")
    normalized_overcomplete_policy = str(overcomplete_policy).strip().lower()
    if normalized_overcomplete_policy not in {"error", "zero_pad"}:
        raise ValueError("overcomplete_policy must be 'error' or 'zero_pad'")

    rank, d_in = lora_A.shape
    d_out, b_rank = lora_B.shape
    if rank != b_rank:
        raise ValueError(
            f"LoRA rank mismatch: A has {rank} rows but B has {b_rank} columns"
            f"{_context_suffix(context)}"
        )
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive{_context_suffix(context)}")
    intrinsic_rank = min(rank, d_in, d_out)
    if rank > intrinsic_rank and normalized_overcomplete_policy == "error":
        raise ValueError(
            f"LoRA rank {rank} exceeds min(d_in={d_in}, d_out={d_out}); "
            f"over-dimension padding is not supported{_context_suffix(context)}"
        )

    _check_finite(lora_A, "LoRA A", context)
    _check_finite(lora_B, "LoRA B", context)
    work_dtype = _select_compute_dtype(lora_A, lora_B, compute_dtype)
    a_work = lora_A.to(dtype=work_dtype)
    b_work = lora_B.to(dtype=work_dtype)

    q_b, r_b = torch.linalg.qr(b_work, mode="reduced")
    q_a, r_a = torch.linalg.qr(a_work.transpose(0, 1), mode="reduced")
    core = r_b @ r_a.transpose(0, 1)
    _check_finite(core, "compact LoRA core", context)

    u, singular_values, vh = _compact_svd(
        core,
        svd_fallback=svd_fallback,
        context=context,
    )
    # torch.linalg.svd promises descending order; sort defensively so slot zero
    # always means the globally strongest direction.
    order = torch.argsort(singular_values, descending=True, stable=True)
    singular_values = singular_values[order].clamp_min(0)
    u = u[:, order]
    vh = vh[order, :]
    svd_singular_values = singular_values
    sqrt_sigma = torch.sqrt(singular_values)

    # Scaling columns/rows avoids even an unnecessary explicit diagonal matrix.
    b_canonical = (q_b @ u) * sqrt_sigma.unsqueeze(0)
    a_canonical = sqrt_sigma.unsqueeze(1) * (vh @ q_a.transpose(0, 1))

    if deterministic_sign:
        # This fixes only per-vector sign freedom, not rotations inside repeated
        # singular-value subspaces (see the public docstring above).
        a_indices = a_canonical.abs().argmax(dim=1, keepdim=True)
        a_pivots = a_canonical.gather(1, a_indices).squeeze(1)
        b_indices = b_canonical.abs().argmax(dim=0, keepdim=True)
        b_pivots = b_canonical.gather(0, b_indices).squeeze(0)
        pivots = torch.where(a_pivots.abs() >= eps, a_pivots, b_pivots)
        signs = torch.where(pivots < 0, -torch.ones_like(pivots), torch.ones_like(pivots))
        a_canonical = signs.unsqueeze(1) * a_canonical
        b_canonical = b_canonical * signs.unsqueeze(0)

    activation_diagnostics: dict[str, Any] | None = None
    if normalized_ordering == "activation_aware":
        (
            a_canonical,
            b_canonical,
            singular_values,
            activation_diagnostics,
        ) = _apply_activation_aware_ordering(
            a_canonical,
            b_canonical,
            singular_values,
            activation_input,
            activation_chunk_size=activation_chunk_size,
            activation_fallback=activation_fallback,
            eps=eps,
            context=context,
        )

    canonical_rank = int(singular_values.numel())
    if canonical_rank < rank:
        # The dense update has at most min(d_in, d_out) non-zero singular
        # directions.  Keep those canonical directions as a leading prefix and
        # restore the configured LoRA shape with exact zero slots.  This keeps
        # strict state-dict loading and heterogeneous prefix broadcast intact.
        padded_a = a_canonical.new_zeros((rank, d_in))
        padded_b = b_canonical.new_zeros((d_out, rank))
        padded_singular_values = singular_values.new_zeros(rank)
        padded_a[:canonical_rank, :] = a_canonical
        padded_b[:, :canonical_rank] = b_canonical
        padded_singular_values[:canonical_rank] = singular_values
        a_canonical = padded_a
        b_canonical = padded_b
        singular_values = padded_singular_values

    _check_finite(a_canonical, "canonical LoRA A", context)
    _check_finite(b_canonical, "canonical LoRA B", context)
    _check_finite(singular_values, "LoRA singular values", context)

    reconstructed_core = (u * svd_singular_values.unsqueeze(0)) @ vh
    core_norm = torch.linalg.vector_norm(core)
    core_error = torch.linalg.vector_norm(core - reconstructed_core) / core_norm.clamp_min(eps)
    b_gram = b_canonical.transpose(0, 1) @ b_canonical
    a_gram = a_canonical @ a_canonical.transpose(0, 1)
    sigma_norm = torch.linalg.vector_norm(singular_values)
    balance_error = torch.linalg.vector_norm(b_gram - a_gram) / sigma_norm.clamp_min(eps)

    sigma_sum = singular_values.sum()
    if bool((sigma_sum > eps).item()):
        probabilities = singular_values / sigma_sum
        positive = probabilities > 0
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        effective_rank = float(torch.exp(entropy).item())
    else:
        effective_rank = 0.0
    energy = singular_values.square()
    total_energy = energy.sum()
    if bool((total_energy > eps).item()):
        prefix_energy = (energy.cumsum(dim=0) / total_energy).tolist()
    else:
        prefix_energy = [0.0] * rank

    minimum = float(singular_values.min().item())
    maximum = float(singular_values.max().item())
    diagnostics: dict[str, Any] = {
        "effective_rank": effective_rank,
        "prefix_energy": prefix_energy,
        "core_reconstruction_error": float(core_error.item()),
        "factor_balance_error": float(balance_error.item()),
        "minimum_singular_value": minimum,
        "maximum_singular_value": maximum,
        "condition_indicator": maximum / max(minimum, eps),
    }
    if canonical_rank < rank:
        diagnostics.update(
            {
                "configured_rank": rank,
                "intrinsic_rank": intrinsic_rank,
                "canonical_rank": canonical_rank,
                "padding_slots": rank - canonical_rank,
                "overcomplete": True,
            }
        )
    if activation_diagnostics is not None:
        diagnostics.update(activation_diagnostics)

    output_a = a_canonical.to(device=lora_A.device, dtype=lora_A.dtype)
    output_b = b_canonical.to(device=lora_B.device, dtype=lora_B.dtype)
    _check_finite(output_a, "cast canonical LoRA A", context)
    _check_finite(output_b, "cast canonical LoRA B", context)
    return CanonicalizationResult(
        lora_A=output_a,
        lora_B=output_b,
        singular_values=singular_values,
        diagnostics=diagnostics,
    )


def _parse_lora_key(
    key: str,
    *,
    suffix_a: str,
    suffix_b: str,
) -> tuple[str, int, list[str]] | None:
    parts = key.split(".")
    positions = [index for index, part in enumerate(parts) if part in (suffix_a, suffix_b)]
    if not positions:
        return None
    if len(positions) != 1:
        raise ValueError(f"ambiguous LoRA key contains multiple factor tokens: '{key}'")
    position = positions[0]
    return ("A" if parts[position] == suffix_a else "B", position, parts)


@torch.no_grad()
def canonicalize_lora_state_dict(
    state_dict: Mapping[str, Any],
    *,
    suffix_a: str = "lora_A",
    suffix_b: str = "lora_B",
    compute_dtype: torch.dtype | None = None,
    deterministic_sign: bool = True,
    svd_fallback: bool = True,
    eps: float = 1e-12,
    strict_pairs: bool = True,
    ordering: str = "singular_value",
    activation_inputs: Mapping[str, torch.Tensor] | None = None,
    activation_chunk_size: int | None = None,
    activation_fallback: bool = True,
    overcomplete_policy: str = "error",
) -> StateDictCanonicalizationResult:
    """Canonicalize complete LoRA pairs while preserving every other entry.

    Pairing replaces one exact dot-delimited factor token and preserves the
    complete remaining path.  Thus PEFT adapter names such as ``default`` and
    layer paths are part of the pair identity and cannot be cross-matched.

    ``activation_inputs``, when supplied, uses the complete matched ``lora_A``
    state-dict key as its key. Missing entries are handled independently per
    layer according to ``activation_fallback``.
    """

    if suffix_a == suffix_b or "." in suffix_a or "." in suffix_b:
        raise ValueError("LoRA suffixes must be distinct single key components")
    if activation_inputs is not None and not isinstance(activation_inputs, Mapping):
        raise TypeError("activation_inputs must be a mapping keyed by full lora_A state-dict key")

    parsed: dict[str, tuple[str, int, list[str]]] = {}
    for key in state_dict:
        if not isinstance(key, str):
            continue
        match = _parse_lora_key(key, suffix_a=suffix_a, suffix_b=suffix_b)
        if match is not None:
            parsed[key] = match

    missing_messages: list[str] = []
    pairs: list[tuple[str, str]] = []
    for key, (role, position, parts) in parsed.items():
        counterpart_parts = list(parts)
        counterpart_parts[position] = suffix_b if role == "A" else suffix_a
        counterpart = ".".join(counterpart_parts)
        expected_role = "B" if role == "A" else "A"
        if counterpart not in parsed or parsed[counterpart][0] != expected_role:
            missing_messages.append(
                f"incomplete LoRA pair: key '{key}' is missing exact counterpart '{counterpart}'"
            )
        elif role == "A":
            pairs.append((key, counterpart))

    if missing_messages:
        message = "; ".join(missing_messages)
        if strict_pairs:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        # ``pairs`` was only populated after finding an exact counterpart, so
        # incomplete entries are already excluded without parsing error text.

    # A shallow mapping copy intentionally keeps non-LoRA tensors bitwise and
    # object-wise unchanged. Only recognized complete factor pairs are replaced.
    output = state_dict.copy() if hasattr(state_dict, "copy") else OrderedDict(state_dict.items())
    if hasattr(state_dict, "_metadata"):
        output._metadata = state_dict._metadata.copy()  # type: ignore[attr-defined]

    singular_values: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for a_key, b_key in pairs:
        pair_context = f"LoRA pair A='{a_key}', B='{b_key}'"
        try:
            result = canonicalize_lora_factor_pair(
                state_dict[a_key],
                state_dict[b_key],
                compute_dtype=compute_dtype,
                deterministic_sign=deterministic_sign,
                svd_fallback=svd_fallback,
                eps=eps,
                context=pair_context,
                ordering=ordering,
                activation_input=(
                    activation_inputs.get(a_key) if activation_inputs is not None else None
                ),
                activation_chunk_size=activation_chunk_size,
                activation_fallback=activation_fallback,
                overcomplete_policy=overcomplete_policy,
            )
        except (TypeError, ValueError, RuntimeError, FloatingPointError) as error:
            raise type(error)(f"canonicalization failed for {pair_context}: {error}") from error
        output[a_key] = result.lora_A
        output[b_key] = result.lora_B
        singular_values[a_key] = result.singular_values
        diagnostics[a_key] = result.diagnostics

    return StateDictCanonicalizationResult(
        state_dict=output,
        singular_values=singular_values,
        diagnostics=diagnostics,
    )
