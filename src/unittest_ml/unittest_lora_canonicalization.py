from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from flex.fl_algorithms import FedAggregatorArgs, FedAggregatorFactory
from flex.ml_algorithms.lora.canonicalization import (
    CanonicalizationConfig,
    canonicalize_lora_factor_pair,
    canonicalize_lora_state_dict,
)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual - expected)
    denominator = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).eps)
    return float((numerator / denominator).item())


def _random_pair(dtype: torch.dtype, *, rank: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260716)
    a = torch.randn(rank, 9, dtype=dtype, generator=generator)
    b = torch.randn(7, rank, dtype=dtype, generator=generator)
    return a, b


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float32, 2e-6), (torch.float64, 1e-12)],
)
def test_function_preservation(dtype: torch.dtype, tolerance: float) -> None:
    a, b = _random_pair(dtype)
    result = canonicalize_lora_factor_pair(a, b)
    assert _relative_error(result.lora_B @ result.lora_A, b @ a) < tolerance


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dense_svd_equivalence(dtype: torch.dtype) -> None:
    a, b = _random_pair(dtype)
    result = canonicalize_lora_factor_pair(a, b)
    dense_values = torch.linalg.svdvals(b @ a)[: a.shape[0]]
    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    assert torch.allclose(result.singular_values, dense_values, rtol=tolerance, atol=tolerance)


def test_global_ordering() -> None:
    a, b = _random_pair(torch.float64)
    singular_values = canonicalize_lora_factor_pair(a, b).singular_values
    assert torch.all(singular_values[:-1] >= singular_values[1:])


def test_balanced_factorization_and_diagnostics() -> None:
    a, b = _random_pair(torch.float64)
    result = canonicalize_lora_factor_pair(a, b)
    expected = torch.diag(result.singular_values)
    assert torch.allclose(result.lora_B.T @ result.lora_B, expected, rtol=1e-11, atol=1e-11)
    assert torch.allclose(result.lora_A @ result.lora_A.T, expected, rtol=1e-11, atol=1e-11)
    expected_diagnostics = {
        "effective_rank",
        "prefix_energy",
        "core_reconstruction_error",
        "factor_balance_error",
        "minimum_singular_value",
        "maximum_singular_value",
        "condition_indicator",
    }
    assert expected_diagnostics == result.diagnostics.keys()
    assert len(result.diagnostics["prefix_energy"]) == a.shape[0]


def test_top_k_optimality() -> None:
    a, b = _random_pair(torch.float64)
    dense = b @ a
    result = canonicalize_lora_factor_pair(a, b)
    dense_u, dense_s, dense_vh = torch.linalg.svd(dense, full_matrices=False)
    for k in range(1, a.shape[0]):
        canonical_prefix = result.lora_B[:, :k] @ result.lora_A[:k, :]
        dense_prefix = (dense_u[:, :k] * dense_s[:k].unsqueeze(0)) @ dense_vh[:k, :]
        assert torch.allclose(canonical_prefix, dense_prefix, rtol=1e-10, atol=1e-10)


def test_gauge_equivalent_inputs_have_the_same_distinct_singular_basis() -> None:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(9182)
    rank = 4
    q_b = torch.linalg.qr(torch.randn(8, rank, dtype=dtype, generator=generator)).Q
    q_a = torch.linalg.qr(torch.randn(10, rank, dtype=dtype, generator=generator)).Q
    singular_values = torch.tensor([13.0, 7.0, 2.0, 0.5], dtype=dtype)
    sqrt_values = singular_values.sqrt()
    b = q_b * sqrt_values.unsqueeze(0)
    a = sqrt_values.unsqueeze(1) * q_a.T

    gauge_q = torch.linalg.qr(torch.randn(rank, rank, dtype=dtype, generator=generator)).Q
    gauge = gauge_q @ torch.diag(torch.tensor([0.4, 0.8, 1.7, 2.3], dtype=dtype))
    a_gauged = gauge @ a
    b_gauged = b @ torch.linalg.inv(gauge)
    assert torch.allclose(b_gauged @ a_gauged, b @ a, rtol=1e-12, atol=1e-12)

    first = canonicalize_lora_factor_pair(a, b, deterministic_sign=True)
    second = canonicalize_lora_factor_pair(a_gauged, b_gauged, deterministic_sign=True)
    assert torch.allclose(first.singular_values, second.singular_values, rtol=1e-12, atol=1e-12)
    assert torch.allclose(first.lora_B @ first.lora_A, second.lora_B @ second.lora_A, rtol=1e-12, atol=1e-12)
    assert torch.allclose(first.lora_A, second.lora_A, rtol=1e-11, atol=1e-11)
    assert torch.allclose(first.lora_B, second.lora_B, rtol=1e-11, atol=1e-11)


def test_deterministic_sign_is_repeatable() -> None:
    a, b = _random_pair(torch.float64)
    first = canonicalize_lora_factor_pair(a, b, deterministic_sign=True)
    second = canonicalize_lora_factor_pair(a, b, deterministic_sign=True)
    assert torch.equal(first.lora_A, second.lora_A)
    assert torch.equal(first.lora_B, second.lora_B)
    assert torch.equal(first.singular_values, second.singular_values)


def test_rank_deficient_factors_are_finite_and_zero_slots_are_last() -> None:
    a, b = _random_pair(torch.float64)
    a[-1] = a[-2]
    b[:, -1] = -b[:, -2]
    dense = b @ a
    result = canonicalize_lora_factor_pair(a, b)
    assert torch.isfinite(result.lora_A).all()
    assert torch.isfinite(result.lora_B).all()
    assert torch.isfinite(result.singular_values).all()
    assert torch.allclose(result.lora_B @ result.lora_A, dense, rtol=1e-11, atol=1e-11)
    assert result.singular_values[-1] <= result.singular_values[0] * 1e-12


@pytest.mark.parametrize("zero_side", ["A", "B"])
def test_zero_update(zero_side: str) -> None:
    a, b = _random_pair(torch.float32)
    if zero_side == "A":
        a.zero_()
    else:
        b.zero_()
    result = canonicalize_lora_factor_pair(a, b)
    assert torch.isfinite(result.lora_A).all()
    assert torch.isfinite(result.lora_B).all()
    assert torch.count_nonzero(result.singular_values) == 0
    assert torch.count_nonzero(result.lora_B @ result.lora_A) == 0


def test_state_dict_preserves_non_lora_and_pairs_exact_adapter_paths() -> None:
    generator = torch.Generator().manual_seed(411)
    non_lora = torch.tensor([1, 2, 3], dtype=torch.int64)
    state = OrderedDict(
        [
            ("encoder.0.weight", non_lora),
            ("encoder.0.lora_A.adapter_one.weight", torch.randn(3, 7, generator=generator)),
            ("encoder.0.lora_B.adapter_one.weight", torch.randn(6, 3, generator=generator)),
            ("encoder.0.lora_A.adapter_two.weight", torch.randn(2, 7, generator=generator)),
            ("encoder.0.lora_B.adapter_two.weight", torch.randn(6, 2, generator=generator)),
            ("encoder.1.lora_A.adapter_one.weight", torch.randn(3, 5, generator=generator)),
            ("encoder.1.lora_B.adapter_one.weight", torch.randn(8, 3, generator=generator)),
            ("encoder.0.not_lora_Aish", torch.tensor([9.0])),
        ]
    )
    old_products = {
        key: state[key.replace(".lora_A.", ".lora_B.")] @ value
        for key, value in state.items()
        if ".lora_A." in key
    }
    result = canonicalize_lora_state_dict(state)
    assert list(result.state_dict) == list(state)
    assert result.state_dict["encoder.0.weight"] is non_lora
    assert torch.equal(result.state_dict["encoder.0.weight"], state["encoder.0.weight"])
    assert result.state_dict["encoder.0.not_lora_Aish"] is state["encoder.0.not_lora_Aish"]
    assert len(result.singular_values) == 3
    for a_key, old_product in old_products.items():
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        new_product = result.state_dict[b_key] @ result.state_dict[a_key]
        assert torch.allclose(new_product, old_product, rtol=2e-5, atol=2e-5)


def test_state_dict_incomplete_pair_and_nonfinite_errors_name_exact_keys() -> None:
    missing = {
        "layer.lora_A.adapter_a.weight": torch.randn(2, 4),
        "layer.lora_B.adapter_b.weight": torch.randn(5, 2),
    }
    with pytest.raises(ValueError, match="layer\\.lora_A\\.adapter_a\\.weight"):
        canonicalize_lora_state_dict(missing)

    nonfinite = {
        "layer.lora_A": torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]),
        "layer.lora_B": torch.eye(2),
    }
    with pytest.raises(FloatingPointError, match="layer\\.lora_A"):
        canonicalize_lora_state_dict(nonfinite)


def _client_data(state_dicts: list[dict[str, torch.Tensor]]) -> list[dict]:
    return [
        {"updated_weights": state, "train_record": {"data_sample_num": index + 1}}
        for index, state in enumerate(state_dicts)
    ]


def test_disabled_aggregator_path_is_bitwise_unchanged() -> None:
    a1, b1 = _random_pair(torch.float32)
    a2, b2 = _random_pair(torch.float32)
    a2 = a2 + 0.75
    states = [
        OrderedDict([("layer.lora_A", a1), ("layer.lora_B", b1), ("other", torch.tensor([2.0]))]),
        OrderedDict([("layer.lora_A", a2), ("layer.lora_B", b2), ("other", torch.tensor([5.0]))]),
    ]
    expected = FedAggregatorFactory.create_aggregator(
        FedAggregatorArgs({"method": "rbla", "device": "cpu"})
    ).aggregate(_client_data(states))
    disabled_aggregator = FedAggregatorFactory.create_aggregator(
        FedAggregatorArgs(
            {
                "method": "rbla",
                "device": "cpu",
                "canonicalization": {"enabled": False},
            }
        )
    )
    disabled = disabled_aggregator.aggregate(_client_data(states))
    assert not disabled_aggregator.canonicalization_applied_last_round
    for key in expected:
        assert torch.equal(disabled[key], expected[key])


def test_start_round_interval_and_rank_prefix_broadcast() -> None:
    a, b = _random_pair(torch.float64)
    state = OrderedDict([("layer.lora_A", a), ("layer.lora_B", b)])
    aggregator = FedAggregatorFactory.create_aggregator(
        FedAggregatorArgs(
            {
                "method": "rbla",
                "device": "cpu",
                "canonicalization": {
                    "enabled": True,
                    "start_round": 1,
                    "interval": 2,
                    "log_diagnostics": False,
                },
            }
        )
    )
    expected_schedule = [False, True, False, True, False]
    results = []
    for expected in expected_schedule:
        global_state = aggregator.aggregate(_client_data([state]))
        results.append(global_state)
        assert aggregator.canonicalization_applied_last_round is expected

    canonical_state = results[1]
    local_state = OrderedDict(
        [
            ("layer.lora_A", torch.empty(2, a.shape[1], dtype=a.dtype)),
            ("layer.lora_B", torch.empty(b.shape[0], 2, dtype=b.dtype)),
        ]
    )
    broadcast = aggregator.broadcast_lora_state_dict(canonical_state, local_state)
    assert torch.equal(broadcast["layer.lora_A"], canonical_state["layer.lora_A"][:2, :])
    assert torch.equal(broadcast["layer.lora_B"], canonical_state["layer.lora_B"][:, :2])


def test_support_scaling_runs_before_shared_canonicalization_hook() -> None:
    a1, b1 = _random_pair(torch.float64)
    a2, b2 = _random_pair(torch.float64)
    a2 = a2 * 0.6
    b2 = b2 + 0.2
    states = [
        OrderedDict([("layer.lora_A", a1), ("layer.lora_B", b1)]),
        OrderedDict([("layer.lora_A", a2), ("layer.lora_B", b2)]),
    ]
    common = {
        "method": "rbla_p10_freeze_a_support_scaling",
        "device": "cpu",
        "gamma": 0.5,
        "scaling_type": "q_power",
    }
    before = FedAggregatorFactory.create_aggregator(FedAggregatorArgs(common)).aggregate(
        _client_data(states)
    )
    enabled_args = dict(common)
    enabled_args["canonicalization"] = {"enabled": True, "log_diagnostics": False}
    enabled = FedAggregatorFactory.create_aggregator(FedAggregatorArgs(enabled_args))
    after = enabled.aggregate(_client_data(states))
    assert enabled.canonicalization_applied_last_round
    assert torch.allclose(
        after["layer.lora_B"] @ after["layer.lora_A"],
        before["layer.lora_B"] @ before["layer.lora_A"],
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_uses_stable_compute_and_returns_original_dtype_device(dtype: torch.dtype) -> None:
    a, b = _random_pair(torch.float32)
    a = a.to(dtype)
    b = b.to(dtype)
    result = canonicalize_lora_factor_pair(a, b)
    assert result.lora_A.dtype == dtype
    assert result.lora_B.dtype == dtype
    assert result.lora_A.device == a.device
    assert result.lora_B.device == b.device
    assert result.singular_values.dtype == torch.float32
    assert torch.isfinite(result.lora_A).all()
    assert torch.isfinite(result.lora_B).all()
    dense_original = b.float() @ a.float()
    dense_canonical = result.lora_B.float() @ result.lora_A.float()
    assert _relative_error(dense_canonical, dense_original) < 5e-3


def test_svd_cpu_float64_fallback_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    real_svd = torch.linalg.svd
    call_count = 0

    def fail_once(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("injected first SVD failure")
        return real_svd(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", fail_once)
    a, b = _random_pair(torch.float32)
    result = canonicalize_lora_factor_pair(a, b, svd_fallback=True)
    assert call_count == 2
    assert torch.isfinite(result.singular_values).all()


def test_over_dimension_rank_defaults_to_error() -> None:
    with pytest.raises(ValueError, match="exceeds min"):
        canonicalize_lora_factor_pair(torch.randn(4, 3), torch.randn(5, 4))


@pytest.mark.parametrize(("d_in", "d_out"), [(2, 128), (128, 2), (2, 3)])
def test_over_dimension_zero_pad_preserves_shape_product_and_prefix(
    d_in: int,
    d_out: int,
) -> None:
    generator = torch.Generator().manual_seed(804)
    rank = 8
    intrinsic_rank = min(rank, d_in, d_out)
    a = torch.randn(rank, d_in, dtype=torch.float64, generator=generator)
    b = torch.randn(d_out, rank, dtype=torch.float64, generator=generator)

    result = canonicalize_lora_factor_pair(
        a,
        b,
        overcomplete_policy="zero_pad",
    )

    assert result.lora_A.shape == a.shape
    assert result.lora_B.shape == b.shape
    assert result.singular_values.shape == (rank,)
    assert torch.allclose(result.lora_B @ result.lora_A, b @ a, rtol=1e-11, atol=1e-11)
    assert torch.count_nonzero(result.lora_A[intrinsic_rank:, :]) == 0
    assert torch.count_nonzero(result.lora_B[:, intrinsic_rank:]) == 0
    assert torch.count_nonzero(result.singular_values[intrinsic_rank:]) == 0
    assert result.diagnostics["configured_rank"] == rank
    assert result.diagnostics["intrinsic_rank"] == intrinsic_rank
    assert result.diagnostics["canonical_rank"] == intrinsic_rank
    assert result.diagnostics["padding_slots"] == rank - intrinsic_rank
    assert result.diagnostics["overcomplete"] is True


def test_sp_plus_overcomplete_policy_is_switchable() -> None:
    generator = torch.Generator().manual_seed(805)
    a = torch.randn(8, 2, dtype=torch.float64, generator=generator)
    b = torch.randn(128, 8, dtype=torch.float64, generator=generator)
    state = OrderedDict([("layer.lora_A", a), ("layer.lora_B", b)])

    sp_plus = FedAggregatorFactory.create_aggregator(
        FedAggregatorArgs({"method": "sp_plus", "device": "cpu"})
    )
    padded = sp_plus.aggregate(_client_data([state]))
    assert padded["layer.lora_A"].shape == a.shape
    assert padded["layer.lora_B"].shape == b.shape
    assert torch.allclose(
        padded["layer.lora_B"] @ padded["layer.lora_A"],
        b @ a,
        rtol=1e-11,
        atol=1e-11,
    )

    strict_sp_plus = FedAggregatorFactory.create_aggregator(
        FedAggregatorArgs(
            {
                "method": "sp_plus",
                "device": "cpu",
                "canonicalization": {"overcomplete_policy": "error"},
            }
        )
    )
    with pytest.raises(ValueError, match="exceeds min"):
        strict_sp_plus.aggregate(_client_data([state]))


def test_invalid_overcomplete_policy_and_schedule_are_rejected() -> None:
    with pytest.raises(ValueError, match="overcomplete_policy"):
        CanonicalizationConfig.from_mapping({"overcomplete_policy": "truncate"})
    with pytest.raises(ValueError, match="interval"):
        CanonicalizationConfig.from_mapping({"enabled": True, "interval": 0})


def _diagonal_pair(
    singular_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_values = singular_values.sqrt()
    return torch.diag(sqrt_values), torch.diag(sqrt_values)


def test_default_ordering_is_exactly_the_existing_singular_value_path() -> None:
    a, b = _random_pair(torch.float64)
    existing = canonicalize_lora_factor_pair(a, b)
    explicit = canonicalize_lora_factor_pair(a, b, ordering="singular_value")
    assert torch.equal(existing.lora_A, explicit.lora_A)
    assert torch.equal(existing.lora_B, explicit.lora_B)
    assert torch.equal(existing.singular_values, explicit.singular_values)
    assert existing.diagnostics == explicit.diagnostics
    assert "ordering_requested" not in existing.diagnostics


def test_activation_aware_reorder_preserves_full_product() -> None:
    singular_values = torch.tensor([9.0, 4.0, 1.0], dtype=torch.float64)
    a, b = _diagonal_pair(singular_values)
    activation = torch.tensor([[0.0, 5.0, 0.0], [0.0, -2.0, 0.0]], dtype=a.dtype)
    result = canonicalize_lora_factor_pair(
        a,
        b,
        ordering="activation_aware",
        activation_input=activation,
    )
    assert result.diagnostics["ordering_indices"] == [1, 0, 2]
    assert result.diagnostics["functional_scores"][1] > result.diagnostics["functional_scores"][0]
    assert torch.allclose(result.lora_B @ result.lora_A, b @ a, rtol=1e-12, atol=1e-12)


def test_isotropic_activation_reduces_to_singular_value_ordering() -> None:
    a, b = _random_pair(torch.float64)
    singular_order = canonicalize_lora_factor_pair(a, b)
    activation_order = canonicalize_lora_factor_pair(
        a,
        b,
        ordering="activation_aware",
        activation_input=torch.eye(a.shape[1], dtype=a.dtype),
    )
    assert activation_order.diagnostics["ordering_indices"] == list(range(a.shape[0]))
    assert torch.allclose(activation_order.lora_A, singular_order.lora_A, rtol=1e-12, atol=1e-12)
    assert torch.allclose(activation_order.lora_B, singular_order.lora_B, rtol=1e-12, atol=1e-12)
    assert torch.equal(activation_order.singular_values, singular_order.singular_values)


def test_existing_broadcast_reuses_one_global_nested_activation_prefix() -> None:
    generator = torch.Generator().manual_seed(702)
    rank = 8
    a = torch.randn(rank, 10, dtype=torch.float64, generator=generator)
    b = torch.randn(9, rank, dtype=torch.float64, generator=generator)
    activation = torch.randn(17, 10, dtype=torch.float64, generator=generator)
    result = canonicalize_lora_factor_pair(
        a,
        b,
        ordering="activation_aware",
        activation_input=activation,
    )
    global_state = OrderedDict(
        [("layer.lora_A", result.lora_A), ("layer.lora_B", result.lora_B)]
    )
    ordering = result.diagnostics["ordering_indices"]
    prefixes: dict[int, list[int]] = {}
    for local_rank in (2, 4, 8):
        local_state = OrderedDict(
            [
                ("layer.lora_A", torch.empty(local_rank, a.shape[1], dtype=a.dtype)),
                ("layer.lora_B", torch.empty(b.shape[0], local_rank, dtype=b.dtype)),
            ]
        )
        broadcast = FedAggregatorFactory.create_aggregator(
            FedAggregatorArgs({"method": "rbla", "device": "cpu"})
        ).broadcast_lora_state_dict(global_state, local_state)
        assert torch.equal(broadcast["layer.lora_A"], result.lora_A[:local_rank, :])
        assert torch.equal(broadcast["layer.lora_B"], result.lora_B[:, :local_rank])
        prefixes[local_rank] = ordering[:local_rank]
    assert prefixes[2] == prefixes[4][:2]
    assert prefixes[4] == prefixes[8][:4]


def test_functional_error_curve_matches_explicit_small_matrix_error() -> None:
    singular_values = torch.tensor([8.0, 3.0, 1.0], dtype=torch.float64)
    a, b = _diagonal_pair(singular_values)
    activation = torch.tensor(
        [[0.1, 3.0, 0.5], [0.2, -2.0, 1.5], [-0.1, 1.0, 2.0]],
        dtype=a.dtype,
    )
    eps = 1e-12
    result = canonicalize_lora_factor_pair(
        a,
        b,
        ordering="activation_aware",
        activation_input=activation,
        eps=eps,
    )
    dense_update = b @ a
    for rank in range(1, a.shape[0] + 1):
        dense_prefix = result.lora_B[:, :rank] @ result.lora_A[:rank, :]
        residual_output = activation @ (dense_update - dense_prefix).T
        full_output = activation @ dense_update.T
        explicit_error = residual_output.square().sum() / (full_output.square().sum() + eps)
        diagnosed_error = result.diagnostics["functional_error_curve"][rank - 1]
        assert diagnosed_error == pytest.approx(float(explicit_error.item()), rel=1e-12, abs=1e-12)


def test_missing_and_invalid_activation_fallback_or_raise_with_context() -> None:
    a, b = _random_pair(torch.float64)
    baseline = canonicalize_lora_factor_pair(a, b)
    missing = canonicalize_lora_factor_pair(
        a,
        b,
        ordering="activation_aware",
        activation_input=None,
        activation_fallback=True,
        context="layer.lora_A",
    )
    assert torch.equal(missing.lora_A, baseline.lora_A)
    assert missing.diagnostics["ordering_applied"] == "singular_value"
    assert missing.diagnostics["fallback_reason"] == "activation_input is None"
    with pytest.raises(ValueError, match=r"layer\.lora_A.*activation_input is None"):
        canonicalize_lora_factor_pair(
            a,
            b,
            ordering="activation_aware",
            activation_fallback=False,
            context="layer.lora_A",
        )

    invalid = canonicalize_lora_factor_pair(
        a,
        b,
        ordering="activation_aware",
        activation_input=torch.randn(3, a.shape[1] + 1, dtype=a.dtype),
        activation_fallback=True,
    )
    assert invalid.diagnostics["ordering_applied"] == "singular_value"
    assert "does not match" in invalid.diagnostics["fallback_reason"]
    with pytest.raises(ValueError, match="does not match d_in"):
        canonicalize_lora_factor_pair(
            a,
            b,
            ordering="activation_aware",
            activation_input=torch.randn(3, a.shape[1] + 1, dtype=a.dtype),
            activation_fallback=False,
        )


def test_chunked_and_non_chunked_activation_scores_are_consistent() -> None:
    a, b = _random_pair(torch.float64)
    activation = torch.randn(5, 7, a.shape[1], dtype=a.dtype, generator=torch.Generator().manual_seed(81))
    full = canonicalize_lora_factor_pair(
        a,
        b,
        ordering="activation_aware",
        activation_input=activation,
        activation_chunk_size=None,
    )
    chunked = canonicalize_lora_factor_pair(
        a,
        b,
        ordering="activation_aware",
        activation_input=activation,
        activation_chunk_size=6,
    )
    assert full.diagnostics["ordering_indices"] == chunked.diagnostics["ordering_indices"]
    assert torch.allclose(
        torch.tensor(full.diagnostics["functional_scores"]),
        torch.tensor(chunked.diagnostics["functional_scores"]),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.allclose(full.lora_A, chunked.lora_A, rtol=1e-12, atol=1e-12)
    assert torch.allclose(full.lora_B, chunked.lora_B, rtol=1e-12, atol=1e-12)


def test_state_dict_activation_mapping_is_per_exact_lora_a_key() -> None:
    first_a, first_b = _diagonal_pair(torch.tensor([7.0, 3.0], dtype=torch.float64))
    second_a, second_b = _diagonal_pair(torch.tensor([6.0, 2.0], dtype=torch.float64))
    state = OrderedDict(
        [
            ("encoder.0.lora_A.adapter.weight", first_a),
            ("encoder.0.lora_B.adapter.weight", first_b),
            ("encoder.1.lora_A.adapter.weight", second_a),
            ("encoder.1.lora_B.adapter.weight", second_b),
            ("other", torch.tensor([5.0])),
        ]
    )
    result = canonicalize_lora_state_dict(
        state,
        ordering="activation_aware",
        activation_inputs={
            "encoder.0.lora_A.adapter.weight": torch.tensor(
                [[0.0, 4.0]], dtype=torch.float64
            )
        },
        activation_fallback=True,
    )
    first_diag = result.diagnostics["encoder.0.lora_A.adapter.weight"]
    second_diag = result.diagnostics["encoder.1.lora_A.adapter.weight"]
    assert first_diag["ordering_applied"] == "activation_aware"
    assert first_diag["ordering_indices"] == [1, 0]
    assert second_diag["ordering_applied"] == "singular_value"
    assert second_diag["fallback_reason"] == "activation_input is None"
    assert result.state_dict["other"] is state["other"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_activation_aware_output_preserves_dtype_and_device(dtype: torch.dtype) -> None:
    a, b = _random_pair(dtype)
    activation = torch.randn(13, a.shape[1], dtype=dtype)
    result = canonicalize_lora_factor_pair(
        a,
        b,
        ordering="activation_aware",
        activation_input=activation,
        activation_chunk_size=4,
    )
    assert result.lora_A.dtype == dtype
    assert result.lora_B.dtype == dtype
    assert result.lora_A.device == a.device
    assert result.lora_B.device == b.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_activation_aware_cuda_device_preservation() -> None:
    a, b = _random_pair(torch.float32)
    a, b = a.cuda(), b.cuda()
    result = canonicalize_lora_factor_pair(
        a,
        b,
        ordering="activation_aware",
        activation_input=torch.randn(11, a.shape[1], device="cuda"),
    )
    assert result.lora_A.device.type == "cuda"
    assert result.lora_B.device.type == "cuda"


def test_aggregator_activation_injection_reorders_before_existing_broadcast() -> None:
    singular_values = torch.tensor([9.0, 4.0, 1.0], dtype=torch.float64)
    a, b = _diagonal_pair(singular_values)
    state = OrderedDict([("layer.lora_A", a), ("layer.lora_B", b)])
    aggregator = FedAggregatorFactory.create_aggregator(
        FedAggregatorArgs(
            {
                "method": "rbla",
                "device": "cpu",
                "canonicalization": {
                    "enabled": True,
                    "ordering": "activation_aware",
                    "activation_chunk_size": 2,
                    "activation_fallback": False,
                    "log_diagnostics": False,
                },
            }
        )
    )
    aggregator.set_canonicalization_activation_inputs(
        {"layer.lora_A": torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64)}
    )
    global_state = aggregator.aggregate(_client_data([state]))
    diagnostics = aggregator.canonicalization_diagnostics["layer.lora_A"]
    assert diagnostics["ordering_indices"] == [1, 0, 2]
    local_state = OrderedDict(
        [
            ("layer.lora_A", torch.empty(1, 3, dtype=torch.float64)),
            ("layer.lora_B", torch.empty(3, 1, dtype=torch.float64)),
        ]
    )
    local = aggregator.broadcast_lora_state_dict(global_state, local_state)
    assert torch.equal(local["layer.lora_A"], global_state["layer.lora_A"][:1, :])
    assert torch.equal(local["layer.lora_B"], global_state["layer.lora_B"][:, :1])
