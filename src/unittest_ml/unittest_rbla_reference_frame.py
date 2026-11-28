"""Focused tests for the isolated RBLA reference-frame implementations."""
from __future__ import annotations

import torch
import torch.nn as nn

from flex.fl_algorithms.aggregation.fed_aggregator_args import FedAggregatorArgs
from flex.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from flex.fl_algorithms.aggregation.methods._fed_aggregator_rbla import FedAggregator_RBLA
from flex.fl_algorithms.aggregation.methods.rbla_problem._fed_aggregator_rbla_support_scaling import (
    FedAggregator_RBLAFreezeASupportGamma,
    FedAggregator_RBLAP10FreezeASupportScaling,
    FedAggregator_RBLARefDiagSupportScaling,
    FedAggregator_RBLAStrongASupportScaling,
)
from flex.ml_algorithms.rbla_problem import (
    RblaReferenceDiagnostics,
    StrongAConfig,
    StrongAProximalLoss,
    run_reparameterization_stress_test,
    support_scaled_discrepancy,
    support_scaled_discrepancy_metrics,
    coefficient_for_eligible,
)
from flex.ml_models.nn_model_factory import NNModelFactory


def test_reparameterization_stress() -> None:
    metrics = run_reparameterization_stress_test()
    assert metrics["per_client_function_error"] < 1e-5
    assert metrics["dense_update_invariance_error"] < 1e-5
    assert metrics["rbla_reparameterization_sensitivity"] > 1e-3


def test_shared_a_eliminates_factor_discrepancy() -> None:
    generator = torch.Generator().manual_seed(7)
    global_a = torch.randn(4, 6, generator=generator)
    global_b = torch.zeros(5, 4)
    clients = []
    for rank in (2, 3, 4):
        clients.append({
            "layer.lora_A": global_a[:rank].clone(),
            "layer.lora_B": torch.randn(5, rank, generator=generator),
        })
    metrics = RblaReferenceDiagnostics().compute(
        clients,
        [1.0, 2.0, 3.0],
        {"layer.lora_A": global_a, "layer.lora_B": global_b},
    )
    assert metrics["ref_a_prox_drift"] == 0.0
    assert metrics["ref_agg_discrepancy"] < 1e-6


def test_strong_a_loss_controls_direction_and_scale() -> None:
    anchor = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    parameter = nn.Parameter(anchor.clone())
    loss_fn = StrongAProximalLoss(StrongAConfig(lambda_a=0.1))
    zero = loss_fn({"layer.lora_A": parameter}, {"layer.lora_A": anchor})
    assert zero.item() == 0.0

    with torch.no_grad():
        parameter.copy_(torch.tensor([[0.0, 2.0], [-1.0, 0.0]]))
    loss = loss_fn({"layer.lora_A": parameter}, {"layer.lora_A": anchor})
    assert loss.item() > 0.0
    loss.backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


def test_new_aggregators_are_registered() -> None:
    for method in (
        "rbla_refdiag", "rbla_freeze_a", "rbla_strong_a",
        "rbla_p8_refdiag_support_scaling",
        "rbla_p8_strong_a_support_scaling",
        "rbla_p9_freeze_a_support_scaling",
        "rbla_p10_freeze_a_support_scaling",
    ):
        aggregator = FedAggregatorFactory.create_aggregator(
            FedAggregatorArgs({"method": method, "device": "cpu", "pad_mode": "nan"})
        )
        assert aggregator.aggregated_method == method


def test_freeze_a_excludes_a_from_training() -> None:
    args = NNModelFactory.create_args({
        "name": "mnist_lora_reference_mlp",
        "rank_ratio": 0.1,
        "lora_scale": 1.0,
        "share_model": False,
    })
    model = NNModelFactory.create(args)
    anchors = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if "lora_A" in key.split(".")
    }
    for key, parameter in model.named_parameters():
        if "lora_A" in key.split("."):
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    inputs = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))
    loss = nn.CrossEntropyLoss()(model(inputs), labels)
    loss.backward()
    optimizer.step()
    for key, anchor in anchors.items():
        assert torch.equal(model.state_dict()[key], anchor)


def test_reference_model_has_constant_scaling_across_ranks() -> None:
    low_args = NNModelFactory.create_args({
        "name": "mnist_lora_reference_mlp",
        "rank_ratio": 0.2,
        "lora_scale": 1.0,
        "share_model": False,
    })
    high_args = NNModelFactory.create_args({
        "name": "mnist_lora_reference_mlp",
        "rank_ratio": 1.0,
        "lora_scale": 1.0,
        "share_model": False,
    })
    low = NNModelFactory.create(low_args)
    high = NNModelFactory.create(high_args)
    low_scales = [module.scaling for module in low.modules() if hasattr(module, "scaling")]
    high_scales = [module.scaling for module in high.modules() if hasattr(module, "scaling")]
    assert low_scales and high_scales
    assert all(scale == 1.0 for scale in low_scales + high_scales)


def _support_scaling_fixture():
    generator = torch.Generator().manual_seed(23)
    shared_a = torch.randn(3, 4, generator=generator)
    states = []
    for rank in (1, 2, 3):
        states.append({
            "layer.lora_A": shared_a[:rank].clone(),
            "layer.lora_B": torch.randn(2, rank, generator=generator),
        })
    return states, [0.2, 0.3, 0.5]


def test_support_gamma_zero_matches_freeze_a_rbla() -> None:
    states, weights = _support_scaling_fixture()
    baseline = FedAggregator_RBLA.aggregate_state_dicts(states, weights, pad_mode="nan")
    scaled = FedAggregator_RBLAFreezeASupportGamma.aggregate_state_dicts_gamma(
        states, weights, gamma=0.0
    )
    for key in baseline:
        assert torch.allclose(baseline[key], scaled[key], atol=1e-7, rtol=1e-6)


def test_support_gamma_one_is_zero_missing_weighting() -> None:
    states, weights = _support_scaling_fixture()
    scaled = FedAggregator_RBLAFreezeASupportGamma.aggregate_state_dicts_gamma(
        states, weights, gamma=1.0
    )
    expected = torch.zeros_like(scaled["layer.lora_B"])
    for state, weight in zip(states, weights):
        expected[:, : state["layer.lora_B"].shape[1]] += weight * state["layer.lora_B"]
    assert torch.allclose(scaled["layer.lora_B"], expected, atol=1e-7, rtol=1e-6)


def test_support_gamma_shared_a_matches_direct_rank_one_sum() -> None:
    states, weights = _support_scaling_fixture()
    for gamma in (0.0, 0.5, 1.0):
        scaled = FedAggregator_RBLAFreezeASupportGamma.aggregate_state_dicts_gamma(
            states, weights, gamma=gamma
        )
        discrepancy = support_scaled_discrepancy(
            states, weights, scaled, gamma=gamma
        )
        assert discrepancy < 1e-6


def test_support_gamma_is_identical_when_q_is_one() -> None:
    generator = torch.Generator().manual_seed(29)
    shared_a = torch.randn(2, 4, generator=generator)
    states = [
        {"layer.lora_A": shared_a.clone(), "layer.lora_B": torch.randn(3, 2, generator=generator)}
        for _ in range(3)
    ]
    weights = [0.2, 0.3, 0.5]
    outputs = [
        FedAggregator_RBLAFreezeASupportGamma.aggregate_state_dicts_gamma(states, weights, gamma)
        for gamma in (0.0, 0.5, 1.0)
    ]
    for later in outputs[1:]:
        for key in outputs[0]:
            assert torch.allclose(outputs[0][key], later[key], atol=1e-7, rtol=1e-6)


def test_support_gamma_outputs_are_finite() -> None:
    states, weights = _support_scaling_fixture()
    for gamma in (0.0, 0.5, 1.0):
        output = FedAggregator_RBLAFreezeASupportGamma.aggregate_state_dicts_gamma(
            states, weights, gamma
        )
        assert all(torch.isfinite(value).all() for value in output.values())


def test_all_new_modes_gamma_zero_match_original_aggregation() -> None:
    states, weights = _support_scaling_fixture()
    baseline = FedAggregator_RBLA.aggregate_state_dicts(states, weights, pad_mode="nan")
    for aggregator in (
        FedAggregator_RBLARefDiagSupportScaling,
        FedAggregator_RBLAStrongASupportScaling,
        FedAggregator_RBLAP10FreezeASupportScaling,
    ):
        actual = aggregator.aggregate_state_dicts_scaled(
            states, weights, gamma=0.0, scaling_type="q_power"
        )
        for key in baseline:
            assert torch.allclose(baseline[key], actual[key], atol=1e-7, rtol=1e-6)


def test_effective_support_coefficient_matches_definition() -> None:
    weights = [0.5, 0.3, 0.2]
    coefficient, q_s, n_eff, n_eff_full = coefficient_for_eligible(
        weights, [1, 2], scaling_type="effective_support", gamma=0.5
    )
    assert abs(q_s - 0.5) < 1e-12
    assert abs(n_eff - 1.0 / (0.6 ** 2 + 0.4 ** 2)) < 1e-12
    assert abs(n_eff_full - 1.0 / sum(value ** 2 for value in weights)) < 1e-12
    assert abs(coefficient - (n_eff / n_eff_full) ** 0.5) < 1e-12


def test_all_scaling_types_are_identical_when_q_is_one() -> None:
    generator = torch.Generator().manual_seed(31)
    shared_a = torch.randn(2, 4, generator=generator)
    states = [
        {"layer.lora_A": shared_a.clone(), "layer.lora_B": torch.randn(3, 2, generator=generator)}
        for _ in range(3)
    ]
    weights = [0.5, 0.3, 0.2]
    outputs = [
        FedAggregator_RBLAP10FreezeASupportScaling.aggregate_state_dicts_scaled(
            states, weights, gamma=gamma, scaling_type=scaling_type
        )
        for scaling_type, gamma in (
            ("conditional", 0.0),
            ("q_power", 0.5),
            ("effective_support", 0.5),
            ("population", 1.0),
        )
    ]
    for actual in outputs[1:]:
        for key in outputs[0]:
            assert torch.allclose(outputs[0][key], actual[key], atol=1e-7, rtol=1e-6)


def test_effective_support_shared_a_matches_direct_and_unshared_a_does_not() -> None:
    states, weights = _support_scaling_fixture()
    shared = FedAggregator_RBLAP10FreezeASupportScaling.aggregate_state_dicts_scaled(
        states, weights, gamma=0.5, scaling_type="effective_support"
    )
    metrics = support_scaled_discrepancy_metrics(
        states, weights, shared, gamma=0.5, scaling_type="effective_support"
    )
    assert metrics["ref_agg_discrepancy"] < 1e-6
    assert metrics["ref_agg_discrepancy_abs_numerator"] < 1e-5
    assert metrics["ref_global_a_norm"] > 0
    assert metrics["ref_global_b_norm"] > 0
    assert metrics["ref_global_delta_w_norm"] > 0

    unshared_states = [
        {key: value.clone() for key, value in state.items()}
        for state in states
    ]
    unshared_states[1]["layer.lora_A"] *= -1.7
    unshared_states[2]["layer.lora_A"] += 0.8
    unshared = FedAggregator_RBLARefDiagSupportScaling.aggregate_state_dicts_scaled(
        unshared_states, weights, gamma=0.5, scaling_type="q_power"
    )
    discrepancy = support_scaled_discrepancy(
        unshared_states, weights, unshared, gamma=0.5, scaling_type="q_power"
    )
    assert discrepancy > 1e-3
    assert all(torch.isfinite(value).all() for value in unshared.values())


if __name__ == "__main__":
    test_reparameterization_stress()
    test_shared_a_eliminates_factor_discrepancy()
    test_strong_a_loss_controls_direction_and_scale()
    test_new_aggregators_are_registered()
    test_freeze_a_excludes_a_from_training()
    test_reference_model_has_constant_scaling_across_ranks()
    test_support_gamma_zero_matches_freeze_a_rbla()
    test_support_gamma_one_is_zero_missing_weighting()
    test_support_gamma_shared_a_matches_direct_rank_one_sum()
    test_support_gamma_is_identical_when_q_is_one()
    test_support_gamma_outputs_are_finite()
    test_all_new_modes_gamma_zero_match_original_aggregation()
    test_effective_support_coefficient_matches_definition()
    test_all_scaling_types_are_identical_when_q_is_one()
    test_effective_support_shared_a_matches_direct_and_unshared_a_does_not()
    print("RBLA reference-frame tests passed")
