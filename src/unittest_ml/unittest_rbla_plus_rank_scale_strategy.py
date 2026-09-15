from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import torch
from torch import nn

from flex.fed_strategy.client_strategy_impl._rbla_plus_rank_scale_client import (
    RblaPlusRankScaleClientTrainingStrategy,
)
from flex.fed_strategy.server_strategy_impl._rbla_plus_rank_scale_server import (
    RblaPlusRankScaleServerStrategy,
)
from flex.fed_strategy.server_strategy_impl._sp_plus_server import SpPlusServerStrategy
from flex.fed_strategy.runner_strategy_impl._rbla_plus_rank_scale_runner_strategy import (
    RblaPlusRankScaleRunnerStrategy,
)
from flex.fed_strategy.strategy_args import StrategyArgs
from flex.fed_strategy.strategy_factory import StrategyFactory
from flex.ml_algorithms.lora.impl.lora_ms import MSLoRAConv2d, MSLoRALinear
from flex.ml_algorithms.lora.rank_scale_alignment import (
    LoRALayerScaleProfile,
    align_lora_state_dict_scale,
    build_lora_scale_profile,
)


class _LinearLoRAModel(nn.Module):
    def __init__(self, rank: int, alpha: int) -> None:
        super().__init__()
        self.layer = MSLoRALinear(
            5,
            4,
            r=rank,
            lora_alpha=alpha,
            merge_weights=False,
        )

    def forward(self, x):
        return self.layer(x)


class _ClientNode:
    def __init__(self, node_id: str, model: nn.Module) -> None:
        self.node_id = node_id
        self.node_var = SimpleNamespace(
            model=model,
            model_weight=model.state_dict(),
            cache_weight=None,
        )
        self.strategy = RblaPlusRankScaleClientTrainingStrategy(
            StrategyArgs(
                {"role": "client", "strategy_name": "rbla_plus_rank_scale"}
            ),
            self,
        )

    def receive_weight(self, weight) -> None:
        self.strategy.receive_weight(weight)

    def set_local_weight(self) -> None:
        self.strategy.set_local_weight()


def _randomize_lora(model: nn.Module, seed: int = 123) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("lora_A", "lora_B")):
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                        device=parameter.device,
                    )
                )


def test_rank_scale_alignment_preserves_the_effective_update() -> None:
    client_model = _LinearLoRAModel(rank=2, alpha=8)  # scaling=4
    server_model = _LinearLoRAModel(rank=2, alpha=4)  # scaling=2
    _randomize_lora(client_model)

    client_profile = build_lora_scale_profile(client_model)
    server_profile = build_lora_scale_profile(server_model)
    original = client_model.state_dict()
    original_A = original["layer.lora_A"].clone()
    original_B = original["layer.lora_B"].clone()

    aligned = align_lora_state_dict_scale(
        original,
        source_profile=client_profile,
        target_profile=server_profile,
    )

    source_delta = 4.0 * (original_B @ original_A)
    target_delta = 2.0 * (
        aligned["layer.lora_B"] @ aligned["layer.lora_A"]
    )
    assert torch.allclose(source_delta, target_delta, atol=1e-6, rtol=1e-6)
    assert torch.equal(original["layer.lora_A"], original_A)
    assert torch.equal(original["layer.lora_B"], original_B)
    assert aligned["layer.lora_A"].data_ptr() != original["layer.lora_A"].data_ptr()
    assert aligned["layer.lora_B"].data_ptr() != original["layer.lora_B"].data_ptr()


def test_client_owns_and_caches_its_rank_scale_profile() -> None:
    first_model = _LinearLoRAModel(rank=2, alpha=8)
    client = _ClientNode("client-2", first_model)
    strategy = client.strategy

    assert strategy._rank_scale_profile is None
    first_profile = strategy.rank_scale_profile
    assert strategy.rank_scale_profile is first_profile
    assert first_profile["layer"].rank == 2
    assert first_profile["layer"].scaling == 4.0

    second_model = _LinearLoRAModel(rank=4, alpha=8)
    client.node_var.model = second_model
    second_profile = strategy.rank_scale_profile
    assert second_profile is not first_profile
    assert second_profile["layer"].rank == 4
    assert second_profile["layer"].scaling == 2.0

    second_model.layer.scaling = 3.0
    assert strategy.rank_scale_profile["layer"].scaling == 2.0
    refreshed_profile = strategy.refresh_rank_scale_profile()
    assert refreshed_profile["layer"].scaling == 3.0


def test_client_exposes_explicit_upload_and_broadcast_transforms() -> None:
    server_model = _LinearLoRAModel(rank=2, alpha=4)  # scaling=2
    client_model = _LinearLoRAModel(rank=2, alpha=8)  # scaling=4
    _randomize_lora(client_model, seed=789)
    client = _ClientNode("client-2", client_model)
    server_profile = build_lora_scale_profile(server_model)

    uploaded = client_model.state_dict()
    normalized = client.strategy.normalize_upload_for_server(
        uploaded,
        server_profile,
    )
    restored = client.strategy.prepare_broadcast_from_server(
        normalized,
        server_profile,
    )

    assert torch.allclose(restored["layer.lora_A"], uploaded["layer.lora_A"])
    assert torch.allclose(restored["layer.lora_B"], uploaded["layer.lora_B"])


def test_rank_scale_broadcast_preserves_the_server_rank_prefix() -> None:
    server_model = _LinearLoRAModel(rank=4, alpha=8)  # scaling=2
    client_model = _LinearLoRAModel(rank=2, alpha=8)  # scaling=4
    _randomize_lora(server_model, seed=321)

    client = _ClientNode("client-2", client_model)
    server_node = SimpleNamespace(
        node_var=SimpleNamespace(
            model=server_model,
            model_weight=server_model.state_dict(),
        ),
        client_nodes=[client],
    )
    strategy = RblaPlusRankScaleServerStrategy(
        StrategyArgs(
            {"role": "server", "strategy_name": "rbla_plus_rank_scale"}
        ),
        server_node,
    )
    strategy.broadcast()

    server_state = server_model.state_dict()
    client_state = client.node_var.model_weight
    expected = 2.0 * (
        server_state["layer.lora_B"][:, :2]
        @ server_state["layer.lora_A"][:2, :]
    )
    received = 4.0 * (
        client_state["layer.lora_B"] @ client_state["layer.lora_A"]
    )
    assert torch.allclose(received, expected, atol=1e-6, rtol=1e-6)


def test_server_normalizes_upload_without_mutating_the_client_state() -> None:
    server_model = _LinearLoRAModel(rank=2, alpha=4)  # scaling=2
    client_model = _LinearLoRAModel(rank=2, alpha=8)  # scaling=4
    _randomize_lora(client_model, seed=456)
    uploaded = client_model.state_dict()
    uploaded_A = uploaded["layer.lora_A"].clone()
    uploaded_B = uploaded["layer.lora_B"].clone()

    class _CaptureAggregator:
        def __init__(self) -> None:
            self.updates = None

        def aggregate(self, updates):
            self.updates = updates
            return updates[0]["updated_weights"]

    aggregator = _CaptureAggregator()
    client = _ClientNode("client-2", client_model)
    server_node = SimpleNamespace(
        node_var=SimpleNamespace(
            model=server_model,
            aggregation_method=aggregator,
            client_updates=[
                {
                    "updated_weights": uploaded,
                    "train_record": {
                        "node_id": "client-2",
                        "data_sample_num": 10,
                    },
                }
            ],
            aggregated_weight=None,
        ),
        client_nodes=[client],
    )
    strategy = RblaPlusRankScaleServerStrategy(
        StrategyArgs(
            {"role": "server", "strategy_name": "rbla_plus_rank_scale"}
        ),
        server_node,
    )
    strategy.aggregation()

    normalized = aggregator.updates[0]["updated_weights"]
    source_delta = 4.0 * (uploaded_B @ uploaded_A)
    normalized_delta = 2.0 * (
        normalized["layer.lora_B"] @ normalized["layer.lora_A"]
    )
    assert torch.allclose(source_delta, normalized_delta, atol=1e-6, rtol=1e-6)
    assert torch.equal(uploaded["layer.lora_A"], uploaded_A)
    assert torch.equal(uploaded["layer.lora_B"], uploaded_B)


def test_conv_profile_uses_logical_rank_instead_of_factor_shape() -> None:
    model = nn.Sequential(
        OrderedDict(
            [
                (
                    "conv",
                    MSLoRAConv2d(
                        3,
                        5,
                        kernel_size=3,
                        r=2,
                        lora_alpha=8,
                        merge_weights=False,
                    ),
                )
            ]
        )
    )
    profile = build_lora_scale_profile(model)["conv"]

    assert model.conv.lora_A.shape[0] == 6
    assert profile.rank == 2
    assert profile.alpha == 8.0
    assert profile.scaling == 4.0


def test_rank_scale_alignment_has_finite_gradients() -> None:
    source = {
        "layer": LoRALayerScaleProfile(
            "layer", "layer.lora_A", "layer.lora_B", 2, 8.0, 4.0
        )
    }
    target = {
        "layer": LoRALayerScaleProfile(
            "layer", "layer.lora_A", "layer.lora_B", 2, 4.0, 2.0
        )
    }
    A = torch.randn(2, 5, requires_grad=True)
    B = torch.randn(4, 2, requires_grad=True)
    aligned = align_lora_state_dict_scale(
        OrderedDict((('layer.lora_A', A), ('layer.lora_B', B))),
        source,
        target,
    )
    loss = (2.0 * aligned["layer.lora_B"] @ aligned["layer.lora_A"]).square().mean()
    loss.backward()

    assert A.grad is not None and torch.isfinite(A.grad).all()
    assert B.grad is not None and torch.isfinite(B.grad).all()


def test_new_strategy_registration_does_not_replace_sp_plus() -> None:
    new_server = StrategyFactory.create_server_strategy(
        StrategyArgs(
            {"role": "server", "strategy_name": "rbla_plus_rank_scale"}
        ),
        SimpleNamespace(),
    )
    old_server = StrategyFactory.create_server_strategy(
        StrategyArgs({"role": "server", "strategy_name": "sp_plus"}),
        SimpleNamespace(),
    )
    new_client = StrategyFactory.create_client_strategy(
        StrategyArgs(
            {"role": "client", "strategy_name": "rbla_plus_rank_scale"}
        ),
        SimpleNamespace(),
    )
    runner_server = SimpleNamespace(set_client_nodes=lambda clients: None)
    new_runner = StrategyFactory.create_runner_strategy(
        StrategyArgs(
            {
                "role": "runner",
                "strategy_name": "rbla_plus_rank_scale",
                "training_rounds": 0,
            }
        ),
        SimpleNamespace(),
        [],
        runner_server,
    )

    assert type(new_server) is RblaPlusRankScaleServerStrategy
    assert type(old_server) is SpPlusServerStrategy
    assert type(new_client) is RblaPlusRankScaleClientTrainingStrategy
    assert type(new_runner) is RblaPlusRankScaleRunnerStrategy
    assert new_server._strategy_type == "rbla_plus_rank_scale"
    assert old_server._strategy_type == "sp_plus"
