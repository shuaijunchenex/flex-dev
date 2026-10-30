"""Isolated runner registrations for the RBLA reference experiments."""
from __future__ import annotations

from .._rbla_runner_strategy import RblaRunnerStrategy


class RblaRefDiagRunnerStrategy(RblaRunnerStrategy):
    def __init__(self, runner, args, client_node, server_node) -> None:
        super().__init__(runner, args, client_node, server_node)
        self._strategy_type = "rbla_refdiag"


class RblaFreezeARunnerStrategy(RblaRunnerStrategy):
    def __init__(self, runner, args, client_node, server_node) -> None:
        super().__init__(runner, args, client_node, server_node)
        self._strategy_type = "rbla_freeze_a"


class RblaStrongARunnerStrategy(RblaRunnerStrategy):
    def __init__(self, runner, args, client_node, server_node) -> None:
        super().__init__(runner, args, client_node, server_node)
        self._strategy_type = "rbla_strong_a"
