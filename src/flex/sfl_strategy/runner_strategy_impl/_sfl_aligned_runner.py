from __future__ import annotations

from typing import Any, Iterable

from flex.sfl_strategy.runner_strategy_impl._sfl_roundavg_runner import SflRoundAvgRunnerStrategy


class SflAlignedRunnerStrategy(SflRoundAvgRunnerStrategy):
    """
    SFL-Aligned runner strategy.

    Identical to SflRoundAvgRunnerStrategy in all behaviour.
    Exists as a named type so the strategy_factory can dispatch to
    sfl_aligned client / server strategies via strategy_name = 'sfl_aligned'.
    """

    def __init__(self, runner, args, client_nodes, server_node) -> None:
        super().__init__(runner, args, client_nodes, server_node)
        self._strategy_type = "sfl_aligned"
