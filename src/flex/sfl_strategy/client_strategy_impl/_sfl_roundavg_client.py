from __future__ import annotations

import copy
from typing import Any

from flex.fed_strategy.strategy_args import StrategyArgs
from flex.sfl_strategy.client_strategy_impl._sfl_client_example import SflClientStrategy


class SflRoundAvgClientStrategy(SflClientStrategy):
    def __init__(self, args: StrategyArgs, client_node: Any) -> None:
        super().__init__(args, client_node)
        self._strategy_type = "sfl_roundavg"

    def receive_weight(self, global_weight) -> None:
        self._obj.node_var.cache_weight = copy.deepcopy(global_weight)

    def set_local_weight(self) -> None:
        self._obj.node_var.model_weight = copy.deepcopy(self._obj.node_var.cache_weight)
