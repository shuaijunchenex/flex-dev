from __future__ import annotations

from typing import Any

from flex.fed_strategy.strategy_args import StrategyArgs
from flex.sfl_strategy.client_strategy_impl._sfl_roundavg_client import SflRoundAvgClientStrategy


class SflAlignedClientStrategy(SflRoundAvgClientStrategy):
    """
    SFL-Aligned client strategy.

    Identical to SflRoundAvgClientStrategy in all behaviour.
    The only difference is the model it uses (sfl_aligned_client),
    whose single layer _fc1 matches NNModel_MnistNNBrenden exactly.
    """

    def __init__(self, args: StrategyArgs, client_node: Any) -> None:
        super().__init__(args, client_node)
        self._strategy_type = "sfl_aligned"
