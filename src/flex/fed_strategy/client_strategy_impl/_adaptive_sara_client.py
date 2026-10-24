from __future__ import annotations

from ._sara_client import SaraClientTrainingStrategy


class AdaptiveSaraClientTrainingStrategy(SaraClientTrainingStrategy):
    """Client strategy for Adaptive SARA.

    It reuses SARA's anchor caching and context injection. The adaptive behavior
    lives in the adaptive SARA trainer.
    """

    def __init__(self, args, client_node):
        super().__init__(args, client_node)
        self._strategy_type = "adaptive_sara"

    def _create_inner(self, args, client_node) -> None:
        super()._create_inner(args, client_node)
        self._strategy_type = "adaptive_sara"
