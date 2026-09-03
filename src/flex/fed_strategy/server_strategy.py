from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict

from ..fed_strategy.strategy_args import StrategyArgs
from ..ml_utils import TrainingLogger, EventHandler, console, String, ObjectMap, KeyValueArgs
from .base_strategy import BaseStrategy

class ServerStrategy(BaseStrategy):

    def __init__(self) -> None:
        super().__init__()
        self._strategy_type: str = "server" 
        self._obj = None

    def create(self, args: StrategyArgs, server_node):
        self._args = args
        self._create_inner(args, server_node)  # create dataset loader

        return self

    @abstractmethod
    def aggregation(self) -> dict:
        """
        Aggregate weights from clients.
        :param client_weights: List of weights from clients.
        :return: Aggregated weights.
        """
        pass

    @abstractmethod
    def broadcast(self) -> None:
        """
        Broadcast aggregated weights to clients.
        :param aggregated_weights: The aggregated weights to be broadcast.
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """
        Main loop/step for the strategy (e.g., one FL round orchestration).
        """
        pass

    @abstractmethod
    def evaluate(self) -> None:
        """
        Evaluate server-side performance/metrics.
        """
        pass

    @abstractmethod
    def select_clients(self, available_clients) -> list:
        """
        Select a subset of clients for the current round.
        :param available_clients: List of available client nodes.
        :return: List of selected client nodes.
        """
        pass

    @abstractmethod
    def record_evaluation(self)-> None:
        """
        Record evaluation metrics.
        """
        pass

    @abstractmethod
    def receive_client_updates(self, client_updates) -> None:
        """
        Receive updates from clients.
        :param client_updates: List of updates from clients.
        """
        pass

    @abstractmethod
    def prepare(self, logger_header, client_nodes_in) -> None:
        """
        Prepare the strategy before starting the training rounds.
        :param logger_header: Header information for logging.
        :param client_nodes_in: List of client nodes to be used in the strategy.
        """
        pass

    def apply_weight(self):
        """
        Apply the aggregated weights to the server's model evaluator.

        Default pipeline:
        1. Fetch ``aggregated_weight`` from ``node_var`` and store as ``model_weight``.
        2. Call :meth:`_prepare_weight_for_model` to convert the weight dict into a
           format compatible with the evaluator model (e.g. decompose SP-aggregated
           keys back to LoRA A/B).
        3. Pass the prepared weight to ``model_evaluator.update_model()``.

        Subclasses that need a fundamentally different flow (e.g. SFL strategies
        that compose client+server model parts) should override this method
        entirely.  Subclasses that only need to customise the weight-conversion
        step can override :meth:`_prepare_weight_for_model` instead.
        """
        aggregated = getattr(self._obj.node_var, "aggregated_weight", None)
        if aggregated is None:
            return

        self._obj.node_var.model_weight = aggregated
        prepared = self._prepare_weight_for_model(aggregated)
        self._obj.node_var.model_evaluator.update_model(prepared)

    # ------------------------------------------------------------------
    # Hooks for weight-format conversion (override-friendly)
    # ------------------------------------------------------------------

    def _prepare_weight_for_model(self, weight: Dict[str, object]) -> Dict[str, object]:
        """
        Convert an aggregated weight dict into a format that can be loaded
        by the evaluator model via ``load_state_dict``.

        Default behaviour:
        * If the dict contains any key ending with ``.sp_aggregated``, perform
          SVD decomposition via :meth:`_decompose_sp_weight`.
        * Otherwise return the weight unchanged.

        Subclasses may override this to implement custom conversions
        (e.g. ``replace_w`` mode, or SFL full-model composition).
        """
        if self._has_sp_aggregated_keys(weight):
            return self._decompose_sp_weight(weight)
        return weight

    @staticmethod
    def _has_sp_aggregated_keys(weight: Dict[str, object]) -> bool:
        """Return ``True`` if *weight* contains at least one ``*.sp_aggregated`` key."""
        return any(k.endswith(".sp_aggregated") for k in weight)

    def _decompose_sp_weight(self, weight: Dict[str, object]) -> Dict[str, object]:
        """
        SVD-decompose every ``{prefix}.sp_aggregated`` entry into
        ``{prefix}.lora_A`` / ``{prefix}.lora_B`` using the rank information
        from ``node_var.inference_model`` (falls back to ``node_var.model``).

        Non-sp_aggregated keys (LayerNorm, embeddings, bias, …) are passed
        through unchanged so that ``load_state_dict(strict=True)`` succeeds.
        """
        from ..ml_algorithms.lora.lora_utils import LoRAUtils

        model = getattr(self._obj.node_var, "inference_model", None)
        if model is None:
            model = getattr(self._obj.node_var, "model", None)

        if model is None:
            raise RuntimeError(
                f"[{self._strategy_type}] Cannot decompose sp_aggregated weights: "
                "no inference_model or model available for rank inference."
            )

        rank_dict = LoRAUtils.get_lora_ranks(model)
        lora_cfg = getattr(model, "lora_config", None) or {}

        return LoRAUtils.svd_split_global_weight(
            weight,
            rank_dict,
            lora_suffix_A=lora_cfg.get("suffix_A", "lora_A"),
            lora_suffix_B=lora_cfg.get("suffix_B", "lora_B"),
            sp_suffix=lora_cfg.get("sp_suffix", "sp_aggregated"),
        )
