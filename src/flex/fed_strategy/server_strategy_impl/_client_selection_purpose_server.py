from __future__ import annotations

from typing import Any, Dict, List

from flex.fed_strategy.server_strategy import ServerStrategy
from flex.fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from flex.fl_algorithms.selection.fed_client_selector_args import FedClientSelectorArgs
from flex.ml_utils import console


# ---------------------------------------------------------------------------
# Selector methods supported by this strategy
# ---------------------------------------------------------------------------
_SUPPORTED_SELECTORS = frozenset({
    "high_loss",
    "low_loss",
    "high_weight_divergence",
    "random",
    "all",
})


class ClientSelectionPurposeServerStrategy(ServerStrategy):
    """
    Server strategy dedicated to client-selection experiments.

    Supports the following selector methods out-of-the-box:
        - ``high_loss``              – pick top-k by training loss
        - ``low_loss``               – pick bottom-k by training loss
        - ``high_weight_divergence`` – pick top-k by ‖w_local − w_global‖₂
        - ``random``                 – uniform random selection
        - ``all``                    – select every available client

    The active selector is determined by ``config_dict["client_selection"]["method"]``
    and is validated at construction time so misconfigurations fail early.

    Feedback path
    -------------
    ``receive_client_updates()`` stores updates for aggregation AND
    immediately feeds ``train_record`` data to the selector via
    ``with_clients_data()``.  This means the strategy works correctly with
    both the standard ``FedAvgRunnerStrategy`` (which calls
    ``receive_client_updates`` after real training) **and** the
    ``ObserverRunnerStrategy`` (which calls ``receive_client_updates_for_selection``
    after the observation pass).
    """

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "client_selection_purpose"
        self._obj = server_node
        self._validate_selector_method()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_selector_method(self) -> None:
        cfg = getattr(self._obj, "node_var", None)
        config_dict = getattr(cfg, "config_dict", {}) if cfg else {}
        method = (
            config_dict.get("client_selection", {}).get("method", "random").lower()
        )
        if method not in _SUPPORTED_SELECTORS:
            raise ValueError(
                f"[ClientSelectionPurpose] Unsupported selector method '{method}'. "
                f"Supported: {sorted(_SUPPORTED_SELECTORS)}"
            )
        console.info(
            f"[ClientSelectionPurpose] Active selector: '{method}'"
        )

    @staticmethod
    def _extract_feedback(client_updates: List[Dict[str, Any]]) -> Dict[str, dict]:
        """
        Parse a list of client update dicts into the ``{node_id: {...}}`` format
        expected by ``FedClientSelector.with_clients_data()``.

        Handles two forms of ``train_record``:
          - plain ``dict``  (normal training path)
          - ``(weights, dict)`` tuple  (observation_step path)
        """
        feedback: Dict[str, dict] = {}
        for item in client_updates:
            if not isinstance(item, dict):
                continue
            train_rec = item.get("train_record", {})
            # Unwrap (weights, record) tuple produced by observation_step
            if isinstance(train_rec, tuple) and len(train_rec) == 2:
                train_rec = train_rec[1]
            if not isinstance(train_rec, dict):
                continue
            node_id = train_rec.get("node_id") or item.get("node_id")
            if node_id is None:
                continue
            feedback[str(node_id)] = {
                "train_record": train_rec,
                "latency": item.get("latency", 1.0),
                "data_sample_num": (
                    train_rec.get("data_sample_num")
                    or item.get("data_sample_num")
                ),
            }
        return feedback

    def _push_feedback_to_selector(self, client_updates: List[Dict[str, Any]]) -> None:
        """Push extracted feedback into the selector if it supports it."""
        selector = getattr(self._obj.node_var, "client_selection", None)
        if selector is None or not hasattr(selector, "with_clients_data"):
            return
        feedback = self._extract_feedback(client_updates)
        if feedback:
            selector.with_clients_data(feedback)
            console.debug(
                f"[ClientSelectionPurpose] Fed {len(feedback)} client records "
                f"to selector '{selector.select_method}'"
            )

    # ------------------------------------------------------------------
    # ServerStrategy interface
    # ------------------------------------------------------------------

    def aggregation(self) -> None:
        aggregator = self._obj.node_var.aggregation_method
        aggregated_weights = aggregator.aggregate(self._obj.node_var.client_updates)
        self._obj.node_var.aggregated_weight = aggregated_weights

    def select_clients(self, available_clients) -> list:
        selector = self._obj.node_var.client_selection
        number = self._obj.node_var.config_dict["client_selection"]["number"]
        selected = selector.select(available_clients, number)
        ids = [str(c.node_id) for c in selected]
        console.info(
            f"[ClientSelectionPurpose] Selected {len(selected)} / {len(available_clients)} clients: "
        ).ok(", ".join(ids))
        return selected

    def receive_client_updates(self, client_updates: List[Dict[str, Any]]) -> None:
        """
        Store updates for aggregation and simultaneously feed metrics to the
        selector so that metric-based selectors (high_loss, low_loss,
        high_weight_divergence) have up-to-date data for the *next* round.
        """
        self._obj.node_var.client_updates = client_updates
        self._push_feedback_to_selector(client_updates)

    def receive_client_updates_for_selection(self, client_updates: List[Dict[str, Any]]) -> None:
        """
        Called by ObserverRunnerStrategy after the lightweight observation pass.
        Only updates the selector; does NOT overwrite ``client_updates`` used
        for aggregation.
        """
        self._push_feedback_to_selector(client_updates)

    def record_evaluation(self) -> None:
        self._obj.node_var.training_logger.record(self._obj.eval_results)

    def broadcast(self) -> None:
        for client in self._obj.client_nodes:
            client.receive_weight(self._obj.node_var.model_weight)
            client.set_local_weight()

    def evaluate(self) -> None:
        self._obj.eval_results = self._obj.node_var.model_evaluator.evaluate()
        self._obj.node_var.model_evaluator.print_results()
        console.info("Server Evaluation Completed.\n")

    def prepare(self, logger_header, client_nodes_in) -> None:
        self._obj.node_var.training_logger.begin(logger_header)
        self._obj.set_client_nodes(client_nodes_in)

    def run(self) -> None:
        raise NotImplementedError(
            "ClientSelectionPurposeServerStrategy does not implement run(). "
            "Use a RunnerStrategy (e.g. observer or fedavg) to drive the training loop."
        )
