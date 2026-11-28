from __future__ import annotations

from ..ml_utils import console
from ..ml_data_process import DataDistribution
from .fed_node import FedNode
from .fed_node_type import EFedNodeType
from ..fed_strategy.strategy_factory import StrategyFactory
from .fed_node_event_args import FedNodeEventArgs

# from train_strategy.client_strategy.fedavg_client import FedAvgClientTrainingStrategy
# from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter
# from model_extractor.advanced_model_extractor import AdvancedModelExtractor 


class FedNodeServer(FedNode):
    def __init__(self, node_id: str, node_group: str = ""):
        super().__init__(node_id, node_group)

        # Server node type
        self.node_type = EFedNodeType.server
        self._eval_round_counter: int = 0
        self._selected_clients_this_round: list | None = None
        return

    def get_client_nodes_from_switcher(self):
        self.client_nodes = self.simu_switcher._node_dict
        return self.simu_switcher._node_dict

    def set_client_nodes(self, client_nodes):
        self.client_nodes = client_nodes
        return

    # override
    def run(self) -> None:
        console.info(f"{self._node_id}: Run...")
        self.strategy.run()
        return

    def broadcast(self):
        self.strategy.broadcast()
        return
    
    def apply_weight(self):
        self.strategy.apply_weight()
        return

    def prepare_strategy(self):
        self.declare_events('on_prepare_strategy')
        if "strategy" in self.node_var.config_dict:
            self.strategy = self.node_var.config_dict["strategy"]
        args = FedNodeEventArgs("strategy", self.node_var.config_dict).with_sender(self)
        self.strategy = StrategyFactory.create(StrategyFactory.create_args(self.node_var.config_dict["strategy"]), self.node_var.owner_nodes[0])
        self.raise_event("on_prepare_strategy", args)
        return
    
    def receive_client_updates(self, client_updates) -> None:
        self.strategy.receive_client_updates(client_updates)
        return

    def receive_client_updates_for_selection(self, client_updates) -> None:
        self.strategy.receive_client_updates_for_selection(client_updates)
        return

    def aggregation(self) -> None:
        self.strategy.aggregation()
        return

    def evaluate(self) -> None:
        self.strategy.evaluate()
        # results = self.node_var.model_evaluator.evaluate()
        # self.eval_results = results
        # self.node_var.model_evaluator.print_results()
        # console.info("Server Evaluation Completed.\n")
        return
    
    def record_evaluation(self)-> None:
        self.strategy.record_evaluation()
        # ── Auto-log selected clients + EWM weights when enabled ──
        logger = getattr(self.node_var, "training_logger", None)
        if logger is not None:
            if self._selected_clients_this_round:
                logger.record_selected_clients(
                    self._eval_round_counter,
                    [str(c) for c in self._selected_clients_this_round]
                )
            # EWM weights from FedGRA selector (if available)
            selector = getattr(self.node_var, "client_selection", None)
            ewm_weights = getattr(selector, "_last_ewm_weights", None)
            if ewm_weights:
                logger.record_ewm_weights(self._eval_round_counter, ewm_weights)
            else:
                # No EWM weights (e.g. high_loss selector) → flush directly
                logger.flush_selected_clients()
        self._eval_round_counter += 1
        self._selected_clients_this_round = None
        return
    
    def select_clients(self, available_clients) -> list:
        selected = self.strategy.select_clients(available_clients)
        self._selected_clients_this_round = selected
        return selected
    
    def prepare(self, logger_header, client_nodes) -> None:
        self.strategy.prepare(logger_header, client_nodes)
        # self.node_var.training_logger.begin(logger_header)
        # self.set_client_nodes(client_nodes)
        return