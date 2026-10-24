from __future__ import annotations

from .fed_node import FedNode
from .fed_node_type import EFedNodeType
from ..fed_strategy.strategy_factory import StrategyFactory
from ..ml_utils import console
from .fed_node_event_args import FedNodeEventArgs
# from train_strategy.client_strategy.fedavg_client import FedAvgClientTrainingStrategy
# from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter
# from model_extractor.advanced_model_extractor import AdvancedModelExtractor


class FedNodeClient(FedNode):
    def __init__(self, node_id: str, node_group:str = ""):
        super().__init__(node_id, node_group)

        # Client node type
        self.node_type = EFedNodeType.client
        self.node_strategy = None
        return

    def set_server_node(self, server_node):
        self.server_node = server_node
        return

    # override
    def run(self) -> None:
        return

    def run_local_training(self):
        """
        Run local training on the client node
        """
        updated_weights, train_record = self.strategy.run_local_training()
        return updated_weights, train_record

    def receive_weight(self, broadcast_weight):
        """
        Receive new weight from server
        """
        #self.node_var.strategy.receive_weight(broadcast_weight)
        self.strategy.receive_weight(broadcast_weight)
        console.info(f"{self._node_id}: Received new weight from server.")
        return
    
    def set_local_weight(self):
        """
        Set local weight to the current model weight
        """
        #self.node_var.strategy.set_local_weight()
        self.strategy.set_local_weight()
        return
    
    def prepare_strategy(self):
        self.declare_events('on_prepare_strategy')
        if "strategy" in self.node_var.config_dict:
            self.strategy = self.node_var.config_dict["strategy"]
        # Raise strategy event
        args = FedNodeEventArgs("strategy", self.node_var.config_dict).with_sender(self)
        self.strategy = StrategyFactory.create(StrategyFactory.create_args(self.node_var.config_dict["strategy"]), self.node_var.owner_nodes[0])
        self.raise_event("on_prepare_strategy", args)
        return