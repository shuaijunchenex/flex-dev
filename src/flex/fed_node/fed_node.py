from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from .fed_node_vars import FedNodeVars
from .fed_node_type import EFedNodeType
from ..ml_simu_switcher import SimuNode, SimuSwitcher
from ..ml_utils import EventHandler
from ..fed_strategy.client_strategy_impl._fedavg_client import FedAvgClientTrainingStrategy
from ..model_trainer.model_evaluator import ModelEvaluator

class FedNode(ABC, EventHandler):
    '''
    Node class interface declare(virtual class as interface)
    '''
    
    def __init__(self, node_id: str, node_group: str = ""):
        EventHandler.__init__(self)

        self._node_id: str = node_id  # Unique Node ID
        self.node_group: str = node_group  # Belong to group
        self.node_type: EFedNodeType = EFedNodeType.unknown  # Node Type
        self.simu_switcher: Optional[SimuSwitcher] = None  # switcher
        self.simu_node: Optional[SimuNode] = None  # Simu node of switcher

        # Node var associated to node
        self.node_var: Optional[FedNodeVars] = None

        #
        #self.evaluator = ModelEvaluator(self.node_var)  # Model evaluator

        return

    @property
    def node_id(self): return self._node_id

    @property
    def node_full_id(self):
        """
        Node full id with group id, like "client_1@group_1"
        """
        return f"{self._node_id}@{self.node_group}"

    def with_node_var(self, var: FedNodeVars):
        """
        DI node var
        """
        self.node_var = var
        return self

    def create_simu_node(self, simu_switcher: SimuSwitcher):
        """
        Create node's simu node for data exchange
        """
        self.simu_switcher = simu_switcher
        self.simu_node = self.simu_switcher.create_node(self._node_id)
        return

    def evaluate_model(self, test_data):
        """
        Evaluate model with test data
        """
        if self.node_var is not None:
            return self.node_var.evaluate_model(test_data)
        return

    def connect(self, node_id: str):
        """
        Make connection of this node to specified simu node(node id)
        """
        if self.simu_node is not None:
            self.simu_node.connect(node_id)
        return

    @abstractmethod
    def run(self) -> None:
        """
        run node
        """
        return

    def training(self):
        """
        Client node executes local operations.
        """
        return self.strategy.run_local_training()

    def __str__(self):
        return self.node_full_id
