from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Any, List, Optional
from ..fed_runner.fed_runner import FedRunner
from ..fed_strategy.strategy_args import StrategyArgs
from .base_strategy import BaseStrategy

class RunnerStrategy(BaseStrategy):
    
    def __init__(self, runner):
        super().__init__()
        self._obj = runner 
        self._strategy_type = "runner"
        self.client_nodes = None
        self.server_node = None
        return

    def create(self, args: StrategyArgs, client_nodes: List = None, server_nodes: List = None):
        self._args = args
        self._create_inner(client_nodes, server_nodes)
        return self

    @abstractmethod
    def run(self, runner: FedRunner) -> None:

        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def simulate_client_local_training_process():
        """
        Local training simulation method.
        This method should be overridden by subclasses to implement local training logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def simulate_server_broadcast_process():
        """
        Server broadcast simulation method.
        This method should be overridden by subclasses to implement server broadcast logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def simulate_server_update_process():
        """
        Server update simulation method.
        This method should be overridden by subclasses to implement server update logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
