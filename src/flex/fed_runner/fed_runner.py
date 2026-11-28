from __future__ import annotations
from abc import ABC, abstractmethod

import yaml

from ..ml_utils import console
from ..ml_utils.tqdm_utils import pbar
from ..ml_simu_switcher import SimuSwitcher
from ..fed_node import FedNodeClient, FedNodeEdge, FedNodeServer
from ..fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from ..fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from ..ml_utils.text_logger import TextLogger
from ..ml_utils.training_logger import TrainingLogger
from ..fed_strategy.strategy_factory import StrategyFactory
from ..fed_strategy.strategy_args import StrategyArgs

class FedRunner(ABC):
    def __init__(self):
        self._switcher = SimuSwitcher()          #Simulate net switcher

        self._yaml: dict = {}
        self.training_rounds = 50

        self.client_node_list = []
        self.edge_node_list = []
        self.server_node: FedNodeServer|None = None
        self.runner_strategy = None #TODO
        self.train_logger = TrainingLogger(self._yaml.get("logger", None))
        #self.__create_aggregator_selector_from_yaml()

        return

    #------------------------------------------
    @property
    def client_node_count(self):
        return len(self.client_node_list)

    @property
    def edge_node_count(self):
        return len(self.edge_node_list)

    def with_switcher(self, switcher):
        self._switcher = switcher
        return self

    def with_yaml(self, runner_yaml):
        self._yaml = runner_yaml
        return self

    #------------------------------------------
    def create_run_strategy(self):
        self.__create_run_strategy_from_yaml()

    def create_nodes(self):
        # Create server node(only 1 node)
        self.__create_server_nodes(self._yaml)

        # Create edge nodes
        self.__create_edge_nodes(self._yaml)

        # Create client nodes
        self.__create_client_nodes(self._yaml)
        return

    #private
    def __create_client_nodes(self, runner_yaml: dict):
        # Check 'server_node' if defined
        if "client_nodes" not in runner_yaml:
            console.error_exception("'client_nodes' not defined in node config yaml")

        node_count = 1

        if "client_nodes" in runner_yaml:
            client_section = runner_yaml["client_nodes"]
        else:
            client_section = runner_yaml

        for group in client_section:
            g = client_section[group]
            num: int = g["number"]
            id_prefix = g.get("id_prefix", "")
            link_to = g.get("link_to", "server")

            for index in range(1, num + 1):
                node_id = f"{id_prefix}.{node_count}"
                client = FedNodeClient(node_id, group)

                self.client_node_list.append(client)

                print("Create Node", node_id, "with group", group)

                # Create simu node and connect to node
                client.create_simu_node(self._switcher)

                # Create local strategy
                #client.create_local_strategy(self._yaml)
                
                # Create topology link
                client.connect(link_to)
                
                node_count += 1

        return

    #private
    def __create_edge_nodes(self, runner_yaml: dict):
        # TODO:
        return

    #private
    def __create_server_nodes(self, runner_yaml: dict):
        # Check 'server_node' if defined
        if "server_node" not in runner_yaml:
            console.warn("'server_node' not defined in runner yaml")
            return

        if "server_node" in runner_yaml:
            server_section = runner_yaml["server_node"]
        else:
            server_section = runner_yaml

        server_id = server_section.get("id", "server")

        self.server_node = FedNodeServer(server_id)

        # Create simu node
        self.server_node.create_simu_node(self._switcher)
        return

    def __create_run_strategy_from_yaml(self):

        args = StrategyFactory.create_runner_args(self._yaml)
        self.runner_strategy = StrategyFactory.create_runner_strategy(args, self, self.client_node_list, self.server_node)
        return 

    def run(self):
        if self.runner_strategy is None:
            raise RuntimeError("runner_strategy is not set")
        
        self.runner_strategy.run()
