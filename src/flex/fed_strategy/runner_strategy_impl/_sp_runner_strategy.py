import time

from ...ml_utils.tqdm_utils import pbar

from flex.fed_strategy.strategy_args import StrategyArgs
from ...ml_utils import console
from ...ml_utils.training_utils import TrainingUtils
from ...fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from ...fl_algorithms.aggregation.methods._fed_aggregator_sp import FedAggregator_SP
from ...fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from ...fed_runner import FedRunner
from ...fed_strategy.runner_strategy import RunnerStrategy 
from ...fed_node import FedNodeClient, FedNodeServer

class SpRunnerStrategy(RunnerStrategy):

    def __init__(self, runner: FedRunner, args, client_node, server_node) -> None:
        super().__init__(runner) #TODO: modify runner object declaration

        self._strategy_type = "sp"
        self.args = args
        self.client_nodes : list[FedNodeClient]= client_node
        self.server_node : FedNodeServer = server_node
        self.set_node_connection()

    def _create_inner(self, client_node, server_node) -> None:
        return self

    def set_node_connection(self) -> None:
        self.server_node.set_client_nodes(self.client_nodes)
        for client in self.client_nodes:
            client.set_server_node(self.server_node)
        return

    def prepare(self, logger_header) -> None:
        self.server_node.prepare(logger_header, self.client_nodes)
        return

    def simulate_client_local_training_process(self, participants):
        for client in participants:
            console.info(f"\n[{client.node_id}] Local training started")
            updated_weights, train_record = client.strategy.run_local_training()
            yield {
                "updated_weights": updated_weights,
                "train_record": train_record
            }

    def simulate_server_broadcast_process(self):
        self.server_node.broadcast(self.client_nodes)
        return

    def simulate_server_update_process(self):
        return

    def run(self) -> None:
        print("Running [Sum-Product] strategy...")
        header_data = TrainingUtils.build_training_header(self.server_node)
        self.server_node.prepare(header_data, self.client_nodes)
        self.server_node.broadcast()
        for round in pbar(range(self.args.key_value_dict.data['training_rounds'] + 1)):
           
            console.out(f"\n{'='*10} Training round {round}/{self.args.key_value_dict.data['training_rounds']}, Total participants: {len(self.client_nodes)} {'='*10}")
            
            self.participants = self.server_node.select_clients(self.client_nodes)
            
            console.info(f"Round: {round}, Select {len(self.participants)} clients: ', '").ok(f"{', '.join(map(str, self.participants))}")

            client_updates = list(self.simulate_client_local_training_process(self.participants))         

            self.server_node.receive_client_updates(client_updates)

            self.server_node.aggregation()

            self.server_node.apply_weight()

            self.server_node.broadcast()

            self.server_node.evaluate()

            self.server_node.record_evaluation()

            console.out(f"{'='*10} Round {round}/{self.args.key_value_dict.data['training_rounds']} End{'='*10}")

        return
        
