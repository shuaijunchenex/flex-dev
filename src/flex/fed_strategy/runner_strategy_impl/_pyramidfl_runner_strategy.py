import time

from ...ml_utils.tqdm_utils import pbar

from flex.fed_strategy.strategy_args import StrategyArgs
from ...ml_utils import console
from ...ml_utils.training_utils import TrainingUtils
from ...fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from ...fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from ...fed_runner import FedRunner
from ...fed_strategy.runner_strategy import RunnerStrategy
from ...fed_node import FedNodeClient, FedNodeServer


class PyramidFLRunnerStrategy(RunnerStrategy):
    """Runner wrapper for PyramidFL that mirrors Oort runner flow."""

    def __init__(self, runner: FedRunner, args, client_node, server_node) -> None:
        super().__init__(runner)
        self._strategy_type = "pyramidfl"
        self.args = args
        self.client_nodes: list[FedNodeClient] = client_node
        self.server_node: FedNodeServer = server_node
        self.set_node_connection()

    def _create_inner(self, client_node, server_node) -> None:
        return self

    def prepare(self, logger_header) -> None:
        self.server_node.prepare(logger_header, self.client_nodes)
        return

    def set_node_connection(self) -> None:
        self.server_node.set_client_nodes(self.client_nodes)
        for client in self.client_nodes:
            client.set_server_node(self.server_node)
        return

    def simulate_client_local_training_process(self, participants):
        for client in participants:
            start_time = time.time()
            console.info(f"\n[{client.node_id}] Local training started")
            updated_weights, train_record = client.run_local_training()
            training_duration = time.time() - start_time
            yield {
                "updated_weights": updated_weights,
                "train_record": train_record,
                "latency": training_duration,
            }

    def simulate_server_broadcast_process(self):
        self.server_node.broadcast(self.client_nodes)
        return

    def simulate_server_update_process(self, weight):
        self.server_node.strategy.server_update(weight)
        return

    def run(self) -> None:
        print("Running [PyramidFL] strategy...")
        header_data = TrainingUtils.build_training_header(self.server_node)
        self.server_node.prepare(header_data, self.client_nodes)
        self.participants = []
        for round in pbar(range(self.args.key_value_dict.data['training_rounds'] + 1)):
            console.out(f"\n{'='*10} Training round {round}/{self.args.key_value_dict.data['training_rounds']}, Total participants: {len(self.client_nodes)} {'='*10}")

            selector = getattr(self.server_node.node_var, "client_selection", None)
            selection_interval = getattr(selector, "client_selection_round", 1)
            if round % selection_interval == 0:
                self.participants = self.server_node.select_clients(self.client_nodes)
                console.info(f"Round: {round}, Select {len(self.participants)} clients: ', '").ok(f"{', '.join(map(str, self.participants))}")

                # Deliver per-client optimisation params (adaptive_iter, shard_keep_ratio)
                # to selected clients BEFORE local training begins (Algorithm 1 Lines 14-17).
                # These are computed by the selector during select_clients() above.
                selector = getattr(self.server_node.node_var, "client_selection", None)
                local_params = (selector._client_local_params or {}) if (selector and hasattr(selector, "_client_local_params")) else {}
                for client in self.participants:
                    params = local_params.get(str(client.node_id), {})
                    if params and hasattr(client.strategy, "receive_pyramidfl_params"):
                        client.strategy.receive_pyramidfl_params(params)

            client_updates = list(self.simulate_client_local_training_process(self.participants))

            self.server_node.receive_client_updates(client_updates)

            self.server_node.aggregation()

            self.server_node.apply_weight()

            self.server_node.broadcast()

            self.server_node.evaluate()

            self.server_node.record_evaluation()
            
            console.out(f"{'='*10} Round {round}/{self.args.key_value_dict.data['training_rounds']} End{'='*10}")

        return
