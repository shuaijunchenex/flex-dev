"""
rblasa runner strategy �?identical to RBLA runner but injects round index
into client config for lambda-schedule decay.
"""
from __future__ import annotations

from ...ml_utils.tqdm_utils import pbar

from flex.fed_strategy.strategy_args import StrategyArgs
from ...ml_utils import console
from ...ml_utils.training_utils import TrainingUtils
from ...fed_runner import FedRunner
from ...fed_strategy.runner_strategy import RunnerStrategy
from ...fed_node import FedNodeClient, FedNodeServer


class rblasaRunnerStrategy(RunnerStrategy):
    """Runner for rblasa (uses RBLA aggregation + alignment-aware clients)."""

    def __init__(self, runner: FedRunner, args, client_node, server_node) -> None:
        super().__init__(runner)
        self._strategy_type = "rblasa"
        self.args = args
        self.client_nodes: list[FedNodeClient] = client_node
        self.server_node: FedNodeServer = server_node
        self.set_node_connection()

    def _create_inner(self, client_node, server_node) -> None:
        return self

    def prepare(self, logger_header) -> None:
        self.server_node.prepare(logger_header, self.client_nodes)

    def set_node_connection(self) -> None:
        self.server_node.set_client_nodes(self.client_nodes)
        for client in self.client_nodes:
            client.set_server_node(self.server_node)

    def simulate_client_local_training_process(self, participants):
        for client in participants:
            console.info(f"\n[{client.node_id}] Local training started")
            updated_weights, train_record = client.strategy.run_local_training()
            yield {
                "updated_weights": updated_weights,
                "train_record": train_record,
            }

    def simulate_server_broadcast_process(self):
        self.server_node.broadcast_weight(self.client_nodes)

    def simulate_server_update_process(self, weight):
        self.server_node.strategy.server_update(weight)

    def run(self) -> None:
        print("Running [rblasa] strategy...")
        header_data = TrainingUtils.build_training_header(self.server_node)
        self.server_node.prepare(header_data, self.client_nodes)
        self.server_node.broadcast()

        total_rounds = self.args.key_value_dict.data.get("training_rounds", 100)
        for round_idx in pbar(range(total_rounds + 1)):
            # Inject round index into every client config for lambda decay
            for client in self.client_nodes:
                client.node_var.config_dict["_round_idx"] = round_idx

            console.out(
                f"\n{'='*10} Training round {round_idx}/{total_rounds}, "
                f"Total participants: {len(self.client_nodes)} {'='*10}"
            )

            self.participants = self.server_node.select_clients(self.client_nodes)

            console.info(
                f"Round: {round_idx}, Select {len(self.participants)} clients: ', '"
            ).ok(f"{', '.join(map(str, self.participants))}")

            client_updates = list(
                self.simulate_client_local_training_process(self.participants)
            )

            self.server_node.receive_client_updates(client_updates)
            self.server_node.aggregation()
            self.server_node.apply_weight()
            self.server_node.broadcast()
            self.server_node.evaluate()
            self.server_node.record_evaluation()

            console.out(
                f"{'='*10} Round {round_idx}/{total_rounds} End{'='*10}"
            )
