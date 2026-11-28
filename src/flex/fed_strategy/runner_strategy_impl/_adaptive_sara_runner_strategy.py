from __future__ import annotations

from ...ml_utils.tqdm_utils import pbar

from ._sara_runner_strategy import SaraRunnerStrategy
from ...ml_utils import console
from ...ml_utils.training_utils import TrainingUtils


class AdaptiveSaraRunnerStrategy(SaraRunnerStrategy):
    """Runner for Adaptive SARA.

    Same round-index injection as SARA; kept separate so experiments can select
    an independent strategy route.
    """

    def __init__(self, runner, args, client_node, server_node) -> None:
        super().__init__(runner, args, client_node, server_node)
        self._strategy_type = "adaptive_sara"

    def run(self) -> None:
        print("Running [Adaptive SARA] strategy...")
        header_data = TrainingUtils.build_training_header(self.server_node)
        self.server_node.prepare(header_data, self.client_nodes)
        self.server_node.broadcast()

        total_rounds = self.args.key_value_dict.data.get("training_rounds", 100)
        for round_idx in pbar(range(total_rounds + 1)):
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
