from __future__ import annotations

from ...ml_utils import console
from ...ml_utils.tqdm_utils import pbar
from ...ml_utils.training_utils import TrainingUtils
from ._rbla_runner_strategy import RblaRunnerStrategy


class SpPlusRunnerStrategy(RblaRunnerStrategy):
    """Round orchestration for the standalone SP+ strategy."""

    def __init__(self, runner, args, client_node, server_node) -> None:
        super().__init__(runner, args, client_node, server_node)
        self._strategy_type = "sp_plus"

    def run(self) -> None:
        print("Running [SP+] strategy...")
        header_data = TrainingUtils.build_training_header(self.server_node)
        self.server_node.prepare(header_data, self.client_nodes)
        self.server_node.broadcast()
        training_rounds = self.args.key_value_dict.data["training_rounds"]

        for round_index in pbar(range(training_rounds + 1)):
            console.out(
                f"\n{'=' * 10} Training round {round_index}/{training_rounds}, "
                f"Total participants: {len(self.client_nodes)} {'=' * 10}"
            )
            self.participants = self.server_node.select_clients(self.client_nodes)
            console.info(
                f"Round: {round_index}, Select {len(self.participants)} clients: ', '"
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
                f"{'=' * 10} Round {round_index}/{training_rounds} End{'=' * 10}"
            )
        return

