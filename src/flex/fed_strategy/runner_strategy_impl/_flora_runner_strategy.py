from __future__ import annotations

from ...ml_utils.tqdm_utils import pbar

from flex.fed_strategy.strategy_args import StrategyArgs
from ...ml_utils import console
from ...ml_utils.training_utils import TrainingUtils
from ...fed_runner import FedRunner
from ...fed_strategy.runner_strategy import RunnerStrategy
from ...fed_node import FedNodeClient, FedNodeServer


class FloraRunnerStrategy(RunnerStrategy):
    """FLoRA (Wang et al., NeurIPS 2024) — runner / orchestration strategy.

    **Per‑round lifecycle:**

    1. Server selects participating clients.
    2. Each participating client performs local training:
       - Freeze backbone *W*
       - Freshly initialise LoRA A/B with a zero product
       - Train only A/B
    3. Server receives client updates (A/B + non‑LoRA params).
    4. Server runs FLoRA stacking aggregation → ΔW.
    5. Server **merges** ΔW into the backbone: :math:`W \\leftarrow W + \\Delta W`.
    6. Server broadcasts the updated full *W* to **all** clients.
    7. Server evaluates the merged model.
    8. Logging.
    """

    def __init__(
        self,
        runner: FedRunner,
        args: StrategyArgs,
        client_nodes,
        server_node,
    ) -> None:
        super().__init__(runner)
        self._strategy_type = "flora"
        self.args = args
        self.client_nodes: list[FedNodeClient] = client_nodes
        self.server_node: FedNodeServer = server_node
        self.set_node_connection()

    def _create_inner(self, client_node=None, server_node=None) -> None:
        return self

    # ------------------------------------------------------------------
    # Node wiring
    # ------------------------------------------------------------------

    def set_node_connection(self) -> None:
        self.server_node.set_client_nodes(self.client_nodes)
        for client in self.client_nodes:
            client.set_server_node(self.server_node)

    def prepare(self, logger_header: str) -> None:
        self.server_node.prepare(logger_header, self.client_nodes)

    # ------------------------------------------------------------------
    # Simulation helpers (required by RunnerStrategy ABC)
    # ------------------------------------------------------------------

    def simulate_client_local_training_process(self, participants):
        """Run local training sequentially for each participant."""
        for client in participants:
            console.info(f"\n[{client.node_id}] Local training started")
            updated_weights, train_record = client.strategy.run_local_training()
            yield {
                "updated_weights": updated_weights,
                "train_record": train_record,
            }

    def simulate_server_broadcast_process(self):
        self.server_node.broadcast()

    def simulate_server_update_process(self):
        return

    # ------------------------------------------------------------------
    # Main FLoRA round loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        console.out("Running [FLoRA] strategy (merge-ΔW-into-backbone each round)...")
        header_data = TrainingUtils.build_training_header(self.server_node)
        self.server_node.prepare(header_data, self.client_nodes)

        # Heterogeneous-rank clients are independently constructed.  Align
        # their frozen backbone with the authoritative server model before
        # round 0 while retaining each client's own LoRA tensor shapes.
        self.server_node.broadcast()

        total_rounds: int = self.args.key_value_dict.data["training_rounds"]

        for round_idx in pbar(range(total_rounds + 1)):
            console.out(
                f"\n{'=' * 10} FLoRA Round {round_idx}/{total_rounds}"
                f", Total clients: {len(self.client_nodes)} {'=' * 10}"
            )

            # ---- 1. Client selection ----
            participants = self.server_node.select_clients(self.client_nodes)
            console.info(
                f"Round {round_idx}: Selected {len(participants)} clients: "
            ).ok(f"{', '.join(map(str, participants))}")

            # ---- 2 & 3. Client local training → server receive ----
            client_updates = list(
                self.simulate_client_local_training_process(participants)
            )
            self.server_node.receive_client_updates(client_updates)

            # ---- 4. FLoRA stacking aggregation → ΔW ----
            self.server_node.aggregation()

            # ---- 5. W ← W + ΔW (merge) + update evaluator ----
            self.server_node.apply_weight()

            # ---- 6. Broadcast merged W to ALL clients ----
            self.server_node.broadcast()

            # ---- 7. Evaluation & logging ----
            self.server_node.evaluate()
            self.server_node.record_evaluation()

            console.out(
                f"{'=' * 10} FLoRA Round {round_idx}/{total_rounds} End {'=' * 10}"
            )

        return
