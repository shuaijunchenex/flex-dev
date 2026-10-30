"""
RBLA-SASG runner strategy.

Orchestrates the FL round loop: select → broadcast → train → aggregate.
"""

from __future__ import annotations

from ...ml_utils.tqdm_utils import pbar

from flex.fed_strategy.strategy_args import StrategyArgs
from ...ml_utils import console
from ...ml_utils.training_utils import TrainingUtils
from ...fed_runner import FedRunner
from ...fed_strategy.runner_strategy import RunnerStrategy
from ...fed_node import FedNodeClient, FedNodeServer
from ...ml_utils.model_utils import ModelUtils


class RblaSasgRunnerStrategy(RunnerStrategy):

    def __init__(
        self,
        runner: FedRunner,
        args,
        client_node,
        server_node,
    ) -> None:
        super().__init__(runner)
        self._strategy_type = "rbla_sasg"
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
        self.server_node.broadcast()
        return

    def simulate_server_update_process(self, weight):
        self.server_node.strategy.server_update(weight)
        return

    def run(self) -> None:
        print("Running [RBLA-SASG] strategy...")
        header_data = TrainingUtils.build_training_header(self.server_node)
        self.server_node.prepare(header_data, self.client_nodes)

        # ── Initial broadcast ─────────────────────────────────────────
        self.server_node.broadcast()

        for round_idx in pbar(
            range(self.args.key_value_dict.data["training_rounds"] + 1)
        ):
            console.out(
                f"\n{'=' * 10} Training round {round_idx}/"
                f"{self.args.key_value_dict.data['training_rounds']}, "
                f"Total participants: {len(self.client_nodes)} {'=' * 10}"
            )

            self.participants = self.server_node.select_clients(self.client_nodes)
            console.info(
                f"Round: {round_idx}, Select {len(self.participants)} clients: "
            ).ok(f"{', '.join(map(str, self.participants))}")

            client_updates = list(
                self.simulate_client_local_training_process(self.participants)
            )

            # ── Enrich updates with per-client metadata from client strategy ──
            for update, client in zip(client_updates, self.participants):
                tr = update.get("train_record", {})
                # Lift per-prefix metadata to top level (aggregator reads them from here)
                update["data_sample_num"] = tr.get("data_sample_num", 1)
                update["rank_by_prefix"] = tr.get("rank_by_prefix", {})
                update["Phi_by_prefix"] = tr.get("Phi_by_prefix", {})
                # Backward-compat: global r_i / Phi_i
                if hasattr(client.strategy, "_r_i"):
                    update["r_i"] = client.strategy._r_i
                if hasattr(client.strategy, "_Phi_i"):
                    update["Phi_i"] = list(client.strategy._Phi_i)

            self.server_node.receive_client_updates(client_updates)
            self.server_node.aggregation()
            self.server_node.apply_weight()
            self.server_node.broadcast()
            self.server_node.evaluate()
            self.server_node.record_evaluation()

            # ── Save weights ───────────────────────────────────────────
            node_vars = self.server_node.node_var
            logger = getattr(node_vars, "training_logger", None)
            weight = getattr(node_vars, "aggregated_weight", None) or getattr(
                node_vars, "model_weight", None
            )
            if logger is not None and weight is not None:
                saved = logger.save_weights_if_enabled(weight, round_idx)
                if saved:
                    console.debug(
                        f"[RBLA-SASG] Round {round_idx} weights saved → {saved}"
                    )

            console.out(
                f"{'=' * 10} Round {round_idx}/"
                f"{self.args.key_value_dict.data['training_rounds']} End{'=' * 10}"
            )
