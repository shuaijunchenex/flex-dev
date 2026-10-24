from __future__ import annotations

from typing import Any, Iterable

from flex.ml_utils import console
from flex.ml_utils.tqdm_utils import pbar
from flex.ml_utils.training_utils import TrainingUtils
from flex.sfl_strategy.runner_strategy_impl._sfl_runner_example import SflRunnerStrategy


class SflRoundAvgRunnerStrategy(SflRunnerStrategy):
    def __init__(self, runner, args, client_nodes, server_node) -> None:
        super().__init__(runner, args, client_nodes, server_node)
        self._strategy_type = "sfl_roundavg"

    def simulate_client_local_training_process(self, participants: Iterable[Any]):
        server_strategy = getattr(self.server_node, "strategy", None)
        if server_strategy is None:
            raise RuntimeError("SFL round-average runner requires server strategy for forward/backward steps.")

        if hasattr(server_strategy, "begin_round_training"):
            server_strategy.begin_round_training()

        for client in participants:
            if hasattr(server_strategy, "prepare_client_training"):
                server_strategy.prepare_client_training()

            console.info(f"\n[{client.node_id}] Local training started")

            # ------------------------------------------------------------------
            # Interleaved per-batch training: forward → server fwd/bwd →
            # client backward, all within a single batch loop.
            # This avoids the stale-activation incoherence caused by
            # forwarding all batches before any backward step.
            # ------------------------------------------------------------------
            train_dl = client.strategy.prepare_interleaved_training()

            loss_sum = 0.0
            metric_acc = {}
            grad_norm_sum = 0.0
            batch_count = 0

            loop = pbar(
                train_dl,
                desc=f"SL [{client.node_id}] interleaved",
                leave=False,
                ncols=120,
                mininterval=0.1,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

            for inputs, labels in loop:
                # 1. Client forward (one batch)
                smashed_data = client.strategy.forward_batch_and_cache(inputs)

                # 2. Server forward + backward (one batch)
                server_input, loss, metrics = server_strategy.run_server_forward(smashed_data, labels, training=True)
                act_grad, loss_value = server_strategy.run_server_backward(server_input, loss, training=True)

                # 3. Client backward (one batch, using fresh activation graph)
                grad_norm = client.strategy.backward_batch(act_grad)

                loss_sum += float(loss_value)
                grad_norm_sum += float(grad_norm)
                batch_count += 1
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metric_acc[key] = metric_acc.get(key, 0.0) + float(value)

                loop.set_postfix(loss=f"{loss_value:.4f}")

            batch_count = max(batch_count, 1)
            if metric_acc:
                metric_acc = {key: value / batch_count for key, value in metric_acc.items()}
            grad_norm_avg = grad_norm_sum / batch_count

            front_weight, client_metrics = client.strategy.finish_interleaved_training()
            server_weight = server_strategy.finish_client_training() if hasattr(server_strategy, "finish_client_training") else None

            train_record = {
                "server_loss_avg": loss_sum / batch_count,
                "server_metrics": metric_acc,
                "client_metrics": client_metrics,
            }
            console.debug(
                f"[SFL-Server-RoundAvg] client {client.node_id} server_loss_avg={loss_sum / batch_count:.4f}, grad_norm_avg={grad_norm_avg:.4f}"
            )

            yield {
                "front_weight": front_weight,
                "server_weight": server_weight,
                "train_record": train_record,
                "data_sample_num": getattr(client.node_var, "data_sample_num", 0),
            }

    def run(self) -> None:
        if self.server_node is None:
            raise ValueError("SFL round-average runner strategy requires a server node.")

        console.info("Running [SFL-RoundAvg] strategy...")
        header_data = TrainingUtils.build_training_header(self.server_node)
        self.server_node.prepare(header_data, self.client_nodes)

        server_strategy = getattr(self.server_node, "strategy", None)
        if server_strategy is None:
            raise RuntimeError("SFL round-average runner requires server strategy.")

        if hasattr(server_strategy, "initialize_shared_roundavg_state"):
            server_strategy.initialize_shared_roundavg_state(self.client_nodes)

        self.simulate_server_broadcast_process()

        total_rounds = int(self.args.get("training_rounds", 0))
        participants = []
        for round_idx in range(total_rounds + 1):
            console.out(
                f"\n{'=' * 10} Training round {round_idx}/{total_rounds}, "
                f"Total participants: {len(self.client_nodes)} {'=' * 10}"
            )

            selector = getattr(self.server_node.node_var, "client_selection", None)
            selection_interval = getattr(selector, "client_selection_round", 1)
            if round_idx % selection_interval == 0:
                participants = self.server_node.select_clients(self.client_nodes)

            ids = [str(client.node_id) for client in participants]
            console.info(f"Round: {round_idx}, Select {len(participants)} clients: ").ok(", ".join(ids))

            client_updates = list(self.simulate_client_local_training_process(participants))
            self.server_node.receive_client_updates(client_updates)

            self.simulate_server_update_process()
            self.server_node.apply_weight()
            self.simulate_server_broadcast_process()
            self.server_node.evaluate()
            self.server_node.record_evaluation()

            # ── Save aggregated global weights for post-hoc norm analysis ──
            node_vars = self.server_node.node_var
            _logger = getattr(node_vars, "training_logger", None)
            _weight = getattr(node_vars, "aggregated_weight", None) \
                   or getattr(node_vars, "aggregated_front_weight", None) \
                   or getattr(node_vars, "model_weight", None)
            if _logger is not None and _weight is not None:
                _saved = _logger.save_weights_if_enabled(_weight, round_idx)
                if _saved:
                    console.debug(f"[SFL-RoundAvg] Round {round_idx} weights saved → {_saved}")

            console.out(f"{'=' * 10} Round {round_idx}/{total_rounds} End{'=' * 10}")
