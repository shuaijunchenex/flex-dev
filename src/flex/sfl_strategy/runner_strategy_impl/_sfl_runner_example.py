from __future__ import annotations

from typing import Any, Iterable, Sequence

from flex.fed_runner.fed_runner import FedRunner
from flex.fed_strategy.runner_strategy import RunnerStrategy
from flex.fed_strategy.strategy_args import StrategyArgs
from flex.ml_utils import console
from flex.ml_utils.tqdm_utils import pbar


class SflRunnerStrategy(RunnerStrategy):
    def __init__(self, runner: FedRunner, args: StrategyArgs, client_nodes: Sequence[Any], server_node: Any) -> None:
        super().__init__(runner)
        self._strategy_type = "sfl"
        self.args = args
        self.client_nodes = list(client_nodes)
        self.server_node = server_node
        self.set_node_connection()

    def _create_inner(self, client_nodes: Sequence[Any] | None, server_nodes: Sequence[Any] | None) -> None:
        return

    def prepare(self, logger_header: dict[str, Any]) -> None:
        self.server_node.prepare(logger_header, self.client_nodes)

    def set_node_connection(self) -> None:
        if self.server_node is None:
            return
        self.server_node.set_client_nodes(self.client_nodes)
        for client in self.client_nodes:
            client.set_server_node(self.server_node)

    def simulate_client_local_training_process(self, participants: Iterable[Any]):
        server_strategy = getattr(self.server_node, "strategy", None)
        if server_strategy is None:
            raise RuntimeError("SFL runner requires server strategy for forward/backward steps.")

        for client in participants:
            console.info(f"\n[{client.node_id}] Local training started")

            # Client forward to get smashed activations
            forward_batches = client.strategy.run_local_forward()

            activation_grads = []
            loss_sum = 0.0
            metric_acc = {}
            grad_norm_sum = 0.0

            for smashed_data, labels in forward_batches:
                server_input, loss, metrics = server_strategy.run_server_forward(smashed_data, labels, training=True)
                act_grad, loss_value = server_strategy.run_server_backward(server_input, loss, training=True)
                
                activation_grads.append(act_grad)
                loss_sum += float(loss_value)
                grad_norm_sum += float(act_grad.norm().item())
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        metric_acc[k] = metric_acc.get(k, 0.0) + float(v)

            batch_count = max(len(forward_batches), 1)
            if metric_acc:
                metric_acc = {k: v / batch_count for k, v in metric_acc.items()}
            grad_norm_avg = grad_norm_sum / batch_count if batch_count > 0 else 0.0

            front_weight, client_metrics = client.strategy.run_local_backward(activation_grads) # change name to: gradient

            train_record = {
                "server_loss_avg": loss_sum / batch_count,
                "server_metrics": metric_acc,
                "client_metrics": client_metrics,
            }
            console.debug(
                f"[SFL-Server] client {client.node_id} server_loss_avg={loss_sum / batch_count:.4f}, grad_norm_avg={grad_norm_avg:.4f}"
            )

            yield {
                "front_weight": front_weight,
                "train_record": train_record,
                "data_sample_num": getattr(client.node_var, "data_sample_num", 0),
            }

    def simulate_server_broadcast_process(self) -> None:
        self.server_node.broadcast()

    def simulate_server_update_process(self) -> None:
        self.server_node.aggregation()

    def run(self) -> None:
        if self.server_node is None:
            raise ValueError("SFL runner strategy requires a server node.")

        console.info("Running [SFL] strategy...")
        header_data = {
            "round": "round",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1_score": "f1_score",
        }
        self.server_node.prepare(header_data, self.client_nodes)

        total_rounds = int(self.args.get("training_rounds", 0))
        for round_idx in pbar(range(total_rounds), disable=total_rounds <= 0):
            console.out(
                f"\n{'=' * 10} Training round {round_idx + 1}/{total_rounds}, "
                f"Total participants: {len(self.client_nodes)} {'=' * 10}"
            )

            participants = self.server_node.select_clients(self.client_nodes)
            
            ids = [str(c.node_id) for c in participants]

            console.info(f"Round: {round_idx + 1}, Select {len(participants)} clients: ").ok(", ".join(ids))

            client_updates = list(self.simulate_client_local_training_process(participants))

            self.server_node.receive_client_updates(client_updates)

            self.simulate_server_update_process()

            self.server_node.apply_weight()

            self.simulate_server_broadcast_process()

            self.server_node.evaluate()

            self.server_node.record_evaluation()

            console.out(
                f"{'=' * 10} Round {round_idx + 1}/{total_rounds} End{'=' * 10}"
            )
