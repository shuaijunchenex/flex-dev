import copy
import time

import torch
from ...ml_utils.tqdm_utils import pbar

from ...ml_utils import console
from ...ml_utils.training_utils import TrainingUtils
from ...fed_runner import FedRunner
from ...fed_strategy.runner_strategy import RunnerStrategy
from ...fed_node import FedNodeClient, FedNodeServer


class FedgraKerasExactRunnerStrategy(RunnerStrategy):
    """KerasFL-state FedGRA runner.

    This runner intentionally reproduces KerasFL's shared-model training state:
    all clients train sequentially from one shared model state, and each client's
    stored weight is updated only after that client trains. The selected-client
    aggregation pass also continues from the shared state left by the selection
    pass, matching KerasFL's shared `model.fit()` behaviour.
    """

    def __init__(self, runner: FedRunner, args, client_node, server_node) -> None:
        super().__init__(runner)
        self._strategy_type = "fedgra_keras_exact"
        self.args = args
        self.client_nodes: list[FedNodeClient] = client_node
        self.server_node: FedNodeServer = server_node
        self._shared_weight = None
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

    def _layerwise_l2_distance(self, weight_a: dict, weight_b: dict) -> float:
        total_norm = 0.0
        for key in sorted(set(weight_a.keys()) & set(weight_b.keys())):
            wa = weight_a[key]
            wb = weight_b[key]
            if isinstance(wa, torch.Tensor) and isinstance(wb, torch.Tensor):
                diff = wa.detach().cpu().float() - wb.detach().cpu().float()
                total_norm += float(torch.norm(diff, p=2).item())
        return total_norm

    def _patch_weight_divergence(self, train_record, before_client_weight, updated_weight) -> None:
        if not isinstance(train_record, dict) or not isinstance(updated_weight, dict):
            return
        inner = train_record.get("train_record", train_record)
        if isinstance(inner, dict) and isinstance(before_client_weight, dict):
            inner["weight_l2_delta_keras"] = self._layerwise_l2_distance(
                before_client_weight,
                updated_weight,
            )

    def _run_client_local_training(self, client):
        if self._shared_weight is None:
            self._shared_weight = copy.deepcopy(self.server_node.node_var.model_weight)

        before_client_weight = copy.deepcopy(client.node_var.model_weight)
        client.node_var.model_weight = copy.deepcopy(self._shared_weight)

        updated_weight, train_record = client.run_local_training()
        self._patch_weight_divergence(train_record, before_client_weight, updated_weight)

        client.node_var.model_weight = copy.deepcopy(updated_weight)
        self._shared_weight = copy.deepcopy(updated_weight)
        return updated_weight, train_record

    def simulate_client_local_training_process(self, participants, label: str = "local", observe: bool = False):
        """Train participants sequentially with shared model state.

        When *observe* is True, each client trains for exactly 1 epoch
        (config ``training.epochs`` is temporarily overridden) so the
        selection pass only gathers quick metrics without burning the
        client's full training budget.  The original epoch count is
        restored after all participants finish.
        """
        saved_epochs: dict = {}
        if observe:
            for client in participants:
                cfg = client.node_var.config_dict
                training = cfg.setdefault("training", {})
                saved_epochs[client] = training.get("epochs")
                training["epochs"] = 1
        try:
            for client in participants:
                start_time = time.time()
                console.info(f"\n[{client.node_id}] KerasFL-state {label} training")
                updated_weights, train_record = self._run_client_local_training(client)
                yield {
                    "updated_weights": updated_weights,
                    "train_record": train_record,
                    "latency": time.time() - start_time,
                }
        finally:
            if observe:
                for client, old_epochs in saved_epochs.items():
                    cfg = client.node_var.config_dict
                    if old_epochs is not None:
                        cfg["training"]["epochs"] = old_epochs
                    else:
                        cfg["training"].pop("epochs", None)

    def simulate_server_broadcast_process(self):
        self.server_node.broadcast(self.client_nodes)
        return

    def simulate_server_update_process(self, weight):
        self.server_node.strategy.server_update(weight)
        return

    def run(self) -> None:
        print("Running [FedGRA-Keras-Exact] strategy...")
        header_data = TrainingUtils.build_training_header(self.server_node)
        self.server_node.prepare(header_data, self.client_nodes)
        self.server_node.broadcast()
        self._shared_weight = copy.deepcopy(self.server_node.node_var.model_weight)
        self.participants = []

        for round_idx in pbar(range(self.args.key_value_dict.data["training_rounds"] + 1)):
            console.out(
                f"\n{'=' * 10} Training round {round_idx}/"
                f"{self.args.key_value_dict.data['training_rounds']}, "
                f"Total participants: {len(self.client_nodes)} {'=' * 10}"
            )

            selector = getattr(self.server_node.node_var, "client_selection", None)
            selection_interval = getattr(selector, "client_selection_round", 1)

            if round_idx % selection_interval == 0:
                all_updates = list(
                    self.simulate_client_local_training_process(
                        self.client_nodes,
                        label="selection",
                        observe=True,
                    )
                )
                self.server_node.receive_client_updates_for_selection(all_updates)
                self.participants = self.server_node.select_clients(self.client_nodes)
                console.info(
                    f"Round: {round_idx}, Select {len(self.participants)} clients: "
                ).ok(f"{', '.join(map(str, self.participants))}")

            client_updates = list(
                self.simulate_client_local_training_process(
                    self.participants,
                    label="aggregation",
                )
            )

            self.server_node.receive_client_updates(client_updates)
            self.server_node.aggregation()
            self.server_node.apply_weight()
            self._shared_weight = copy.deepcopy(self.server_node.node_var.model_weight)
            self.server_node.broadcast()

            self.server_node.evaluate()
            self.server_node.record_evaluation()

            node_vars = self.server_node.node_var
            logger = getattr(node_vars, "training_logger", None)
            weight = getattr(node_vars, "aggregated_weight", None) or getattr(node_vars, "model_weight", None)
            if logger is not None and weight is not None:
                saved = logger.save_weights_if_enabled(weight, round_idx)
                if saved:
                    console.debug(f"[FedGRA-Keras-Exact] Round {round_idx} weights saved -> {saved}")

            console.out(
                f"{'=' * 10} Round {round_idx}/"
                f"{self.args.key_value_dict.data['training_rounds']} End{'=' * 10}"
            )
