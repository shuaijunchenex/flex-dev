import random
from contextlib import ExitStack, contextmanager
from functools import partial
from time import perf_counter

import torch

from flex.fed_strategy.runner_strategy import RunnerStrategy
from flex.ml_utils.model_utils import tensor_bytes
from flex.ml_utils.training_utils import TrainingUtils

TRAFFIC_FIELDS = ("activation_up", "label_up", "cut_grad_down", "adapter_up", "adapter_down")

# Protocol of the yaml strategy names; the runner yaml can override any key.
PRESETS = {
    "sfl": {"rear_mode": "shared", "front_sync": "fedavg"},
    "sfl_roundavg": {"rear_mode": "round_average", "front_sync": "fedavg"},
    "sfl_aligned": {"rear_mode": "round_average", "front_sync": "fedavg"},
}


class _LoaderView:
    """Batches of one client node, taken from its data loader."""

    def __init__(self, node):
        self.node = node

    def train_batches(self, round_id, epoch):
        return iter(self.node.node_var.data_loader.data_loader)


def _sample_optimizer(row, side, params, optimizer, *hook_args):
    grad_bytes = sum(tensor_bytes(p.grad) for p in params if p.grad is not None)
    slot_bytes = 0
    if optimizer is not None:
        slot_bytes = sum(tensor_bytes(v) for slot in optimizer.state.values()
                         for v in slot.values() if torch.is_tensor(v))
    for kind, size in (("grad", grad_bytes), ("optimizer", slot_bytes)):
        key = f"{side}_peak_{kind}_bytes"
        row[key] = max(row.get(key, 0), size)


class SplitRunnerStrategy(RunnerStrategy):
    """
    Run synchronous front broadcasts with one fresh cut graph per minibatch.

    A round is: every participant visits the server batch by batch (forward, server step,
    backward), then the server commits the rear and aggregates the fronts, then the new
    fronts are broadcast. Rounds are 1..protocol["rounds"]; round 0 is only the optional
    evaluation of the initial models done by the entry.
    """

    def __init__(self, runner, args, client_nodes, server_node):
        super().__init__(runner)
        self._strategy_type = "split"
        self.args = args
        self.client_nodes = list(client_nodes)
        self.server_node = server_node
        self.completed_round = 0
        self.server = None
        self.protocol = None
        self.set_node_connection()

    def _create_inner(self, client_nodes, server_nodes) -> None:
        return

    def set_node_connection(self) -> None:
        self.server_node.set_client_nodes(self.client_nodes)
        for client in self.client_nodes:
            client.set_server_node(self.server_node)

    def configure(self, job, views, *, on_round_end, on_batch_end=None):
        self.protocol = job["protocol"]
        run = job["run"]
        self.seed = run["seed"]
        self.profile = run.get("profile", False)
        self.views = views
        self.on_round_end = on_round_end
        self.on_batch_end = on_batch_end

    def _configure_from_yaml(self):
        """
        Configure from the runner yaml, for runs driven by FedRunner and the yaml nodes: batches
        come from each client's data loader, participants from the server's selector, and every
        round ends with the server's evaluation and record.
        """
        preset = PRESETS.get(self.args.strategy_name, PRESETS["sfl"])
        rear_mode = self.args.get("rear_mode", preset["rear_mode"])
        keeps_slots = self.server_node.strategy._args.preserve_optimizer_state and rear_mode == "round_average"
        protocol = {
            "rear_mode": rear_mode,
            "front_sync": self.args.get("front_sync", preset["front_sync"]),
            "rear_optimizer_history": "per_client" if keeps_slots else "shared",
            "rounds": int(self.args.get("training_rounds")),
            "local_epochs": int(self.args.get("local_epochs", 1)),
            "participant_source": "server",
        }
        self.configure({"protocol": protocol, "run": {"seed": 0}},
                       {node.node_id: _LoaderView(node) for node in self.client_nodes},
                       on_round_end=self._evaluate_round)
        self.server_node.prepare(TrainingUtils.build_training_header(self.server_node), self.client_nodes)

    def _evaluate_round(self, round_id, client_ids, updates):
        self.server_node.evaluate()
        self.server_node.record_evaluation()

    def initialize(self):
        if self.server is not None:
            return
        self.server = self.server_node.strategy
        self.server.initialize(self.client_nodes, self.protocol)
        self._check_optimizer_state_alignment()
        self.last_visit_round = {node.node_id: 0 for node in self.client_nodes}
        self._reset_round()
        self.simulate_server_broadcast_process()

    def _check_optimizer_state_alignment(self):
        """
        A round-average rear reproduces FedAvg only if it keeps optimizer slots exactly when the
        clients keep theirs; a silent mismatch drifts from round 2 on, so refuse to run.
        """
        rear = self.server.rear_state
        if rear.mode != "round_average":
            return
        keeps = rear.history == "per_client"
        for node in self.client_nodes:
            if node.strategy.preserve_optimizer_state != keeps:
                raise ValueError(
                    "preserve_optimizer_state must match on the clients and the rear "
                    f"(client={node.strategy.preserve_optimizer_state}, "
                    f"rear per_client history={keeps}); a round-average rear only mirrors FedAvg "
                    "when both share the same optimizer-state policy."
                )

    def _reset_round(self):
        self.train_rows = []
        self.round_times = {}
        self.traffic = {node.node_id: dict.fromkeys(TRAFFIC_FIELDS, 0) for node in self.client_nodes}

    def simulate_server_broadcast_process(self):
        for client_id, size in self.server.broadcast_front(self.client_nodes).items():
            self.traffic[client_id]["adapter_down"] += size

    def simulate_client_local_training_process(self, round_id, participants):
        return [self.run_visit(round_id, node) for node in participants]

    def simulate_server_update_process(self, updates):
        return self.server.finish_round(updates)

    def participants_for(self, round_id):
        if self.protocol.get("participant_source") == "server":
            return self.server_node.select_clients(self.client_nodes)
        rng = random.Random(f"{self.seed}:{round_id}")
        clients = list(self.client_nodes)
        count = self.protocol.get("participants_per_round", len(clients))
        if not 1 <= count <= len(clients):
            raise ValueError("participants_per_round must be within the client roster")
        if count < len(clients):
            clients = [clients[i] for i in sorted(rng.sample(range(len(clients)), count))]
        order = self.protocol.get("participant_order", "seeded_per_round")
        if order == "seeded_per_round":
            rng.shuffle(clients)
        elif order == "reverse":
            clients.reverse()
        elif order != "fixed":
            raise ValueError(f"Unknown participant order: {order}")
        return clients

    @contextmanager
    def _profile_models(self, row, node):
        # Front gradients clear immediately after step, so sample inside the optimizer hook.
        with ExitStack() as hooks:
            if self.profile:
                for side, var in (("front", node.node_var), ("rear", self.server_node.node_var)):
                    params = list(var.model.parameters())
                    row[side + "_parameter_bytes"] = sum(map(tensor_bytes, params))
                    sample = partial(_sample_optimizer, row, side, params)
                    sample(var.optimizer)
                    if var.optimizer is not None:
                        hooks.enter_context(var.optimizer.register_step_post_hook(sample))
            yield

    @contextmanager
    def _profile_step(self, row, phase, device):
        if not self.profile:
            yield
            return
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            initial = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        yield
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            key = phase + "_process_peak_bytes"
            row[key] = max(row.get(key, 0), peak)
            delta_key = phase + "_peak_delta_bytes"
            row[delta_key] = max(row.get(delta_key, 0), peak - initial)
        key = phase + "_seconds"
        row[key] = row.get(key, 0.0) + perf_counter() - start

    def run_visit(self, round_id, node):
        client_id, client = node.node_id, node.strategy
        started = perf_counter()
        rear = self.server.rear_state
        rear.activate(client_id)
        client.begin_visit()
        row = {"client_id": client_id, "loss_sum": 0.0, "correct": 0, "n_examples": 0, "steps": 0}
        with self._profile_models(row, node):
            self._train_batches(round_id, node, row)
        row["front_optimizer_steps"] = row["steps"] if client.needs_cut_grad else 0
        row["rear_optimizer_steps"] = row["steps"] if rear.optimizer is not None else 0
        update = client.end_visit({"n_examples": row["n_examples"], "steps": row["steps"]})
        if self.server.front_sync.method != "local":
            self.traffic[client_id]["adapter_up"] += tensor_bytes(update["updated_weights"])
        rear.finish_visit(client_id, update["train_record"]["data_sample_num"])
        row["visit_seconds"] = perf_counter() - started - row.get("recovery_seconds", 0)
        self.train_rows.append(row)
        return update

    def _train_batches(self, round_id, node, row):
        client_id, client = node.node_id, node.strategy
        view = self.views[client_id]
        rear_device = self.server_node.node_var.device
        counts = self.traffic[client_id]
        for epoch in range(self.protocol["local_epochs"]):
            for inputs, labels in view.train_batches(round_id, epoch):
                with self._profile_step(row, "front_forward", node.node_var.device):
                    activation = client.forward(inputs)
                counts["activation_up"] += tensor_bytes(activation)
                counts["label_up"] += tensor_bytes(labels)
                with self._profile_step(row, "rear_step", rear_device):
                    cut_grad, loss, stats = self.server.step(activation, labels, need_cut_grad=client.needs_cut_grad)
                if cut_grad is not None:
                    counts["cut_grad_down"] += tensor_bytes(cut_grad)
                    with self._profile_step(row, "front_backward", node.node_var.device):
                        client.backward(cut_grad)
                row["loss_sum"] += loss * stats["n_examples"]
                row["correct"] += stats["correct"]
                row["n_examples"] += stats["n_examples"]
                row["steps"] += 1
                if self.on_batch_end is not None:
                    probe_start = perf_counter()
                    visit_bytes = counts["activation_up"] + counts["label_up"] + counts["cut_grad_down"]
                    self.on_batch_end(round_id, client_id, row["steps"], visit_bytes)
                    row["recovery_seconds"] = row.get("recovery_seconds", 0) + perf_counter() - probe_start

    def run_round(self, round_id, participants):
        start = perf_counter()
        self._reset_round()
        self.server.rear_state.begin_round()
        updates = self.simulate_client_local_training_process(round_id, participants)
        visits_end = perf_counter()
        old_version = self.server.front_sync.version
        version = self.simulate_server_update_process(updates)
        aggregate_end = perf_counter()
        if version != old_version:
            self.simulate_server_broadcast_process()
        end = perf_counter()
        probe_seconds = sum(row.get("recovery_seconds", 0) for row in self.train_rows)
        self.round_times = {
            "train_visits_seconds": visits_end - start - probe_seconds,
            "aggregate_seconds": aggregate_end - visits_end,
            "broadcast_seconds": end - aggregate_end,
            "training_seconds": end - start - probe_seconds,
        }
        if self.on_batch_end is not None:
            self.round_times["recovery_seconds"] = probe_seconds
        self.completed_round = round_id
        client_ids = [node.node_id for node in participants]
        self.last_visit_round.update(dict.fromkeys(client_ids, round_id))
        self.on_round_end(round_id, client_ids, updates)

    def run(self):
        if self.protocol is None:
            self._configure_from_yaml()
        self.initialize()
        for round_id in range(self.completed_round + 1, self.protocol["rounds"] + 1):
            self.run_round(round_id, self.participants_for(round_id))
