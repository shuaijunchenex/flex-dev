import time

from ...ml_utils.tqdm_utils import pbar

from flex.fed_strategy.strategy_args import StrategyArgs
from ...ml_utils import console
from ...ml_utils.training_utils import TrainingUtils
from ...fed_runner import FedRunner
from ...fed_strategy.runner_strategy import RunnerStrategy
from ...fed_node import FedNodeClient, FedNodeServer


class ObserverRunnerStrategy(RunnerStrategy):
    """
    Runner strategy that performs a lightweight observation pass over ALL
    available clients before the selector runs each round.

    Observation flow per round
    --------------------------
    1. Broadcast current global weights to all clients.
    2. For every client call ``client.strategy.observation_step()``
       (no weight write-back, pure metric collection).
    3. Feed the collected observation records to the server via
       ``server_node.receive_client_updates()`` so that the selector's
       ``with_clients_data()`` is populated before ``select_clients()`` is called.
    4. Select participants, run real local training, aggregate as usual.
    """

    def __init__(self, runner: FedRunner, args, client_node, server_node) -> None:
        super().__init__(runner)
        self._strategy_type = "observer"
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

    # ------------------------------------------------------------------
    # Observation pass: runs on ALL clients, no weight write-back
    # ------------------------------------------------------------------
    def simulate_observe_all_clients(self, all_clients: list[FedNodeClient]) -> list[dict]:
        """
        Call observation_step() on every client and collect the records.
        Returns a list of observation dicts keyed the same way as train_records.
        """
        observation_updates = []
        for client in all_clients:
            try:
                start_time = time.time()
                console.info(f"[{client.node_id}] Observation started")
                obs_record = client.strategy.observation_step()
                latency = time.time() - start_time
                observation_updates.append({
                    "node_id": client.node_id,  # include node_id so selector can identify the client
                    "updated_weights": None,   # observation does not write back weights
                    "train_record": obs_record,
                    "latency": latency,
                })
            except Exception as exc:
                console.warn(f"[{client.node_id}] observation_step failed: {exc}")
        return observation_updates

    # ------------------------------------------------------------------
    # Training pass: runs only on selected participants
    # ------------------------------------------------------------------
    def simulate_client_local_training_process(self, participants: list[FedNodeClient]):
        for client in participants:
            start_time = time.time()
            console.info(f"\n[{client.node_id}] Local training started")
            updated_weights, train_record = client.run_local_training()
            latency = time.time() - start_time
            yield {
                "updated_weights": updated_weights,
                "train_record": train_record,
                "latency": latency,
            }

    def simulate_server_broadcast_process(self) -> None:
        self.server_node.broadcast(self.client_nodes)
        return

    def simulate_server_update_process(self, weight) -> None:
        self.server_node.strategy.server_update(weight)
        return

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        print("Running [Observer] strategy...")
        header_data = TrainingUtils.build_training_header(self.server_node)
        self.server_node.prepare(header_data, self.client_nodes)
        self.server_node.broadcast()
        self.participants = []

        for round in pbar(range(self.args.key_value_dict.data['training_rounds'] + 1)):

            console.out(
                f"\n{'='*10} Training round {round}/"
                f"{self.args.key_value_dict.data['training_rounds']}, "
                f"Total participants: {len(self.client_nodes)} {'='*10}"
            )

            selector = getattr(self.server_node.node_var, "client_selection", None)
            selection_interval = getattr(selector, "client_selection_round", 1)

            if round % selection_interval == 0:
                # ── Step 1: Observe ALL clients before selection ──────────
                console.info(f"Round {round}: observing all {len(self.client_nodes)} clients...")
                observation_updates = self.simulate_observe_all_clients(self.client_nodes)

                # Feed observation data to the server/selector so that
                # client metrics are available when select_clients() runs.
                self.server_node.receive_client_updates_for_selection(observation_updates)

                # ── Step 2: Select participants based on fresh metrics ────
                self.participants = self.server_node.select_clients(self.client_nodes)
                console.info(
                    f"Round: {round}, Select {len(self.participants)} clients: "
                ).ok(f"{', '.join(map(str, self.participants))}")

            # ── Step 3: Real local training on selected participants ──────
            client_updates = list(self.simulate_client_local_training_process(self.participants))

            self.server_node.receive_client_updates(client_updates)

            self.server_node.aggregation()

            self.server_node.apply_weight()

            self.server_node.broadcast()

            self.server_node.evaluate()

            self.server_node.record_evaluation()

            # ── Save aggregated global weights for post-hoc analysis ──────
            _node_vars = self.server_node.node_var
            _logger = getattr(_node_vars, "training_logger", None)
            _weight = (
                getattr(_node_vars, "aggregated_weight", None)
                or getattr(_node_vars, "model_weight", None)
            )
            if _logger is not None and _weight is not None:
                _saved = _logger.save_weights_if_enabled(_weight, round)
                if _saved:
                    console.debug(f"[Observer] Round {round} weights saved → {_saved}")

            console.out(
                f"{'='*10} Round {round}/{self.args.key_value_dict.data['training_rounds']} End{'='*10}"
            )

        return
