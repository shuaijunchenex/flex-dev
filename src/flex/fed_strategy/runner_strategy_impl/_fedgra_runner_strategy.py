import time

from ...ml_utils.tqdm_utils import pbar

from flex.fed_strategy.strategy_args import StrategyArgs
from ...ml_utils import console
from ...ml_utils.training_utils import TrainingUtils
from ...fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from ...fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from ...fed_runner import FedRunner
from ...fed_strategy.runner_strategy import RunnerStrategy 
from ...fed_node import FedNodeClient, FedNodeServer

class FedgraRunnerStrategy(RunnerStrategy):

    def __init__(self, runner: FedRunner, args, client_node, server_node) -> None:
        super().__init__(runner) #TODO: modify runner object declaration
        self._strategy_type = "fedgra"
        self.args = args
        self.client_nodes : list[FedNodeClient]= client_node
        self.server_node : FedNodeServer = server_node
        # Observation round uses fewer epochs (lightweight probe for selector metrics)
        self.observation_epochs = int(args.key_value_dict.data.get("observation_epochs", 1))
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
    
    def simulate_client_local_training_process(self, participants, label: str = "local", observe: bool = False):
        """Train participants sequentially.

        When *observe* is True, each client trains for ``observation_epochs``
        epochs (config ``training.epochs`` is temporarily overridden) so the
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
                training["epochs"] = self.observation_epochs
        try:
            for client in participants:
                start_time = time.time()
                console.info(f"\n[{client.node_id}] {label} training started")
                updated_weights, train_record = client.run_local_training()
                training_duration = time.time() - start_time
                yield {
                    "updated_weights": updated_weights,
                    "train_record": train_record,
                    "latency": training_duration
                }
        finally:
            if observe:
                for client, old_epochs in saved_epochs.items():
                    cfg = client.node_var.config_dict
                    if old_epochs is not None:
                        cfg["training"]["epochs"] = old_epochs
                    else:
                        cfg["training"].pop("epochs", None)

    def collect_observation_feedback(self):
        """Run a lightweight observation on all clients and feed selector."""
        selector = getattr(self.server_node.node_var, "client_selection", None)
        if selector is None or not hasattr(selector, "with_clients_data"):
            return

        feedback = {}
        for client in self.client_nodes:
            console.info(f"\n[{client.node_id}] Observation started")
            obs = client.strategy.run_observation()
            node_id = str(obs.get("node_id", client.node_id))
            train_record = obs.get("train_record", {}) or {}
            feedback[node_id] = {
                **train_record,
                "node_id": node_id,
                "data_sample_num": obs.get("data_sample_num", train_record.get("data_sample_num")),
                "latency": obs.get("latency", train_record.get("latency", 1.0)),
            }

        selector.with_clients_data(feedback)

    def simulate_server_broadcast_process(self):
        self.server_node.broadcast(self.client_nodes)
        return
    
    def simulate_server_update_process(self, weight):
        self.server_node.strategy.server_update(weight)
        return

    def run(self) -> None:
        print("Running [FedGRA] strategy...")
        header_data = TrainingUtils.build_training_header(self.server_node)
        self.server_node.prepare(header_data, self.client_nodes)
        self.server_node.broadcast()  # ensure all clients have initial weights
        self.participants = []
        for round in pbar(range(self.args.key_value_dict.data['training_rounds'] + 1)):
           
            console.out(f"\n{'='*10} Training round {round}/{self.args.key_value_dict.data['training_rounds']}, Total participants: {len(self.client_nodes)} {'='*10}")

            selector = getattr(self.server_node.node_var, "client_selection", None)
            selection_interval = getattr(selector, "client_selection_round", 1)
            if round % selection_interval == 0:
                # Lightweight observation on ALL clients → fresh metrics for EWM+GRA selection
                all_updates = list(self.simulate_client_local_training_process(
                    self.client_nodes, label="selection", observe=True
                ))
                self.server_node.receive_client_updates_for_selection(all_updates)
                self.participants = self.server_node.select_clients(self.client_nodes)
                console.info(f"Round: {round}, Select {len(self.participants)} clients: ', '").ok(f"{', '.join(map(str, self.participants))}")

            # Real training on selected clients → weights for aggregation
            client_updates = list(self.simulate_client_local_training_process(
                self.participants, label="aggregation"
            ))

            self.server_node.receive_client_updates(client_updates)

            self.server_node.aggregation()

            self.server_node.apply_weight()

            self.server_node.broadcast()

            self.server_node.evaluate()

            self.server_node.record_evaluation()

            # ── Save aggregated global weights for post-hoc norm analysis ──
            _node_vars = self.server_node.node_var
            _logger = getattr(_node_vars, "training_logger", None)
            _weight = getattr(_node_vars, "aggregated_weight", None) \
                   or getattr(_node_vars, "model_weight", None)
            if _logger is not None and _weight is not None:
                _saved = _logger.save_weights_if_enabled(_weight, round)
                if _saved:
                    console.debug(f"[FedGRA] Round {round} weights saved → {_saved}")

            console.out(f"{'='*10} Round {round}/{self.args.key_value_dict.data['training_rounds']} End{'='*10}")

        return
       
