# FLEX: Flexible Distributed ML Simulation

FLEX is a modular framework for simulating federated and split learning workflows. It separates concerns into nodes, strategies, algorithms, and configuration so you can prototype FL/SFL variants quickly without rewiring code.

## Installation

The installer requires Anaconda or Miniconda. It creates a Conda environment named `flex_env`, selects a CPU/MPS or CUDA PyTorch build, installs the project dependencies, and installs FLEX in editable mode.

```bash
python3 requirements/install.py
conda activate flex_env
```

Use `--dev` for test dependencies, or explicitly select `--cpu`, `--cuda121`, or `--cuda124`. The Bash compatibility entry point is `requirements/install.sh`.

Legacy loaders that import `torchtext` are not part of the default environment because the final `torchtext` release is tied to PyTorch 2.3. Current CV and Hugging Face-backed NLP workflows use the maintained PyTorch profiles in `requirements/`.

## Architecture at a Glance
- **Nodes**: `FedNodeServer` orchestrates, `FedNodeClient` trains locally. Each node carries `node_var` (model weights, optimizer builder, trainer, logger, config, client selection, aggregation method, etc.). Strategies operate on nodes.
- **Strategies** (under `src/flex/fed_strategy` and `src/flex/sfl_strategy`):
	- Runner strategies (e.g., FedAvg, Oort, FedGRA) coordinate rounds: broadcast → optional observation → select → local train → aggregate → eval.
	- Server strategies wrap selection + aggregation + evaluation, and forward feedback to selectors.
	- Client strategies wrap observation/local training and weight receive/set.
- **Factories**:
	- `StrategyFactory`: creates runner/server/client strategies based on YAML `strategy_name` and role; also builds strategy args.
	- `FedClientSelectorFactory`: builds client selectors (Oort, FedGRA, AdaFL, PyramidFL, etc.).
	- `FedAggregatorFactory`: builds aggregation methods (FedAvg, etc.).
	- Model/Trainer/Data factories live under `src/flex/model_trainer` and `src/flex/ml_*`.
- **Algorithms**: Client selection methods in `src/flex/fl_algorithms/selection/methods`, aggregation in `src/flex/fl_algorithms/aggregation`.
- **Events**: Nodes can emit events (e.g., `on_prepare_strategy` in client) so external hooks can react when strategies are prepared/weights updated; strategies typically call `declare_events`/`raise_event` on node during lifecycle.

## YAML-Driven Experiments
Experiments are assembled by composing YAML snippets. Test bundles in `src/test/fl_client_selection_test/*.yaml` point to reusable YAMLs under `src/yamls/`.

### YAML folders (src/yamls)
- `aggregation/`: how server aggregates updates (e.g., `aggregation_fedavg.yaml`).
- `client_selection/`: client picker configs (e.g., `fedgra.yaml`, `oort.yaml`, `adafl.yaml`, `pyramidfl.yaml`).
- `client_strategy/`, `server_strategy/`, `runner_strategy/`: pick the strategy class by `strategy_name`.
- `dataset_loader/`, `data_distribution/`, `nn_model/`, `optimizer/`, `training/`, `trainer/`, `training_logger/`: data, model, optimizer, training loop, logging.
- `general/`: global knobs like `training_rounds`.
- `role/`: role defaults for server/client.

### How a test YAML is structured
Example: `src/test/fl_client_selection_test/fedgra_test.yaml`
- Declares folder prefixes (e.g., `yaml_folder_client_selection_path`), then maps filename → alias (e.g., `fedgra.yaml: client_selection_fedgra`).
- `yaml_combination` lists which aliases to merge for runner, client, server. Order matters; later configs can extend/override earlier ones.

### Strategy–Node–Var Collaboration
- `node_var` is the live state a strategy mutates (variables), and `strategies` define behaviors that read/write it: model weights (`model_weight`), cached weight, optimizer builder, trainer, logger, selector (`client_selection`), aggregator (`aggregation_method`), config dictionary, etc.
- Runner strategy drives the order of operations (broadcast/observe/select/train/aggregate/eval) and calls server/client strategy entry points.
- Server strategy reads/writes `node_var.client_updates`, `node_var.aggregated_weight`, and pushes feedback into selectors (via `with_clients_data`). It also updates the evaluator and logger.
- Client strategy consumes `node_var` to build models/optimizers, runs observation/local training, and sets local/global weights.
- Events: nodes can `declare_events` and `raise_event` (e.g., client `on_prepare_strategy`) so external hooks or instrumentation can plug in without modifying strategy code.

### Switching strategies via YAML
- **Client selection**: change `yaml_folder_client_selection_files` entry and the corresponding alias in `yaml_combination.server_yaml` (e.g., swap `client_selection_fedgra` for `client_selection_adafl`).
- **Server strategy**: point to the server strategy YAML (e.g., `fedgra_server_strategy`, `adafl_server_strategy`).
- **Runner strategy**: choose orchestrator in `yaml_combination.runner` (e.g., `fedgra_runner_strategy`, `oort_runner_strategy`).
- **Client strategy**: set `fedavg_client_strategy` or others in `yaml_combination.client_yaml`.

### Example: Minimal FedGRA run bundle
File: `src/test/fl_client_selection_test/fedgra_test.yaml`
```yaml
yaml_folder_client_selection_path: ../../yamls/client_selection/
yaml_folder_server_strategy_path: ../../yamls/server_strategy/
yaml_folder_runner_strategy_path: ../../yamls/runner_strategy/
yaml_folder_client_selection_files:
- fedgra.yaml: client_selection_fedgra
yaml_folder_server_strategy_files:
- fedgra.yaml: fedgra_server_strategy
yaml_folder_runner_strategy_files:
- fedgra.yaml: fedgra_runner_strategy
yaml_combination:
	runner:
	- fedgra_runner_strategy
	client_yaml:
	- fedavg_client_strategy
	server_yaml:
	- client_selection_fedgra
	- fedgra_server_strategy
	- aggregation_fedavg
```
This wires FedGRA selector + FedGRA server + FedGRA runner with FedAvg client training and FedAvg aggregation. Swap any alias (e.g., `client_selection_fedgra` → `client_selection_adafl`, `fedgra_runner_strategy` → `oort_runner_strategy`) to change behavior without code changes.

### Events and hooks in practice
- Client: during `prepare_strategy`, clients can raise `on_prepare_strategy` so listeners can inspect/modify configs before training starts.
- Server: after receiving updates, server strategies push metrics to selectors via `with_clients_data`, which is the hook point for selection algorithms to consume losses/latency/etc.
- Runner: controls the timing of these hooks (e.g., some runners first do observation to populate selector state, then selection, then training).

### Define your own algorithm (example: new client selector)
1) Implement selector in `src/flex/fl_algorithms/selection/methods/_fed_client_selector_myalgo.py` inheriting `FedClientSelector`. Fill `select`, optionally `with_random_seed`, and any helpers.
2) Register it in `src/flex/fl_algorithms/selection/fed_client_selector_factory.py` under a new `case "myalgo":` returning your selector.
3) Add YAML `src/yamls/client_selection/myalgo.yaml`:
	 ```yaml
	 client_selection:
		 method: myalgo
		 number: 5
		 random_seed: 42
	 ```
4) Reference it in a test bundle, e.g., `src/test/fl_client_selection_test/myalgo_test.yaml`:
	 ```yaml
	 yaml_folder_client_selection_files:
	 - myalgo.yaml: client_selection_myalgo
	 yaml_folder_server_strategy_files:
	 - fedavg.yaml: fedavg_server_strategy
	 yaml_folder_runner_strategy_files:
	 - fedavg.yaml: fedavg_runner_strategy
	 yaml_combination:
		 runner:
		 - fedavg_runner_strategy
		 client_yaml:
		 - fedavg_client_strategy
		 server_yaml:
		 - client_selection_myalgo
		 - fedavg_server_strategy
		 - aggregation_fedavg
	 ```
5) Run using that bundle; the new selector will receive client feedback via `with_clients_data` when server strategies forward `train_record`/`latency`.

### Running a preset
1) Pick a test bundle, e.g., `src/test/fl_client_selection_test/fedgra_test.yaml`.
2) Ensure referenced YAML files exist in `src/yamls/` (all included in repo for provided tests).
3) Run the project’s entry point that loads the bundle (see `src/test/run_all.py` or specific test scripts) with the path to the YAML.

## Extending
- Add a new selector: implement under `src/flex/fl_algorithms/selection/methods`, register in `fed_client_selector_factory.py`, add a YAML in `client_selection/`, reference it in a test bundle.
- Add a new runner/server/client strategy: implement under `fed_strategy/...`, register in `strategy_factory.py`, add YAMLs in `runner_strategy/` or `server_strategy/`, then wire a test bundle.

## Key Paths
- Strategies: `src/flex/fed_strategy/` (FL) and `src/flex/sfl_strategy/` (SFL)
- Selection methods: `src/flex/fl_algorithms/selection/methods/`
- Aggregation: `src/flex/fl_algorithms/aggregation/`
- Tests (YAML bundles): `src/test/fl_client_selection_test/`
