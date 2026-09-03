# FLEX Technical Design and Thesis-Suitability Report

**Review baseline:** Git commit `3354430` plus the uncommitted working-tree state inspected on 20 July 2026. This qualification matters: the workspace contains modified RBLA files and untracked LoRA-canonicalisation code and experiments. All statements below describe this local repository, not necessarily the public GitHub default branch.

## 1. Executive assessment

FLEX is an experimental, PyTorch-based federated- and split-learning **simulation and research-prototyping framework**. Its strongest implemented idea is the decomposition of an experiment into node-held state, runner/server/client strategies, client selectors, aggregators, trainers, model/data factories, and composable YAML fragments. The standard execution is a single Python process in which clients train sequentially and the runner calls node objects directly (`src/flex/fed_strategy/runner_strategy_impl/_fedavg_runner_strategy.py:39-88`). It is not a deployment framework or an implemented FL-as-a-Service control plane: no socket/RPC transport, remote worker lifecycle, authentication, tenancy, job scheduler, durable service API, or production failure recovery was found.

The code genuinely implements local training, synchronous orchestration, partial participation, sample-weighted FedAvg, several research selectors, LoRA/heterogeneous-rank aggregation variants, evaluation, logging, YAML composition, and non-IID partition generation. A queue/thread-based simulated switch exists (`src/flex/ml_simu_switcher/simu_switcher.py:10-151`), but the main FedAvg/FedGRA round paths do not send weights through it; they call `client.receive_weight(...)` and `client.strategy.run_local_training()` directly. `FedNodeEdge` and edge-node creation are stubs (`src/flex/fed_node/fed_node_edge.py:26-28`; `src/flex/fed_runner/fed_runner.py:105-109`). Secure aggregation was not found in the current repository.

### Capability classification

| Area | Status | Code evidence and qualification |
|---|---|---|
| Client training | Implemented | `FedAvgClientTrainingStrategy.local_training_step`, `src/flex/fed_strategy/client_strategy_impl/_fedavg_client.py:101-151`; delegates to configured `ModelTrainer.train`. |
| Server orchestration | Implemented, synchronous | `FedAvgRunnerStrategy.run`, `.../_fedavg_runner_strategy.py:60-103`. Orchestration resides mainly in runner strategy, not server thread/process. |
| Aggregation | Implemented | `AbstractFedAggregator`, `src/flex/fl_algorithms/aggregation/fed_aggregator_abc.py:11-100`; implementations under `aggregation/methods`. |
| Client selection | Implemented | `FedClientSelector.select`, `src/flex/fl_algorithms/selection/fed_client_selector_abc.py:11-61`; factory registrations at `fed_client_selector_factory.py:20-68`. |
| Communication | Architectural prototype only | Queue/thread/event transport in `ml_simu_switcher`; standard FL call chain bypasses it. Payloads are live Python objects, not serialised messages. |
| Broadcast/model distribution | Implemented in-process | `FedAvgServerStrategy.broadcast`, `src/flex/fed_strategy/server_strategy_impl/_fedavg_server.py:78-82`, loops over client objects. |
| Evaluation | Implemented | `FedAvgServerStrategy.evaluate`, same file:87-114; `ModelEvaluator` in `src/flex/model_trainer/model_evaluator.py`. |
| Logging | Implemented | CSV/result/weight/selection recording in `src/flex/ml_utils/training_logger.py:11-206`. |
| Configuration | Implemented, weakly validated | YAML loading/composition in `src/flex/ml_utils/app_entry.py:48-60,106-242`; shallow last-writer-wins merge at lines 216-240. |
| Data heterogeneity | Implemented/emulated | Non-IID allocation in `StandardSampleEntry.run`, `src/entries/app/standard_entry.py:62-92`; distributions in `src/yamls/data_distribution`. |
| System heterogeneity | Partly emulated by algorithms | Latency is locally measured in FedGRA (`.../_fedgra_runner_strategy.py:40-75`) and ranks/configs can differ; no real devices/network/resource manager. |
| Hierarchical FL | Planned/incomplete | Edge type exists, but `FedRunner.__create_edge_nodes` is TODO and `FedNodeEdge.run` is `pass`. |
| LoRA/PEFT | Implemented, research-specific | LoRA models/utilities and RBLA/SP/FFA-LoRA/Flora/FedSA-LoRA/RoLoRA/FLoRG aggregators are registered in `fed_aggregator_facotry.py:32-113`. |
| Secure aggregation/privacy | Not found in the current repository | No protocol, cryptographic primitive, DP mechanism, or threat model found by repository-wide search. |
| Personalisation | Not found as a coherent core facility | Client-local weights exist, but standard broadcast overwrites all selected and unselected client model weights; no explicit personalised-model API was found. |
| FLaaS | Unsupported as a runtime claim | Configuration and reusable experiments are FLaaS-adjacent abstractions; service isolation, APIs, persistence, multi-tenancy and deployment are absent. |

**Thesis verdict.** FLEX already supports a substantial implementation section and can support a defensible standalone thesis chapter if framed as an internally developed research execution substrate and backed by quantitative extension-effort, reproducibility, overhead, and scalability evidence. The current implementation alone is not enough for a publishable framework paper: API stability, tests/CI, comparative evaluation, release engineering, and production-grade runtime boundaries are insufficient.

## 2. Repository architecture

| Layer | Important paths | Responsibility and interactions |
|---|---|---|
| Application/entry | `src/entries/standard.py`; `src/entries/app/standard_entry.py`; `src/entries/app/lora_entry.py`; `src/entries/app/sfl_entry.py` | Loads experiment definitions, builds node state, partitions data, attaches hooks, then invokes `FedRunner`. Special experiment scripts also exist under `src/entries`. |
| Configuration | `src/flex/ml_utils/app_entry.py`; `config_loader.py`; `src/yamls/**` | Loads YAML fragments by alias and shallow-merges combinations. Stores combined objects in static `AppEntry.__app_objects`. |
| Runtime/topology | `src/flex/fed_runner/fed_runner.py`; `src/flex/fed_strategy/runner_strategy_impl/**` | Creates server/client objects and selects a runner strategy. The concrete runner owns the actual round loop. |
| Nodes/state | `src/flex/fed_node/fed_node.py`; `fed_node_client.py`; `fed_node_server.py`; `fed_node_vars.py` | Nodes are façades; `FedNodeVars` is a mutable service/state container holding model, loaders, builders, selector, aggregator, evaluator, logs and weights. |
| Strategies | `src/flex/fed_strategy/{base,runner,server,client}_strategy.py`; `*_strategy_impl/**`; `strategy_factory.py` | Encapsulates orchestration and algorithm-specific client/server behaviour. Factory chooses concrete classes from YAML string names. |
| Aggregation | `src/flex/fl_algorithms/aggregation/**` | Converts update envelopes into `(state_dict, sample_count)` pairs and aggregates them. Standard and LoRA-specific implementations share one ABC. |
| Selection | `src/flex/fl_algorithms/selection/**` | Selects client-node objects, optionally using metrics supplied by server strategy. Implementations include all/random/loss/Oort/PyramidFL/AFL/AdaFL/Pow-d/FedGRA/RepuFL/FedSDR. |
| Training/evaluation | `src/flex/model_trainer/**`; `src/flex/ml_algorithms/{optimizer_builder,loss_function_builder}.py`; `src/flex/model_trainer/model_evaluator.py` | Trainer factory binds models, optimisers, loaders and loss. Strategies call `trainer.train`; evaluator loads aggregated state and computes metrics. |
| Models/weights | `src/flex/ml_models/**`; `src/flex/ml_utils/model_utils.py`; `src/flex/ml_algorithms/lora/**` | Model factory and PyTorch `state_dict` operations; LoRA replacement, rank handling, SVD/canonicalisation helpers. No transport-neutral weight DTO exists. |
| Simulated communication | `src/flex/ml_simu_switcher/**` | `SimuSwitcher` routes live Python objects through `queue.Queue` on one thread and raises receive events. It is connected during node creation but is not used by the standard training call chain. |
| Logging/results | `src/flex/ml_utils/training_logger.py`; `csv_data_recorder.py`; `batch_summary.py`; `src/test/experiment_results/**` | Writes CSV metrics, selected-client/EWM logs and optional state dictionaries; batch launcher records subprocess outcomes. |
| Tests/experiments | `src/unittest_ml/**`; `src/unit_test/**`; `src/test/**` | Mixture of true `unittest`/pytest tests, script-style checks, experiment definitions and generated results. Separation is inconsistent. |

The repository contains 298 Python files and approximately 36,644 physical lines under `src/flex` (30,358 nonblank), plus 1,269 YAML files in the whole workspace. These counts show that FLEX is no longer a tiny library even though its runtime is lightweight.

## 3. End-to-end execution flow

### Standard FedAvg path

1. `src/entries/standard.py:18-22` constructs `StandardSampleEntry`, calls `load_app_config(config_path)`, chooses an accelerator, and calls `run(device)`.
2. `AppEntry.load_app_config` (`src/flex/ml_utils/app_entry.py:48-60`) loads the definition YAML. `__parse_app_config_define` (106-242) loads aliased fragments and shallow-merges each `yaml_combination` in order.
3. `StandardSampleEntry.run` obtains `runner`, `client_yaml`, `edge_yaml`, and `server_yaml` from the static app-object map (`src/entries/app/standard_entry.py:28-38`).
4. It creates `FedRunner`, copies the configured round count, calls `with_yaml`, `create_nodes`, and `create_run_strategy` (45-50). `FedRunner.create_nodes` creates one `FedNodeServer` and N `FedNodeClient`s from groups (`src/flex/fed_runner/fed_runner.py:45-101`). Edge creation is a no-op.
5. `StrategyFactory.create_runner_strategy` selects `FedAvgRunnerStrategy` from `strategy_name` (`src/flex/fed_strategy/strategy_factory.py:37-45`). Its constructor directly connects server/client object references (`.../_fedavg_runner_strategy.py:17-37`).
6. The entry creates a server `FedNodeVars`, calls `prepare`, binds it bidirectionally to the node, then creates the server strategy (`standard_entry.py:52-60`). `FedNodeVars.prepare` builds configured data/model/optimiser/loss/selector/trainer/aggregator/evaluator/logger services (`src/flex/fed_node/fed_node_vars.py:580-689`).
7. Server data is partitioned by `NoniidDataGenerator`; each client receives a custom loader and separately prepared `FedNodeVars` and strategy (`standard_entry.py:62-99`).
8. `FedRunner.run` delegates to the concrete runner (`src/flex/fed_runner/fed_runner.py:120-124`). `FedAvgRunnerStrategy.run` starts logging, broadcasts the initial server `model_weight`, and iterates `range(training_rounds + 1)` (`.../_fedavg_runner_strategy.py:60-68`). This is an off-by-one semantic: a configured N produces N+1 aggregation rounds.
9. At the selector interval, `server_node.select_clients` delegates through `FedAvgServerStrategy.select_clients` to the configured selector (`fed_node_server.py:98-101`; `_fedavg_server.py:32-35`).
10. For every participant, the runner directly calls `client.strategy.run_local_training()` sequentially (`_fedavg_runner_strategy.py:39-50`). The client deep-copies its model, loads `node_vars.model_weight`, rebuilds an optimiser, calls `trainer.train(epochs)`, optionally offloads state to CPU, and stores the returned state dict (`_fedavg_client.py:101-151`).
11. The runner wraps each return as `{updated_weights, train_record}` (47-50), then sends the list by direct method call to `server_node.receive_client_updates` (76-78). No serialisation or simulated-network send occurs.
12. `FedAvgServerStrategy.aggregation` invokes `node_var.aggregation_method.aggregate(client_updates)` and stores `aggregated_weight` (`_fedavg_server.py:26-30`). The ABC extracts each update's `train_record.data_sample_num` (`fed_aggregator_abc.py:68-76`).
13. `ServerStrategy.apply_weight` moves `aggregated_weight` to `model_weight`, optionally converts SP-form weights, and updates the evaluator model (`server_strategy.py:86-108`).
14. Broadcast loops over **all** client objects, calls `receive_weight(model_weight)`, then `set_local_weight()` (`_fedavg_server.py:78-82`). Client receive uses `cache_weight`; set copies it to `model_weight` (`_fedavg_client.py:153-159`).
15. Server evaluation calls `ModelEvaluator.evaluate`, prints results, and logger records them (`_fedavg_server.py:87-114,37-39`). `FedNodeServer.record_evaluation` additionally records selected clients and FedGRA EWM weights (`fed_node_server.py:76-95`).
16. The next loop iteration starts the next round. Termination is only exhaustion of the range; no convergence, budget, failure or early-stopping criterion is in this runner.

Control inversion occurs at three points: string-to-class factories choose implementations; runner strategy owns lifecycle sequencing; entry-attached Observer-style events can mutate preparation. However, the round sequence is not a reusable template method: concrete runners duplicate substantial loops.

## 4. Core abstractions

| Boundary | Interface/contract | Caller, state, and real replaceability |
|---|---|---|
| Aggregator | `AbstractFedAggregator.aggregate` plus `_before/_do/_after_aggregation`, `fed_aggregator_abc.py:68-100` | Called by server strategy; owns args and result. A new algorithm needs a subclass **and a factory case** in misspelled `fed_aggregator_facotry.py`; it must accept the fixed update schema. Thus extension is modular but not plugin-open. |
| Selector | `FedClientSelector.select`, `fed_client_selector_abc.py:48-61` | Called by server strategy with live client objects; state can be injected with `with_clients_data`. New selector also requires factory modification. Global `random.seed` is mutated at construction (38-46). |
| Runner strategy | `RunnerStrategy.run` and three simulation methods, `runner_strategy.py:8-50` | Owns orchestration and direct node references. Replacing it can change entire round semantics without `FedRunner` modification, but needs factory registration and commonly duplicates the full loop. |
| Server strategy | Abstract selection, aggregation, broadcast, evaluation, receipt and prepare methods, `server_strategy.py:9-84` | Holds server node in `_obj` and mutates `node_var`. Encodes both algorithm adapter logic and infrastructure concerns. `apply_weight` is useful shared infrastructure (86-166). |
| Client strategy | Abstract observation/local-training steps, `client_strategy.py:14-43` | Holds client node; shared GPU offload/cleanup helpers at 45-194. Concrete strategy controls model copying, training and update schema. |
| Local trainer | `ModelTrainer.train`/`training_step`, `src/flex/model_trainer/model_trainer.py:10-39` | Constructed by `ModelTrainerFactory`; client strategy binds model/optimizer/loader. Trainer implementations vary by CNN/NLP/SFL/LoRA. |
| Evaluation | Concrete `ModelEvaluator`, `src/flex/model_trainer/model_evaluator.py` | Not defined behind an ABC; stored in `FedNodeVars`, updated by server strategy. Replacement is possible through preparation hooks/manual injection, but no evaluator factory/interface comparable to aggregators. |
| Model/weight | PyTorch `nn.Module` and untyped state-dict dictionaries | `NNModelFactory`, `ModelUtils`, client/server strategies. There is no explicit weight interface; algorithm-specific key suffixes and tensor shapes leak into strategies and aggregators. |
| Communication | `SimuNode.send_up/send_down/send_to`, `src/flex/ml_simu_switcher/simu_node.py:98-134` | Queue-backed switcher accepts `Any`. Main algorithms depend on concrete node references instead, so replacing communication alone would not produce distributed execution. |

`FedNodeVars` (`src/flex/fed_node/fed_node_vars.py:21-48,77-110`) is the central mutable container and de facto dependency-injection mechanism. This reduces constructor plumbing but weakens contracts: attributes such as `client_updates`, `aggregated_weight`, `cache_weight`, `data_sample_num`, builders and evaluator are accessed dynamically, with limited static validation.

## 5. Extension mechanisms

### Stable infrastructure versus research logic

The stable portion is: config loading, entry construction, node/state preparation, factories, trainer/evaluator, logging, and the broad select-train-aggregate-evaluate cycle. Switching a plain FedAvg aggregator to another shape-compatible aggregator leaves `StandardSampleEntry`, `FedRunner`, nodes, trainers and evaluator unchanged; YAML and the factory-selected aggregator change.

The separation is incomplete:

- Factories use closed `match` statements (`strategy_factory.py:41-275`; `fed_aggregator_facotry.py:20-115`; `fed_client_selector_factory.py:24-68`). Every new named extension modifies framework code.
- FedGRA adds a selector, server strategy and runner strategy because observation/feedback changes lifecycle (`_fedgra_runner_strategy.py:40-145`). Its runner duplicates the baseline aggregation/evaluation loop.
- LoRA formats leak into shared `ServerStrategy._prepare_weight_for_model` through `.sp_aggregated` detection and SVD decomposition (`server_strategy.py:114-166`). This is research-specific coupling in a nominally generic base class.
- RBLA requires aggregator, client/server/runner strategies, model/rank configuration, and factory registrations. Heterogeneous-rank broadcast is not satisfied by adding only an aggregator because global tensors must be fitted back to client-local shapes (`_fed_aggregator_rbla.py:380 onward`).

### Patterns actually present

- **Strategy:** Runner/server/client families are runtime-selected interchangeable behaviours with meaningful different workflows; this is more than ordinary inheritance (`strategy_factory.py`).
- **Factory Method/Simple Factory:** factories translate YAML names to concrete models, trainers, selectors, aggregators and strategies. They centralise construction but are not registries/plugins.
- **Template Method:** aggregator `aggregate` fixes extract-before-do-after order and delegates hooks (`fed_aggregator_abc.py:68-100`). This is a genuine template method. Runner classes do not share an equivalent template.
- **Observer:** `EventHandler` plus `declare_events`, `attach_event`, `raise_event`; `StandardSampleEntry.__attach_event_handler` registers preparation callbacks (`standard_entry.py:105-155`). Hooks are mainly preparation-time, not a general runtime event bus.
- **Dependency injection/service locator:** `FedNodeVars` owns replaceable service objects prepared from config, while `AppEntry` exposes a static object map. It is pragmatic DI, but also global mutable state/service-location.
- **Adapter:** LoRA/model weight adaptor classes under `src/flex/ml_algorithms/model_adaptor` transform tensor layouts; server SP conversion adapts aggregate format to evaluator format.

No dynamic registry, package-discovered plugin architecture, command pattern, or meaningful State pattern was found. Dependencies generally flow entries → runner/nodes/strategies → algorithm/trainer/model utilities, but base server strategy imports LoRA utilities lazily, concrete factories know all algorithms, and `FedNodeVars` imports most factories. These are central coupling points. Static `AppEntry.__app_objects`, `FedNodeVars.share_model/share_vocab`, and global RNG seeding are global mutable state.

## 6. Communication and runtime model

Implemented execution is synchronous, in-process, and normally sequential per client. The main path uses no multiprocessing, sockets, RPC, cloud worker, shared-memory protocol or serialisation. Model dictionaries are passed by Python reference (although local training deep-copies models and some aggregators clone tensors).

`SimuSwitcher` is a real but limited simulation abstraction: one worker thread waits on a `Queue`, retrieves `SimuNodeData`, then calls the receiver's event handler (`simu_switcher.py:105-151`). `SimuNode` maintains IDs and logical links and can send `Any` payload (`simu_node.py:54-67,98-134`). Yet `FedAvgServerStrategy.broadcast` directly loops through clients, and runners directly invoke training; therefore the transport is not on the critical algorithm path. Replacing it cannot yield remote execution without rewriting runner/server/client interaction.

Synchronization is implicit: the runner waits for each client call, then aggregates the complete list. There are no per-client timeouts, retries, disconnect recovery, quorum logic, dropped-update handling, asynchronous aggregation, or true straggler execution. FedGRA measures elapsed local training and some selectors use latency as an algorithmic feature, but it does not simulate concurrent deadline effects. Oort/PyramidFL parameters do not establish a runtime failure model. Batch subprocess timeout handling in `src/entries/run_all.py` is experiment-launcher logic, not FL communication resilience.

## 7. State and data flow

| Data | Representation and movement |
|---|---|
| Global model | PyTorch state dict in `server.node_var.model_weight`; post-aggregation value also in `aggregated_weight`. Evaluator owns an `nn.Module` updated via `load_state_dict` through `ModelEvaluator.update_model`. |
| Client model | Base `node_var.model` plus current `model_weight`; local training deep-copies the module and loads the state dict strictly (`_fedavg_client.py:101-110`). |
| Broadcast cache | `client.node_var.cache_weight`; then assigned to `model_weight` (`_fedavg_client.py:153-159`). These are dynamic fields rather than declared dataclass fields. |
| Client update envelope | Runner-level dictionary `{ "updated_weights": state_dict, "train_record": record }`; FedGRA may add `latency`. Note that `FedAvgClientTrainingStrategy.run_local_training` itself returns a nested record containing another `updated_weights` field (91-98), so schemas are redundant/inconsistent. |
| Aggregator input | `AbstractFedAggregator.aggregate` expects each envelope to contain `d["updated_weights"]` and `d["train_record"]["data_sample_num"]` (`fed_aggregator_abc.py:68-76`). Missing keys fail with raw exceptions. |
| Training record | Untyped dict from trainer, expected to include `data_sample_num`, usually `avg_loss` and sometimes `node_id`/metrics. Selector feedback reshapes it in `_fedavg_server.py:44-76`. |
| Sample count | `client_var.data_sample_num` is set from loader (`standard_entry.py:90-92`) and expected in train record for weighting. |
| Metrics | Evaluator returns a dict stored as `server.eval_results`; logger appends dict data to CSV. |
| Round | Local loop variable in runner; server separately keeps `_eval_round_counter` for selection logging (`fed_node_server.py:21,76-95`). There is no shared `RoundContext`. |
| Client metadata | Node ID/group and arbitrary selector dictionaries. Heterogeneous ranks/resources are generally YAML/strategy fields, not a typed capability model. |
| Configuration | Nested dicts wrapped by `KeyValueArgs`; composed by shallow `dict.update`, so whole top-level sections overwrite rather than recursively merge. |
| Communication payload | `SimuNodeData(data, from_id, to_id)` for switcher path; standard path uses direct method arguments. |

The update schema is a hidden but critical coupling: created by runner generators, stored by server strategy, destructured by the aggregator ABC, and reused by selection feedback. It should be formalised as a typed dataclass/protocol before claiming a stable framework API.

### Aggregation architecture

`AbstractFedAggregator.aggregate` is model-agnostic only at its outer envelope. It extracts state dictionaries and sample counts, then calls hooks. `FedAggregator_FedAvg._do_aggregation` (`src/flex/fl_algorithms/aggregation/methods/_fed_aggregator_fedavg.py:33-78`) takes keys from the first client, allocates device-local buffers, sums sample volumes, accumulates floating tensors by `vol/total`, accumulates integers in float32, then rounds them back. It assumes a nonempty update list, positive total samples, identical keys, compatible shapes, tensor values and accessible first parameter. These are not explicitly validated; empty input causes indexing failure and zero total causes division by zero.

FedAvg is reusable with any same-shaped PyTorch state dict. RBLA is algorithm-specific: `aggregate_lora_tensors` NaN- or zero-pads heterogeneous LoRA matrices and computes elementwise support-aware weighted means (`_fed_aggregator_rbla.py:291-317`); `aggregate_state_dicts` distinguishes `lora_A/lora_B`, copies or averages non-LoRA values and assumes keys from the first state dict (319-378); `broadcast_lora_state_dict` fits global tensors to each local tensor shape (380 onward). This supports heterogeneous ranks but binds aggregation to naming conventions, A/B tensor layout, PyTorch and matching key sets.

The aggregation layer is independently testable and mostly separate from server strategy, but not independently deployable: it consumes FLEX's update envelope and LoRA broadcast/conversion requires strategy cooperation.

## 8. Case studies

### FedAvg baseline

Core files are `_fedavg_runner_strategy.py`, `_fedavg_server.py`, `_fedavg_client.py`, and `_fed_aggregator_fedavg.py`, plus factory cases and YAML. It reuses entries, nodes, `FedNodeVars`, model/data/trainer factories, evaluator and logger. The baseline demonstrates the component boundaries but also establishes hidden schemas and direct-call execution.

### FedGRA client selection

`src/flex/fl_algorithms/selection/methods/_fed_client_selector_fedgra.py` implements selection; `_fedgra_server.py` adapts metrics and aggregation; `_fedgra_runner_strategy.py` adds an all-client lightweight observation stage, measured latency, selection, then real selected-client training. Framework-core factory files are modified to register names. Much infrastructure is reused, but a new runner/server pair and duplicated round loop were required, so “add one selector class without core changes” would be false.

### RBLA / heterogeneous-rank LoRA

`_fed_aggregator_rbla.py` handles rank-mismatched tensors; `_rbla_client.py`, `_rbla_server.py`, `_rbla_runner_strategy.py`, LoRA model/utilities, rank-distribution YAML and factory cases integrate it. Reference-frame and support-scaling variants subclass/reuse RBLA under `*/rbla_problem`. Tests include `src/unittest_ml/unittest_aggregator_rbla.py`, `unittest_rbla_reference_frame.py`, and the untracked `unittest_lora_canonicalization.py`. This is the strongest evidence for a thesis case study because the same runtime supports baseline and multiple heterogeneous-LoRA hypotheses, but the implementation touches more than the aggregator boundary.

### Split learning

SFL strategies and trainers exist under `src/flex/sfl_strategy` and `src/flex/model_trainer/trainer/_model_trainer_sl_*`. This demonstrates architectural breadth, but several example methods raise `NotImplementedError`, and SFL entry preparation hooks log TODO warnings (`src/entries/app/sfl_entry.py:125-169`). It should be presented as partial/experimental, not a mature supported mode.

Hierarchical FL and secure aggregation cannot serve as implemented case studies: hierarchical edge execution is stubbed and secure aggregation was not found in the current repository.

## 9. Lightweightness assessment

FLEX is lightweight in **runtime topology** but not clearly lightweight in dependency or code-organisation terms.

- Positive: one process, one optional switcher thread, no external coordinator/database/message broker, and a basic algorithm runs through direct Python calls.
- Negative: `requirements/base.txt` has 17 substantial direct packages, including the full PyTorch vision/audio stack, transformers, datasets, accelerate, pandas/scipy/sklearn and matplotlib. `pyproject.toml` incorrectly declares zero dependencies, so install metadata understates the footprint.
- The core has roughly 36.6k physical Python lines across 298 files. There are at least 13 ABC-bearing modules and many factories/args/builders before a round executes.
- A new same-schema aggregator minimally requires one implementation file, one factory branch and YAML. A genuinely new lifecycle algorithm commonly requires selector/aggregator plus client, server and runner strategies, three factory registrations and several YAML fragments.
- Execution is traceable for FedAvg, but indirection through static config objects, `FedNodeVars`, three strategy layers and factories raises cognitive overhead. Duplicated concrete runners make cross-cutting lifecycle changes expensive.

Defensible wording is “lightweight in-process research runtime,” not “lightweight framework” without measured installation size, startup/runtime overhead, memory, extension effort and comparison baselines.

## 10. Code-quality and architectural limitations

Strengths include explicit ABCs for main algorithm families, broad factory/config coverage, reusable training/offload helpers, concrete LoRA aggregation tests, non-IID configurations, result logging and visible failure on unsupported factory names.

Prototype-quality weaknesses are material:

1. **Packaging mismatch:** `pyproject.toml` names `usyd_learning`, packages `src/usyd_learning` (which does not exist), and declares no dependencies, while code is `src/flex` and `requirements/base.txt` lists 17 packages.
2. **No stable schemas:** state and updates are untyped dictionaries/dynamic attributes; comments conflict with actual structures. `FedNodeVars.get_var` omits `return` (`fed_node_vars.py:73-75`).
3. **Closed factories:** extensions require central edits; no registry or entry-point plugins.
4. **Duplicated orchestration:** FedAvg, FedGRA, Oort, PyramidFL and RBLA runners repeat select/train/aggregate/broadcast/evaluate loops.
5. **Incomplete abstractions:** `FedRunner` inherits `ABC` but contains no abstract method; `AppEntry.run` simply passes; server implementations contain duplicate `run` definitions raising `NotImplementedError`; edge creation is stubbed.
6. **Direct-call/transport mismatch:** logical network objects exist but algorithms bypass them, preventing credible deployment substitution.
7. **Global mutable state:** static application object/config maps (`app_entry.py:17-20`), shared model/vocab (`fed_node_vars.py:26-28`) and global RNG resetting in `BaseStrategy` and selectors can contaminate multi-experiment processes.
8. **Seed weaknesses:** base strategies repeatedly call `TrainingUtils.set_seed(42)` (`base_strategy.py:7-14`); selector calls global `random.seed`; data-loader seed derives partly from Python `hash`, whose cross-process reproducibility depends on hash seed (`fed_node_vars.py:283-293`).
9. **Configuration:** composition is shallow and weakly schema-validated; many failures become `KeyError`/attribute errors. There is no emitted fully resolved immutable config manifest.
10. **Round semantics:** `range(training_rounds + 1)` is surprising and likely inconsistent with configuration language.
11. **Error handling:** FedAvg aggregation does not validate empty lists, zero samples, key equality, shapes or finite tensors. No client failure isolation exists.
12. **Naming/encoding:** `fed_aggregator_facotry.py`, `client_manger.py`, class naming such as `rblasaRunnerStrategy`, mixed absolute/relative imports, duplicated imports, and mojibake in comments/README reduce maintainability.
13. **Tests:** there are useful unit tests, especially for aggregators and LoRA canonicalisation, but directories mix pytest, `unittest`, manual `main()` scripts, full experiments and generated outputs. No authoritative test command, coverage threshold or visible CI workflow was established from the inspected core. Many “test” files download datasets or depend on working-directory mutations.
14. **Platform assumptions:** `standard.py` changes CWD to `src/test`; relative dataset paths are embedded; `install.sh` is Bash-centric while the repository is also used on Windows.
15. **Dirty research state:** important canonicalisation code/tests/configs are untracked and RBLA sources are modified. This is valid during research but unsuitable as thesis evidence until versioned and tagged.

## 11. Thesis suitability

| Level | Assessment |
|---|---|
| A. Short implementation section | **Yes now.** There is enough concrete architecture and execution code. |
| B. Substantial thesis section | **Yes now, with critical limitations.** FedAvg/FedGRA/RBLA provide meaningful examples of reused infrastructure and altered algorithm components. |
| C. Standalone thesis chapter | **Conditionally yes.** Add formal architecture, measured extension effort, reproducibility, overhead/scalability evidence, a clean release and tests. Frame it as research infrastructure supporting the thesis methods, not a production platform. |
| D. Publishable framework paper | **Not yet supported.** Missing comparative evidence, stable/public API, transport/runtime evaluation, documentation, CI/coverage, packaging, release and user study/adoption evidence. |

| Possible claim | Support | Reason |
|---|---|---|
| Lightweight FL research framework | **Partially supported** | Runtime is light; dependency/code/config footprint is not demonstrated as light. |
| Modular separation of FL components | **Strongly supported, with coupling caveats** | Separate selector/aggregator/strategy/trainer/model/config modules and real variants exist. Closed factories and LoRA leakage qualify the claim. |
| Rapid prototyping of aggregation/selection | **Partially supported** | Many implementations reuse infrastructure, but no measured implementation-time/LOC comparison and lifecycle algorithms require multiple files. |
| FLaaS-oriented abstraction | **Weakly supported** | Configurable roles/components are compatible with an FLaaS research narrative; no service plane exists. |
| Heterogeneous LoRA support | **Strongly supported for simulation** | Rank-aware aggregation/broadcast, configs, strategies, tests and experiments exist. |
| Reusable infrastructure across published studies | **Partially supported** | Multiple methods share runtime in code; publication provenance and version-to-paper mapping are not documented. |
| Reproducible experimentation platform | **Partially supported** | YAML/seeds/logs/results exist; global/static state, shallow merges, generated/untracked files and environment/version gaps impair exact reproduction. |

## 12. Missing evidence and recommended additions

### Essential before a standalone chapter

1. Tag a clean, immutable thesis version and record commit, environment lock, datasets/checksums and resolved configs.
2. Add a component/dependency diagram and a sequence diagram that accurately show direct in-process calls.
3. Formalise APIs and data contracts: `ClientUpdate`, `TrainRecord`, `RoundContext`, `ClientCapability`, and aggregator/selector tables.
4. Provide three controlled extension case studies (FedAvg, FedGRA, RBLA) with files changed, non-comment LOC, core edits and implementation effort.
5. Measure runtime overhead against equivalent direct PyTorch scripts, peak memory, and scaling with clients/model size.
6. Add deterministic end-to-end smoke tests and CI; publish pass/coverage results.
7. Fix packaging/dependency metadata, round-count semantics, factory spelling, and a single documented test/install/run path.
8. Explicitly delimit non-goals: real networking, production deployment, privacy/security, fault tolerance and multi-tenancy.

### Desirable

- Replace closed matches with typed registries and validate YAML schemas.
- Extract a shared round template/pipeline with overridable phases.
- Make transport a real port and run the standard path through it, or remove deployment implications.
- Compare extension effort with a plain bespoke script and one general framework, without universal-superiority claims.
- Add failure/timeout experiments only if resilience becomes a contribution.
- Produce public API docs, examples, changelog, semantic versions and architecture decision records.
- Run a reproducibility study on a clean machine and archive raw outputs.

### Unnecessary unless claimed as a contribution

- Building a production cloud service, UI, multi-tenant scheduler or cryptographic secure aggregation.
- Supporting every FL algorithm or every deployment backend.
- Repository-size-only comparisons or unsupported benchmark claims against Flower/FedML.
- An ablation of every software class; evaluate only design mechanisms tied to stated claims.

## 13. Proposed thesis narrative

**Suggested title:** *FLEX: A Configurable In-Process Execution Substrate for Federated-Learning Algorithm Research*

**Motivation.** The thesis repeatedly changes selection logic, round sequencing, aggregation mathematics and heterogeneous LoRA state while retaining common PyTorch training, data partitioning, evaluation and experiment recording. FLEX was developed to make those stable concerns reusable and to expose the changing concerns as explicit strategy, selector and aggregator boundaries. The contribution should be evaluated by reuse and experimental consistency across the thesis case studies, not by claims of production deployment or universal superiority over established frameworks.

**Design principles:** (1) configuration-composed experiments; (2) explicit separation of round, server, client, selection and aggregation behaviours; (3) a common mutable node context for research instrumentation; (4) PyTorch-native state dictionaries and trainers; (5) minimal in-process runtime assumptions; (6) specialised but replaceable heterogeneous-LoRA transformations.

**Main architectural contribution.** FLEX provides a common synchronous execution substrate in which conventional FedAvg, feedback-driven selection such as FedGRA, and heterogeneous-rank LoRA aggregation such as RBLA can reuse node construction, data partitioning, training, evaluation and logging while replacing defined algorithm components and, when necessary, the round lifecycle.

**Suitable case studies:** FedAvg establishes the baseline contract; FedGRA demonstrates lifecycle/selection feedback extension; RBLA and reference/canonicalisation variants demonstrate model-heterogeneous aggregation and client-specific broadcast fitting. SFL may be mentioned as architectural exploration, not mature evidence.

**Limitations to acknowledge:** single-process sequential execution; communication abstraction bypassed in the main path; no secure aggregation, fault tolerance or FLaaS service plane; closed factories; weakly typed state/update contracts; incomplete hierarchy/SFL areas; heavy scientific dependencies; prototype packaging/testing/reproducibility.

**Conclusion.** The repository supports a defensible claim that FLEX reduced repeated engineering across this thesis's FL algorithm studies. It does not yet support claims of a production-ready, transport-independent or generally superior federated-learning framework. A standalone chapter becomes credible when the architectural account is paired with clean versioned artefacts and quantitative evidence of reuse, overhead and reproducibility.

## 14. Appendix: important classes, interfaces, and call chains

### Important classes

| Symbol | Location | Role |
|---|---|---|
| `AppEntry` | `src/flex/ml_utils/app_entry.py:12-242` | Static config/object composition and application base. |
| `StandardSampleEntry` | `src/entries/app/standard_entry.py:16-155` | Builds a conventional FL experiment. |
| `FedRunner` | `src/flex/fed_runner/fed_runner.py:17-124` | Node construction and runner-strategy delegation. |
| `FedNode`, `FedNodeClient`, `FedNodeServer` | `src/flex/fed_node/fed_node.py`; `fed_node_client.py`; `fed_node_server.py` | Node façades and strategy delegation. |
| `FedNodeVars` | `src/flex/fed_node/fed_node_vars.py:21-690` | Mutable runtime state and prepared services. |
| `RunnerStrategy` | `src/flex/fed_strategy/runner_strategy.py:8-50` | Round-lifecycle abstraction. |
| `ServerStrategy` | `src/flex/fed_strategy/server_strategy.py:9-166` | Server algorithm operations and shared weight application. |
| `ClientStrategy` | `src/flex/fed_strategy/client_strategy.py:14-194` | Local observation/training abstraction and GPU helpers. |
| `AbstractFedAggregator` | `src/flex/fl_algorithms/aggregation/fed_aggregator_abc.py:11-100` | Update-envelope adapter and aggregation template. |
| `FedClientSelector` | `src/flex/fl_algorithms/selection/fed_client_selector_abc.py:11-61` | Stateful client-selection interface. |
| `SimuSwitcher`, `SimuNode` | `src/flex/ml_simu_switcher/simu_switcher.py`; `simu_node.py` | Optional in-process event/queue topology simulation. |
| `ModelTrainer`, `ModelEvaluator` | `src/flex/model_trainer/model_trainer.py`; `model_evaluator.py` | Training contract and global evaluation. |

### Compact call chain

```text
entries.standard.main
  -> StandardSampleEntry.load_app_config
     -> ConfigLoader.load
     -> AppEntry.__parse_app_config_define
  -> StandardSampleEntry.run
     -> FedRunner.create_nodes
     -> StrategyFactory.create_runner_strategy
     -> FedNodeVars.prepare [server and each client]
     -> FedNodeServer/FedNodeClient.prepare_strategy
     -> FedRunner.run
        -> FedAvgRunnerStrategy.run
           -> FedAvgServerStrategy.broadcast
           -> FedAvgServerStrategy.select_clients
              -> FedClientSelector.select
           -> FedAvgClientTrainingStrategy.run_local_training [sequential]
              -> local_training_step
              -> ModelTrainer.train
           -> FedAvgServerStrategy.receive_client_updates
           -> FedAvgServerStrategy.aggregation
              -> AbstractFedAggregator.aggregate
              -> FedAggregator_FedAvg._do_aggregation
           -> ServerStrategy.apply_weight
              -> ModelEvaluator.update_model
           -> FedAvgServerStrategy.broadcast
           -> FedAvgServerStrategy.evaluate
              -> ModelEvaluator.evaluate
           -> FedAvgServerStrategy.record_evaluation
              -> TrainingLogger.record
```

### Features explicitly not found or incomplete

- Secure aggregation, differential privacy and cryptographic transport: **not found in the current repository**.
- Remote RPC/socket/cloud execution and model serialisation protocol: **not found in the current repository**.
- Multi-tenant FLaaS APIs, authentication, durable job state and scheduling: **not found in the current repository**.
- Complete hierarchical FL: **incomplete**; edge node/runtime methods are stubs.
- Asynchronous FL, quorum aggregation, client retry and per-round timeout recovery: **not found in the current repository**.
- Stable typed client-update/capability API: **not found in the current repository**.

