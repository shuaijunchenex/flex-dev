# Flex-Core Architecture V2: Distributed PEFT Operator Substrate

> **Status:** Revised design proposal, pre-implementation  
> **Date:** 2026-07-23  
> **Scope:** `flex-core` and its contracts with `flex-runtime`, `flex-integrations`, and `flex-lab`  
> **Supersedes:** `flex-core-architecture.md` for new implementation decisions  
> **Non-goal:** This document does not implement any package or migrate existing FLEX code.

---

## 1. Executive Summary

Flex-Core is a framework-independent mathematical substrate for distributed parameter-efficient fine-tuning (PEFT). It defines typed update representations, pure operators, explicit operator state, validation, diagnostics, cost descriptions, codecs, and composition rules. It does not own training loops, datasets, optimizers, client selection, cloud launchers, or experiment matrices.

The central execution model is:

```text
External PEFT state
  → Integration Codec
  → immutable Core IR
  → validated OperatorRequest
  → pure Operator or compiled Pipeline
  → OperatorResult
  → Integration Codec
  → External PEFT state
```

The primary heterogeneous-rank RBLA+ path is explicitly modeled as:

```text
RBLAReducer
  → CanonicalizeLoRAFactors
  → ClientRankProjector
```

V2 makes the following binding decisions:

1. Client-specific projection targets are typed mathematical inputs, not untyped context tags.
2. Adapter updates are grouped by logical module; representation belongs to a module update, not redundantly to both collection and tensor.
3. Core values are logically immutable and must not share writable tensor storage across operator boundaries.
4. Operator configuration is schema-versioned and typed; arbitrary `dict[str, Any]` is not a public Core contract.
5. Operators are deterministic pure computations. Retry, idempotency, checkpoint commit, and exactly-once visibility are Runtime responsibilities.
6. Validation is separated from execution. Runtime policy decides whether invalid client inputs abort a round or are excluded.
7. Cost models and diagnostics are structured serializable data, never executable callables or log-only output.
8. Datasets, Dirichlet distributions, rank-assignment experiments, optimizers, random-seed matrices, AWS launchers, and statistical reports belong to `flex-lab`.

---

## 2. System Boundaries

### 2.1 Layering

```text
┌──────────────────────────────────────────────────────────────────────┐
│ flex-lab                                                             │
│ experiment specifications, datasets, distributions, rank assignment,│
│ optimizers, seed matrices, launchers, metrics, statistical analysis │
├──────────────────────────────────────────────────────────────────────┤
│ flex-runtime                                                         │
│ pipeline execution, rounds, retries, transactional checkpoints,     │
│ placement, concurrency, streaming, communication, resource control  │
├──────────────────────────────────────────────────────────────────────┤
│ flex-integrations                                                    │
│ FLEX, Flower, NVFlare, HuggingFace PEFT, transport and model bridges │
├──────────────────────────────────────────────────────────────────────┤
│ flex-core                                                            │
│ IR, schemas, operators, validation, state values, diagnostics, codecs│
└──────────────────────────────────────────────────────────────────────┘
```

Dependency direction is strictly downward toward `flex-core`. `flex-core` must not import any other layer.

### 2.2 Flex-Core owns

- PEFT update identity and representation semantics.
- Heterogeneous-rank aggregation and projection mathematics.
- Typed operator requests, results, configuration, and state values.
- Shape, dtype, representation, weight, and compatibility validation.
- Reference implementations of mathematical operators.
- Structured diagnostics and serializable cost estimates.
- Codec protocols and Core-side round-trip requirements.
- Pipeline type checking and semantic compatibility rules.
- State schema versioning and integrity definitions.

### 2.3 Flex-Core does not own

- Model training, optimizers, schedulers, losses, or data loaders.
- Dataset partitioning, Dirichlet sampling, or rank-assignment generation.
- Federated client selection, availability, stragglers, or dropouts.
- Round scheduling, timeout retry, checkpoint storage, or transaction commit.
- Network transport, serialization framing, compression negotiation, or RPC.
- GPU allocation, process management, cloud launchers, or AWS configuration.
- Experiment sweeps, random-seed orchestration, CSV logging, plotting, or significance testing.
- Framework lifecycle objects such as Flower `ClientProxy` or NVFlare `Shareable`.

### 2.4 Meaning of “framework-independent”

V2 distinguishes training-framework independence from tensor-backend independence.

- Phase 1 is PyTorch-tensor-first.
- Core operators do not depend on FLEX, Flower, NVFlare, HuggingFace Trainer, or a particular training loop.
- A future tensor backend abstraction is possible, but is not required before mathematical contracts stabilize.
- Serialized Core artifacts must not require a live training-framework object.

---

## 3. Architectural Invariants

The following are mandatory and testable.

### 3.1 Semantic invariants

1. **Self-description:** A module update identifies its PEFT type, module identity, representation, roles, rank, dimensions, dtype, scaling, and base-model compatibility without parsing an external framework key.
2. **Representation clarity:** Factor, product, compact-factor, spectral, stacked, and generic-delta forms are never conflated.
3. **Heterogeneous rank:** No reducer, canonicalizer, projector, codec, or pipeline may assume equal client ranks unless its `OperatorSpec` explicitly rejects heterogeneous rank.
4. **Target explicitness:** Any client-specific output is a function of explicit `AdapterTarget` inputs.
5. **Weight explicitness:** Aggregation weighting and normalization policy are declared by the operator configuration.

### 3.2 Execution invariants

1. **Logical immutability:** Operators never mutate requests, inputs, configuration, targets, or state.
2. **Storage isolation:** Operator results do not expose writable tensor storage shared with an input unless the result type explicitly declares read-only identity passthrough.
3. **Pure transition:** The same validated request and pre-state under the same determinism scope produce the same result and new state.
4. **Atomic state value:** An operator returns a complete replacement state or no state. It never partially advances a caller-owned state.
5. **No hidden round:** Round, run, and invocation identities are explicit.
6. **No hidden RNG:** Random behavior derives only from the declared execution seed and deterministic substream rules.
7. **Structured observability:** Mathematical and resource diagnostics are returned as data, not only printed.

### 3.3 Boundary invariants

1. External names and framework objects terminate at codecs or integration adapters.
2. Runtime retry policy never changes operator mathematics.
3. Experiment configuration never leaks into Core as an unvalidated arbitrary mapping.
4. Every serialized artifact carries schema and producer versions.
5. Every lossy transform declares its approximation source and error metric.

---

## 4. Package Structure

```text
src/flex-core/
├── pyproject.toml
├── README.md
├── src/flex_core/
│   ├── __init__.py
│   ├── errors.py
│   ├── ir/
│   │   ├── __init__.py
│   │   ├── identity.py
│   │   ├── representation.py
│   │   ├── tensor_value.py
│   │   ├── module_update.py
│   │   ├── adapter_update.py
│   │   ├── source.py
│   │   └── target.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── protocol.py
│   │   ├── schema.py
│   │   └── migration.py
│   ├── operator/
│   │   ├── __init__.py
│   │   ├── protocol.py
│   │   ├── request.py
│   │   ├── result.py
│   │   ├── spec.py
│   │   ├── context.py
│   │   ├── state.py
│   │   ├── diagnostics.py
│   │   ├── cost.py
│   │   └── pipeline.py
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── report.py
│   │   ├── shapes.py
│   │   ├── weights.py
│   │   ├── compatibility.py
│   │   └── determinism.py
│   ├── codec/
│   │   ├── __init__.py
│   │   ├── protocol.py
│   │   ├── manifest.py
│   │   └── errors.py
│   └── ops/
│       ├── __init__.py
│       ├── identity.py
│       ├── weighted_average.py
│       ├── zero_padding.py
│       ├── rbla.py
│       ├── product_average.py
│       ├── canonicalize_lora.py
│       └── client_rank_projector.py
└── tests/
    ├── test_ir/
    ├── test_config/
    ├── test_operator/
    ├── test_validation/
    ├── test_codec/
    ├── test_ops/
    └── test_integration/
```

The package structure expresses ownership. For example, an AWS launcher cannot be added beneath `flex_core/ops`, and a canonicalization implementation cannot live in a federated server strategy.

---

## 5. Core Intermediate Representation

### 5.1 RepresentationKind

```python
class RepresentationKind(Enum):
    FACTOR = "factor"                    # LoRA A/B or equivalent factors
    PRODUCT = "product"                  # dense ΔW
    COMPACT_FACTOR = "compact_factor"    # canonical ordered factors
    SPECTRAL = "spectral"                # U, singular values, Vh
    STACKED = "stacked"                  # concatenated/stacked factor form
    DELTA = "delta"                      # generic parameter delta
```

Representation is declared per logical module update. A top-level update may contain different representations for different modules and is therefore not assigned one redundant global representation.

### 5.2 AdapterIdentity

```python
@dataclass(frozen=True)
class AdapterIdentity:
    module_path: str
    adapter_name: str
    peft_type: str
    base_model_fingerprint: str | None = None
```

`module_path` is a normalized logical path chosen by the codec. It is not required to equal a HuggingFace state-dict key.

### 5.3 TensorRole and TensorSpec

```python
class TensorRole(Enum):
    LORA_A = "lora_A"
    LORA_B = "lora_B"
    SINGULAR_VALUES = "singular_values"
    LEFT_SINGULAR_VECTORS = "left_singular_vectors"
    RIGHT_SINGULAR_VECTORS = "right_singular_vectors"
    SCALING = "scaling"
    MAGNITUDE = "magnitude"
    BIAS = "bias"
    DELTA = "delta"

@dataclass(frozen=True)
class TensorSpec:
    role: TensorRole
    shape: tuple[int, ...]
    dtype: str
    layout: str = "dense"
```

Device placement is intentionally not part of mathematical identity. The physical tensor value reports its current placement, while serialized manifests record storage placement independently.

### 5.4 TensorValue

```python
@dataclass(frozen=True)
class TensorValue:
    spec: TensorSpec
    data: torch.Tensor

    @property
    def device(self) -> str: ...

    def isolated_copy(self, *, device: str | None = None) -> "TensorValue": ...
```

Immutability rules:

- `frozen=True` alone is insufficient because PyTorch tensors remain mutable.
- Public Core APIs treat `data` as read-only.
- Operator implementations may use internal mutable work buffers but never mutate caller-owned storage.
- Outputs must be cloned or newly allocated when mutation of one output could affect an input or another output.
- Codecs define whether encoding adopts or copies external storage; the default is safe copy/detach.
- Validation tests compare storage identities and input checksums before and after execution.

### 5.5 AdapterModuleUpdate

```python
@dataclass(frozen=True)
class AdapterModuleUpdate:
    identity: AdapterIdentity
    representation: RepresentationKind
    tensors: ImmutableMapping[TensorRole, TensorValue]
    rank: int | None
    input_dim: int | None
    output_dim: int | None
    scaling: float
    metadata: ImmutableMapping[str, ScalarValue]
```

The module is the smallest unit on which factor pairing, rank, scaling, product reconstruction, canonicalization, and projection have coherent meaning.

Required role sets depend on representation:

| Representation | Required roles | Rank meaning |
|---|---|---|
| `FACTOR` | `LORA_A`, `LORA_B` | factor inner dimension |
| `COMPACT_FACTOR` | `LORA_A`, `LORA_B` | ordered canonical prefix length |
| `PRODUCT` | `DELTA` | matrix effective rank may be diagnostic only |
| `SPECTRAL` | left vectors, singular values, right vectors | number of spectral components |
| `DELTA` | `DELTA` | optional/not applicable |

The validator rejects missing pairs, inconsistent dimensions, incompatible scaling, or representation-role conflicts.

### 5.6 SourceMetadata

```python
class WeightSemantics(Enum):
    RAW = "raw"
    NORMALIZED = "normalized"

@dataclass(frozen=True)
class SourceMetadata:
    source_id: str
    sample_count: int | None
    aggregation_weight: float | None
    weight_semantics: WeightSemantics
    produced_round: int | None
    update_id: str
```

Rules:

- `source_id` is opaque to Core.
- Raw sample counts and precomputed aggregation weights are distinct.
- An operator config declares which field it consumes.
- Negative or non-finite weights are invalid.
- Whether weights are normalized before aggregation is explicit.
- If invalid clients are excluded by Runtime policy, remaining-weight renormalization follows operator configuration and is reported.

### 5.7 AdapterUpdate

```python
@dataclass(frozen=True)
class AdapterUpdate:
    modules: ImmutableMapping[AdapterIdentity, AdapterModuleUpdate]
    source: SourceMetadata
    schema_version: str
    update_fingerprint: str
```

An update may contain mixed module representations. Compatibility is checked module-by-module. `update_fingerprint` covers semantic metadata and tensor content and is used for replay verification, caching, and artifact integrity.

### 5.8 Client projection targets

```python
@dataclass(frozen=True)
class ModuleTarget:
    identity: AdapterIdentity
    target_rank: int | None
    accepted_representations: frozenset[RepresentationKind]
    dtype: str | None = None

@dataclass(frozen=True)
class AdapterTarget:
    target_id: str
    modules: ImmutableMapping[AdapterIdentity, ModuleTarget]
    rank_budget: int | None = None
    metadata: ImmutableMapping[str, ScalarValue] = EMPTY_MAPPING
```

Targets are mathematical inputs because rank and representation change projected values. Physical device placement and transport details remain Runtime concerns.

---

## 6. Typed Configuration

### 6.1 Configuration contract

Every configurable operator defines an immutable configuration value with:

- a stable schema name;
- a schema version;
- field types and constraints;
- canonical serialization;
- a fingerprint;
- explicit migration rules between supported schema versions.

```python
class OperatorConfig(Protocol):
    schema_name: ClassVar[str]
    schema_version: ClassVar[str]

    def validate(self) -> "ValidationReport": ...
    def canonical_dict(self) -> ImmutableMapping[str, ScalarValue]: ...
    def fingerprint(self) -> str: ...
```

Operators receive their typed configuration during construction or through an immutable request field. They do not read arbitrary YAML mappings.

### 6.2 Reference configurations

```python
@dataclass(frozen=True)
class WeightedReductionConfig:
    weight_source: Literal["sample_count", "aggregation_weight"]
    normalize_weights: bool
    zero_total_policy: Literal["error", "uniform"]
    accumulation_dtype: str

@dataclass(frozen=True)
class RBLAConfig:
    pad_mode: Literal["nan_masked", "zero"]
    weight_source: Literal["sample_count", "aggregation_weight"]
    normalize_weights: bool
    accumulation_dtype: str

@dataclass(frozen=True)
class CanonicalizationConfig:
    ordering: Literal["singular_value", "activation_aware"]
    deterministic_sign: bool
    compute_dtype: str
    eps: float
    svd_fallback: bool
    start_round: int
    interval: int
    activation_fallback: bool

@dataclass(frozen=True)
class ProjectionConfig:
    undersized_policy: Literal["zero_pad", "error"]
    oversized_policy: Literal["prefix", "energy_truncate", "error"]
```

Optimizer settings such as SGD learning rate and momentum are not Core operator configurations.

---

## 7. Operator Model

### 7.1 Cardinality

```python
class Cardinality(Enum):
    ONE_TO_ONE = "1:1"
    MANY_TO_ONE = "N:1"
    ONE_TO_MANY = "1:N"
    MANY_TO_MANY = "N:M"
```

Cardinality is machine-checkable metadata. Pipelines verify that adjacent stages can be bound without silently dropping or duplicating updates.

### 7.2 ExecutionContext

```python
@dataclass(frozen=True)
class ExecutionContext:
    run_id: str
    invocation_id: str
    round_index: int
    attempt_index: int
    seed: int
    determinism_scope: "DeterminismScope"
    tags: ImmutableMapping[str, str]
```

`tags` are observational labels only. Values that alter mathematical output must appear in typed config, inputs, targets, or state.

### 7.3 OperatorRequest

```python
@dataclass(frozen=True)
class OperatorRequest:
    inputs: tuple[AdapterUpdate, ...]
    targets: tuple[AdapterTarget, ...]
    context: ExecutionContext
    state: "OperatorState | None"
```

Rules:

- Reducers normally require inputs and no targets.
- Client projectors require one input and one or more targets.
- Pipelines bind intermediate outputs and preserve required targets.
- Target order is not used as identity; `target_id` is.

### 7.4 Operator protocol

```python
class Operator(ABC):
    @property
    @abstractmethod
    def spec(self) -> "OperatorSpec": ...

    @abstractmethod
    def validate(self, request: OperatorRequest) -> "ValidationReport": ...

    @abstractmethod
    def apply(self, request: "ValidatedOperatorRequest") -> "OperatorResult": ...
```

`apply` accepts only a validated request. Direct execution on unvalidated external data is not part of the public protocol.

### 7.5 OperatorResult

```python
@dataclass(frozen=True)
class TargetedUpdate:
    target_id: str
    update: AdapterUpdate

@dataclass(frozen=True)
class OperatorResult:
    outputs: tuple[AdapterUpdate, ...]
    targeted_outputs: tuple[TargetedUpdate, ...]
    new_state: "OperatorState | None"
    diagnostics: "Diagnostics"
    result_fingerprint: str
```

Reducers return `outputs`. Client projectors return `targeted_outputs`. A result may not ambiguously rely on tuple position to identify a client.

### 7.6 OperatorSpec

```python
@dataclass(frozen=True)
class OperatorSpec:
    name: str
    version: str
    config_schema_name: str
    config_schema_version: str
    cardinality: Cardinality
    accepted_input_representations: frozenset[RepresentationKind]
    produced_representations: frozenset[RepresentationKind]
    stateless: bool
    deterministic_scope: "DeterminismScope"
    exact: bool
    supports_heterogeneous_rank: bool
    requires_targets: bool
    commutative_mathematically: bool | None
    associative_mathematically: bool | None
    supports_streaming: bool
    requires_all_inputs: bool
    approximation: "ApproximationSpec | None"
    cost_model: "CostModelSpec"
```

`commutative_mathematically` does not claim floating-point bitwise equivalence across input orders.

---

## 8. Reference Operators

### 8.1 IdentityOp

- Cardinality: `1:1`
- Purpose: codec, request, result, and pipeline boundary testing.
- Exact: yes.
- Output isolation behavior is explicit: safe copy by default; verified read-only passthrough may be offered separately.

### 8.2 WeightedAverageReducer

- Cardinality: `N:1`
- Input: same-shape compatible module updates.
- Output: representation-preserving weighted average where mathematically defined.
- Rejects heterogeneous factor shapes unless a preceding transform makes them compatible.
- Reports effective weights, accumulation dtype, input order, and numerical warnings.

### 8.3 ZeroPaddingReducer

- Cardinality: `N:1`
- Input: heterogeneous-rank factor updates.
- Behavior: pads absent factor slots with zero, then applies configured weighted reduction.
- Output rank: maximum valid input rank unless config introduces an explicit bound.
- Purpose: baseline and ablation, not an alias for RBLA.

### 8.4 RBLAReducer

- Cardinality: `N:1`
- Input: compatible LoRA factor modules with heterogeneous rank.
- Behavior: mask-aware weighted factor aggregation; missing padded positions do not dilute present factor positions.
- Output: noncanonical `FACTOR` representation with rank equal to the maximum valid client rank per module.
- Exactness: exact with respect to the declared RBLA factor-space definition, not necessarily equivalent to averaging dense products.
- Reports per-module input ranks, output rank, valid weight mass per factor position, and excluded inputs.

### 8.5 ProductAverageReducer

- Cardinality: `N:1`
- Input: factor or product updates.
- Behavior: reconstructs each scaled dense update, aggregates in product space, emits `PRODUCT`.
- Purpose: SP-style baseline and a reference against which factor-space methods can be compared.
- Cost model must expose potentially much larger dense intermediate memory and communication size.

### 8.6 CanonicalizeLoRAFactors

- Cardinality: `1:1`
- Input: `FACTOR` or supported `SPECTRAL` modules.
- Output: `COMPACT_FACTOR` with deterministic ordered factor slots.
- Primary method: stable QR/SVD-based canonicalization.
- Scheduling: controlled by typed config using explicit round context.
- When schedule says “do not run,” output remains semantically explicit; either representation remains `FACTOR`, or the operator returns an identity diagnostic. It must not falsely label noncanonical values as compact.
- Singular-value ordering defines prefix semantics for later client projection.
- Activation-aware ordering requires typed activation summary input or state; it may not read hidden server objects.
- Reports singular values, effective rank, reconstruction error, factor balance error, sign choices, fallback usage, and execution cost.

Exactness rules:

- Without truncation and within numerical tolerance, reconstructed scaled product should be preserved.
- With truncation, `OperatorSpec.exact` is false and the result reports absolute and relative reconstruction error.
- Fallback changes are recorded but must preserve the declared output contract.

### 8.7 ClientRankProjector

- Cardinality: `1:N`
- Input: one canonical compact-factor or spectral global update.
- Targets: one or more `AdapterTarget` values.
- Output: one `TargetedUpdate` per target ID.
- Prefix projection is valid only for an ordered canonical representation.
- A noncanonical factor input is rejected unless config explicitly selects a separately defined noncanonical projection rule.
- Zero-padding when a target rank exceeds available global rank is explicit and diagnostic.
- Reports requested rank, delivered rank, discarded spectral energy, added zero slots, dtype conversion, and output byte estimate per target.

### 8.8 RBLA+ pipeline

RBLA+ is an identity-bearing pipeline, not a hidden flag on RBLA:

```text
PipelineSpec(name="rbla_plus", version="1.0.0")

Stage 1: RBLAReducer(RBLAConfig(...))
Stage 2: CanonicalizeLoRAFactors(CanonicalizationConfig(...))
Stage 3: ClientRankProjector(ProjectionConfig(...))
```

The pipeline exposes both the global canonical update after Stage 2 and client-targeted outputs after Stage 3 when requested by Runtime. This avoids recomputing canonicalization separately for every client.

RBLA and RBLA+ remain distinct experiment identities even when canonicalization scheduling skips a particular round.

---

## 9. Pipeline Composition

### 9.1 PipelineSpec

```python
@dataclass(frozen=True)
class PipelineStageSpec:
    stage_id: str
    operator_name: str
    operator_version: str
    config_fingerprint: str
    input_binding: "InputBinding"
    target_binding: "TargetBinding"

@dataclass(frozen=True)
class PipelineSpec:
    name: str
    version: str
    stages: tuple[PipelineStageSpec, ...]
    exposed_outputs: tuple["OutputBinding", ...]
```

### 9.2 Compile-time validation

Before execution, a pipeline compiler verifies:

- cardinality compatibility;
- representation compatibility;
- required target availability;
- config and operator version availability;
- state namespace uniqueness;
- exact/lossy transition visibility;
- deterministic-scope compatibility;
- output binding uniqueness;
- whether target projection is attempted on a representation with valid prefix semantics.

### 9.3 Phase scope

- Phase 1 supports a linear pipeline with named exposed outputs.
- Phase 2 may add a typed DAG when a demonstrated use case requires branching.
- Pipeline fusion is a Runtime optimization. It must be proven equivalent to the unfused reference stages within declared tolerance.

---

## 10. State, Replay, and Transaction Semantics

### 10.1 OperatorState

```python
@dataclass(frozen=True)
class OperatorState:
    schema_name: str
    schema_version: str
    operator_name: str
    operator_version: str
    revision: int
    last_committed_round: int
    data: ImmutableMapping[str, StateValue]
    state_fingerprint: str
```

State is a value. Core defines validation, migration, fingerprints, and serialization semantics. Core does not decide where or when a checkpoint becomes committed.

### 10.2 Why round-index replay detection is insufficient

Comparing only `state.last_committed_round` with `context.round_index` cannot distinguish:

- a first execution whose result has not been committed;
- a retry after computation but before commit acknowledgement;
- a retry using the same pre-state;
- a duplicate request using a different input;
- an incorrect attempt to skip a round.

Therefore, Operator code must not implement exactly-once behavior by returning a cached result solely when round indices match.

### 10.3 Runtime transaction protocol

Runtime owns the following conceptual transaction:

```text
1. Build request with invocation_id, input fingerprints, and pre-state fingerprint.
2. Validate request.
3. Check durable invocation journal.
4. If an identical invocation is committed, return its committed result.
5. Otherwise execute the pure operator.
6. Atomically commit:
     invocation record
     result artifact/fingerprint
     new state artifact/fingerprint
7. Publish the committed result to downstream stages.
```

A duplicate `invocation_id` with different inputs, configuration, targets, or pre-state is a conflict and must fail.

### 10.4 Pipeline state

Pipeline state is namespaced by stage ID:

```text
pipeline_state
  ├── rbla_reducer → state or null
  ├── canonicalizer → state or null
  └── projector → state or null
```

The entire pipeline transition is committed atomically when Runtime offers pipeline-level transactions. Otherwise each stage commit is explicit and recoverable by binding fingerprints.

### 10.5 Artifact format

The canonical checkpoint artifact contains:

- a JSON or equivalent canonical manifest;
- tensor payloads in safetensors or another non-executable tensor format;
- schema names and versions;
- operator and config versions;
- content hashes for every payload;
- environment fingerprint where determinism scope requires it;
- parent/pre-state fingerprint;
- result or state fingerprint.

Pickle is not a portable or trusted checkpoint contract.

---

## 11. Determinism

### 11.1 DeterminismScope

```python
class DeterminismScope(Enum):
    MATHEMATICAL = "mathematical"
    SAME_SOFTWARE = "same_software"
    SAME_HARDWARE = "same_hardware"
    BITWISE_SAME_ENVIRONMENT = "bitwise_same_environment"
```

Operators declare the strongest scope they support. “Given a seed” alone is not a sufficient determinism definition.

### 11.2 Environment fingerprint

For bitwise or same-environment claims, Runtime records at least:

- Python version;
- PyTorch version;
- CUDA and cuDNN versions;
- device model and capability;
- deterministic algorithm flags;
- operator and codec versions;
- relevant numeric backend configuration.

### 11.3 RNG derivation

Any stochastic operator derives a substream from:

```text
hash(run_id, invocation_id, operator_name, stage_id, seed)
```

It does not mutate process-global RNG state. Reference RBLA+ operators should be deterministic and normally require no stochastic sampling.

---

## 12. Validation and Error Policy

### 12.1 ValidationReport

```python
@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Literal["error", "warning"]
    scope: Literal["request", "input", "module", "target", "state"]
    source_id: str | None
    module_path: str | None
    message: str
    details: ImmutableMapping[str, ScalarValue]

@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]
    valid_input_ids: tuple[str, ...]
    invalid_input_ids: tuple[str, ...]
    request_fingerprint: str
```

### 12.2 Separation of concerns

- Operator validation discovers and classifies issues.
- Runtime policy decides `fail_all`, `exclude_invalid`, or a declared quorum behavior.
- Runtime constructs a new validated request after exclusions.
- Operator execution does not silently drop an input.
- Diagnostics preserve the original validation report and effective input set.

### 12.3 Error hierarchy

```text
FlexCoreError
├── SchemaError
│   ├── UnsupportedSchemaVersionError
│   └── SchemaMigrationError
├── ConfigurationError
├── ValidationError
│   ├── ShapeMismatchError
│   ├── MissingTensorRoleError
│   ├── InvalidWeightError
│   ├── RepresentationMismatchError
│   ├── IncompatibleBaseModelError
│   └── InvalidTargetError
├── StateError
│   ├── StateVersionMismatchError
│   └── StateIntegrityError
├── CodecError
│   ├── UnsupportedExternalFormatError
│   └── RoundTripError
├── NumericalError
└── ResourceEstimateError
```

Unexpected internal invariant violations raise exceptions. Recoverable numerical conditions may produce warnings only when the operator contract defines a valid fallback.

---

## 13. Diagnostics and Cost Model

### 13.1 Diagnostics

```python
@dataclass(frozen=True)
class Diagnostics:
    operator_name: str
    operator_version: str
    invocation_id: str
    validation: ValidationReport
    input_count: int
    effective_input_count: int
    output_count: int
    effective_weights: ImmutableMapping[str, float]
    module_diagnostics: ImmutableMapping[str, "ModuleDiagnostics"]
    numerical_warnings: tuple["DiagnosticWarning", ...]
    approximation: "ApproximationReport | None"
    measured_cost: "MeasuredCost | None"
    state_transition: "StateTransition | None"
```

Canonicalization module diagnostics include:

- input and output rank;
- singular-value spectrum summary;
- effective rank;
- maximum and relative reconstruction error;
- factor balance error;
- ordering method actually applied;
- activation-aware fallback status;
- deterministic sign decisions;
- decomposition fallback status.

Projection diagnostics include requested/delivered rank and discarded spectral energy per target.

### 13.2 Serializable cost estimates

Cost metadata is data, not a callable:

```python
@dataclass(frozen=True)
class CostEstimate:
    flops: int | None
    input_bytes: int
    temporary_bytes: int | None
    output_bytes: int | None
    decomposition_count: int
    communication_bytes_by_target: ImmutableMapping[str, int]
    confidence: Literal["exact", "upper_bound", "estimated", "unknown"]
    assumptions: tuple[str, ...]
```

An operator exposes a deterministic estimator method that returns this schema. The estimator itself is not embedded in serialized `OperatorSpec`.

### 13.3 MeasuredCost

Runtime may enrich results with:

```python
@dataclass(frozen=True)
class MeasuredCost:
    wall_time_ms: float
    device_time_ms: float | None
    peak_host_memory_bytes: int | None
    peak_device_memory_bytes: int | None
    input_bytes: int
    output_bytes: int
    upload_bytes: int | None
    download_bytes: int | None
```

Core defines the schema; Runtime performs measurement. Task accuracy, loss, convergence AUC, and time-to-accuracy remain `flex-lab` metrics.

---

## 14. Codec and Integration Boundaries

### 14.1 Codec protocol

```python
class Codec(ABC):
    @property
    def spec(self) -> "CodecSpec": ...

    def encode(
        self,
        external: ExternalPEFTState,
        source: SourceMetadata,
    ) -> AdapterUpdate: ...

    def decode(self, update: AdapterUpdate) -> ExternalPEFTState: ...

    def validate_round_trip(self, external: ExternalPEFTState) -> ValidationReport: ...
```

Concrete HuggingFace/FLEX codecs live in `flex-integrations`. Test codecs may live under Core tests.

### 14.2 Base-model parameters

Core IR represents adapter updates, not a complete model state. A codec must declare how non-adapter entries are handled:

- rejected as out of scope;
- preserved in an integration-owned side channel;
- or handled by a separate generic-delta codec.

It must not silently discard unknown external keys while claiming full-state round-trip equality.

### 14.3 Naming and pairing

The codec is responsible for:

- parsing external tensor keys;
- normalizing module and adapter identities;
- pairing A/B roles;
- determining scaling semantics;
- detecting missing or ambiguous adapter names;
- reconstructing the external key format on decode;
- recording unsupported external metadata.

Core operators act only on normalized semantic identities.

### 14.4 Integration adapter

An integration adapter owns lifecycle translation:

```text
framework client result
  → SourceMetadata + external adapter state
  → Codec.encode
  → OperatorRequest
  → Runtime execution
  → Codec.decode targeted result
  → framework broadcast/update object
```

It also maps framework failure lists, sample counts, and client IDs without exposing framework classes to Core.

---

## 15. Registry and Versioning

### 15.1 Registry key

Operators, configs, codecs, and state schemas use stable compound identities:

```text
(namespace, name, semantic_version, schema_version)
```

### 15.2 Registry behavior

- Direct construction is sufficient in the first minimal implementation.
- Before external plugins or declarative pipelines are supported, a registry becomes mandatory.
- Registration rejects duplicate identities.
- Lookup never silently selects a different major version.
- Pipeline manifests pin exact operator and configuration versions.
- Migrations are explicit functions between known schema versions.

### 15.3 Compatibility

- Patch version: bug fix preserving declared semantics and compatibility.
- Minor version: backward-compatible capability addition.
- Major version: mathematical or contract change.
- A change in numerical ordering, normalization, scaling, truncation, or fallback semantics requires at least a new operator minor version and may require a major version.

---

## 16. Flex-Lab Experiment Design Boundary

The current and planned verification work maps to `flex-lab`, not Core.

### 16.1 ExperimentSpec

A future lab-level design should define a declarative experiment value containing:

```text
experiment identity
dataset and dataset version
data-distribution generator and seed
Dirichlet alpha or other heterogeneity parameters
client count and participation policy
rank-assignment generator, target correlation, realized correlation, and seed
model and adapter configuration
optimizer, learning rate, momentum, and local epochs
operator/pipeline identity and config fingerprint
training seeds
round count
runtime backend and resource request
metrics and output artifact location
```

### 16.2 Mapping recent verification experiments

| Experiment concern | Owning layer |
|---|---|
| MNIST/KMNIST/FMNIST/QMNIST loading | `flex-lab` + integration loader |
| Dirichlet α = 0.4/0.8 generation | `flex-lab` |
| Rank Spearman targets +0.5/0/−0.5 | `flex-lab` |
| SGD lr=0.01, momentum=0 | training integration / `flex-lab` |
| RBLA, RBLA+, SP, ZeroPadding identity | `flex-core` operator/pipeline specs |
| 96-run matrix expansion | `flex-lab` |
| AWS batch enumeration | `flex-lab` launcher + `flex-runtime` |
| Accuracy, AUC, confidence intervals | `flex-lab` |
| SVD error and effective rank | `flex-core` diagnostics |
| SVD time and memory | Runtime measurement using Core schema |

### 16.3 Reproducibility manifest

Each lab run records:

- the canonical `ExperimentSpec` fingerprint;
- generated distribution matrix and its fingerprint;
- realized rank list and Spearman correlation;
- Core pipeline manifest and versions;
- all training and data seeds;
- environment fingerprint;
- source revision;
- result artifact fingerprints;
- completion/failure status.

Generated YAML may remain an integration format, but it is not the semantic source of truth unless it validates against the lab schema.

---

## 17. Migration from Existing FLEX

Migration is incremental. Existing experiment execution must remain available until parity tests pass.

### Phase 0: Contract fixtures

- Capture representative existing RBLA, ZeroPadding, SP, and SP+/RBLA+ inputs and outputs.
- Capture heterogeneous ranks, raw sample weights, base tensor behavior, and broadcast shapes.
- Freeze golden fixtures with environment and source revision metadata.
- Document whether existing behavior is intentional semantics or an implementation accident.

### Phase 1: Core foundation

- Create package and immutable IR.
- Add typed config and validation contracts.
- Add operator request/result/spec/context values.
- Add IdentityOp and WeightedAverageReducer.
- Add manifest serialization and a test codec.
- Prove input storage invariance and codec round-trip boundaries.

### Phase 2: Heterogeneous-rank reference operators

- Implement ZeroPaddingReducer.
- Implement RBLAReducer.
- Implement ProductAverageReducer.
- Implement CanonicalizeLoRAFactors.
- Implement ClientRankProjector.
- Compile and test the RBLA+ linear pipeline.
- Compare against golden fixtures and mathematical reference calculations.

### Phase 3: Native FLEX integration

- Add a FLEX codec and lifecycle adapter outside Core.
- Introduce `LegacyAggregatorAdapter` only as a temporary bridge.
- Route one opt-in experiment path through Runtime and Core.
- Keep old and new paths side-by-side for differential testing.
- Remove server-strategy SVD and broadcast special cases after parity is demonstrated.

### Phase 4: Runtime reliability

- Add invocation journal and atomic result/state commit.
- Add pipeline checkpoints and recovery.
- Add structured resource measurements.
- Add validated input exclusion/quorum policies.
- Add placement and streaming optimizations behind equivalence tests.

### Phase 5: Lab migration

- Define `ExperimentSpec` and matrix-expansion schema.
- Migrate generated verification YAMLs to lab-owned manifests.
- Add seed matrices and statistical aggregation.
- Add AWS launcher adapter using the same canonical experiment manifest.

### Phase 6: External integrations

- Add HuggingFace PEFT integration codec.
- Add Flower lifecycle adapter.
- Add NVFlare lifecycle adapter.
- Verify the same Core pipeline and fixtures across integrations.

---

## 18. Testing Strategy

### 18.1 IR tests

- Schema validation for every representation and required role set.
- Mixed-module representations without ambiguity.
- Tensor shape, dtype, scaling, rank, and dimension validation.
- Immutable mapping behavior.
- Input tensor checksum and storage-alias tests.
- Stable update fingerprint under canonical serialization.
- Different content produces different fingerprints.

### 18.2 Operator contract tests

Every reference operator must test:

- known-input mathematical correctness;
- empty input and single input;
- heterogeneous rank where supported;
- rejection where heterogeneous rank is unsupported;
- raw and normalized weights;
- zero, negative, missing, infinite, and NaN weights;
- float32, float64, bfloat16, and declared accumulation dtype;
- CPU and CUDA where available;
- determinism within declared scope;
- input immutability and output storage isolation;
- diagnostics completeness;
- cost estimate consistency;
- schema serialization and restoration;
- state transition integrity when stateful.

### 18.3 Canonicalization properties

- `B_out @ A_out` preserves the configured scaled product within tolerance when no truncation occurs.
- Canonical slot order follows descending singular values for singular-value ordering.
- Deterministic sign is stable across repeated executions.
- Prefix projection to rank `r` uses the first `r` canonical components.
- Reconstruction error is monotonic nonincreasing with increasing retained rank within numerical tolerance.
- Activation-aware fallback is explicit.
- Degenerate and repeated singular values follow documented deterministic tie behavior.

### 18.4 RBLA+ pipeline tests

- Differential comparison with the current SP+/RBLA+ computation on frozen fixtures.
- Correct targeted shapes for clients with ranks 1–10.
- No target output is identified by tuple position alone.
- Changing one target rank changes only that target's projected output.
- Global canonical output is computed once per pipeline invocation.
- Stage diagnostics and aggregate pipeline diagnostics remain attributable.
- Scheduled canonicalization skips never mislabel noncanonical factors.

### 18.5 Runtime contract tests

Runtime, outside Core, must test:

- duplicate identical invocation returns the committed result;
- duplicate invocation with changed input is rejected;
- crash before commit does not expose partial state;
- crash after commit is recoverable without recomputation visibility;
- pipeline stage fingerprints bind the correct predecessor result;
- state and result corruption is detected.

### 18.6 Integration and lab tests

- External codec round-trip for adapter-only state.
- Explicit handling of unknown/non-adapter keys.
- Same Core fixture produces equivalent results through native FLEX and external integration paths.
- Experiment matrix expansion produces the expected unique count.
- Every referenced dataset, distribution, rank, optimizer, operator, and runtime config exists.
- Distribution totals and rank-correlation realizations match manifest declarations.

---

## 19. Acceptance Criteria

### 19.1 Minimal Core acceptance

1. `flex-core` installs independently of the existing FLEX package.
2. Core contains zero imports from FLEX, Flower, NVFlare, HuggingFace Trainer, dataset libraries, or experiment launchers.
3. IR values serialize, restore, validate, and preserve fingerprints.
4. IdentityOp and WeightedAverageReducer satisfy all applicable operator contract tests.
5. A test codec completes an adapter-only encode/operator/decode round trip.
6. No operator mutates or aliases writable caller-owned tensor storage.

### 19.2 Heterogeneous-rank acceptance

1. ZeroPaddingReducer, RBLAReducer, CanonicalizeLoRAFactors, ProductAverageReducer, and ClientRankProjector have independent specs and tests.
2. The RBLA+ pipeline is declared as three stages and passes frozen-fixture differential tests.
3. One invocation produces correctly keyed outputs for multiple target ranks.
4. Canonicalization reports reconstruction, rank, ordering, and fallback diagnostics.
5. Cost estimates distinguish factor-space and product-space memory/output size.

### 19.3 Integration acceptance

1. Existing FLEX can opt into the Core path without changing legacy experiments by default.
2. Legacy and Core paths can run on identical frozen client updates.
3. Mathematical differences are either within declared tolerance or documented as intentional corrected semantics.
4. Server and client strategies no longer contain method-specific SVD or heterogeneous-rank slicing after migration completes.

### 19.4 Experiment-system acceptance

1. Dataset, Dirichlet, rank assignment, optimizer, seed, and cloud-launch configuration live outside Core.
2. Experiment manifests pin the exact Core pipeline and config fingerprints.
3. Generated experiment count, distribution totals, rank correlations, and referenced files are validated before launch.
4. Results contain enough provenance to reproduce a run in the declared determinism scope.

---

## 20. Deferred Decisions

The following remain deliberately deferred until reference implementations and measurements exist:

1. General DAG pipelines beyond named linear stages.
2. A tensor backend abstraction beyond PyTorch.
3. Distributed/streaming reduction protocols and numerical order guarantees.
4. Quantized adapter representations.
5. Secure aggregation and privacy-preserving diagnostics.
6. Cross-device automatic placement and offload planning.
7. Custom binary artifact formats beyond manifest plus safetensors.
8. Plugin marketplace and remote operator distribution.
9. General support for every PEFT family; V2 structures the IR to permit extension but initially validates LoRA semantics.

Deferred features may not weaken current invariants or introduce untyped escape hatches into public contracts.

---

## 21. First Design-Conformant Delivery

The first implementation PR should create only the minimal closed loop:

```text
Adapter-only external fixture
  → test Codec
  → immutable AdapterUpdate
  → validated OperatorRequest
  → IdentityOp or WeightedAverageReducer
  → OperatorResult + Diagnostics
  → test Codec
  → restored external adapter state
```

It should not yet migrate RBLA, add an AWS launcher, import datasets, build a general DAG, or optimize GPU execution. Its purpose is to prove that the V2 boundaries, immutability rules, schema contracts, and test strategy are implementable before algorithm migration begins.

