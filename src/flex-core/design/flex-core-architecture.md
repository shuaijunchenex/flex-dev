# Flex-Core: Distributed PEFT Operator Substrate — Architecture Design

> **Author:** Architecture Review  
> **Date:** 2026-07-22  
> **Status:** Design Proposal (pre-implementation)  
> **Scope:** `flex-core` — framework-independent operator substrate for distributed PEFT

---

## Table of Contents

1. [Motivation & Long-Term Vision](#1-motivation--long-term-vision)
2. [Repository Findings](#2-repository-findings)
3. [Current Architectural Risks](#3-current-architectural-risks)
4. [Core Requirements & Invariants](#4-core-requirements--invariants)
5. [Candidate Designs & Recommendation](#5-candidate-designs--recommendation)
6. [Package Structure](#6-package-structure)
7. [Core IR (Intermediate Representation)](#7-core-ir-intermediate-representation)
8. [Operator Protocol](#8-operator-protocol)
9. [State & Replay Semantics](#9-state--replay-semantics)
10. [Capability & Cost Metadata](#10-capability--cost-metadata)
11. [Codec & Integration Boundaries](#11-codec--integration-boundaries)
12. [Diagnostics & Error Semantics](#12-diagnostics--error-semantics)
13. [Reference Operators (Phase 1)](#13-reference-operators-phase-1)
14. [Validation with Use Cases](#14-validation-with-use-cases)
15. [Incremental Migration Plan](#15-incremental-migration-plan)
16. [Testing Strategy](#16-testing-strategy)
17. [Open Design Decisions](#17-open-design-decisions)
18. [Recommended First PR](#18-recommended-first-pr)

---

## 1. Motivation & Long-Term Vision

### 1.1 Flex 的长期定位

> A framework-independent operator substrate for distributed PEFT.

Flex Core 是面向分布式 PEFT 的、与上层训练框架解耦的算子基础设施。它**不是**另一个包含完整客户端管理、训练循环、网络通信和任务调度的重型 FL 框架。

Flex Core 负责：

- PEFT update 的统一表示（IR）
- 聚合、变换、规范化、压缩、重构和投影
- Heterogeneous rank 的表达与处理
- Client-specific broadcast
- 算子状态管理
- 数学语义和近似误差描述
- 序列化、验证、诊断和可重放执行
- 为后续 Planner 提供可分析的算子元数据

### 1.2 长期分层

```
flex-integrations     (Flower, NVFlare, HuggingFace PEFT adapters)
flex-runtime          (execution graph, streaming, fusion, checkpoint/recovery)
flex-lab              (research algorithms, benchmarks, experiments)
        ↓
    flex-core          (IR, operator protocol, codec, state, diagnostics, reference ops)
```

**依赖方向：** `flex-core` 永不反向依赖 Flower、NVFlare、具体训练任务或论文实验代码。

### 1.3 当前阶段目标

验证最小闭环：

```
External PEFT state  →  Codec  →  Flex Core IR  →  Pure Operator  →  OperatorResult  →  Codec  →  External PEFT state
```

还需证明：

- 同一份输入可以在 Native 和外部框架适配层中执行
- 算子的数学语义不依赖具体训练框架
- 执行结果可以序列化、恢复和重放
- 算子可以声明自己的执行特性
- 后续可在不修改数学语义的情况下增加优化实现

---

## 2. Repository Findings

> 基于对 `src/flex/` 下 298 个 Python 文件、~36,600 行代码的全面审查。

### 2.1 当前 LoRA/Aggregation/Broadcast 调用链

```
experiment_entrency.py:main(config_path)
  → LoRAAppEntry.run(device)
    → FedRunner.create_nodes()           # 1×FedNodeServer + N×FedNodeClient
    → StrategyFactory.create_runner_strategy()
    → FedRunner.run()
      → RblaRunnerStrategy.run()         # synchronous round loop
        → server_node.broadcast()        # initial round-0 broadcast
        → [per round]:
          → server_node.select_clients()
          → simulate_client_local_training_process(participants)
            → client.strategy.run_local_training()
              → deepcopy(model).load_state_dict(node_var.model_weight)
              → optimizer_builder.rebuild(...)
              → trainer.train(epochs) → (updated_weights, train_record)
          → server_node.receive_client_updates(client_updates)
          → server_node.aggregation()
            → aggregator.aggregate(node_var.client_updates)
              → AbstractFedAggregator.aggregate(client_data_dict)
                → extract (sd["updated_weights"], sd["train_record"]["data_sample_num"])
                → _before_aggregation() → _do_aggregation() → _after_aggregation()
          → server_node.apply_weight()
            → node_var.model_weight = aggregated_weight
            → model_evaluator.update_model(prepared)
          → server_node.broadcast()
            → for client in client_nodes:
              → client.receive_weight(node_var.model_weight)
              → client.set_local_weight()
                → FedAggregator_RBLA.broadcast_lora_state_dict(cache, local)
                  → fit_tensor_to_local_shape()
          → server_node.evaluate() → record_evaluation()
```

**关键观察：** 数据在 client update → global weight → client broadcast 全链路中以**裸 `dict[str, Tensor]`** 流通。聚合和广播被硬编码耦合在策略类中，无序列化边界。

### 2.2 当前核心对象及问题

| 对象 | 位置 | 职责 | 问题 |
|------|------|------|------|
| `AbstractFedAggregator` | `fl_algorithms/aggregation/fed_aggregator_abc.py` | 聚合模板方法 + 数据提取 | 无类型 I/O；输入格式硬编码 `d["updated_weights"]` |
| `FedAggregator_RBLA` | `aggregation/methods/_fed_aggregator_rbla.py` | RBLA 聚合 + 广播 | 聚合和广播耦合在同一个类中；`broadcast_lora_state_dict` 是静态方法 |
| `FedAggregator_SP` | `aggregation/methods/_fed_aggregator_sp.py` | SP 聚合（ΔW 产品空间） | 输出格式与 RBLA 完全不同（`.sp_aggregated` vs `lora_A/lora_B`），但共享同一 ABC |
| `ServerStrategy` | `fed_strategy/server_strategy.py` | 聚合/广播/评估编排 | 混合编排逻辑与 SP 特化的权重格式转换（`_prepare_weight_for_model`） |
| `ClientStrategy` | `fed_strategy/client_strategy.py` | 本地训练 + 权重接收 | `receive_weight`/`set_local_weight` 紧耦合 broadcast 逻辑 |
| `FedNodeVars` | `fed_node/fed_node_vars.py` | 可变状态容器 | **God object**：持有 model、weights、optimizer、trainer、aggregator、selector、logger 等一切 |

### 2.3 Framework-Specific Logic 泄漏到 Core 的位置

| 位置 | 泄漏内容 | 严重度 |
|------|---------|--------|
| `ServerStrategy._prepare_weight_for_model` (`server_strategy.py:114-166`) | SP `.sp_aggregated` 检测 + SVD 分解 | **高** |
| `FedNodeVars` 构造函数 (`fed_node_vars.py:580-689`) | 导入并构建 trainer、aggregator、selector、evaluator 等全部服务 | **高** |
| `AbstractFedAggregator.aggregate` (`fed_aggregator_abc.py:68-76`) | 硬编码 `d["updated_weights"]` 和 `d["train_record"]["data_sample_num"]` | **中** |
| `FedAggregatorFactory` (`fed_aggregator_facotry.py:20-115`) | 封闭的 `match` 语句，新增方法需修改核心代码 | **中** |

### 2.4 隐式状态

| 位置 | 隐式状态 | 风险 |
|------|---------|------|
| `AbstractFedAggregator._aggregated_weight` | 缓存于实例变量 | 重入不安全；无法序列化 |
| `FedAggregator_RBLA._aggregation_round` | 轮次计数器 | 重放时计数错误 |
| `FedAggregator_RBLA._canonicalization_applied_last_round` | 上次 canonicalization 标记 | 外部不可感知 |
| `FedNodeVars` 所有属性 | `model_weight`, `cache_weight`, `aggregated_weight`, `client_updates` | 无 schema；无类型；无事务 |
| `BaseStrategy.__init__` 调用 `TrainingUtils.set_seed(42)` | 全局 RNG 污染 | 多实验不可共存 |

### 2.5 语义模糊的数据结构

- **`client_updates`**: 有时是 `list[dict]`，有时期望特定 key（`updated_weights`、`train_record`）。不同 runner 产生的 envelope 格式不一致。
- **`aggregated_weight`**: 可能是 `OrderedDict[str, Tensor]`（FedAvg）、带 `.sp_aggregated` 键（SP）、或 factored form（SP server）。
- **`model_weight`**: 可能是完整 state_dict、factored state_dict、或 SP 格式。

### 2.6 原地修改和 Aliasing 风险

| 位置 | 风险 |
|------|------|
| `FedAggregator_FedAvg._do_aggregation` | `new_weights` 原地累加；输入 tensor `.to(device)` 后可能共享存储 |
| `FedAggregator_RBLA.broadcast_lora_state_dict` | `fitted[common_slices] = global_tensor[common_slices]` 可能导致 aliasing |
| Client `local_training_step` | `deepcopy(model)` 后 `load_state_dict`，但 optimizer state 可能与原始 model 共享 |

### 2.7 无法重放或序列化的位置

- 聚合器的 `_aggregation_data_dict` 是运行时构建，不保存
- `_aggregation_round` 计数器不可外部设置
- Canonicalization 结果只在 `_after_aggregation` 中产生，无持久化
- 广播是直接方法调用，无序列化

### 2.8 当前测试缺口

- 无 operator pipeline 集成测试
- 无 round-trip serialization 测试
- 无 replay/determinism 测试
- 无状态迁移测试
- 无错误恢复测试
- 最完整的测试（`unittest_aggregator_rbla.py`）仅覆盖 RBLA 的数学正确性

### 2.9 未来最可能导致大规模重构的设计

1. **`dict[str, Tensor]` 作为通用协议** — 无法区分 factor-space vs product-space vs factored representations
2. **`FedNodeVars` God object** — 任何重构都需要修改它
3. **Strategy 层直接调用聚合/广播** — 无法插入 planner、无法替换通信后端
4. **封闭工厂** — 无法独立发布新算子
5. **隐式状态和全局 RNG** — 不可重放、不可并行

---

## 3. Current Architectural Risks

按严重程度排列：

| # | 风险 | 影响 | 缓解优先级 |
|---|------|------|-----------|
| 1 | `dict[str, Tensor]` 无类型协议 | 任何 IR 变更都是 breaking；无法静态验证 | **Phase 1** |
| 2 | 聚合与广播耦合在单一类中 | 无法独立替换广播策略 | **Phase 1** |
| 3 | `FedNodeVars` God object | 无法独立使用聚合算子 | **Phase 1** |
| 4 | 封闭工厂 | 新算法需修改框架代码 | **Phase 1** |
| 5 | 隐式状态 | 不可重放、不可检查点 | **Phase 2** |
| 6 | 无序列化协议 | 无法跨进程/跨框架 | **Phase 2** |
| 7 | Strategy 层混合编排与算子逻辑 | 无法独立演进 | **Phase 3** |

---

## 4. Core Requirements & Invariants

### 必须满足的不变量

1. **IR 自描述性**: 给定 `AdapterUpdate`，无需外部上下文即可判断表示形式、rank、dtype、device 和 PEFT 类型
2. **State 显式性**: 有状态算子的状态必须显式输入输出，不可隐藏在对象内部
3. **输入不可变性**: Operator 永不原地修改输入
4. **原子状态迁移**: 状态变更要么完全成功，要么完全不发生
5. **确定性**: 相同 `(inputs, context, state)` 必须产生相同 `(outputs, new_state)`
6. **框架无关性**: 核心算子不 import Flower、NVFlare 或 FLEX 策略类
7. **可重放性**: 任何执行结果可通过保存的 `(inputs, context, state)` 精确重放
8. **异构 Rank 是一等语义**: 不允许假设所有 client 同 rank

### 设计原则

**必须保证：**
- Core 与训练框架解耦
- 数学语义显式
- State 显式
- 输入不可被意外原地修改
- 失败不会留下部分 state transition
- Reference implementation 优先于早期优化
- 算子可独立测试
- 同一输入可重放
- 外部格式可以 round-trip
- Heterogeneous rank 是一等语义
- Diagnostics 是结构化结果

**避免：**
- God object
- 过深继承层次
- 为每种算法创建一套不兼容接口
- `dict[str, Any]` 作为长期核心协议
- 隐藏的 global/round state
- 在 Core 中引入 Flower/NVFlare 对象
- 算子直接打印日志作为唯一诊断
- 一个函数同时做 decode、aggregate、project、encode
- 在数学 contract 未稳定前做过早性能优化

---

## 5. Candidate Designs & Recommendation

### 方案 A：统一 Operator Protocol + Capability Metadata

```python
class Operator(Protocol):
    @property
    def spec(self) -> OperatorSpec: ...

    def apply(
        self,
        inputs: Sequence[AdapterUpdate],
        context: ExecutionContext,
        state: Optional[OperatorState],
    ) -> tuple[Sequence[AdapterUpdate], Optional[OperatorState], Diagnostics]: ...
```

**优点：** 单一协议；统一组合；Planner 友好；扩展成本低
**缺点：** 运行时需验证输入数量（1:1, N:1, 1:N）

### 方案 B：Typed Category Protocols（Reducer / Broadcaster / Transformer）

```python
class Reducer(Protocol):
    def reduce(self, inputs: Sequence[AdapterUpdate], ...) -> tuple[AdapterUpdate, ...]: ...

class Broadcaster(Protocol):
    def broadcast(self, input: AdapterUpdate, targets: Sequence[AdapterSpec], ...) -> Sequence[AdapterUpdate]: ...
```

**优点：** 类型更精确；意图更明确
**缺点：** 多协议增加概念负担；Pipeline 组合复杂；某些算子跨类别；可能出现分类学争论

### 对比

| 维度 | 方案 A | 方案 B |
|------|--------|--------|
| API 复杂度 | 低 — 单一协议 | 中 — 3+ 协议 |
| 类型安全 | 中 — 运行时验证 | 高 — 静态类型 |
| 扩展成本 | 低 | 中 |
| Pipeline composition | 简单 | 需适配器 |
| Planner compatibility | 高 | 中 |
| 过度设计风险 | 低 | 中-高 |

### ✅ 推荐：方案 A

**理由：**
1. 13 种聚合器 + 4 种广播模式本质上都是 `AdapterUpdate → AdapterUpdate` 变换
2. Capability metadata (`OperatorSpec.cardinality: "1:1"|"N:1"|"1:N"`) 表达 I/O 约束，无需类型系统编码
3. 统一协议使 Pipeline 组合极其简单
4. 参考 JAX `lax` 和 Apache Beam `PTransform` 的设计

---

## 6. Package Structure

```
src/flex-core/
├── pyproject.toml
├── README.md
├── src/flex_core/
│   ├── __init__.py
│   │
│   ├── ir/                           # Core IR — 稳定数据类型
│   │   ├── __init__.py
│   │   ├── adapter_spec.py           # AdapterSpec (frozen dataclass)
│   │   ├── adapter_tensor.py         # AdapterTensor (frozen dataclass)
│   │   ├── adapter_update.py         # AdapterUpdate (frozen dataclass)
│   │   ├── source_metadata.py        # SourceMetadata
│   │   └── representation.py         # RepresentationKind enum
│   │
│   ├── operator/                     # Operator protocol + metadata
│   │   ├── __init__.py
│   │   ├── protocol.py               # Operator ABC
│   │   ├── spec.py                   # OperatorSpec
│   │   ├── context.py                # ExecutionContext
│   │   ├── state.py                  # OperatorState, StateVersion
│   │   ├── pipeline.py               # Pipeline composition (Phase 2)
│   │   └── diagnostics.py            # Diagnostics, DiagnosticWarning
│   │
│   ├── codec/                        # Codec protocol (boundary)
│   │   ├── __init__.py
│   │   ├── protocol.py               # Codec ABC, CodecSpec
│   │   └── errors.py                 # CodecError, RoundTripError
│   │
│   ├── ops/                          # Reference operator implementations
│   │   ├── __init__.py
│   │   ├── identity.py               # IdentityOp
│   │   ├── weighted_average.py       # WeightedAverageReducer
│   │   ├── zero_padding.py           # ZeroPaddingReducer (Phase 2)
│   │   ├── rbla.py                   # RBLAReducer (Phase 2)
│   │   └── sp_projector.py           # SPProjector (Phase 2)
│   │
│   ├── errors.py                     # Error hierarchy
│   └── validation/                   # Validation utilities
│       ├── __init__.py
│       ├── shapes.py
│       └── determinism.py
│
└── tests/
    ├── test_ir/
    ├── test_operator/
    ├── test_codec/
    ├── test_ops/
    └── test_integration/
```

---

## 7. Core IR (Intermediate Representation)

### 7.1 RepresentationKind

```python
class RepresentationKind(Enum):
    """Adapter tensor 的物理表示形式"""
    FACTOR = "factor"             # LoRA A/B 因子 (r×in, out×r)
    PRODUCT = "product"           # ΔW = B@A (out×in)
    COMPACT_FACTOR = "compact"    # canonicalized QR+SVD form
    STACKED = "stacked"           # FLoRA stacking form
    SPECTRAL = "spectral"         # U, Σ, Vh decomposed
    DELTA = "delta"               # generic (non-LoRA-specific)
```

### 7.2 AdapterSpec

```python
@dataclass(frozen=True)
class AdapterSpec:
    """唯一标识一个 adapter tensor。不可变、可哈希、可序列化。"""
    
    # -- Identity --
    module_path: str              # "model.layers.0.self_attn.q_proj"
    adapter_name: str             # "default"
    peft_type: str                # "lora"
    tensor_role: str              # "lora_A" | "lora_B" | "scaling" | "bias"
    
    # -- Shape --
    shape: tuple[int, ...]        # (r, in_dim) or (out_dim, r)
    dtype: str                    # "float32", "bfloat16", ...
    
    # -- PEFT-specific --
    rank: int                     # LoRA rank r
    input_dim: int                # in_dim
    output_dim: int               # out_dim
    scaling: float = 1.0          # lora_alpha/r
    
    # -- Compatibility --
    base_model_fingerprint: str | None = None
```

**设计要点：**
- **不要**用 HuggingFace `state_dict` key（如 `base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight`）作为 Core 唯一语义——这只是 Codec 的外部表示
- `AdapterSpec` 只包含数学和结构信息，不包含框架特定命名
- `frozen=True` 保证不可变、可哈希、线程安全

### 7.3 AdapterTensor

```python
@dataclass(frozen=True)
class AdapterTensor:
    """带标识的 tensor。不可变——data 不应被原地修改。"""
    spec: AdapterSpec
    data: Tensor
    representation: RepresentationKind = RepresentationKind.FACTOR
    
    def __post_init__(self):
        if tuple(self.data.shape) != self.spec.shape:
            raise ValueError(
                f"Shape mismatch: spec={self.spec.shape}, data={self.data.shape}"
            )
```

### 7.4 AdapterUpdate

```python
@dataclass(frozen=True)
class AdapterUpdate:
    """一个客户端或全局的 PEFT update 集合。不可变。"""
    
    tensors: dict[str, AdapterTensor]    # key = "{module_path}.{tensor_role}"
    source: SourceMetadata
    representation: RepresentationKind = RepresentationKind.FACTOR
    
    def filter_role(self, role: str) -> "AdapterUpdate": ...
    def filter_prefix(self, prefix: str) -> "AdapterUpdate": ...
    def to(self, device) -> "AdapterUpdate": ...

@dataclass(frozen=True)
class SourceMetadata:
    """更新的来源信息——不属于数学语义但影响权重和重放"""
    source_id: str                  # client_id or "server"
    aggregation_weight: float      # normalized aggregation weight
    sample_count: int | None = None
    round_index: int | None = None
```

### 7.5 Update Identity 区分

需要明确区分以下不同概念（不允许都用 `dict[str, Tensor]` 表达）：

| 概念 | IR 类型 | 说明 |
|------|---------|------|
| Model parameter | (不属于 Core IR) | 基础模型权重，由 Codec 层管理 |
| Trainable adapter parameter | `AdapterUpdate(representation=FACTOR)` | 可训练的 LoRA A/B |
| Parameter delta | `AdapterUpdate(representation=DELTA)` | 通用增量 |
| Factor representation | `AdapterUpdate(representation=FACTOR)` | A/B 因子形式 |
| Reconstructed dense update | `AdapterUpdate(representation=PRODUCT)` | ΔW = B@A 的完整矩阵 |
| Aggregated update | `AdapterUpdate` (any representation) | 聚合后的全局更新 |
| Client-specific broadcast | `AdapterUpdate` (any representation) | 适配到特定 client rank 的更新 |

### 7.6 Metadata Boundary

以下信息**属于 Core IR**（影响数学语义或重放）：

- `rank`（影响聚合和广播的数学）
- `aggregation_weight`（影响加权平均结果）
- `representation`（影响算子选择）
- `round_index`（影响有状态算子的重放）

以下信息**仅属于 ExecutionContext**（不影响算子数学语义）：

- `seed`（随机种子）
- `config`（算子配置快照）

以下信息**仅由 Integration Layer 保留**（Core 不应感知）：

- Client ID（Core 用 `SourceMetadata.source_id` 作为不透明标识）
- Optimizer information
- Framework-specific objects（Flower `ClientProxy`, NVFlare `Shareable`）

---

## 8. Operator Protocol

### 8.1 Core Contract

```python
class Operator(ABC):
    """
    Flex Core 统一算子协议。
    
    Contract:
      (outputs, new_state, diagnostics) = operator.apply(inputs, context, state)
    
    Invariants:
      1. 永不原地修改 inputs
      2. 相同 (inputs, context, state) → 相同结果
      3. state 被完整替换（非原地更新）
      4. 所有运行时错误通过 diagnostics 报告
         （仅配置错误如 shape mismatch 抛异常）
    """
    
    @property
    @abstractmethod
    def spec(self) -> "OperatorSpec": ...
    
    @abstractmethod
    def apply(
        self,
        inputs: Sequence[AdapterUpdate],
        context: "ExecutionContext",
        state: Optional["OperatorState"],
    ) -> tuple[Sequence[AdapterUpdate], Optional["OperatorState"], "Diagnostics"]:
        ...
```

### 8.2 ExecutionContext

```python
@dataclass(frozen=True)
class ExecutionContext:
    """不可变的执行上下文。"""
    round_index: int
    seed: int                       # deterministic random seed
    config: frozendict              # immutable operator configuration snapshot
    tags: frozendict[str, str] = field(default_factory=frozendict)
```

### 8.3 OperatorSpec

```python
@dataclass(frozen=True)
class OperatorSpec:
    """算子的声明性元数据。供 Planner、Runtime 和验证器使用。"""
    
    # -- Identity --
    name: str                       # "flex_core.ops.rbla"
    version: str                    # "1.0.0"
    
    # -- I/O constraints --
    cardinality: str                # "1:1" | "N:1" | "1:N" | "N:M"
    input_representation: frozenset[RepresentationKind]
    output_representation: frozenset[RepresentationKind]
    
    # -- Capabilities (Phase 1 required) --
    stateless: bool
    deterministic: bool
    exact: bool
    supports_heterogeneous_rank: bool = False
    
    # -- Capabilities (Phase 2+) --
    commutative: bool | None = None
    associative: bool | None = None
    supports_streaming: bool = False
    requires_all_inputs: bool = True
    
    # -- Mathematical semantics --
    approximation_source: str | None = None  # "svd_truncation", "quantization"
    error_metric: str | None = None          # "frobenius_norm", "relative_error"
    
    # -- Resource estimators (Phase 2+) --
    peak_memory_per_input: Callable | None = None
    compute_cost_per_input: Callable | None = None
```

### 8.4 关键设计决策

| 问题 | 决策 | 理由 |
|------|------|------|
| 算子应是纯函数还是可持有状态？ | **纯函数风格** — 状态显式传入传出 | 可重放、可序列化、可检查点 |
| State 是否必须显式 I/O？ | **是** — `(state) → (new_state)` | 不允许隐藏在对象内部 |
| 如何区分 config 与 state？ | `ExecutionContext.config` (不可变配置) vs `OperatorState.data` (可变状态) | Config 随 round 固定；state 跨 round 演化 |
| Aggregation 和 Broadcast 是否都建模为 Operator？ | **是** — Aggregation = N:1；Broadcast = 1:N | Cardinality 在 OperatorSpec 中声明 |
| Validation 应在哪里执行？ | Operator 内（shape 验证）+ Runtime 可额外验证 | Operator 对自己最了解 |
| Operator 是否直接接收 `AdapterUpdate`？ | **是** — 类型化的 collection | 不接受原始 `dict` |

### 8.5 算子不应依赖的内容

- ❌ Flower 参数类型（`Parameters`, `FitRes`）
- ❌ Server/Client strategy 对象
- ❌ Trainer / DataLoader
- ❌ 网络通信对象
- ❌ 全局单例（`AppEntry.__app_objects`）
- ❌ 隐式当前轮次

---

## 9. State & Replay Semantics

### 9.1 State 定义

```python
@dataclass(frozen=True)
class OperatorState:
    """可序列化的算子状态。不可变——每次 apply 返回新实例。"""
    schema_version: str             # "1.0"
    operator_name: str              # matches OperatorSpec.name
    operator_version: str           # matches OperatorSpec.version
    round_index: int                # last execution round
    data: frozendict                # operator-specific immutable data
    checksum: str                   # data integrity hash
    
    def checkpoint(self, path: Path) -> None: ...
    
    @classmethod
    def from_checkpoint(cls, path: Path) -> "OperatorState": ...
```

### 9.2 防重复推进（Idempotency via round_index）

```python
def apply(self, inputs, context, state):
    if state is not None and state.round_index == context.round_index:
        # 重放请求 — 不推进状态
        return self._replay(inputs, context, state)
    
    # 正常执行 — 推进状态
    result, new_state = self._compute(inputs, context, state)
    return result, new_state, diagnostics
```

**场景：** 外部 runtime 因超时重试同一轮 Round 5，传入相同的 `context.round_index=5` 和 `state.round_index=5` → 算子检测到重复，返回缓存结果而不推进状态。

### 9.3 状态序列化格式

| Phase | 格式 | 用途 |
|-------|------|------|
| Phase 1 | JSON metadata + safetensors (tensors) | 可读、可调试 |
| Phase 2 | 自定义二进制格式 + 增量状态 | 性能优化 |

### 9.4 Schema Versioning

```
OperatorState.schema_version: "1.0"
  ↑
  Must be checked on deserialization.
  If version mismatch → StateVersionMismatchError
  Migration functions can be registered per (old_version, new_version) pair.
```

---

## 10. Capability & Cost Metadata

### 10.1 Phase 1 必须实现的能力

| Capability | Type | Purpose |
|-----------|------|---------|
| `cardinality` | `str` | Planner 验证 I/O 数量 |
| `stateless` | `bool` | Planner 判断是否需要 checkpoint |
| `deterministic` | `bool` | Runtime 判断是否可重放 |
| `input_representation` | `frozenset[RepresentationKind]` | Codec 选择编码 |
| `output_representation` | `frozenset[RepresentationKind]` | Codec 选择解码 |
| `exact` | `bool` | 下游算子可补偿近似误差 |

### 10.2 Phase 2+ 预留能力

| Capability | Purpose |
|-----------|---------|
| `commutative` | Planner 决定是否支持乱序/部分聚合 |
| `associative` | Planner 决定分层聚合策略 |
| `supports_streaming` | Runtime 决定执行模式 |
| `peak_memory_per_input` | Planner 决定 offload 策略 |

### 10.3 为什么不是简单的 class-level boolean

以 `commutative` 为例：
- `WeightedAverage` 在数学上可交换：`Σ w_i T_i = Σ w_{σ(i)} T_{σ(i)}`
- 但在浮点环境下，不同求和顺序产生不同 bitwise 结果
- 某些 `nanmean` 算子对 NaN 位置的处理依赖输入顺序

因此：
```python
commutative: bool | None = None
# True  = mathematically commutative
# False = order-sensitive
# None  = not applicable (non-reducer operator)
```

而 `deterministic` 声明 **bitwise** 确定性：
```python
deterministic: bool  # True = given seed, bitwise identical output
```

---

## 11. Codec & Integration Boundaries

### 11.1 Codec Protocol

```python
class Codec(ABC):
    """PEFT 格式编解码器。外部格式 ↔ AdapterUpdate。"""
    
    @property
    @abstractmethod
    def spec(self) -> "CodecSpec": ...
    
    @abstractmethod
    def encode(self, external: Any, source: SourceMetadata) -> AdapterUpdate:
        """External → Core IR"""
    
    @abstractmethod
    def decode(self, update: AdapterUpdate) -> Any:
        """Core IR → External"""
    
    def round_trip(self, external: Any, source: SourceMetadata) -> bool:
        """Verify encode→decode invariance"""

@dataclass(frozen=True)
class CodecSpec:
    name: str                       # "huggingface_peft_lora"
    version: str
    peft_type: str                  # "lora"
    supported_roles: frozenset[str] # {"lora_A", "lora_B", "scaling"}
```

### 11.2 职责边界

| 组件 | 职责 | 不负责 |
|------|------|--------|
| **Codec** | 解析外部命名 → `AdapterSpec`；识别 A/B/scaling；round-trip | 聚合算法；client selection |
| **Adapter** (integrations) | 框架生命周期集成（Flower `aggregate_fit` 等） | 算子数学语义 |
| **Runtime** | 执行调度、设备管理、pipeline | PEFT 命名约定 |

### 11.3 HuggingFace PEFT Codec 示例

```python
class HuggingFacePEFTCodec(Codec):
    """HF PEFT state_dict → AdapterUpdate → HF PEFT state_dict"""
    
    def encode(self, peft_state_dict: dict[str, Tensor], source) -> AdapterUpdate:
        """Parse HF keys into structured AdapterSpec objects"""
        tensors = {}
        for key, tensor in peft_state_dict.items():
            spec = self._parse_key(key)  # key → AdapterSpec
            tensors[f"{spec.module_path}.{spec.tensor_role}"] = AdapterTensor(spec, tensor)
        return AdapterUpdate(tensors=tensors, source=source)
    
    def decode(self, update: AdapterUpdate) -> dict[str, Tensor]:
        """Reconstruct HF key format from AdapterSpec"""
        result = {}
        for _, at in update.tensors.items():
            hf_key = self._format_key(at.spec)  # AdapterSpec → HF key
            result[hf_key] = at.data
        return result
```

### 11.4 Integration Adapter 示例（Flower）

```python
class FlowerOperatorStrategy(flwr.server.strategy.Strategy):
    """Bridge between Flower Strategy and flex-core Operator"""
    
    def __init__(self, operator: Operator, codec: Codec):
        self._op = operator
        self._codec = codec
        self._state: OperatorState | None = None
    
    def aggregate_fit(self, server_round, results, failures):
        # Flower FitRes → AdapterUpdate (via Codec)
        updates = [
            self._codec.encode(
                parameters_to_ndarrays(res.parameters),
                SourceMetadata(source_id=str(res.cid), 
                              aggregation_weight=res.num_examples,
                              round_index=server_round)
            ) for _, res in results
        ]
        # Core Operator
        outputs, self._state, diag = self._op.apply(
            updates,
            ExecutionContext(round_index=server_round, seed=42, config=frozendict()),
            self._state,
        )
        # AdapterUpdate → Flower Parameters
        return ndarrays_to_parameters(self._codec.decode(outputs[0]))
```

---

## 12. Diagnostics & Error Semantics

### 12.1 Diagnostics Schema

```python
@dataclass(frozen=True)
class Diagnostics:
    """Structured diagnostics — NOT log lines."""
    
    operator_name: str
    execution_time_ms: float
    peak_memory_bytes: int | None = None
    
    # Input validation
    inputs_received: int
    inputs_valid: int
    inputs_dropped: list[str] = field(default_factory=list)
    
    # Mathematical info
    rank_distribution: dict[str, int] = field(default_factory=dict)
    effective_weights: dict[str, float] = field(default_factory=dict)
    approximation_error: dict[str, float] = field(default_factory=dict)
    
    # Warnings
    warnings: list["DiagnosticWarning"] = field(default_factory=list)
    
    # State changes
    state_transition: "StateTransition | None" = None

@dataclass(frozen=True)
class DiagnosticWarning:
    level: str    # "numerical" | "approximation" | "resource" | "validation"
    message: str
    detail: dict = field(default_factory=dict)

@dataclass(frozen=True)
class StateTransition:
    state_before: str
    state_after: str
    changed_keys: frozenset[str]
```

### 12.2 Error Hierarchy

```python
class FlexCoreError(Exception): ...

class ConfigurationError(FlexCoreError): ...         # 应阻止执行
class UnsupportedRepresentationError(ConfigurationError): ...
class InvalidInputError(FlexCoreError): ...
class ShapeMismatchError(InvalidInputError): ...
class MissingTensorError(InvalidInputError): ...
class StateError(FlexCoreError): ...
class StateVersionMismatchError(StateError): ...
class StateCorruptionError(StateError): ...
class CodecError(FlexCoreError): ...
class RoundTripError(CodecError): ...
class ResourceExhaustionError(FlexCoreError): ...
```

### 12.3 错误分类与处理策略

| 错误类型 | 处理方式 | 示例 |
|---------|---------|------|
| Configuration error | 抛出异常，阻止执行 | 不支持的 representation |
| Invalid input (single) | 标记为 dropped，继续执行 | 某个 client 的 shape 不匹配 |
| Recoverable numerical warning | 记录 warning，继续执行 | 低秩近似误差 > 阈值 |
| Resource exhaustion | 抛出异常 | OOM |
| Internal invariant violation | 抛出异常（bug） | 算子内部状态不一致 |

**Core 不直接决定** server 是否跳过一轮——但它返回足够信息让 runtime 决策。

---

## 13. Reference Operators (Phase 1)

### 13.1 第一阶段只构建 5 个参考算子

| 算子 | Cardinality | 用途 |
|------|-------------|------|
| `IdentityOp` | 1:1 | Codec 和 runtime 测试基准 |
| `WeightedAverageReducer` | N:1 | 标准 FedAvg |
| `ZeroPaddingReducer` (Phase 2) | N:1 | 异构 rank 零填充 |
| `RBLAReducer` (Phase 2) | N:1 | RBLA NaN-padding 聚合 |
| `SPProjector` (Phase 2) | 1:1 | ΔW → SVD → factored |

### 13.2 每个 Reference Operator 必须满足

- 数学定义（docstring）
- Typed input/output（`AdapterUpdate`）
- Reference implementation（可读性优先于性能）
- Validation（shape/dtype/rank 检查）
- Deterministic test（相同输入 → bitwise 相同输出）
- Shape/rank property test
- Dtype/device test
- Serialization/replay test（如 stateful）
- Diagnostics 输出
- Complexity description
- Exactness/approximation declaration

### 13.3 不做的

- ❌ 不要一次移植所有 13 种聚合器
- ❌ RBLA、SP/SP+ 等暂不迁移（Phase 2）
- ❌ 不要为某一个算法特化 Core API

---

## 14. Validation with Use Cases

### Case A: Stateless Weighted Aggregation

```python
op = WeightedAverageReducer()

# 两个相同形状的 client update
u1 = make_update(specs, vals=[1.0, 2.0], client_id="c0", weight=10.0)
u2 = make_update(specs, vals=[3.0, 4.0], client_id="c1", weight=30.0)

outputs, state, diag = op.apply(
    [u1, u2],
    ExecutionContext(round_index=0, seed=42, config=frozendict()),
    state=None,  # stateless
)

# Verify: lora_A avg = (10*1.0 + 30*3.0) / 40 = 2.5
assert torch.allclose(outputs[0].tensors["layer.0.lora_A"].data,
                       torch.full((4, 64), 2.5))
assert state is None
assert not diag.has_errors
```

**证明：** 简单算子不会因抽象层过多而笨重（~10 行核心逻辑）。

### Case B: Heterogeneous-Rank LoRA Aggregation + Client Projection

```python
rbla = RBLAReducer(pad_mode="nan")

# 3 个不同 rank (2, 4, 8) 的 client
global_update, state, diag = rbla.apply([u1, u2, u3], ctx, state=None)
# global_update.tensors["layer.0.lora_A"].shape == (8, 64)  # max rank

assert diag.rank_distribution == {"layer.0": 8}
assert diag.supports_heterogeneous_rank

# 投影到 client 0 的 local rank=2
projector = SPProjector(target_rank=2)
local, _, _ = projector.apply([global_update], ctx, None)
assert local[0].tensors["layer.0.lora_A"].shape == (2, 64)
```

**证明：** IR 可表达异构 rank；Global → local projection 是独立算子。

### Case C: Stateful Adaptive Operator with Retry Safety

```python
op = AdaptiveThresholdReducer()

# Round 0
result1, state1, diag1 = op.apply(inputs, ExecutionContext(round_index=0, ...), None)
state1.checkpoint("/tmp/r0.state")

# Round 1
result2, state2, diag2 = op.apply(inputs, ExecutionContext(round_index=1, ...), state1)

# Simulated timeout retry: same context, same state
result2_retry, state2_retry, _ = op.apply(
    inputs,
    ExecutionContext(round_index=1, ...),  # same round
    state1,                                 # same pre-round state
)
# Operator detects: state.round_index == context.round_index → replay mode
# State is NOT advanced twice
assert len(state2_retry.data["threshold_history"]) == len(state1.data["threshold_history"]) + 1
```

**证明：** State 显式；Checkpoint/replay 可行；Retry 不会隐式推进状态。

---

## 15. Incremental Migration Plan

### Phase 1: Core Foundation（最小闭环）

| Step | Content | New/Modified Files |
|------|---------|-------------------|
| 1.1 | Create `src/flex-core/` package structure + `pyproject.toml` | New |
| 1.2 | Implement IR types | `flex_core/ir/*.py` |
| 1.3 | Implement Operator protocol + OperatorSpec + ExecutionContext | `flex_core/operator/*.py` |
| 1.4 | Implement Diagnostics + OperatorState | `flex_core/operator/*.py` |
| 1.5 | Implement `IdentityOp` + `WeightedAverageReducer` | `flex_core/ops/*.py` |
| 1.6 | Implement HF PEFT Codec | (in flex-integrations or test dir) |
| 1.7 | End-to-end test: HF state_dict → Codec → Operator → Codec → HF state_dict | `tests/test_integration/` |

**Phase 1 完成标志：**
```python
hf_sd = load_peft_model("path").state_dict()
ir = codec.encode(hf_sd, source)
result, _, _ = op.apply([ir], ctx, None)
restored = codec.decode(result[0])
assert torch.allclose(hf_sd["lora_A"], restored["lora_A"])
```

### Phase 2: Operator Expansion + State

| Step | Content |
|------|---------|
| 2.1 | Implement `ZeroPaddingReducer` |
| 2.2 | Implement `RBLAReducer` |
| 2.3 | Implement `SPProjector` |
| 2.4 | Implement `Pipeline` combinator |
| 2.5 | Native Adapter (wrap existing aggregators as Operators) |

### Phase 3: Current Codebase Integration

| Step | Content |
|------|---------|
| 3.1 | Add `flex-core` dependency to FLEX project |
| 3.2 | Add `flex_core_operator` option in `FedNodeVars` |
| 3.3 | Provide `LegacyAggregatorAdapter` |
| 3.4 | Trial new Operator in RBLA server strategy |
| 3.5 | Gradual migration of remaining strategies |

### Phase 4: External Framework Integration

| Step | Content |
|------|---------|
| 4.1 | Create `flex-integrations/flower/` |
| 4.2 | Implement Flower Strategy wrapper |
| 4.3 | Create `flex-integrations/nvflare/` |
| 4.4 | Implement NVFlare Aggregator wrapper |

---

## 16. Testing Strategy

### Test Matrix per Reference Operator

```
✅ Correctness       — 固定输入 → 已知输出
✅ Shape             — 不同 rank 组合
✅ dtype             — float32, bfloat16, float64
✅ device            — cpu, cuda
✅ Determinism       — 相同 seed → bitwise 相同
✅ Empty inputs      — 空列表处理
✅ Single input      — N:1 算子接收 1 client
✅ Diagnostics       — 结构完整、字段正确
✅ Serialize/replay  — stateful 算子 checkpoint/restore
✅ Input invariance  — 输入不被原地修改
```

### Test Directory Map

```
tests/
├── test_ir/              # AdapterSpec/Tensor/Update unit tests
├── test_operator/        # Operator protocol conformance, state checkpoint
├── test_codec/           # Codec round-trip tests
├── test_ops/             # Per-operator correctness + property tests
└── test_integration/     # End-to-end: Codec + Operator + Codec
```

---

## 17. Open Design Decisions

| # | Decision | Recommendation | Risk |
|---|----------|---------------|------|
| 1 | `AdapterSpec` frozen dataclass? | **Yes** — immutable, hashable, serializable | One extra copy on construction |
| 2 | Tensor storage: PyTorch-first? | **Yes** — defer storage backend abstraction | Changing later touches `AdapterTensor` |
| 3 | Pipeline: DAG or linear only? | **Linear only in Phase 1** | Complex orchestration not yet needed |
| 4 | Codec in flex-core or separate? | **Protocol in core; HF impl in tests/integrations** | Need clear boundary |
| 5 | State serialization format? | **Phase 1: JSON + safetensors; Phase 2: binary** | JSON inefficient for large tensors |
| 6 | Need OperatorRegistry? | **Phase 2** — Phase 1 uses direct imports | Plugin architecture important but not blocking |
| 7 | ExecutionPlan timeline? | **Phase 3** — after Operator protocol stable | Premature abstraction harmful |

---

## 18. Recommended First PR

### Goal

Establish flex-core's minimal existence: IR types + Operator protocol + 1 reference operator + Codec protocol + 1 test codec + end-to-end test.

### File List (all new)

```
src/flex-core/pyproject.toml
src/flex-core/README.md
src/flex-core/src/flex_core/__init__.py
src/flex-core/src/flex_core/ir/__init__.py
src/flex-core/src/flex_core/ir/adapter_spec.py
src/flex-core/src/flex_core/ir/adapter_tensor.py
src/flex-core/src/flex_core/ir/adapter_update.py
src/flex-core/src/flex_core/ir/representation.py
src/flex-core/src/flex_core/ir/source_metadata.py
src/flex-core/src/flex_core/operator/__init__.py
src/flex-core/src/flex_core/operator/protocol.py
src/flex-core/src/flex_core/operator/spec.py
src/flex-core/src/flex_core/operator/context.py
src/flex-core/src/flex_core/operator/diagnostics.py
src/flex-core/src/flex_core/codec/__init__.py
src/flex-core/src/flex_core/codec/protocol.py
src/flex-core/src/flex_core/ops/__init__.py
src/flex-core/src/flex_core/ops/weighted_average.py
src/flex-core/src/flex_core/errors.py
src/flex-core/tests/test_ir/test_adapter_spec.py
src/flex-core/tests/test_ir/test_adapter_update.py
src/flex-core/tests/test_ops/test_weighted_average.py
src/flex-core/tests/test_codec/test_hf_peft_codec.py
src/flex-core/tests/test_integration/test_round_trip.py
```

### Acceptance Criteria

1. ✅ `pip install -e src/flex-core` succeeds
2. ✅ `AdapterSpec`, `AdapterTensor`, `AdapterUpdate` constructable, JSON-serializable, deserializable
3. ✅ `WeightedAverageReducer` passes all test matrix items
4. ✅ HF PEFT codec passes round-trip test
5. ✅ End-to-end test runs < 100ms (excluding model loading)
6. ✅ **Zero** `from flex.` imports (core has no dependency on existing FLEX code)
