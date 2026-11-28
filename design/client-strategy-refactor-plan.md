# Client Strategy 架构重构计划

## 目标

1. **解耦 FL 与通用训练工具** — 提取 `TrainSession` 到 `ml_utils`，纯 LoRA 微调不引入 FL 框架
2. **消除子类重复代码** — 5 个 client strategy 共 ~710 行缩减到 ~180 行（-75%）
3. **统一架构** — 所有子类共享一套模板方法，差异点通过覆写 hook 实现

---

## 当前架构（问题）

```
fed_strategy/
├── client_strategy.py              ← FL + 训练工具混在一起
│   ├── receive_weight()            [FL]
│   ├── set_local_weight()          [FL]
│   ├── run_local_training()        [FL]  abstract
│   ├── run_observation()           [FL]  abstract
│   ├── observation_step()          [FL]  abstract
│   ├── local_training_step()       [FL]  abstract
│   ├── train_and_offload()         [通用] ← 和 FL 耦合
│   ├── offload_weights()           [通用] ← 和 FL 耦合
│   └── cleanup_training_resources()[通用] ← 和 FL 耦合
│
└── client_strategy_impl/
    ├── _fedavg_client.py           ~170 行，大量重复模板
    ├── _oort_client.py             ~130 行，同上
    ├── _rbla_client.py             ~130 行，同上（仅 set_local_weight 不同）
    ├── _sp_client.py               ~130 行，同上（仅 set_local_weight 不同）
    ├── _pyramidfl_client.py        ~150 行，同上（仅 epoch 逻辑不同）
    └── _client_selection_purpose_client.py  ~140 行，继承 FedAvg
```

**问题清单**:
- 做纯 LoRA 微调必须 import `ClientStrategy` → 引入整个 FL 框架
- 5 个子类的 `observation_step`/`local_training_step` 代码 ~80% 相同
- PyramidFL 的 epoch 逻辑散落在 50 行样板代码中，不易维护

---

## 目标架构

```
ml_utils/
├── gpu_memory_cleaner.py           ← 已有，不改动
├── model_utils.py                  ← 已有，不改动
└── train_session.py                ← 新建：通用训练会话（零 FL 依赖）

fed_strategy/
├── client_strategy.py              ← 重写：模板方法 + 组合 TrainSession
└── client_strategy_impl/
    ├── _fedavg_client.py           ~30 行（零覆写）
    ├── _oort_client.py             ~25 行（零覆写）
    ├── _rbla_client.py             ~35 行（覆写 set_local_weight）
    ├── _sp_client.py               ~35 行（覆写 set_local_weight）
    ├── _pyramidfl_client.py        ~55 行（覆写 _training_epochs + _enrich_train_record）
    └── _client_selection_purpose_client.py  ~110 行（覆写 _enrich_train_record）
```

---

## 详细步骤

### Step 1: 新建 `src/flex/ml_utils/train_session.py`

**目的**: 零 FL 依赖的通用训练会话，可独立用于 LoRA 微调。

**类名**: `TrainSession`

**属性**:

| 属性 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `offload_to_cpu` | `bool` | `True` | 全局 offload 开关 |
| `_auto_offload` | `bool` | `True` | 实例级覆盖 |

**方法**:

| 方法 | 签名 | 来源 | 说明 |
|------|------|------|------|
| `is_offload_enabled()` | `() -> bool` | 从 `ClientStrategy.is_offload_weights_to_cpu` 简化 | 不再读 `self._args` |
| `prepare()` | `(base_model, weights, optimizer_builder, device, ...) -> (model, optimizer)` | 新 | deepcopy + .to(device) + load_state_dict + rebuild optimizer + clear_all |
| `train_and_offload()` | `(trainer, epochs) -> (weights, record)` | 从 `ClientStrategy` 搬家 | trainer.train() + offload_weights() |
| `offload_weights()` | `(weights) -> weights` | 从 `ClientStrategy` 搬家 | `.detach().cpu()` 每个 tensor |
| `cleanup()` | `(model, optimizer, trainer) -> None` | 从 `ClientStrategy.cleanup_training_resources` 搬家 | 委托到 `GPUMemoryCleaner.cleanup_all` |
| `release_gpu()` | `() -> None` | 委托 | 委托到 `ModelUtils.release_gpu_memory` |

**使用示例**:

```python
from flex.ml_utils.train_session import TrainSession

session = TrainSession(offload_to_cpu=True)
model, optimizer = session.prepare(base_model, weights, optimizer_builder, device="cuda")
try:
    weights, record = session.train_and_offload(trainer, epochs=3)
finally:
    session.cleanup(model, optimizer, trainer)
```

---

### Step 2: 重写 `src/flex/fed_strategy/client_strategy.py`

**删除**: 以下 5 个方法（搬家到 `TrainSession`）

- `is_offload_weights_to_cpu()`
- `offload_weights()`
- `release_gpu_after_training()`
- `train_and_offload()`
- `cleanup_training_resources()`

**新增/改写**:

#### 2a. 组合 `TrainSession`

```python
class ClientStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self._strategy_type = "client"
        self._obj = None
        self._session = TrainSession(offload_to_cpu=True)
```

#### 2b. Hook 方法（子类可视需要覆写）

| Hook | 默认行为 | 覆写场景 |
|------|----------|----------|
| `_device()` | `getattr(node_vars, "device", "cpu")` | 极少 |
| `_training_epochs(cfg)` | `cfg["training"]["epochs"]` 或 1 | PyramidFL: adaptive_iter 逻辑 |
| `_observation_epochs(cfg)` | `cfg["training"]["local_epochs"]` 或 1 | 极少 |
| `_enrich_train_record(record)` | 原样返回 | PyramidFL: 注入 shard_keep_ratio; ClientSelectionPurpose: 注入 node_id 等 |
| `_on_training_done(weights, record)` | `node_vars.model_weight = weights` | 极少 |

#### 2c. 模板方法 `_prepare_and_train()`

```python
def _prepare_and_train(
    self, trainer, cfg, base_model, weights, optimizer_builder,
    epoch_hook, *, preserve_opt_state=False, restore_opt_state=False
) -> Tuple[dict, dict]:
```

**流程**:
```
1. _device()                              → 获取 device
2. _session.prepare()                      → deepcopy + optimizer
3. 保存 trainer 原始绑定 (orig_model, orig_optimizer, orig_device)
4. try:
     trainer.set_model(model)
     trainer.set_optimizer(optimizer)
     epoch_hook(cfg)                       → 子类定制的 epoch 数
     _session.train_and_offload(trainer)   → 训练 + offload
     snapshot_opt_state (if preserve)
   finally:
     恢复 trainer 原始绑定
     _session.cleanup(model, optimizer, trainer)
5. 返回 (weights, record)
```

#### 2d. Concrete 方法（不再 abstract）

```python
def observation_step(self) -> Tuple[dict, Any]:
    """调用 _prepare_and_train，epoch_hook=_observation_epochs"""
    ...

def local_training_step(self) -> Tuple[dict, Any]:
    """调用 _prepare_and_train，epoch_hook=_training_epochs，处理 preserve_opt_state"""
    ...

def run_observation(self) -> dict:
    """调用 observation_step + _enrich_train_record"""
    ...

def run_local_training(self) -> dict:
    """调用 local_training_step + _enrich_train_record"""
    ...

def receive_weight(self, global_weight):
    """self._obj.node_var.cache_weight = global_weight"""

def set_local_weight(self):
    """self._obj.node_var.model_weight = self._obj.node_var.cache_weight"""
```

**说明**: `observation_step`/`local_training_step`/`run_observation`/`run_local_training` 全部从 `abstractmethod` → `concrete`。

---

### Step 3: 简化子类

#### 3a. `_fedavg_client.py` — 170 行 → 30 行

**删除**: `run_observation`, `observation_step`, `run_local_training`, `local_training_step`, `receive_weight`, `set_local_weight`

**保留**: `__init__`, `_create_inner`

**改动理由**: FedAvg 是标准实现，所有默认行为已满足。

```python
class FedAvgClientTrainingStrategy(ClientStrategy):
    def __init__(self, args, client_node):
        super().__init__()
        self._args = args
        self._strategy_type = "fedavg"
        self._obj = client_node

    def _create_inner(self, args, client_node) -> None:
        self._args = args
        self._strategy_type = "fedavg"
        self._obj = client_node
```

#### 3b. `_oort_client.py` — 130 行 → 25 行

**完全同 FedAvg**。Oort 的 client 侧与 FedAvg 一致。

#### 3c. `_rbla_client.py` — 130 行 → 35 行

**删除**: `run_observation`, `observation_step`, `run_local_training`, `local_training_step`, `receive_weight`

**保留并覆写**:

```python
def set_local_weight(self):
    self._obj.node_var.model_weight = FedAggregator_RBLA.broadcast_lora_state_dict(
        self._obj.node_var.cache_weight, self._obj.node_var.model_weight
    )
```

#### 3d. `_sp_client.py` — 130 行 → 35 行

**同上**，覆写 `set_local_weight`（SP 有 LoRA 分解逻辑）。

#### 3e. `_pyramidfl_client.py` — 150 行 → 55 行

**覆写**:

```python
def _training_epochs(self, cfg: dict) -> int:
    """adaptive_iter + max_epoch_ratio 逻辑"""
    config_epochs = int(cfg.get("training", {}).get("epochs", 1))
    strategy_cfg = cfg.get("strategy", {})
    disable_adaptive = bool(strategy_cfg.get("disable_adaptive_epochs", False))
    max_epoch_ratio = strategy_cfg.get("max_epoch_ratio", None)
    
    adaptive_iter = self._pyramidfl_params.get("adaptive_iter", None)
    if (not disable_adaptive) and adaptive_iter is not None and adaptive_iter > 0:
        local_epochs = int(adaptive_iter)
    else:
        local_epochs = config_epochs
    
    if max_epoch_ratio is not None and float(max_epoch_ratio) > 0:
        max_epochs = int(float(max_epoch_ratio) * config_epochs)
        local_epochs = min(local_epochs, max(1, max_epochs))
    
    return local_epochs

def _enrich_train_record(self, record: dict) -> dict:
    """注入 shard_keep_ratio"""
    shard = self._pyramidfl_params.get("shard_keep_ratio", 1.0)
    return {**record, "shard_keep_ratio": shard}
```

**保留**: `__init__`, `_create_inner`, `receive_pyramidfl_params`（PyramidFL 特有）

**删除**: `run_observation`, `observation_step`, `run_local_training`, `local_training_step`, `receive_weight`, `set_local_weight`

#### 3f. `_client_selection_purpose_client.py` — 140 行 → 110 行

**覆写**: `_enrich_train_record` — 调用现有的 `_enrich_train_record` 自由函数

**删除**: `observation_step` 覆写（不再需要——基类 `observation_step` 自动调用 `_enrich_train_record`）

---

### Step 4: 不变的文件

| 文件 | 说明 |
|------|------|
| `ml_utils/gpu_memory_cleaner.py` | 不改动 |
| `ml_utils/model_utils.py` | 不改动 |
| `ml_utils/model_ewma.py` | 不改动 |
| `fed_strategy/server_strategy_impl/*` | 不改动 |
| `fed_strategy/runner_strategy_impl/*` | 不改动 |
| `sfl_strategy/*` | 不改动 |

---

## 改动量汇总

| 文件 | 操作 | 行数变化 |
|------|------|----------|
| `ml_utils/train_session.py` | **新建** | +80 |
| `fed_strategy/client_strategy.py` | 重写 | ~170 → ~180 |
| `client_strategy_impl/_fedavg_client.py` | 大幅删减 | ~170 → ~30 |
| `client_strategy_impl/_oort_client.py` | 大幅删减 | ~130 → ~25 |
| `client_strategy_impl/_rbla_client.py` | 大幅删减 | ~130 → ~35 |
| `client_strategy_impl/_sp_client.py` | 大幅删减 | ~130 → ~35 |
| `client_strategy_impl/_pyramidfl_client.py` | 大幅删减 | ~150 → ~55 |
| `client_strategy_impl/_client_selection_purpose_client.py` | 中等改动 | ~140 → ~110 |
| **总计** | | **-470 行** |

---

## 向后兼容

- `self.train_and_offload()` → 通过 `self._session.train_and_offload()` 仍可用
- `self.cleanup_training_resources()` → 通过 `self._session.cleanup()` 仍可用
- `self.offload_weights()` → 通过 `self._session.offload_weights()` 仍可用
- `self._auto_offload` → 仍在（`TrainSession._auto_offload`）
- `receive_weight` / `set_local_weight` → 从基类继承默认实现，RBLA/SP 覆写

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| `preserve_optimizer_state` 逻辑丢失 | `_prepare_and_train` 参数化 support |
| `_client_selection_purpose` 的 `observation_step` 覆写失效 | 逻辑移到 `_enrich_train_record` |
| RBLA/SP 的 `set_local_weight` 签名不匹配 | 基类接口统一为无参数返回 `dict` |
| SFL 策略受影响 | 不改动 SFL |
| 外部调用 `strategy.train_and_offload()` | 基类保留 proxy 方法向后兼容 |

---

## 实施顺序

1. Step 1 — 新建 `train_session.py`（无外部依赖，可独立测试）
2. Step 2 — 重写 `client_strategy.py`（依赖 Step 1）
3. Step 3a-3f — 简化子类（依赖 Step 2）
4. 全量运行现有测试确认无回归
5. 运行 RoBERTa+CoLA 验证 GPU 内存不再累积
