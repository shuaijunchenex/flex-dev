# RBLA+ 补充实验评估与执行方案

> 最新活动方案：[RBLA / 新版 RBLA+ 两路径四组合](rbla_plus_two_path_plan_2026-09-15_zh.md)。RBLA+ 已统一为带 RankScale 的版本，首批 6 次完整训练；下文及此前 9 次方案均为历史讨论。

> 已被用户后续要求缩减：当前活动方案见 [correct 单次补充实验方案](rbla_plus_correct_single_run_plan_2026-09-15_zh.md)。当前首批为 CIFAR-10、MNLI、QQP 三方法各一次，共 9 次；去掉多 seed、mismatch 与 FLoRIST。下文保留为历史方案。

日期：2026-09-15。状态：完成源码、配置、日志核查，以及现有 RankScale 单元测试；以下补充训练、checkpoint 诊断、服务器基准和 FLoRIST 适配尚未执行。本文是针对“是否必要、如何安排”的方案，不是新增实验结果。

## 1. 建议

建议做这四组实验，但先修正实验配对条件，再分阶段运行。推荐完整预算为 36 次主消融训练，加 6 次 FLoRIST 训练。真实 checkpoint 重构和服务器基准通常不需要额外训练。

| 项目 | 必要性 | 要回答的问题 | 不能据此推出的结论 |
|---|---|---|---|
| RBLA / 普通 RBLA+ / RankScale 配对消融 | 核心 | 谱重分发整体是否有益；增加双向 scale 对齐后有何变化 | 上传与下发各自贡献、协同作用、对所有任务都有效 |
| 原始 RBLA 真实因子重构 | 核心，成本较低 | 真实训练因子上固定前缀是否损失更多；compact 是否达到 dense 最优误差 | Frobenius 误差改善必然提高准确率 |
| 服务器耗时、峰值内存 | 如果论文保留效率主张，则必须 | 当前实现和真实模型尺寸上是否节省资源 | 理论倍率等于实测倍率；所有 dense 算法都一样慢 |
| FLoRIST 近邻比较 | 强烈建议，投稿前优先完成 | 与已有紧凑谱聚合方法相比，完整方法的精度和资源表现如何 | 结果差异全部来自 QR 与 SVD 的不同 |

FLoRIST 已正式发表于 MLSys 2026，其核心包含堆叠适配器的紧凑分解与阈值选 rank，因此与本工作直接相关，不能仅以已有 FLoRA 结果替代。来源：[正式论文页面](https://proceedings.mlsys.org/paper_files/paper/2026/hash/f38cb4cf9a5eaa92b3cfa481832719c6-Abstract-Conference.html)、[论文 PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/f38cb4cf9a5eaa92b3cfa481832719c6-Paper-Conference.pdf)、[官方仓库](https://github.com/DASS-Lab-Group/FLoRIST)。

## 2. 已核查的代码行为和实验问题

### 算法行为

- `src/flex/fl_algorithms/aggregation/methods/_fed_aggregator_rbla.py`：分别聚合 A、B；NaN padding 配合逐坐标有效支持的权重分母。因子平均的乘积一般不等于客户端更新矩阵的加权平均。
- `src/flex/fl_algorithms/aggregation/methods/_fed_aggregator_sp_plus.py`：继承 RBLA 聚合并启用 canonicalization。普通 RBLA+ 的差异主要在聚合后的谱因子重构和前缀下发；不会消除因子平均产生的交叉项。
- `src/flex/ml_algorithms/lora/canonicalization.py`：通过因子 QR 与小核 SVD 重构有序谱因子；需显式固定 `ordering: singular_value`、`overcomplete_policy: zero_pad`，并记录实际计算 dtype。
- `src/flex/ml_algorithms/lora/rank_scale_alignment.py` 及 RankScale server/client：读取实例化层的 scaling，对两个因子分别乘 `sqrt(source_scaling / target_scaling)`。上传与下发均转换；上传通常改变聚合输入和聚合结果。
- TinyBERT 当前工厂按 `max(1, round(8 * ratio))` 构造 rank，LoRA alpha 为 16，因此服务器 scaling 为 2，低 rank 客户端通常不同。
- VGG9 当前工厂路径中卷积和全连接 alpha 默认分别等于各自实际逻辑 rank，scaling 为 1。2026-09-15 按历史 SVHN ratio 实例化客户端模型，所有检查到的 LoRA 层 scaling 均为 1。
- MSLoRA 卷积 A/B 的矩阵化宽度为 `q = r * kernel_size`；3×3 卷积的矩阵尺寸为 `(C_out*3) × (C_in*3)`，不能直接将逻辑 r 当成矩阵截断宽度。

### 当前 correct/mismatch 并非相同 rank 集合的排列

现有 mismatch ratio 使用反向映射，不能直接作为新增预算匹配消融的配置。

| 任务 | 旧 correct | 旧 mismatch | 核查口径 |
|---|---:|---:|---|
| QQP | 客户端 rank 总和 52 | 客户端 rank 总和 48 | 从日志 ratio 和当前 TinyBERT 工厂公式推导；相同层布局下 LoRA A/B 参数数与 rank 成正比 |
| SVHN | 1,027,898 | 1,588,002 | 以历史日志 ratio 实例化当前 VGG9 工厂，累计 10 个客户端全部 LoRA A/B 参数；不含其他可训练参数 |

SVHN 后者比前者多约 54.5%。上述 VGG9 数字是当前工厂实例化核查，不是对历史 checkpoint 的参数清点。这会混淆 correct 与 mismatch 之间的比较，但不自动否定某个固定 assignment 内、配置一致的方法比较。

### 种子、轮数与保存机制

- `src/entries/lora.py`、`src/flex/fed_strategy/base_strategy.py`、`src/flex/model_trainer/model_trainer.py` 存在固定 seed 42 调用。只增加三份 YAML 或在外层设置三个 seed，不足以保证有效的三种随机实现。
- `src/entries/lora_reference_seeded.py` 可参考其种子注入和元数据记录方式，但它不能替代显式保存的配对对象。
- 三种现有 runner 均使用包含 0 的循环并在第 0 轮训练。配置 100 对应 101 次更新。新增协议建议使用初始化评估 t=0、真实训练更新 t=1…100；历史结果保持原样并标注旧口径。
- 当前配置每轮选择 10 个客户端，共 10 个客户端，实际是全参与。仍需固定客户端处理顺序、数据顺序和随机状态。
- 现有实验配置 `save_weights: false`；在此次检索中未发现可直接用于这组 TinyBERT/SVHN 原始 RBLA 诊断的聚合前谱处理 checkpoint。已有人工 gauge/MNIST 产物不能替代。
- 现有权重保存接口不构成完整恢复：还需要优化器、调度器、随机状态、轮次及客户端可变状态。
- `src/test/rbla+/server_memory/runtime_profiler.py` 的 CPU RSS 是开始/结束值及差值，不是 CPU 阶段峰值。已有分析估算也不能冒充实测。

### 已执行的轻量验证

运行了 `src/unittest_ml/unittest_rbla_plus_rank_scale_strategy.py`：8 passed，pytest 报告 8.05 秒。覆盖有效更新一致性、上传/下发转换、前缀、输入不变性、卷积逻辑 rank、梯度与注册等已有测试。这不等于完成了真实模型各层数值审计或训练收益实验。

### 最新日志说明为什么需要最终指标

数据来自 `design/rbla_plus_tables_update_sources_2026-09-15.json` 指向的最新完整 RankScale CSV，数值为百分比，日志末轮为 100。

| 任务与 assignment | 最高准确率 | 末轮准确率 |
|---|---:|---:|
| QQP correct | 82.48 | 79.28 |
| QQP mismatch | 81.02 | 76.16 |
| SVHN correct | 92.98 | 92.63 |
| SVHN mismatch | 93.20 | 92.18 |
| MNLI mismatch | 64.37 | 44.86 |
| SST2 correct | 80.62 | 70.87 |

这是已经观察到的曲线现象，不能仅凭它断言过拟合、scaling 错误或谱重分发导致退化。新实验不得据这些测试峰值选一个有利的截止轮数。QQP 的最终值也低于峰值，不能描述成没有退化。

## 3. 推荐主消融配置：QQP + SVHN

选择 QQP 是因为它提供实际 scale 不同的 TinyBERT 分类路径，且评估集较大；选择 SVHN 是因为它提供当前工厂 scaling=1 的卷积控制路径。任务选择用于覆盖机制，不以期望 RBLA+ 获胜为准。

| 设置 | QQP | SVHN |
|---|---|---|
| 模型 | `prajjwal1/bert-tiny`，现有 TinyBERT LoRA 工厂 | VGG9 LoRA，现有 SVHN 工厂 |
| 模型修订 | 固定本地权重和 tokenizer 文件哈希 | 固定模型构造代码与初始化张量 |
| LoRA 作用层 | 沿用现有全 Linear + embedding 配置，并导出真实逐层清单 | 沿用现有 6 个卷积 + 3 个全连接层 |
| 服务器 rank | 8；alpha 16，scaling 2 | 卷积 8，全连接 16；实际 scaling 1 |
| 客户端 rank | 使用旧 correct 的实际逐层 rank profile 作为固定池 | 使用旧 correct 的实际逐层 rank profile 作为固定池 |
| 客户端数 / 每轮参与 | 10 / 10 | 10 / 10 |
| 分区 | 沿用现有 Dirichlet 0.4 命名配置的类别数量矩阵，保存实际样本 ID | 同左 |
| 本地训练 | 1 epoch，batch 64 | 5 epochs，batch 64 |
| 优化器 | AdamW，lr=1e-4，weight_decay=0.05 | Adam，lr=1e-3，weight_decay=1e-4 |
| 精度 | 沿用 GLUE FP32 路径 | 沿用 BF16 AMP 路径，记录实际启用状态 |
| 服务器计算 | CPU，compact 计算先固定 FP32；诊断另用 FP64 | 同左 |
| 更新预算 | 100 次真实更新，t=0 只评估 | 同左 |
| assignments | correct / mismatched | correct / mismatched |
| seeds | 42、2026、3407 | 42、2026、3407 |
| 方法 | A 原始 RBLA；B 普通 RBLA+/SP+；C 双向 RankScale | 同左 |

这些是待实施的新协议。运行前需要从最终实例化对象导出实际参数，不能把上述拟定值当成已验证的历史运行配置。尤其优化器实际 lr 来自 optimizer 配置，不能误用 training 字段里遗留的 lr=0.1。

QQP 当前 loader 使用 GLUE validation 作为评估集；新增报告须准确称为 validation/held-out accuracy，不应把本地变量 `test` 当作官方有标签测试集。若需调参，从训练数据中另建固定开发集，并保存它的索引；主 held-out 集不用于选参。若引入开发集，应在扣除后重新统计客户端训练样本数并匹配 rank。

### 明确的预算匹配 assignment

保留旧 correct 的 ratio 池，先实例化为完整逐层 rank profile，再按客户端实际训练样本数排序分配；mismatched 反转这些完整 profile。不能对每一层独立乱序，也不能继续使用 `1.25-ratio`。

在沿用旧配置样本数量、客户端顺序不变时，拟定的新 mismatch ratio 如下（便于复核；真正验收依据仍是实际 profile）：

```text
QQP correct:
[1.0, 0.9242, 0.6143, 0.7486, 0.6832, 0.7107, 0.6039, 0.4386, 0.4593, 0.25]
QQP new mismatched:
[0.25, 0.4386, 0.6832, 0.4593, 0.6143, 0.6039, 0.7107, 0.9242, 0.7486, 1.0]

SVHN correct:
[0.25, 0.7096, 0.4521, 0.4801, 1.0, 0.2709, 0.5763, 0.2596, 0.3634, 0.483]
SVHN new mismatched:
[1.0, 0.2596, 0.4801, 0.4521, 0.25, 0.5763, 0.2709, 0.7096, 0.483, 0.3634]
```

验收时比较每层 rank 多重集合、因子形状多重集合、总 adapter 参数数，以及全部可训练参数数。相同参数预算不保证相同训练 FLOPs，因为不同数据量与 rank 的关联正是实验处理因素。

### 显式配对与初始化

每个 seed 先生成不可变 manifest：客户端训练样本 ID、基础权重、分类头与完整 LoRA 初始因子、逐层 rank/scaling、客户端处理顺序、每轮各客户端的 minibatch 顺序与随机流。三种方法及两个 assignment 复用这组基础对象；用哈希和运行时断言核验，不能只核验 seed 字段。

消除内部构造函数将 RNG 重置为 42 的影响，为数据划分、模型初始化、参与/顺序和本地训练分配独立 RNG；记录 Python、NumPy、Torch CPU/CUDA 与 DataLoader 状态。确定性设置和 TF32/cuDNN 设置全方法统一。不同方法的额外计算不得偷偷消耗同一共享 RNG，导致下一客户端的数据或 dropout 流不同。

“相同初始化”指共享转换前的基础对象及相同初始有效模型；RankScale 的初始化下发若属于当前算法语义，应保留它。即使初始 B=0，C 仍可能缩放非零 A，这影响之后的参数化和梯度。应保存转换前后因子哈希并验证初始有效更新一致，不能强制转换后三方法因子逐元素相同。Embedding 可能采用 A=0、B 非零，需按层处理。初始化不做把零更新分解成两个零因子的谱平衡。

### 报告方式

主指标：t=100 的 accuracy，三个 seed 的均值 ± 样本标准差（ddof=1）。QQP 同时报告 F1。保存每轮 held-out accuracy/loss 和训练损失。

逐 seed 报告 `B-A`、`C-B` 的配对差值，并报告它们的均值与样本标准差，单位为百分点。最高评估准确率及其轮次仅为辅助描述。三种 seed 不足以支撑强统计显著性主张；不把两种 assignment 或相邻轮次当作额外独立样本。

## 4. 开跑前的数值检查

先在实际层形状和实际 scaling 上测试，覆盖 Linear、embedding、卷积、过宽分类头、零范数和低秩退化输入。

上传检查：`X_ref = s_i B_i A_i`，`X_test = s_s B_tilde A_tilde`。

下发检查：先选定同一个目标谱前缀 `(B_hat, A_hat)`，然后比较 `s_s B_hat A_hat` 与 `s_i B_next A_next`。卷积和 embedding 遵循实际更新矩阵的方向/reshape，不能把 rank 截断误差混入 scale 转换误差。

保存 Frobenius 绝对误差、最大元素绝对误差、参考范数及参考非零时的相对 Frobenius 误差。零范数参照把相对误差记为 N/A，并单独检查绝对误差。

拟定验收线：标准化测试输入在 FP64 使用相对误差 1e-10、元素 atol 1e-12；FP32 使用相对误差 1e-5、元素 atol 1e-6。容差适用输入幅值需随测试记录，不能把固定绝对阈值套到任意幅值。误差接近阈值时报告原始值并调查，不事后放宽容差来隐藏问题。BF16 训练输入需额外报告转换/量化前后的误差，不以 FP32 测试通过替代 BF16 路径验证。

scale=1 的 SVHN：先对同一因子确认 B/C 转换和聚合结果一致，再在恢复同一训练状态、固定随机流后运行两个完整联邦轮次检查。若结果不一致，先排查数据顺序、初始化、随机数、状态或实现，不能解释为 RankScale 的训练收益。

## 5. 真实 checkpoint 重构

优先从可验证的原始 RBLA 历史 checkpoint 复用；若仍没有，在新 A 方法训练中顺带保存真实更新 t=1、10、50、100 的聚合因子，位置为 `_do_aggregation` 产生有序聚合结果之后、`_after_aggregation` 的 canonicalization 之前。不为此单独重训，也不使用只剩普通 RBLA+ 有序因子的 checkpoint 冒充原始输入。

建议保存这几个时点的全部 LoRA 因子，离线汇总所有层，论文展示预先规定的代表层：TinyBERT embedding、attention projection、FFN、分类头；VGG 第一/中间/最后卷积及分类层。不要在观察误差后只选有利层。

对同一 `X=BA`、相同内部目标宽度 q，计算：原始前缀、compact 重压缩前缀、`torch.linalg.svd(X, full_matrices=False)` 后截断的 dense 参考。按实际客户端请求 rank 和满宽度测试。

统一定义：

```text
relative_error = ||X - X_q||_F / ||X||_F
excess_error   = relative_error(method) - relative_error(dense reference)
optimal_error = sqrt(sum_{j>q} sigma_j^2) / ||X||_F
```

若论文使用平方相对误差，额外输出平方版本，并在图表中选择一致定义。`||X||=0` 单独报告绝对误差。不要将“额外误差”除以可能为零的 dense 最优误差。

FP64 dense 作为离线数值参考，同时报告生产 FP32 compact 的误差。满宽度包含实际过宽因子的零填充语义；重复奇异值跨截断边界时比较最优误差与奇异子空间性质，不要求因子或截断矩阵逐元素相同。卷积只对实现的矩阵化对象作低秩结论，不泛化为整个卷积算子的最佳近似。

## 6. 服务器耗时和峰值内存

当前实际聚合在 CPU，CPU 是主报告；RTX 5080 GPU 后端可作为补充。固定 CPU 线程数、GPU、dtype、软件版本、输入哈希、10 个客户端及请求 rank 集合。每次都使用同一份未经过谱排序的输入。

两种比较边界：

1. 重分发：从同一聚合因子出发，到所有请求 rank 的输出因子构造完成。A 固定前缀；B compact；D dense 经济型完整 SVD 后截断。包括该方法必要的因子/稠密矩阵构造和输出复制，每层公共分解只做一次。
2. 完整服务器步骤：从已驻留服务器的同一组客户端 state 开始，包含聚合、重分发及所有下发对象构造。客户端训练、网络传输和磁盘日志不计入；CPU/GPU 传输和序列化单列计时。B/D 使用相同 RBLA 聚合，避免把不同聚合算法成本混入此微基准。

RankScale C 在其实际训练中另记录完整服务器开销，用于评估 scale 对齐新增成本。FLoRIST 也记录完整方法自己的服务器步骤，不混入上述同聚合的分解微基准。

形状覆盖实际 TinyBERT 和 VGG9 所有 LoRA 层。先运行实际宽度；扩展压力测试可用逻辑 rank 4、8、16、32、64，分别计算卷积内部 q 并标注合成宽度。显存或 RAM 不足的点标为 OOM/TBD，不能换成更小输入却保留原标签。

建议每种形状、方法预热 10 次，正式计时 50 次；耗时很长的大型 dense 点可预先采用预热 3 次、正式 10 次，明确列出。报告中位数、IQR 和全部原始时长，随机交错方法测量顺序。GPU 前后同步，固定并记录 TF32 配置。

内存与耗时分开测：独立服务器进程，输入预加载后记录基线；CPU 采样阶段 RSS，报告开始、采样峰值、结束及峰值减基线，记录采样间隔，并说明短时峰值可能漏采。不能把整个含所有客户端的模拟器进程 RSS 称为服务器独占内存。GPU 报告基线和峰值 allocated/reserved，正确 reset peak 和同步，再计算增量；缓存/冷启动状态统一。进程高水位不得混入模型构造阶段后直接称作该阶段峰值。

Dense 对照的标签必须写明“经济型完整 SVD 后截断”，不称为所有 dense/部分 SVD 算法的速度。理论 `O((m+n)q^2+q^3)` 与 dense 复杂度讨论、测得加速比各自独立报告。

## 7. FLoRIST 适配与比较

先固定官方仓库 commit，核对论文和实际聚合代码，再做 TinyBERT 适配。论文的阈值按累计奇异值能量确定保留宽度，官方常用设置为 tau=0.9；不要把它实现成逐个奇异值减去常数的 soft-thresholding。其加权堆叠聚合、因子构造方式和客户端截取/补零须保留，不能改成 RBLA 因子平均，也不能为了统一格式静默改成 RBLA+ 的平衡因子。参见[论文](https://proceedings.mlsys.org/paper_files/paper/2026/file/f38cb4cf9a5eaa92b3cfa481832719c6-Paper-Conference.pdf)与[官方入口](https://github.com/DASS-Lab-Group/FLoRIST/blob/main/main.py)。

官方主要实验是生成式 LLM 和 attention LoRA；当前 TinyBERT 是分类任务，且包含 embedding、FFN 与分类头。它是模型适配后的比较，不是直接复现官方任务。本次已定位官方仓库和入口，但聚合文件的网页读取失败；尚未完成代码级逐公式核对，也没有执行官方实现，不能写为“官方实现已验证可运行”。

在共享 Linear 小例子上，先与官方实现对齐输入和输出有效更新，然后覆盖 embedding 的转置、分类头的过宽 rank、初始化与 scaling。采用官方 alpha/scaling 语义，记录其与 RBLA+ 的差别；任何适配都必须公开。若决定只比较 attention LoRA，应在主消融开始前统一四种方法的作用层；若事后改变层集合，则相关 A/B/C 也要重跑，预算不能仍只写“增加 6 次”。

优先在 QQP 上采用同一套分区、初始化基础对象、两种预算匹配 assignment、三个 seed、训练与评估预算。先冻结 tau=0.9，不看 held-out 结果调参。若需要预算曲线，预先选 tau={0.8,0.9,0.95}；这会从一个阈值的 6 次变为三个阈值的 18 次，额外增加 12 次训练，不在最小 42 次预算内。也可先做离线阈值/通信扫描，但不能用它代替不同阈值的训练准确率。

记录每层每轮服务器实际宽度、客户端实际 rank、阈值、全部可训练参数、上行/下行字节、完整服务器耗时和峰值内存。通信区分逻辑 LoRA 张量字节、当前模拟器实际传递的完整 state 张量字节、可选序列化字节；同一广播发给 10 个客户端应计 10 份接收负载，内存共享不能当作真实网络通信压缩。若没有真实网络测量，应称为 payload accounting，不能称实测网络流量。

官方实现适配失败时，记录明确错误和未支持路径，继续主消融与离线实验；FLoRIST 结果留 TBD，不能用近似算法顶替。

## 8. 执行顺序与预算

| 阶段 | 内容 | 新增完整训练 |
|---|---|---:|
| 0 | 固定协议、生成配对 manifest、种子/轮次/恢复改造、实际 profile 验证、数值检查、两轮小规模端到端验证；尽早完成 FLoRIST 作用层可行性判断 | 0 |
| 1 | seed 42，A/B/C × QQP/SVHN × 两种 assignment；A 顺带存真实聚合因子 | 12 |
| 2 | seed 2026、3407，完全相同协议；进行离线重构与服务器基准 | 24 |
| 3 | FLoRIST，QQP × 两种 assignment × 三种 seed，单个预定阈值 | 6 |
| 合计 | 完整推荐方案 | 42 |

第一种 seed 的完整训练若协议最终未变，可计入总预算；两轮 smoke test 不计入论文训练。分阶段用于发现程序和可复现性问题，不能因第一种 seed 的准确率不利而停止某方法、换数据集或改主指标。

更省预算的可选方案：QQP 完整三方法 18 次，SVHN 只跑 A/B 的 12 次，并以数值检查及短程配对训练验证 scale=1 恒等性，共 30 次；加 FLoRIST 6 次为 36 次。这个方案不能声称完成了 SVHN C 的三 seed 长程实验，也不能把 B 的结果复制成 C 的实测值。若资源允许，优先选择完整 42 次。

另一个可选增强：既然现有主表中的 FLoRA 是重要强基线，可在同一 QQP 协议下追加它的两种 assignment × 三个 seed，共 6 次。这有助于把当前强基线也纳入最终指标与配对比较，但不属于原定 42 次最小完整方案；追加后为 48 次。已有 FLoRA 最高准确率与新三 seed 最终准确率不能混用为严格配对证据。

现有机器批次日志中的 RankScale 用时：QQP correct/mismatch 为 7838.79/7776.13 秒，平均约 2.17 小时；SVHN correct 为 4699.02 秒，约 1.31 小时。机械套用到 18 次 QQP + 18 次 SVHN，约 62.5 小时，只是排期量级参考。新 rank 分配、确定性设置、检查点、算法差异、100 与历史 101 次更新都会影响时间；FLoRIST 适配时间和训练耗时尚为 TBD，不承诺总工期。先按实测单轮/单次用时更新 ETA。

只使用已有机器，单张 GPU 同时一个训练进程，不购买付费计算。CPU 基准与训练错开，避免共享 CPU 争用污染测量。

已有单次 RankScale 结果保留作历史主表与背景观察。只有数据索引、初始状态、随机序列、实际 rank、代码和预算均能验证匹配时，才能计入新配对实验；当前缺少这些配对产物，不能直接充当新矩阵中的一个 seed。

## 9. 建议实现边界与交付

新增独立补充入口与目录，不直接改动正在使用的历史批次列表。建议入口命名为 `src/test/rbla_plus_supplement.py`，分为 `prepare / check / run / resume / reconstruct / benchmark / summarize / plot` 子命令；这些名称目前是设计，并未实现。

最少新增模块：配对 manifest 构建/验证器；显式 seed 和真实更新轮数 runner；完整 round checkpoint 与恢复；聚合前谱处理因子保存 hook；数值检查与重构脚本；隔离服务器基准；FLoRIST 适配层；统一结果汇总和矢量 PDF 作图。

恢复在完整联邦轮次提交后进行，checkpoint 包含服务器、所有客户端的可变模型状态、optimizer/scheduler、AMP scaler（如有）、RNG、数据顺序位置、rank profile、轮次及配置哈希。共享冻结基础权重可按哈希只存一份，但可训练 head、BN buffers 等不得遗漏。原子写入 checkpoint 和状态，恢复后继续下一未提交轮，CSV 按 run_id/真实轮次去重；已完成 run 不重跑。验证恢复后的连续两轮与不中断参考一致。

每次 run 保存：完整解析配置、git commit 与未提交差异快照/哈希、环境版本、硬件信息、seed、配对对象哈希、日志、checkpoint、状态和失败原因。仅记录 commit 不足以覆盖工作树改动。

输出建议：

```text
experiment_output/<protocol_id>/
  manifest/                 # 样本 ID、基础初始化、rank profiles、随机流
  runs/<run_id>/            # config、metadata、metrics、状态、恢复 checkpoint
  raw_aggregates/           # 原始 RBLA、早中晚轮次
  diagnostics/              # 数值一致性与 checkpoint 重构 CSV/JSON
  benchmarks/               # 每次计时、内存样本、payload 统计
  summaries/                # final accuracy、paired delta、资源比较
  figures/                  # 矢量 PDF，与生成脚本/命令
  README.md
```

论文至少增加：主消融最终准确率表、配对差值与训练曲线、真实 checkpoint 误差图、服务器耗时/内存图、FLoRIST 精度和资源表。全部从真实 CSV/JSON 生成，误差线注明样本标准差或 IQR，缺失结果写 TBD。保留负结果，不修改 Abstract 和 Introduction，也不覆盖历史表格与日志。

## 10. 当前交付状态

已完成：源码/日志审查、当前 VGG9 实例化 rank/scaling/参数预算核对、8 项既有 RankScale 单元测试、正式 FLoRIST 文献与仓库定位、上述分阶段方案。

尚未完成：新配对入口与恢复实现、真实模型全层数值报告、36/42 次补充训练、原始 checkpoint 重构、隔离服务器基准、FLoRIST 代码逐公式验证与运行。它们在本文中均为计划，没有新增准确率、加速比或内存收益结论。
