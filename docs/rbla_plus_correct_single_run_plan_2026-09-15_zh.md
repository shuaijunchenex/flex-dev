# RBLA+ correct 单次补充实验：当前选定方案

> 已被后续命名和设计要求取代：当前活动方案见 [两条训练路径上的广播交叉消融](rbla_plus_two_path_plan_2026-09-15_zh.md)。新版 RBLA+ 唯一指带 RankScale 的版本；首批三数据集只训练 RBLA/RBLA+ 两条路径，共 6 次。每条路径在同一份聚合后、谱处理前因子上分别应用两种广播，记录四组客户端模型准确率；不重新聚合。下文三方法、9 次方案保留为历史记录。

更新：2026-09-15。按用户最新要求，本方案取代前一版主消融/对照的三种种子、两种 assignment 和 FLoRIST 计划。只用一个固定 seed=42，不做 seed sweep，不跑 mismatch，不加入 FLoRIST 或 MRPC。真实 checkpoint 重构与服务器实测仍保留。

状态：已完成现有日志与配置检查，选定首批 9 个配置；本轮没有启动完整训练，也没有实现新的配对训练/恢复入口。

## 1. 最优先配置与首批任务

如果只选一个对 RBLA+ 有利的场景，选 **CIFAR-10 / correct / Dirichlet alpha=0.4**。

考虑到还需要检查 RankScale 增量效果，首批确定为 **CIFAR-10、MNLI、QQP**，每个任务分别运行：

- A：原始 RBLA；
- B：普通 RBLA+，即 SP+，不启用 RankScale；
- C：RBLA+，启用双向 RankScale。

共 **3 个数据集 × 3 个方法 × 1 个 assignment × 1 个固定种子 = 9 次完整训练**。

这是利用已有结果筛选场景的探索性消融。任务固定后，保留全部方法的最终结果和曲线，包括不利或不显著的差异；单次实验不报告跨 seed 均值、标准差或普遍优势。

## 2. 现有结果支持怎样选择

下表全部为 correct、Dirichlet alpha=0.4，单位为百分比。RBLA 列来自用户原有论文表，经已有更新脚本保留；本地此次未找到能逐一核验这些单元格的原始训练 CSV。RankScale 列则重新读取最新完整 CSV 核验。两类来源不能写成已经验证配对的新消融结果。

| 数据集 | 原表 RBLA 最高值 | RankScale 最高值 | 两个最高值之差，百分点 | RankScale 末轮 | 选择 |
|---|---:|---:|---:|---:|---|
| CIFAR-10 | 38.33 | 79.12 | +40.79 | 78.76 | 首批，最强候选、scale=1 对照 |
| CINIC-10 | 31.54 | 69.65 | +38.11 | 69.12 | 扩展候选，与 CIFAR 共用 CNN 架构 |
| SVHN | 72.39 | 92.98 | +20.59 | 92.63 | 扩展候选，不同 VGG9 架构 |
| MNLI | 50.25 | 65.16 | +14.91 | 59.84 | 首批，语言任务中历史差距最大 |
| QQP | 78.50 | 82.48 | +3.98 | 79.28 | 首批，补充一个语言分类任务 |
| SST-2 | 78.44 | 80.62 | +2.18 | 70.87 | 暂不选，峰值在日志 round 1，后期下降明显 |
| RTE | 55.23 | 57.04 | +1.81 | 56.32 | 暂不选，评估集小，单次差值解释有限 |
| AG News | 90.68 | 91.33 | +0.65 | 88.42 | 暂不选，历史差距小 |
| IMDB | 84.15 | 84.32 | +0.17 | 80.80 | 暂不选，历史差距小 |

表中 RankScale 末轮均为现有日志的 round=100。最高值差仅用于筛选，不能用来代替固定预算最终准确率差。尤其不能据此宣称 C 已经优于 B：本地缺少对应的普通 RBLA+ 完整训练日志，这正是新增三方法消融要解决的问题。

MNIST/QMNIST 在原表 correct、alpha=0.1 中也有较大差距，但这些表格单元是三种种子的统计量，不能挑其中最好 seed 充当新的单次结果；同时此次没有找到对应可复核的普通 RBLA+/RBLA 原始日志，暂不纳入首批。

## 3. 为什么首选 CIFAR-10

当前工厂实例化已确认：CIFAR-10 所有服务器/客户端 LoRA 层实际 scaling 均为 1。因此同一因子输入下，RankScale 是恒等变换。B/C 应在一致的初始化、训练状态和随机流程下得到一致行为；它适合检验谱重分发及 scale=1 对照，不能用于宣称 RankScale 本身额外提升准确率。

这个配置还有清晰的机制诊断价值：服务器第一卷积层的 A 为 `[192,9]`，B 为 `[96,192]`，逻辑 rank=64，但 `BA` 矩阵的 rank 最多为 9。最小客户端逻辑 rank=16，对应内部宽度 q=48，足以容纳这个矩阵的全部有效谱成分。compact 重压缩后前缀能在数值精度内保留该层整个聚合更新；原始 192 个因子槽位的固定前缀则不自动具有这种保证。

这是对当前矩阵形状的数学解释，不是已经测出的真实 checkpoint 误差或新训练收益。建议在新 A 训练中记录这层与后续卷积层的真实聚合因子，直接验证机制。

## 4. 首批固定配置

三方法组合配置已通过项目 `AppEntry` 实际解析，并逐项比较。模型、数据分区、rank ratio、训练、优化器、评估、客户端选择配置一致；差异集中在策略注册、聚合方法、canonicalization 开关和过宽因子处理策略。

普通 B 使用 `sp_plus`，谱处理为 `ordering: singular_value`、`overcomplete_policy: zero_pad`；C 使用同一 SP+ 聚合并启用 RankScale。A 的谱处理关闭，因此其 `overcomplete_policy: error` 不会参与谱重压缩。不要把 A 误配置为仍启用 canonicalization。

| 设置 | CIFAR-10 | MNLI | QQP |
|---|---|---|---|
| assignment / 分区 alpha | correct / 0.4 | correct / 0.4 | correct / 0.4 |
| 模型 | 当前 CifarLoRACNN | `prajjwal1/bert-tiny`，3 类 | `prajjwal1/bert-tiny`，2 类 |
| LoRA 层 | 6 个卷积 + 1 个全连接 | 当前 Linear + embedding 作用层 | 同 MNLI |
| 服务器逻辑 rank | 卷积 64、FC 16 | 8 | 8 |
| 服务器实际 scaling | 1 | 2 | 2 |
| 客户端 rank | 见下方实际序列 | 见下方实际序列 | 见下方实际序列 |
| 本地 epochs | 5 | 1 | 1 |
| batch size | 64 | 64 | 64 |
| 优化器 | Adam，lr=1e-3，wd=1e-4 | AdamW，lr=1e-4，wd=0.05 | 同 MNLI |
| 训练精度 | 现有 AMP trainer | 现有 GLUE FP32 trainer | 同 MNLI |
| 服务器聚合 | CPU | CPU | CPU |
| 客户端数 / 每轮参与 | 10 / 10 | 10 / 10 | 10 / 10 |
| 随机配置 | 固定 42，一次运行 | 同左 | 同左 |
| 预算 | 沿用当前 training_rounds=100 | 同左 | 同左 |

为复用当前配置并保持历史预算口径，本轮方案保留现有 runner 的 round 0…100，即 **101 次实际训练更新**。不修改历史 runner，也不把它描述成恰好 100 次更新。三方法预算完全相同，主指标为最后一次更新后的准确率。若以后改成恰好 100 次更新，应作为新协议三方法统一修改，不能只改一个方法。

实际 rank 序列按客户端 1…10：

```text
CIFAR-10 conv: [16,56,30,42,64,21,49,18,30,44]
CIFAR-10 FC:   [4,14,8,11,16,5,12,4,8,11]
MNLI:         [6,6,6,7,6,8,5,5,5,2]
QQP:          [8,7,5,6,5,6,5,4,4,2]
```

CNN 序列由当前模型工厂实例化读取；TinyBERT 序列根据当前工厂公式和已解析配置计算，开跑时仍须导出实际层对象复核，不在本次审计中下载模型。

MNLI/QQP 的 LoRA alpha=16，低 rank 客户端实际 scaling 大于服务器；这提供 C 相对 B 可产生不同训练行为的条件，但不保证 C 的准确率更高。

## 5. 已选配置文件

根目录为 `src/test/rbla+/experiment/rank_match/dirichlet_0_4/`，首批 9 个路径已验证存在并可解析：

```text
cifar10/cifar10_dirichlet_0_4_rank_correct_rbla.yaml
cifar10/cifar10_dirichlet_0_4_rank_correct_rbla_plus.yaml
cifar10/cifar10_dirichlet_0_4_rank_correct_rbla_plus_rank_scale.yaml

mnli/mnli_dirichlet_0_4_rank_correct_rbla.yaml
mnli/mnli_dirichlet_0_4_rank_correct_rbla_plus.yaml
mnli/mnli_dirichlet_0_4_rank_correct_rbla_plus_rank_scale.yaml

qqp/qqp_dirichlet_0_4_rank_correct_rbla.yaml
qqp/qqp_dirichlet_0_4_rank_correct_rbla_plus.yaml
qqp/qqp_dirichlet_0_4_rank_correct_rbla_plus_rank_scale.yaml
```

首批不修改 rank 分配，也不再构造 mismatch，所以前一版跨 assignment 参数预算修正不在本批任务内。保留单次实验内部三方法的实际逐层 rank/参数预算核验。

选中的路径、完整解析配置、方法差异、日志哈希、当前 CNN 实际层 profile，以及筛选指标保存在 `design/rbla_plus_correct_selection_2026-09-15.json`。其中 `selected_protocol.jobs` 是当前 9 个任务清单；状态均为 planned。

## 6. 运行前仍需保留的最小核验

一次运行也需要控制初始化与随机流程。保存并复用客户端样本 ID、转换前的基础模型/adapter 初始化、客户端顺序和本地 minibatch 随机状态，并记录哈希。当前 YAML 一致和 seed=42 一致还不是已完成配对的证据。RankScale 初始化下发保留自身语义，不对零初始化有效更新先做令两个因子都为零的谱平衡。

在 CIFAR 上做 scale=1 的同输入因子数值检查和短程 B/C 配对训练；在 MNLI/QQP 实际层上检查双向 scaling 的有效更新一致性。下发参考使用已选谱前缀，避免混入截断损失。再开始固定配置的完整训练。

现有 C 的日志可以用于选配置与历史曲线展示；由于缺少与新 A/B 对应的显式配对对象，当前不将它们自动抵扣为已完成的新消融任务。前述 9 次预算包含 C 的新配对运行。

所有任务保存最终准确率、完整 accuracy/loss 曲线和辅助最高准确率。MNLI/QQP 当前本地评估使用 GLUE validation，应在报告中准确注明 split；不根据该评估集的峰值改学习率、训练轮数或挑选 checkpoint。

## 7. 扩展与仍保留的补充实验

首批之后若扩展数据集，可固定追加 CINIC-10 或 SVHN 的 A/B/C，各增加 3 次；两个都加入共 15 次。两者已有结果均提示较大优势。CINIC-10 延续同一 CNN 架构，SVHN 增加不同 VGG9 架构，不能把它们当成三个随机种子的替代品。

原始 RBLA 新训练中顺带保存实际更新 1、10、50、101 的聚合后、谱处理前因子，复用做原始前缀/compact/dense 重构对比，无额外完整训练。保留所有层结果，论文展示事先选好的代表层。

服务器实测也保留：CPU 主路径上比较前缀、compact 和经济型完整 dense SVD 后截断，测重分发与完整服务器步骤、阶段峰值增量内存。实际 CIFAR 某些因子宽度接近矩阵尺寸，compact 未必总更快；TinyBERT 的窄因子层也纳入尺寸覆盖。精度上有利的配置不自动意味着服务器速度也有利。

既有 `src/test/rbla_plus_experiment_local.py` 仍是历史 RankScale 补跑入口，包含 mismatch，与本次三方法清单不同。此次没有改动或启动它。新的配对执行、checkpoint 恢复及测量工具仍需在正式运行前落实。

## 8. 本次核查产物

- `design/audit_rbla_plus_correct_candidates.py`：只读审计脚本，重新读取当前配置/日志、实例化 CNN 并将结果 JSON 输出到 stdout；不下载 TinyBERT、不启动训练。
- `design/rbla_plus_correct_selection_2026-09-15.json`：本次审计快照及首批 9 个任务清单。
- 本文：新的活动方案。前一版 42 次方案保留作为历史记录，FLoRIST 与多 seed/mismatch 不属于当前任务。

本次完成的证据是配置核查、日志核查和 CNN 实际对象核查。普通 RBLA+ 的新训练结果、RankScale 相对普通 RBLA+ 的增量、真实 checkpoint 误差和服务器性能仍为待测。
