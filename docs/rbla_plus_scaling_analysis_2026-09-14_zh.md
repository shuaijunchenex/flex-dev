本文分析 2026-09-14 工作区 `H:/flex-src` 中的当前实现，依据源码、实验 YAML、实际模型实例和已有单元测试。它描述当前代码行为；没有重跑完整联邦训练，也不据此追溯证明历史实验使用了相同代码。

最关键的结论是：仓库中普通 `rbla_plus` 实验和 `rbla_plus_rank_scale` 是两条不同路径。普通 RBLA+ 的实现名是 `sp_plus`，执行 RBLA 因子聚合、紧凑 SVD canonicalization 和前缀广播，不显式补偿客户端与服务器的 LoRA scaling 差异。独立的 RankScale 版本在这一流程外增加上传、下发两个方向的尺度变换。另一个实际问题是，部分 CNN 模型没有读取到 YAML 中的固定 alpha，实际 scaling 始终为 1，因此这些 CNN 上的额外变换退化为恒等变换。

先明确代码和实验名称之间的对应关系。

| 配置/入口 | 实际行为 |
| --- | --- |
| 普通 `*_rbla_plus.yaml` / robustness 下的 `rbla_plus.yaml` | client/server/runner 使用 `sp_plus`，聚合使用 `aggregation_sp_plus.yaml` |
| `*_rbla_plus_rank_scale.yaml` | client/server/runner 使用 `rbla_plus_rank_scale`，聚合仍使用 `aggregation_sp_plus.yaml` |
| `src/test/rbla_plus_experiment_aws.py` | 当前选择 `TARGET_METHOD = "rbla_plus_rank_scale"`，收集 10 个 CNN/NLP 数据集 × correct/mismatch，共 20 个配置 |
| `src/test/rbla_plus_experiment_g4.py` | 当前 robustness 方法列表包含普通 `rbla_plus`，没有 RankScale |

实际逐个读取了上述 20 个 RankScale YAML：三类 strategy 引用一致，聚合器均为 SP+，每个 rank 分配列表的最大 ratio 均为 1.0。实验名字里有 RBLA+ 并不意味着启用了 RankScale，需要检查实际引用。

对应源码：[SP+ 聚合器](../src/flex/fl_algorithms/aggregation/methods/_fed_aggregator_sp_plus.py)、[AWS 入口](../src/test/rbla_plus_experiment_aws.py)、[普通 CIFAR 配置](../src/test/rbla+/experiment/rank_match/dirichlet_0_4/cifar10/cifar10_dirichlet_0_4_rank_correct_rbla_plus.yaml)、[RankScale CIFAR 配置](../src/test/rbla+/experiment/rank_match/dirichlet_0_4/cifar10/cifar10_dirichlet_0_4_rank_correct_rbla_plus_rank_scale.yaml)。

这里的 scaling 指每一层 forward 真正使用的乘数。省略层下标，线性层的有效 LoRA 权重更新是

\[
\Delta W_i=s_i B_iA_i,\qquad s_i=\alpha_i/r_i.
\]

`MSLoRALinear`、`MSLoRAConv2d` 和 `MSEmbedding` 在构建时设置 `self.scaling = self.lora_alpha / self.r`；若构建参数 `use_lora_scaling=False`，则设置为 1。它们没有把这个标量定义为可训练的 `nn.Parameter`。线性层在 forward 的 LoRA 分支乘它；卷积层对 `B @ A` reshape 后乘它；Embedding 对乘积转置，代数上的标量补偿仍然成立。merge/unmerge 权重时也使用同一个 scaling。

应区分四种量：

| 符号/量 | 用途 |
| --- | --- |
| `lora_alpha` | 构造 LoRA forward scaling 的超参数 |
| `module.scaling`，记为 s | forward 实际使用的标量；RankScale 直接读取它 |
| `sqrt(s_source / s_target)` | 跨端转换时 A、B 各自乘的标量 |
| SVD 的 `sqrt(singular_value)` | 将奇异值平衡分配给 A 和 B，保持聚合后的乘积 |

数据量权重、NaN padding 的有效支持权重和 Dirichlet 数据分布 alpha 也都不等于 LoRA scaling。当前这条 RankScale 路径没有额外学习 gamma、搜索最佳缩放系数、按范数自适应缩放或自动调整学习率。

来源：[LoRA 层实现](../src/flex/ml_algorithms/lora/impl/lora_ms.py)。

普通 RBLA+ 的一轮计算可写为以下流程。客户端以自己的 `s_i` 本地训练，上传训练完成后的 A、B 参数，`updated_weights` 并不是相对于广播参数的因子差分。服务器直接对上传因子做 RBLA 聚合，然后 canonicalize，并向客户端广播前缀。

默认 `pad_mode: nan`。不同秩的 A、B 补齐到参与者中的最大张量形状，缺失位置填 NaN。设参与客户端的数据量归一化权重为 p_i，具有第 j 个因子槽位的集合为 S_j，则

\[
\bar A_{j,:}=\frac{\sum_{i\in S_j}p_i A_{i,j,:}}{\sum_{i\in S_j}p_i},\qquad
\bar B_{:,j}=\frac{\sum_{i\in S_j}p_i B_{i,:,j}}{\sum_{i\in S_j}p_i}.
\]

这是 A、B 分别按有效支持集合平均。若切换为 zero padding，分母变为全体权重和；当前 SP+ YAML 用的是 NaN 模式。非 LoRA 浮点参数默认按数据量平均，整数 buffer 复制第一个客户端的值；`lora_only=True` 时非 LoRA 参数均从第一个客户端复制。

对于聚合后的因子乘积 M = B_bar A_bar，canonicalization 使用

\[
\bar B=Q_BR_B,\quad \bar A^T=Q_AR_A,\quad
C=R_BR_A^T=U_C\Sigma V_C^T,
\]

\[
B_c=Q_BU_C\Sigma^{1/2},\qquad
A_c=\Sigma^{1/2}V_C^TQ_A^T.
\]

它通过 QR 和小核心 SVD 计算，主路径不构造完整稠密 M；在数值误差内保持 B_c A_c = M。当前配置按奇异值从大到小排序，每次聚合后运行，并固定每个方向的符号。`overcomplete_policy: zero_pad` 会在薄分解后将尾部补零，保留配置要求的因子形状；不会据此修改模块的逻辑 rank、alpha 或 scaling。

广播时，普通 RBLA+ 对 A 取前若干行、B 取前若干列。客户端用自己的 s_i，因此收到的有效更新是

\[
\Delta W_i^{\mathrm{plain}}=s_i B_c^{(q_i)} A_c^{(q_i)},
\]

而同一个前缀在服务器尺度下代表 s_s B_c^(q_i) A_c^(q_i)。两者相差 s_i/s_s；相同 alpha 时才可简化为 r_s/r_i。如果 s_i=s_s，则不存在这个额外的幅值差异。SVD 虽然保持 BA，但没有读取 s_i，不能自行消除跨端 scaling 差异。

来源：[RBLA 聚合与广播](../src/flex/fl_algorithms/aggregation/methods/_fed_aggregator_rbla.py)、[canonicalization](../src/flex/ml_algorithms/lora/canonicalization.py)、[普通 SP+ 客户端](../src/flex/fed_strategy/client_strategy_impl/_sp_plus_client.py)、[普通 SP+ 服务器](../src/flex/fed_strategy/server_strategy_impl/_sp_plus_server.py)。

RankScale 版本的完整流程如下。

1. `server.prepare()` 建立服务器逐层 profile，并调用每个客户端的 `refresh_rank_scale_profile()`。profile 记录层路径、A/B state-dict key、逻辑 rank、alpha 和实际 scaling。服务器建立 client ID 到客户端对象的映射。
2. 训练开始前先做一次带尺度补偿的广播。初始广播没有经过聚合后的 SVD；初始 LoRA 通常有一个零因子，因此初始有效更新可以为零，但非零因子的缩放仍可能改变后续梯度。
3. 客户端继续执行继承自 RBLA/SP+ 的本地训练。模型保留自己的 rank、alpha 和 scaling，每轮重建并清空优化器，不跨轮继承旧槽位的动量。
4. 服务器 `aggregation()` 根据更新中的 node_id 匹配客户端，在调用原 SP+ 聚合器前转换上传因子。当前模拟器是在服务器收到更新后调用客户端 strategy 的转换方法，不是上传方法预先改写了本地参数。
5. 原聚合器完成 NaN padding 支持加权平均、QR/SVD canonicalization，结果留在服务器 scaling 坐标中。
6. `apply_weight()` 将聚合结果存为服务器 `model_weight`，并交给 evaluator。广播时逐客户端转换回本地尺度，再调用原 SP+ 的 shape fit / 前缀截断。随后服务器评估。后续每轮重复训练、上传转换、聚合、下发转换。

上传的具体变换为

\[
c_i=\sqrt{\frac{s_i}{s_s}},\qquad
\widetilde A_i=c_iA_i,\quad \widetilde B_i=c_iB_i.
\]

因此

\[
s_s\widetilde B_i\widetilde A_i=s_s c_i^2 B_iA_i=s_iB_iA_i.
\]

上传对齐保持每个客户端原本的有效更新。它不需要改变因子形状、不改变本地 rank，也不把所有模型的 scaling 属性改成统一值。

服务器随后对 A_tilde、B_tilde 使用上文相同的 RBLA 公式。客户端下发补偿为

\[
d_i=\sqrt{\frac{s_s}{s_i}},\qquad
A_i^{\mathrm{next}}=d_i A_c^{(q_i)},\quad
B_i^{\mathrm{next}}=d_i B_c^{(q_i)}.
\]

因此

\[
s_i B_i^{\mathrm{next}} A_i^{\mathrm{next}}
=s_s B_c^{(q_i)} A_c^{(q_i)}.
\]

代码的顺序是先缩放完整服务器因子，再按本地张量形状截断或补零；对单层标量来说，与先截断再缩放等价。服务器会向所有已注册客户端广播，不限于当轮参与者。

在双方 alpha 相同、均启用 alpha/r scaling 的特例下，c_i=sqrt(r_s/r_i)，d_i=sqrt(r_i/r_s)。若 alpha 不同，则必须使用实际比例 sqrt(alpha_i r_s / (alpha_s r_i))，或更直接地使用实际 `module.scaling`。当前实现采取后者，不按 rank_ratio 近似，也不由因子形状反推 scaling。

来源：[逐层尺度转换](../src/flex/ml_algorithms/lora/rank_scale_alignment.py)、[RankScale 客户端](../src/flex/fed_strategy/client_strategy_impl/_rbla_plus_rank_scale_client.py)、[RankScale 服务器](../src/flex/fed_strategy/server_strategy_impl/_rbla_plus_rank_scale_server.py)、[RankScale runner](../src/flex/fed_strategy/runner_strategy_impl/_rbla_plus_rank_scale_runner_strategy.py)。

一个数值例子：服务器 alpha=16、r=8，因此 s_s=2；客户端 alpha=16、r=2，因此 s_i=8。上传时 A、B 各乘 2，乘积变成原来的 4 倍，服务器再乘 2，恰好得到原客户端的 8BA。广播时客户端前缀 A、B 各乘 0.5，乘积缩为四分之一，本地再乘 8，得到服务器尺度下的 2 倍前缀乘积。普通 RBLA+ 若直接广播该前缀，会使用 8 倍前缀乘积，是服务器尺度的 4 倍。

这个等式保持的是所选前缀的有效更新，不是所有低秩客户端都获得完整服务器更新。默认奇异值排序下，线性层选取的是聚合乘积的截断 SVD；舍弃尾部造成的损失仍存在。对于卷积，要将下文的 q_i 个因子槽位代入，谱性质对应代码采用的 BA 矩阵布局。

实际配置的 scaling 需要特别说明。`NNModelArgs` 把额外配置保存在 DictPath 中，没有将任意 LoRA 配置字段自动暴露为属性。CIFAR CNN 和 VGG9 构建器使用 `getattr(args, "lora_alpha_conv", ...)` 等方式读取这些字段，因此当前正常 factory 路径会走默认值。CIFAR 的默认 alpha 直接取本地 rank；VGG9 的默认 alpha 为 None，再由内部模型设为本地 rank。CNN YAML 中写了固定 alpha，但并未在这条路径生效。

直接调用正常 `NNModelFactory.create_args()` / `create()`、无需数据集和训练，得到以下实际结果：

| 模型/层 | rank_ratio | 实际逻辑 r | 实际 alpha | 实际 s |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 / CINIC-10 CNN 卷积 | 1.00 | 64 | 64 | 1 |
| CIFAR-10 / CINIC-10 CNN 卷积 | 0.25 | 16 | 16 | 1 |
| CIFAR-10 / CINIC-10 CNN FC | 1.00 | 16 | 16 | 1 |
| CIFAR-10 / CINIC-10 CNN FC | 0.25 | 4 | 4 | 1 |
| SVHN VGG9 卷积 | 1.00 | 8 | 8 | 1 |
| SVHN VGG9 卷积 | 0.25 | 2 | 2 | 1 |
| SVHN VGG9 FC | 1.00 | 16 | 16 | 1 |
| SVHN VGG9 FC | 0.25 | 4 | 4 | 1 |
| MNIST canonical MLP 隐藏层 | 1.00 | 160 | 2 | 0.0125 |
| MNIST canonical MLP 隐藏层 | 0.25 | 40 | 2 | 0.05 |
| MNIST canonical MLP 输出层 | 1.00 | 10 | 2 | 0.2 |
| MNIST canonical MLP 输出层 | 0.25 | 10 | 2 | 0.2 |

因此，这组 CNN 配置在当前 factory 构建路径中，各层 s_i=s_s=1，RankScale 的 c_i、d_i 全为 1。在相同输入与状态下，新增的尺度变换没有数值效果；不能仅凭名称断言 CNN 已进行了非平凡的异秩幅值补偿，也不能从此推断不同训练运行必然逐位复现。

MNIST MLP 使用默认 `lora_rank=4`、`lora_scaling=0.5` 计算 alpha=int(4×0.5)=2；这两个名字不能直接当作每层最终 rank 和最终 scaling。真正的隐藏层 rank 为 int(160×ratio)，输出层为 int(100×ratio)，canonical 配置启用矩阵维度上限，因此输出层在上述比例下均被截到 10。隐藏层缩放不同，而输出层缩放相同。不能对所有层只套一个 ratio。

TinyBERT 则使用 `args.get()` 正确读取字典配置。当前 NLP YAML 的 base rank=8、alpha=16，实际 r=max(1, round(8×ratio))、s=16/r；服务器 ratio=1 时 s_s=2，ratio=0.25 的客户端 s_i=8。这里根据配置和构建源码确认，未实例化完整预训练 TinyBERT 或下载模型。它使用仓库自定义的张量型 MSLoRA Linear/Embedding，因此与 profile 的接口匹配。

来源：[NNModelArgs](../src/flex/ml_models/nn_model_args.py)、[KeyValueArgs](../src/flex/ml_utils/key_value_args.py)、[CIFAR CNN 构建器](../src/flex/ml_models/lora/_nn_model_cifar_lora_cnn.py)、[VGG9 构建器](../src/flex/ml_models/lora/_nn_model_vgg_lora_cnn.py)、[MNIST MLP 构建器](../src/flex/ml_models/lora/_nn_model_simple_lora_mlp.py)、[TinyBERT 构建器](../src/flex/ml_models/lora/_nn_model_tinybert_lora.py)。此处仅记录配置读取问题，没有修改实现或实验配置。

卷积有一个容易混淆的维度。kernel_size=k 时，A 的形状为 (r×k, C_in×k)，B 为 (C_out×k, r×k)。forward 的分母是模块逻辑 r，不是 A 的行数 r×k。因此 profile 读取 `module.r` 和 `module.scaling`；例如 r=2、k=3 时 A 有 6 行，alpha=8 时 scaling 仍为 8/2=4。前缀广播则按张量形状保留 q_i=r_i×k 个因子槽位；线性层和 Embedding 的 q_i=r_i。canonicalization 的薄秩或零奇异值个数也不替代模块的逻辑 r。

还需要准确限制这项补偿的数学含义。

- 它统一跨端因子的尺度表示，不保证 RBLA 因子聚合等于有效更新矩阵的 FedAvg。即便所有客户端同秩，(sum p_i B_tilde_i)(sum p_i A_tilde_i) 仍包含交叉项，通常不等于 sum p_i B_tilde_i A_tilde_i。NaN 模式还对不同槽位使用不同支持分母。
- canonicalization 在聚合之后进行，不是先将每个客户端独立做 SVD 再聚合。因此它也不自动消除本地训练产生的所有因子基底差异。
- 保持 sBA 不意味着保持梯度、优化器轨迹或不同 rank 的训练行为。以矩阵梯度 G 为例，A 的梯度含 s B^T G，B 的梯度含 s G A^T；重参数化会改变因子坐标。代码没有额外的梯度或学习率补偿。
- 当前下发先形成每客户端的完整服务器形状变换结果，再在接收端 fit 到本地形状。因子数值缩放本身不减少传输张量的形状，也不是直接在服务器只构造本地前缀的优化实现。
- canonicalization 日志中的奇异值首先对应 BA。若分析有效更新能量，应使用 ||sBA||_F²=s²||BA||_F²。`src/test/rbla+/spmi.py` 支持 `layer_scalings`，默认是 1；缩放一致时单层相对比例可抵消，但跨层能量权重和绝对能量不能直接忽略 s²。

工程上还存在以下明确边界：profile 要求 A/B 为直接 Tensor 且键成对存在，层名一致，逻辑 rank>0，实际 scaling 和 alpha 为正有限标量。容器型 PEFT 多 adapter 因子不被这个 profile 构建器支持。转换函数新建 LoRA 张量，不原地修改上传或服务器因子；非 LoRA 张量在这一转换步骤中共享引用，后续依旧按原聚合规则处理。

客户端 profile 按模型对象缓存，模型对象替换时自动重建；同一对象的 scaling 被原地改变不会自动刷新，需显式调用 refresh。服务器 profile 也在 prepare/lazy setup 后缓存，不自动跟踪同一对象的元数据变化。它适合当前固定秩、固定 scaling 流程，不能未经处理就宣称支持动态 rank 或动态 alpha。

服务器 profile 读取的是 `node_var.model`，评估使用的 inference model 在另一路按 rank_distribution 的最大 ratio 构建。当前审计的 20 个 RankScale 配置最大 ratio 全是 1，与初始服务器 ratio 对齐；如果另配最大 ratio 小于 1 的实验，仍需确认这两个模型的实际 rank/scaling 一致。RankScale 本身没有显式校验 evaluator 的 scale profile。

现有测试验证了有效更新不变、上传/下发往返、广播前缀、上传状态不被修改、客户端缓存/刷新、卷积逻辑 rank、有限梯度、strategy 独立注册，以及 SP+ 与原 RBLA+ 计算的一致性和 canonicalization 性质。执行命令为：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -B -m pytest -q -p no:cacheprovider src/unittest_ml/unittest_rbla_plus_rank_scale_strategy.py src/unittest_ml/unittest_sp_plus_strategy.py src/unittest_ml/unittest_lora_canonicalization.py
```

结果：48 passed、1 skipped（CUDA 不可用）。此外检查了 20 个配置引用，并实际构建 CIFAR/CINIC CNN、SVHN VGG9 和 MNIST MLP 读取逐层 profile。测试结果支持上述实现机制，不构成端到端训练准确率或收敛收益的验证。
