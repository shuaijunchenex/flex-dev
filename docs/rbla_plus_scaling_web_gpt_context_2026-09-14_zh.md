请基于下面的当前代码事实理解我的 RBLA+ 项目。你无法访问我的本地仓库，因此以这些已核实事实为背景，并明确区分代码行为、数学推论和改进建议。不要把标准 LoRA 惯例当成所有模型当前都已正确执行的配置。

我的项目是 FLEX 异构秩 LoRA 联邦学习。对一层，客户端 i 的有效更新为 ΔW_i=s_i B_i A_i，线性层 A_i∈R^(r_i×d_in)、B_i∈R^(d_out×r_i)。模块通常 s_i=alpha_i/r_i，关闭 use_lora_scaling 时 s_i=1。scaling 是模块构建时设置的普通属性，不是可训练参数。Embedding 对 BA 转置，卷积对 BA reshape，不影响标量补偿恒等式。

仓库当前有两个独立版本：

1. 普通 RBLA+：实验文件名为 rbla_plus，但实际 client/server/runner strategy 和 aggregation method 是 sp_plus。它就是 RBLA 的异构因子支持加权平均 + 聚合后紧凑 SVD canonicalization + 前缀广播，没有显式的跨端 scaling 补偿。
2. rbla_plus_rank_scale：继承 SP+，增加上传和广播两个方向的逐层 scale alignment。聚合器仍然是原来的 sp_plus。它没有替换普通 rbla_plus，也没有学习额外 gamma 或调整学习率。

普通 RBLA+ 的聚合以训练后的完整 A、B 参数为输入，不是参数差分，也不是客户端稠密 ΔW。默认 NaN padding。设 p_i 为样本量归一化权重，第 j 个因子槽位的有效客户端集合为 S_j：

    A_bar[j,:] = sum_{i in S_j} p_i A_i[j,:] / sum_{i in S_j} p_i
    B_bar[:,j] = sum_{i in S_j} p_i B_i[:,j] / sum_{i in S_j} p_i

A、B 分别平均。聚合后通过 B_bar=Q_B R_B、A_bar^T=Q_A R_A，对小核心 C=R_B R_A^T 做 SVD：C=U_c Σ V_c^T。输出 B_c=Q_B U_c sqrt(Σ)，A_c=sqrt(Σ) V_c^T Q_A^T，数值误差内保持 B_c A_c=B_bar A_bar。主路径不构造完整稠密 BA。当前按奇异值降序，每次聚合后执行；超出薄矩阵秩的槽位补零以保持张量形状，不修改模块 r/alpha/scaling。这里 sqrt(Σ) 是 SVD 因子平衡，不能与跨端尺度补偿混为一谈。

普通版本直接广播 A_c、B_c 的本地形状前缀。客户端实际 forward 使用 s_i，因此同一前缀的有效更新相对于服务器尺度 s_s 多出比例 s_i/s_s。只有两端 alpha 相同且使用 alpha/r 时，这个比例才等于 r_s/r_i；若 s_i=s_s，则没有额外幅值差异。

RankScale 版本先从实际模型逐层读取 prefix、A/B key、module.r、module.lora_alpha、module.scaling 构建 profile，以实际 module.scaling 作为变换依据，不按 YAML 值或 rank_ratio 推测。

上传：

    c_i = sqrt(s_i / s_s)
    A_tilde_i = c_i A_i
    B_tilde_i = c_i B_i
    s_s B_tilde_i A_tilde_i = s_i B_i A_i

然后把 A_tilde、B_tilde 送进同一个支持加权 RBLA/SP+ 聚合器，再做 canonicalization。实现位置是服务器 aggregation() 接收更新后，根据 node_id 调用对应客户端 strategy 的 normalize_upload_for_server()。它不原地修改客户端上传参数。

广播：

    d_i = sqrt(s_s / s_i)
    A_next_i = d_i prefix(A_c, local_shape)
    B_next_i = d_i prefix(B_c, local_shape)
    s_i B_next_i A_next_i = s_s prefix(B_c) prefix(A_c)

实际先将完整服务器因子转换到该客户端尺度，再在接收端按本地形状截断/补零。训练开始前也执行一次这样的广播。每轮本地训练仍使用客户端原有 rank/alpha/scaling，优化器每轮重建清空。转换不改变模型的 scaling 属性。

相同 alpha 时，上传 A、B 各乘 sqrt(r_s/r_i)，下发各乘 sqrt(r_i/r_s)。例如 alpha=16，服务器 r=8、s=2，客户端 r=2、s=8：上传各乘 2，下发各乘 0.5。补偿保持服务器所选前缀的有效更新，不会补回被截断的尾部，也不保证不同 rank 模型有相同输出或训练轨迹。

非常重要的运行时发现：CNN YAML 中的 alpha 不等于当前实际 alpha。NNModelArgs 将额外参数保存在字典中，没有自动创建任意同名属性。CIFAR CNN / VGG9 构建器使用 getattr(args, 'lora_alpha_conv', default) 等读取方式，读不到字典里的固定 alpha；默认最终是 alpha=本地 rank。因此当前正常 factory 路径下，CIFAR-10、CINIC-10 和 SVHN VGG9 的 LoRA scaling 都是 1。这一点已实际构建模型核实，而非只看注释。

实测例子：CIFAR/CINIC 卷积 ratio=1 时 r=64、alpha=64、s=1；ratio=.25 时 r=16、alpha=16、s=1。FC 对应 (r,alpha)=(16,16) 和 (4,4)，s 都是 1。SVHN VGG9 卷积对应 (8,8) 和 (2,2)，FC 对应 (16,16) 和 (4,4)，s 也都是 1。所以当前这些 CNN 上，RankScale 的额外 c_i、d_i 都为 1，尺度变换为恒等变换。该问题尚未在这次分析中修改。

MNIST canonical MLP 的实际 alpha=2（构建器默认 lora_rank=4、lora_scaling=.5 的乘积取整），隐藏层真实 rank=int(160×ratio)，输出层 int(100×ratio) 后受矩阵维度上限约束。实测 ratio=1/.25 时，隐藏层 r=160/40、s=.0125/.05；输出层均 r=10、s=.2。这说明同一个 ratio 在不同层未必导致相同的 scaling 比例。

TinyBERT 构建器使用 args.get()，能正确读取当前 NLP YAML 的 base rank=8、alpha=16；实际 r=max(1,round(8×ratio))、s=16/r。它的异构秩尺度补偿通常不是恒等变换。这里已检查构建源码和配置，未下载/实例化完整预训练 TinyBERT，也未重跑训练。

卷积必须区分逻辑 rank r 与因子槽位数：kernel_size=k 时 A.shape=(r*k,C_in*k)、B.shape=(C_out*k,r*k)，forward scaling 分母仍为 r。因此 profile 读取 module.r；广播按本地形状保留 r_i*k 个槽位。不能拿 A.shape[0] 当成 alpha/r 的分母，也不能把 canonicalization 后的非零奇异值个数当成模块逻辑 rank。

请保留以下边界：

- 这仍是因子平均。即使统一了 scaling，B_bar A_bar 通常也不等于 sum p_i B_tilde_i A_tilde_i；独立平均有交叉项，异秩时还有不同的支持分母。不要称它为精确的稠密更新 FedAvg。
- canonicalization 发生在聚合后，不是先对齐所有客户端的因子基底。有效更新保持也不代表梯度、优化器轨迹保持。
- 这是逐层标量尺度转换。只支持直接 Tensor 型 A/B 和正有限的 scaling/alpha，要求匹配的层名和键；容器型 PEFT 多 adapter 参数不受该 profile 实现支持。
- profile 被缓存；客户端替换模型对象会自动刷新，但同一模型对象中原地改 scaling 需要显式刷新。服务器 profile 也不是动态追踪的。
- 服务器 profile 来源是 node_var.model。inference model 的 rank 在另一条路径按最大 rank_ratio 构建。当前 20 个 RankScale CNN/NLP 配置最大 ratio 均为 1；自定义不同最大 ratio 时还要检查服务器模型与 evaluator 的尺度一致性。
- 统计有效更新 Frobenius 能量时需乘 s²。canonicalization 的原始奇异值对应 BA；仓库 SPMI 工具支持 layer_scalings，但默认是 1，不能直接将未缩放的绝对能量当作实际 forward 更新能量。

当前 AWS 入口 rbla_plus_experiment_aws.py 选择 rbla_plus_rank_scale，共 20 个配置；G4 robustness 入口的方法列表仍选择普通 rbla_plus。不能将所有带 RBLA+ 标签的历史结果一概称为已做 scaling compensation。

验证情况：已有 rank-scale strategy、SP+ strategy、canonicalization 测试共 48 通过、1 因 CUDA 不可用跳过；另检查 20 个配置引用，并实际构建上述 CNN/MLP 读取了 profile。没有进行端到端训练对比，因此这些事实不证明准确率或收敛收益。

关键源码相对路径：

- src/flex/ml_algorithms/lora/rank_scale_alignment.py
- src/flex/fed_strategy/client_strategy_impl/_rbla_plus_rank_scale_client.py
- src/flex/fed_strategy/server_strategy_impl/_rbla_plus_rank_scale_server.py
- src/flex/fl_algorithms/aggregation/methods/_fed_aggregator_rbla.py
- src/flex/fl_algorithms/aggregation/methods/_fed_aggregator_sp_plus.py
- src/flex/ml_algorithms/lora/canonicalization.py
- src/flex/ml_algorithms/lora/impl/lora_ms.py
- src/flex/ml_models/nn_model_args.py
- src/flex/ml_models/lora/_nn_model_cifar_lora_cnn.py
- src/flex/ml_models/lora/_nn_model_vgg_lora_cnn.py
- src/flex/ml_models/lora/_nn_model_simple_lora_mlp.py
- src/flex/ml_models/lora/_nn_model_tinybert_lora.py

请在后续讨论中分别讨论“普通 RBLA+”“显式 RankScale 版本”和“当前各模型实际是否存在 scale 差异”，并在建议改代码或解释实验差异前先说明你依赖哪一种情况。
