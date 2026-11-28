# FLoRIST 主矩阵训练失败分析与补跑计划

## 1. 批次范围与结论

本报告对应 `src/test/rbla_plus_experiment_local.py` 执行的 32 项 FLoRIST 主矩阵实验：14 项 `correct`、4 项 `medium`、14 项 `mismatch`，Dirichlet α 为 0.4。所有时间使用 Australia/Sydney（UTC+11）。

- 批次日志起始时间：2026-10-17 20:23:18。
- 批次结束时间：2026-10-19 17:20:36。
- 总历时约 **44 小时 57 分钟**。
- **13 项完整完成，19 项失败**；19 项失败进程的退出码均为 `1`。
- 失败项中，**10 项留下部分训练 CSV，9 项在生成训练 CSV 前失败**。

原始证据：[本批次汇总日志](../src/test/batch_summary-20261017_202318-2c427f3e85583e73.log#L2)。

配置中的 `training_rounds=100` 经当前循环执行为 `round=0..100`，即 **101 次更新及评估**。本文统一使用“已完成轮数 / 101”表示进度。13 项成功任务均有完整、连续的 101 条评估记录；其余任务的部分结果不能作为完整实验结果使用。

## 2. 失败类别汇总

| 类别 | 数量 | 涉及任务 | 已确认的直接失败原因 |
|---|---:|---|---|
| 客户端训练数值异常 | 10 | FMNIST/QMNIST correct；四项 medium；四项 MNIST 系列 mismatch | 训练 loss 出现 `NaN`，随后 `_fc1.lora_A` 未通过有限值检查 |
| 缺少加载依赖 | 4 | MRPC、AGNews 的 correct/mismatch | 导入 `torchtext.datasets` 时抛出 `ModuleNotFoundError` |
| IMDB 地址解析失败 | 2 | IMDB correct/mismatch | 当前 Hub 解析器拒绝不含命名空间的 `imdb` 仓库地址 |
| CINIC-10 数据准备失败 | 2 | CINIC10 correct/mismatch | 下载解压后的目录布局未被加载器识别 |
| KMNIST 下载失败 | 1 | KMNIST correct | 标签文件下载遇到 HTTP 404 和连接重置 |
| **合计** | **19** | | |

失败任务累计耗时约 **23.80 小时**，其中数值异常任务约 **14.59 小时**，CINIC-10 两项约 **9.19 小时**。这些是子进程的总运行时间，不能进一步当作纯训练、下载或等待时间。

## 3. 原因、证据与待验证事项

### 3.1 客户端训练出现 NaN：10 项

这 10 项在退出前均记录了 `avg_loss=nan`，随后出现同一异常：

```text
ValueError: FLoRIST payload '_fc1.lora_A' must be a finite floating tensor
```

调用路径为客户端完成本地训练后，在 [`local_training_step`](../src/flex/fed_strategy/client_strategy_impl/_florist_client.py#L196) 中构造上传参数，并由 [`_validate_payload`](../src/flex/fed_strategy/client_strategy_impl/_florist_client.py#L87) 拒绝非有限值。异常发生在客户端参数校验阶段，这些堆栈没有报告服务器 SVD 失败。

FMNIST mismatch 的同一客户端在本地 epoch 1、2 的平均 loss 约为 1.58、1.53，epoch 3 开始变为 NaN，随后 epoch 4、5 仍继续执行。参见[汇总日志中的训练与异常记录](../src/test/batch_summary-20261017_202318-2c427f3e85583e73.log#L4000)。

当前配置为 SGD、学习率 `0.01`、每客户端本地训练 5 个 epoch。标准训练器在 [`loss.backward()` / `optimizer.step()`](../src/flex/model_trainer/trainer/_model_trainer_standard.py#L122) 附近没有逐 batch 非有限值中止或梯度裁剪；因此问题直到客户端训练结束、校验上传参数时才被报告。

**已确认**：本地训练产生非有限值，导致任务中途退出。

**尚未确认**：首次异常 batch 的位置，以及学习率、梯度、LoRA 因子尺度和参数更新实现中哪个环节首先失稳。现有日志不足以认定学习率过大，或直接归因于某一处缩放代码。

建议后续处理：

1. 选择一个已失败配置，在相同 seed、数据划分和训练预算下复现。
2. 记录首个非有限值 batch 的输入、loss、梯度范数及 LoRA A/B 参数范数；同步检查广播后和本地更新后的参数。
3. 在确认触发因素后，再验证修复或超参调整；不要删除有限值检查、吞掉异常或把 NaN 替换成零来伪造完成状态。
4. 若修改算法或公共超参，明确新的实验版本，并复验受影响的成功对照。

### 3.2 MRPC、AGNews 缺少 `torchtext`：4 项

两个加载器都在模块顶层直接导入 `torchtext.datasets`，当前环境未安装 `torchtext`：

```text
ModuleNotFoundError: No module named 'torchtext'
```

对应位置：

- [MRPC 加载器导入](../src/flex/ml_data_loader/loader/_dataset_loader_mrpc.py#L5)，[本批错误证据](../src/test/batch_summary-20261017_202318-2c427f3e85583e73.log#L1519)。
- [AGNews 加载器导入](../src/flex/ml_data_loader/loader/_dataset_loader_agnews.py#L5)，[本批错误证据](../src/test/batch_summary-20261017_202318-2c427f3e85583e73.log#L1624)。

这四项均在创建服务端数据加载器时退出，未开始训练。需要验证兼容的数据加载依赖方案，或迁移到统一的数据加载实现；补跑前检查样本数、标签编码和训练/评估划分。

### 3.3 IMDB 的 Hugging Face URI 解析失败：2 项

当前加载器调用 [`load_dataset("imdb")`](../src/flex/ml_data_loader/loader/_dataset_loader_imdb.py#L51)，最终产生形如 `hf://datasets/imdb@.../.huggingface.yaml` 的地址。已安装的 Hub 解析器要求仓库 ID 为 `namespace/name`，因此抛出：

```text
HfUriError: Repository id must be 'namespace/name', got 'imdb'.
```

参见 [correct 的错误](../src/test/batch_summary-20261017_202318-2c427f3e85583e73.log#L1583)和 [mismatch 的错误](../src/test/batch_summary-20261017_202318-2c427f3e85583e73.log#L5344)。

已使用当前环境的 URI 解析函数，在不联网的情况下复现相同错误；带命名空间的虚拟地址可以通过语法解析。这个检查只确认解析规则，没有验证替代仓库的存在或完整下载流程。

下一步需核实正确的完整数据集仓库 ID，检查加载接口与当前依赖组合，并验证实际训练/测试集能完整读取。

### 3.4 CINIC-10 解压布局识别失败：2 项

两项日志均出现下载达到 100%、开始解压，随后仍报告 `All four download sources failed`。参见 [correct 的末尾输出](../src/test/batch_summary-20261017_202318-2c427f3e85583e73.log#L1456)和 [mismatch 的末尾输出](../src/test/batch_summary-20261017_202318-2c427f3e85583e73.log#L5220)。

检查时本地存在大小为 **687,544,992 字节**的 `CINIC-10.tar.gz`。归档读取到的首批条目直接位于 `test/...`，而预期的数据目录是：

```text
.dataset/cinic-10/train/<class>/*.png
.dataset/cinic-10/valid/<class>/*.png
.dataset/cinic-10/test/<class>/*.png
```

当前 [`_extract_tar`](../src/flex/ml_data_loader/loader/_dataset_loader_cinic10.py#L71) 将内容解压到 `.dataset`，之后只查找“包含 train 子目录的外层目录”。对于包根直接包含 split 目录的布局，这个分支不能把内容归整到 `cinic-10`。检查时 `cinic-10` 目录为空。

已执行一个最小复现：用未修改的仓库函数处理仅包含 `train/valid/test` 三个目录的小型压缩包。结果为根目录 split 齐全，但目标 `cinic-10` 为空，存在检查返回 `False`。参见[复现结果](H:/chatgpt_output/training_analysis/20261017_florist/cinic_layout_reproduction.json)。

实际解压目录 `train/valid/test` 的枚举受当前环境权限限制，尚未验证全部真实图片的完整性；最小复现用于验证目录处理缺陷。

两次 CINIC-10 任务累计耗时约 **9 小时 11 分钟**。此外，所有来源失败后代码会进入交互式安装询问；尾日志不足以精确拆分下载、解压、重试和交互等待各占多久。

建议修复平铺与嵌套两种归档布局的识别，并在训练前验证目录、类别及图片数量。批处理的数据准备应避免等待交互输入。

### 3.5 KMNIST correct 下载失败：1 项

`train-labels-idx1-ubyte.gz` 的下载遇到：

- GitHub 镜像返回 `HTTP Error 404: Not Found`。
- 官方镜像连接被重置，记录 `WinError 10054`。

参见[具体下载错误](../src/test/batch_summary-20261017_202318-2c427f3e85583e73.log#L669)和[镜像设置代码](../src/flex/ml_data_loader/loader/_dataset_loader_kmnist.py#L17)。

后续 KMNIST medium/mismatch 已进入训练，说明数据准备状态后来发生了变化；不能因此假定当前缓存一定完整。补跑前应验证缓存文件及样本数，再检查有效下载来源。

## 4. 19 项失败清单

下表的轮数为已写入 CSV 的完整评估次数。例如 `86/101` 表示存在 `round=0..85` 的记录，不代表失败发生于已记录的最后一轮内。`0/101` 表示没有生成训练 CSV。

| 数据集 | 分配组 | 已完成轮数 | 失败类别 |
|---|---|---:|---|
| FMNIST | correct | 86/101 | NaN / 参数非有限值 |
| KMNIST | correct | 0/101 | 数据下载 |
| QMNIST | correct | 83/101 | NaN / 参数非有限值 |
| CINIC10 | correct | 0/101 | 数据准备/目录识别 |
| MRPC | correct | 0/101 | 缺少 torchtext |
| IMDB | correct | 0/101 | URI 解析 |
| AGNews | correct | 0/101 | 缺少 torchtext |
| MNIST | medium | 93/101 | NaN / 参数非有限值 |
| FMNIST | medium | 78/101 | NaN / 参数非有限值 |
| KMNIST | medium | 98/101 | NaN / 参数非有限值 |
| QMNIST | medium | 79/101 | NaN / 参数非有限值 |
| MNIST | mismatch | 64/101 | NaN / 参数非有限值 |
| FMNIST | mismatch | 64/101 | NaN / 参数非有限值 |
| KMNIST | mismatch | 62/101 | NaN / 参数非有限值 |
| QMNIST | mismatch | 72/101 | NaN / 参数非有限值 |
| CINIC10 | mismatch | 0/101 | 数据准备/目录识别 |
| MRPC | mismatch | 0/101 | 缺少 torchtext |
| IMDB | mismatch | 0/101 | URI 解析 |
| AGNews | mismatch | 0/101 | 缺少 torchtext |

补跑路径见[失败配置清单](H:/chatgpt_output/training_analysis/20261017_florist/failed_configs.txt)。该清单保存的是本机绝对路径；在其他工作目录使用时，应按仓库相对路径定位对应 YAML。

## 5. 补跑顺序与结果保留

1. **保留已有完整结果**：MNIST correct，以及 CIFAR10、SVHN、SST2、QQP、RTE、MNLI 的 correct/mismatch，共 13 项。
2. **先修复 9 项数据加载失败**：验证依赖、数据集标识、目录布局、样本数与标签，然后再启动训练。
3. **复现定位 10 项数值失败**：增加首个异常 batch 的诊断信息，确认修复后再运行完整预算。
4. **增加逐任务即时日志保存及恢复检查点**：当前批次汇总在队列结束后才写文件，失败信息仅保留最后最多 400 行输出。
5. **使用独立补跑清单**：避免无差别重复已经完成的实验。如果修复改变训练算法或公共超参，应同时复验受影响的对照，并区分实验版本。

本批配置为 [`save_weights: false`](../src/yamls/training_logger/experiment_training_logger.yaml#L8)，检查时未找到本批权重检查点，现有入口也不提供断点恢复。因此，中断任务不能直接从最后一条 CSV 继续训练，需要从头补跑。

## 6. 复核环境与分析边界

| 依赖 | 检查时的版本 |
|---|---|
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| torchtext | 未安装 |
| datasets | 5.0.1 |
| huggingface-hub | 1.29.0 |
| fsspec | 2026.6.0 |
| transformers | 5.16.1 |

本报告基于本批次最终汇总、23 份训练 CSV、相关代码与两项小型离线复现。已保存评估记录的轮数连续、数值有限，未发现无效标签或未执行评估的行；失败轮次可能尚未产生 CSV 记录，因此不能用 CSV 中没有 NaN 否定客户端训练出现 NaN。

数值异常的首发原因、真实 CINIC-10 全量图片完整性、修复后的完整训练表现仍需后续验证。本报告用于定位问题和安排补跑，不代表上述修复已经实施。
