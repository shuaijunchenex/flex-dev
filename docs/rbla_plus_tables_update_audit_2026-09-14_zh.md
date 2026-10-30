已将用户提供的两张 LaTeX 表中的 12 个 RBLA+ 值替换为当前工作区最新 RankScale 日志的最高 accuracy × 100。

先按数据集与秩分配选择文件名时间戳最新的运行，再取该运行已记录轮次中的最高值；没有在旧、新方法或多次运行之间择优。显示保留两位小数。

10 项日志覆盖 round 0–100。SVHN correct 仅覆盖 0–63，MNLI mismatch 仅覆盖 0–42，按用户确认填入目前最高值并用 † 标记。

MRPC、QQP、RTE、SST-2 和 MNIST 系列没有找到新的 RankScale 日志，保留用户原表值。这些原值没有在本次检查中重新从原始日志验证。其他方法的所有数值保持不变。更新后重新判断受影响行的最佳结果加粗。

| 数据集 | 秩分配 | 原 RBLA+ (%) | 新 RBLA+ (%) | 峰值日志 round | 已记录轮次 | 状态 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| CIFAR-10 | mismatch | 79.21 | 78.95 | 59 | 0–100 | 完整 |
| CINIC-10 | mismatch | 69.25 | 68.89 | 99 | 0–100 | 完整 |
| SVHN | mismatch | 93.45 | 93.20 | 19 | 0–100 | 完整 |
| AG News | mismatch | 90.50 | 91.68 | 18 | 0–100 | 完整 |
| IMDB | mismatch | 81.41 | 83.64 | 13 | 0–100 | 完整 |
| MNLI | mismatch | 60.02 | 64.07† | 15 | 0–42 | 未完整 |
| CIFAR-10 | correct | 78.30 | 79.12 | 94 | 0–100 | 完整 |
| CINIC-10 | correct | 69.13 | 69.65 | 69 | 0–100 | 完整 |
| SVHN | correct | 92.90 | 92.53† | 58 | 0–63 | 未完整 |
| AG News | correct | 90.99 | 91.33 | 10 | 0–100 | 完整 |
| IMDB | correct | 82.25 | 84.32 | 8 | 0–100 | 完整 |
| MNLI | correct | 64.51 | 65.16 | 24 | 0–100 | 完整 |

来源逐项如下。已核对日志中的 server/client strategy 均为 rbla_plus_rank_scale、聚合器为 sp_plus、数据集及分布、客户端有效 rank ratios 与配置列表一致，并根据样本量与 rank ratio 的相关性验证 correct/mismatch 方向。

- CIFAR-10 / mismatch: [cifar10_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260830_154124-3dc18c46d1c4f90e.csv](../src/test/experiment_results/cifar10_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260830_154124-3dc18c46d1c4f90e.csv)，峰值 round=59，rank-volume Spearman=-1.000000。
- CINIC-10 / mismatch: [cinic10_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260831_084024-e5867a139c3fbb79.csv](../src/test/experiment_results/cinic10_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260831_084024-e5867a139c3fbb79.csv)，峰值 round=99，rank-volume Spearman=-1.000000。
- SVHN / mismatch: [svhn_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260901_223021-5135fb3adeee0962.csv](../src/test/experiment_results/svhn_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260901_223021-5135fb3adeee0962.csv)，峰值 round=19，rank-volume Spearman=-1.000000。
- AG News / mismatch: [agnews_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260904_115754-ec642dc2dc42c5ad.csv](../src/test/experiment_results/agnews_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260904_115754-ec642dc2dc42c5ad.csv)，峰值 round=18，rank-volume Spearman=-1.000000。
- IMDB / mismatch: [imdb_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260906_063020-a84a2401ad48de6c.csv](../src/test/experiment_results/imdb_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260906_063020-a84a2401ad48de6c.csv)，峰值 round=13，rank-volume Spearman=-1.000000。
- MNLI / mismatch: [mnli_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260911_091142-3b2cc50e630c1b24.csv](../src/test/experiment_results/mnli_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260911_091142-3b2cc50e630c1b24.csv)，峰值 round=15，rank-volume Spearman=-1.000000。
- CIFAR-10 / correct: [cifar10_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260830_114801-17922021afed9cdb.csv](../src/test/experiment_results/cifar10_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260830_114801-17922021afed9cdb.csv)，峰值 round=94，rank-volume Spearman=1.000000。
- CINIC-10 / correct: [cinic10_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260830_193615-8a64b050e747b3cb.csv](../src/test/experiment_results/cinic10_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260830_193615-8a64b050e747b3cb.csv)，峰值 round=69，rank-volume Spearman=1.000000。
- SVHN / correct: [svhn_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260831_204747-f3810428dff251cc.csv](../src/test/experiment_results/svhn_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260831_204747-f3810428dff251cc.csv)，峰值 round=58，rank-volume Spearman=1.000000。
- AG News / correct: [agnews_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260903_001433-cac77897798a03ce.csv](../src/test/experiment_results/agnews_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260903_001433-cac77897798a03ce.csv)，峰值 round=10，rank-volume Spearman=1.000000。
- IMDB / correct: [imdb_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260905_224859-6a64706e2055c12c.csv](../src/test/experiment_results/imdb_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260905_224859-6a64706e2055c12c.csv)，峰值 round=8，rank-volume Spearman=1.000000。
- MNLI / correct: [mnli_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260906_141523-e9768cd4ae91f2e0.csv](../src/test/experiment_results/mnli_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260906_141523-e9768cd4ae91f2e0.csv)，峰值 round=24，rank-volume Spearman=1.000000。

两张表保留原宏、列序、字号和分组；表下注明 RankScale 替换范围与两项日志不完整的情况。验证了 12 处 RBLA+ 数值替换、2 处 †、MNIST 系列原文完整保留、其他方法数值不变、LaTeX 花括号平衡。当前环境没有发现 LaTeX 编译器，因此未编译 PDF。

