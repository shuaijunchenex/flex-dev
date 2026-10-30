# RBLA random selection and checkpoint collection — 2026-10-28

18/18 training paths completed successfully. The four collection paths produced 40 validated server-input checkpoints.

- Source commit: 7f634e2a19748cd521f2c0f2deb6eb5cbc7ce451.
- Entry: src/test/rbla+/rbla_random_selection_and_reconstruction.py.
- Started: 2026-10-27 19:44:27; finished: 2026-10-28 18:15:38 (Australia/Sydney). Elapsed: 22 h 31 min.
- Device: NVIDIA RTX 5080, CUDA. Seed: 42. Each path has 101 logged rounds (0..100).
- Random selection: CIFAR-10 / MNLI x seven methods; 100 clients, 10 selected per round.
- Checkpoint collection: CIFAR-10 / MNLI x RBLA / RBLA+; 10 clients, all 10 selected per round.
- Local epochs: 5 for CIFAR-10, 1 for MNLI. Training code, seeds and budgets were not changed.

## Training results

Accuracy is a fraction in summary.csv; the table displays percentages. Peak accuracy is selected from a single full trajectory, not an independent test of statistical significance.

| Task | Case | Final accuracy | Peak accuracy | First peak round |
|---|---|---:|---:|---:|
| random_selection | cifar10_flexlora | 69.71% | 73.94% | 88 |
| random_selection | cifar10_flora | 61.25% | 64.07% | 88 |
| random_selection | cifar10_rbla | 71.95% | 75.17% | 88 |
| random_selection | cifar10_rbla_plus | 71.32% | 75.21% | 88 |
| random_selection | cifar10_zeropadding | 68.67% | 72.34% | 88 |
| random_selection | cifar10_rep_padding | 69.99% | 74.54% | 88 |
| random_selection | cifar10_florist | 48.64% | 56.84% | 81 |
| random_selection | mnli_flexlora | 43.38% | 52.93% | 91 |
| random_selection | mnli_flora | 57.75% | 57.91% | 97 |
| random_selection | mnli_rbla | 43.68% | 56.17% | 85 |
| random_selection | mnli_rbla_plus | 58.39% | 61.21% | 94 |
| random_selection | mnli_zeropadding | 42.96% | 56.43% | 85 |
| random_selection | mnli_rep_padding | 43.62% | 56.40% | 85 |
| random_selection | mnli_florist | 52.96% | 56.79% | 94 |
| cross_layer_reconstruction | cifar10_rbla | 46.97% | 56.72% | 51 |
| cross_layer_reconstruction | cifar10_rbla_plus | 81.55% | 82.78% | 81 |
| cross_layer_reconstruction | mnli_rbla | 44.11% | 49.85% | 3 |
| cross_layer_reconstruction | mnli_rbla_plus | 59.78% | 65.23% | 26 |

## Files and validation

The raw/ directory preserves all 124 original CSV/JSON files byte-for-byte, including effective configs, partitions, selected-client traces, statuses, trajectory metrics, and snapshot manifests. Original machine-specific absolute paths remain as recorded provenance; artifact_manifest.json maps every file to its portable path in this bundle.

All 18 primary trajectories have 101 unique ordered rounds with finite accuracy/loss. All random-selection methods share matching partition and selected-client-sequence hashes per dataset. All 40 snapshot SHA-256 values match their worker-produced manifests.

Checkpoint rounds: 9, 19, 29, 39, 49, 59, 69, 79, 89, 100. Offline reconstruction analysis, server benchmarks, and the paused 144-case FedSS batch are outside this entry's completed scope.

## Download checkpoints

Reports, metrics, and CSV/JSON provenance are stored in Git. The 40 binary server-input checkpoints are shared in the [checkpoint Release](https://github.com/shuaijunchenex/flex-src/releases/tag/rbla-checkpoints-20261028), as four archives with 10 checkpoints each. The Release tag pins training source commit 7f634e2a19748cd521f2c0f2deb6eb5cbc7ce451.

Download the four rbla-checkpoints-*.tar.gz archives and CHECKPOINT_SHA256SUMS.txt from that Release. Verify archive hashes, then extract all four into this result-bundle directory to populate raw/cross_layer_reconstruction/<case>/server_inputs/. See checkpoint_assets.json for download URLs, sizes, archive hashes, and relative paths/hashes for all 40 snapshots.

The complete local run contains 182 original files: 124 CSV/JSON files in Git, 40 checkpoint files in the Release, and 18 verbose training logs retained locally. Original files and local archives are preserved. The reports-and-logs archive is not uploaded. artifact_manifest.json records the storage location, size, and hash of every original file.

Original snapshot manifests retain absolute paths from the training machine. After extraction, use the portable paths in checkpoint_assets.json when passing snapshots to analysis tools. These server-input snapshots support offline reconstruction and aggregation analysis; they do not include all client/optimizer state needed to resume training. The scoped .gitattributes rules preserve committed data and checksum-file bytes.

## Scheduling and scope

A short two-worker CNN trial was conducted, then the user requested serial scheduling at 2026-10-27 20:49:15. Active workers finished naturally; later tasks ran serially. See scheduling.json. Per-case elapsed times during concurrency are not isolated algorithm-speed measurements.

The older FedSS batch remains paused at 52/144 completed with 92 configurations remaining. It was not automatically resumed, and its unrelated local CSVs are not included here.
