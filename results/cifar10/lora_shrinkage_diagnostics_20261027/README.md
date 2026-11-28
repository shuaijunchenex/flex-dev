# CIFAR-10 LoRA shrinkage diagnostics — 2026-10-27

Source commit: f287c112. Seed 42; original training logged rounds 0..100, five local epochs. Checkpoints: 9,19,29,39,49,59,69,79,89,100. MPS forward; CPU FP64 norm and response statistics. PyTorch 2.8.0 / torchvision 0.23.0.

Acceptance: 450 valid evaluations, 90 complete paired groups, 10 checkpoints; no missing rounds, failed or invalid treatments. Each evaluation uses 10,000 CIFAR-10 test samples; probes use the fixed first 128 samples. All 720 layer norm matches pass (max relative error 1.188e-7; tolerance 5e-5). All 30 minimum-budget support_zero/RBLA comparisons are identical in accuracy, loss, logits and predictions.

At maximum budget, mean accuracy changes from RBLA are: head_only +9.527 percentage points, first_conv +3.014, all_conv -25.489. The head_only support_zero mean accuracy is 54.269%, versus 46.799% for norm_matched, 26.808% for permuted_reverse, and 38.742% for permuted_halfshift. Medium-budget effects are small and inconsistent; minimum-budget effects are zero. See paired_contrasts.csv for every checkpoint and scope_budget_summary.csv for averages.

These are controlled interventions on the same snapshots of one training trajectory, not independently trained ZeroPadding models or independent-seed significance estimates. Lower loss alone does not imply higher accuracy. Results do not establish long-term regularization or client-bias mechanisms; tail slots are not identified as high-frequency features.

Machine-specific paths are replaced by $RUN_ROOT, $REPO_ROOT and $PYTHON. Large binary server-input snapshots, dataset, environment, and verbose training logs remain in the original external run directory and are not included in this Git result bundle. Snapshot hashes are retained in the checkpoint reports. No training source changes were needed.
