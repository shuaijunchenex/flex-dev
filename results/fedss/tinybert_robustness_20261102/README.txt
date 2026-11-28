FedSS TinyBERT seeded Dirichlet robustness results - 2026-11-02

144/144 configurations completed successfully at 2026-11-02 07:05:06
Australia/Sydney. The batch began on 2026-10-22 and includes user-requested
pauses and restarts; the calendar span is not continuous training time.

Scope: AGNews and IMDB x Dirichlet alpha 0.1, 0.4, 0.8 x partition seeds
1, 2, 3 x eight methods. Rank assignment is rank_match only.
Methods: FedSS lambda 0.2, 0.4, 0.6, 0.8, 1.0; Flora; RBLA; ZeroPadding.
Each dataset has 72 runs. This bundle is separate from the KMNIST/QMNIST
results in ../robustness_20261027/.

Budget: 100 configured rounds with 101 evaluation records (rounds 0..100),
three local epochs, batch size 64, all 10 clients selected each round,
AdamW learning rate 0.0001. The training RNG seed is 42; directory seeds
select data partitions and their rank assignments. Actual runtime used
CUDA on an NVIDIA RTX 5080, although raw config headers retain device=cpu;
the entry selects the accelerator at runtime.

Files:
- raw_csvs/: 144 original CSVs, preserved byte-for-byte, including effective
  configuration metadata and the complete metric trajectories.
- summary.csv: one row per configuration. Accuracy is a fraction in [0,1].
  final_accuracy/final_loss use round 100; max_accuracy uses all 101 rounds;
  first_best_round selects the first occurrence when maxima tie.
- configs.txt: the 144 repository-relative configuration paths in run order.
- manifest.json: counts, source provenance, schedule, validation, metric
  summaries, config hashes, original CSV hashes and portable result paths.

All 144 CSVs passed validation for ordered unique rounds 0..100, finite
accuracy/loss, evaluated status and zero invalid-label samples. Each round
has 7,600 AGNews or 25,000 IMDB test samples. Published CSV bytes match the
hashes recorded when the results were saved. The scoped .gitattributes
preserves those bytes in Git.

Source: 71581e924aadeb56b1df7f7da332a292cd691e9a.
Entry: src/test/fedss_robustness_experiment_tinybert_01.py.
The original queue recorded a9dd7b0f9add6067d0b6e8e8bb737cd1b1e83eb6;
relevant training and matrix files did not change before source 71581e92
was archived. After completing 52 jobs, the remaining 92 used that archived
source, so later repository updates did not change their implementation.
Use the recorded source revision and raw effective configs for reproduction.

Job 53 was interrupted for the RBLA priority batch after 32 recorded rounds;
job 62 was interrupted for a computer restart after 13 recorded rounds.
Both were restarted from round 0 and completed successfully. Only their
successful attempts are published here. All jobs ran serially. Elapsed times
reflect wall time under varying machine workload, not isolated speed tests.
Verbose logs, partial attempts and local scheduler files remain local.
This publication adds result files only; training code and budgets are unchanged.
