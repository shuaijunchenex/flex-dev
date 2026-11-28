FedSS seeded Dirichlet robustness batch, completed2026-10-27 09:08:06 Australia/Sydney.
144/144 configurations completed successfully; no failed configurations.
Scope: KMNIST,QMNIST x alpha0.1,0.4,0.8 x partition seed1,2,3 x8 methods.
Methods: FedSS lambda0.2,0.4,0.6,0.8,1.0 (90 runs); Flora,RBLA,ZeroPadding (54 runs).
Rank assignment: rank_match only. Runtime/model training seed remains42.
Unchanged budget: 100 configured rounds, inclusive evaluation rounds0..100,5 local epochs.
Original CSV bytes and effective configuration metadata are retained in raw_csvs/.
All144 CSVs were checked for101 unique rounds and finite accuracy/loss.
Accuracy values in summary.csv and manifest.json are fractions; max_accuracy uses
the complete trajectory and first_best_round breaks ties by first occurrence.
Manifest records config paths, CSV paths and SHA256 hashes. The local batch summary
independently reported total=144 success=144 fail=0. No logs,tracebacks,environment
snapshots,training source changes or unrelated historical runs are uploaded.
The CSV headers retain source configuration values; actual runtime used Mac MPS.
