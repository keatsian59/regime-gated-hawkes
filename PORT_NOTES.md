# Port notes: `acmccs226/hawkes-regime-repro` to `keatsian59/regime-gated-hawkes`

This folder is a port of the uploaded `hawkes-regime-repro-main.zip` repository to the new repository name `regime-gated-hawkes`.

Additional experiment code added in this port:

- `regime_hawkes/simulate.py` now supports `SimConfig.active_edge_rule` and stores planted directed-edge metadata.
- `regime_hawkes/simulate_sparse_directed_patch.py` is kept as an explicit compatibility module for the sparse directed-edge runners.
- `regime_hawkes/run_sparse_directed_powered_mp.py` runs the powered multiprocessing sparse directed-edge diagnostic.
- `regime_hawkes/evaluate_sparse_directed_edges_with_baselines.py` evaluates directed-edge F1/AUC with chance baselines and lift over chance.
- `regime_hawkes/run_sparse_directed_synthetic.py` and `regime_hawkes/run_sparse_directed_synthetic_mp.py` are the smaller sparse diagnostic runners used before the powered run.
- `regime_hawkes/evaluate_sparse_directed_edges.py` is the smaller diagnostic evaluator.
- `SPARSE_DIRECTED_DIAGNOSTIC.md` documents the new experiment and PowerShell commands.

Completed powered diagnostic summary included:

- `runs/sparse_directed_powered/sparse_directed_edges_summary.csv`
- `paper_outputs/table_sparse_directed_powered_summary.csv`

The powered diagnostic row from the completed reviewer-response run is:

```text
Sparse large & 20 & 90 & 18 & $0.200$ & $0.508\pm0.114$ & $+0.308$ & $0.500$ & $0.729\pm0.045$ & $+0.229$ \\
```
