# Powered sparse directed-edge diagnostic

This repository includes the sparse directed-edge diagnostic added after the original `hawkes-regime-repro` archive. The goal is to test directed-edge recovery on a non-degenerate ordered-edge task, rather than only member-set recovery or hub identification.

## What changed

The simulator now supports a configurable planted active-edge rule via `SimConfig.active_edge_rule`:

- `dense_subgroup`: original completed synthetic benchmark behavior; every non-self ordered pair inside `ring_actors` is positive.
- `cycle`: sparse directed cycle inside the subgroup.
- `hub_to_others`: hub sends to all other subgroup members.
- `hub_plus_cycle`: hub-to-others plus directed cycle. This is the powered diagnostic used for the reviewer response.

The powered diagnostic uses `active_edge_rule=hub_plus_cycle` with `|S*|=10`, yielding 90 ordered non-self candidate pairs and 18 planted directed edges. The random F1@|E*| baseline is therefore 18/90 = 0.200, and the random AUC baseline is 0.5.

## PowerShell commands

Run from the repository root.

Smoke test:

```powershell
python -m regime_hawkes.run_sparse_directed_powered_mp `
  --benchmark large `
  --large-k 10 `
  --large-K 30 `
  --large-T 900 `
  --seeds 1 `
  --max-iters 2 `
  --workers 1
```

Full powered diagnostic:

```powershell
Remove-Item .\results\sparse_directed_powered -Recurse -Force -ErrorAction SilentlyContinue

python -m regime_hawkes.run_sparse_directed_powered_mp `
  --benchmark large `
  --large-k 10 `
  --large-K 30 `
  --large-T 900 `
  --seeds 20 `
  --max-iters 30 `
  --workers 8
```

Evaluate:

```powershell
python -m regime_hawkes.evaluate_sparse_directed_edges_with_baselines `
  --scores .\results\sparse_directed_powered\sparse_directed_scores.csv `
  --truth .\results\sparse_directed_powered\sparse_directed_truth.csv `
  --outdir .\results\sparse_directed_powered
```

## Completed powered diagnostic summary

The completed run used 20 independent simulated datasets and 30 EM iterations per fit.

```text
Sparse large & 20 & 90 & 18 & $0.200$ & $0.508\pm0.114$ & $0.308$ & $0.500$ & $0.729\pm0.045$ & $0.229$ \\
```

The summary CSV is included at:

```text
runs/sparse_directed_powered/sparse_directed_edges_summary.csv
paper_outputs/table_sparse_directed_powered_summary.csv
```

## Suggested paper table row

```latex
Sparse large & 20 & 90 & 18 &
$0.200$ & $0.508\pm0.114$ & $+0.308$ &
$0.500$ & $0.729\pm0.045$ & $+0.229$ \\
```
