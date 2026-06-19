from __future__ import annotations

"""Evaluate sparse directed-edge diagnostics with chance baselines.

Place under ``regime_hawkes/evaluate_sparse_directed_edges_with_baselines.py``
and run from repo root, for example:

    python -m regime_hawkes.evaluate_sparse_directed_edges_with_baselines `
      --scores .\results\sparse_directed_powered\sparse_directed_scores.csv `
      --truth .\results\sparse_directed_powered\sparse_directed_truth.csv `
      --outdir .\results\sparse_directed_powered

The random-ranking baseline for precision/recall/F1 at the planted edge count is
|E*| / number_of_candidate_ordered_pairs. The random AUC baseline is 0.5.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def auc_rank(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)

    pos = score[y_true == 1]
    neg = score[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    wins = 0.0
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def metrics_at_edge_count(df):
    y = df["y_true"].astype(int).to_numpy()
    s = df["score"].astype(float).to_numpy()

    n_pairs = int(len(y))
    m = int(y.sum())
    if m <= 0 or m >= n_pairs:
        raise ValueError(
            f"Degenerate directed-edge truth: positives={m}, total={n_pairs}. "
            "Need both positives and negatives."
        )

    order = np.argsort(-s)
    pred = np.zeros_like(y)
    pred[order[:m]] = 1

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = auc_rank(y, s)

    chance_prf = m / n_pairs
    chance_auc = 0.5
    auc_granularity = 1.0 / (m * (n_pairs - m)) if 0 < m < n_pairs else np.nan

    return {
        "n_pairs": n_pairs,
        "n_edges": int(m),
        "n_non_edges": int(n_pairs - m),
        "edge_density": float(chance_prf),
        "precision_at_m": float(precision),
        "recall_at_m": float(recall),
        "f1_at_m": float(f1),
        "auc": float(auc),
        "chance_precision_at_m": float(chance_prf),
        "chance_recall_at_m": float(chance_prf),
        "chance_f1_at_m": float(chance_prf),
        "chance_auc": float(chance_auc),
        "f1_lift_over_chance": float(f1 - chance_prf),
        "auc_lift_over_chance": float(auc - chance_auc),
        "auc_granularity": float(auc_granularity),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores)
    truth = pd.read_csv(args.truth)

    df = scores.merge(
        truth,
        on=["benchmark", "seed", "src", "dst"],
        how="inner",
        validate="one_to_one",
    )

    if df.empty:
        raise SystemExit("No rows after merging scores and truth.")

    raw_rows = []
    for (benchmark, method, seed), g in df.groupby(["benchmark", "method", "seed"], sort=True):
        row = {
            "benchmark": benchmark,
            "method": method,
            "seed": int(seed),
        }
        row.update(metrics_at_edge_count(g))
        raw_rows.append(row)

    raw = pd.DataFrame(raw_rows)

    metric_cols = [
        "precision_at_m",
        "recall_at_m",
        "f1_at_m",
        "auc",
        "f1_lift_over_chance",
        "auc_lift_over_chance",
    ]

    summary_rows = []
    for (benchmark, method), g in raw.groupby(["benchmark", "method"], sort=True):
        row = {
            "benchmark": benchmark,
            "method": method,
            "n_seeds": int(g["seed"].nunique()),
            "n_pairs": int(g["n_pairs"].iloc[0]),
            "n_edges": int(g["n_edges"].iloc[0]),
            "n_non_edges": int(g["n_non_edges"].iloc[0]),
            "edge_density": float(g["edge_density"].iloc[0]),
            "chance_f1_at_m": float(g["chance_f1_at_m"].iloc[0]),
            "chance_auc": float(g["chance_auc"].iloc[0]),
            "auc_granularity": float(g["auc_granularity"].iloc[0]),
        }
        for col in metric_cols:
            row[col + "_mean"] = float(g[col].mean())
            row[col + "_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    raw.to_csv(outdir / "sparse_directed_edges_raw.csv", index=False)
    summary.to_csv(outdir / "sparse_directed_edges_summary.csv", index=False)

    display_cols = [
        "benchmark", "method", "n_seeds", "n_pairs", "n_edges", "n_non_edges",
        "chance_f1_at_m", "f1_at_m_mean", "f1_at_m_std", "f1_lift_over_chance_mean",
        "chance_auc", "auc_mean", "auc_std", "auc_lift_over_chance_mean", "auc_granularity",
    ]
    print(summary[display_cols].to_string(index=False))

    print("\nLaTeX rows with baselines:")
    for _, r in summary.iterrows():
        print(
            f"{r['benchmark']} & {int(r['n_seeds'])} & {int(r['n_pairs'])} & {int(r['n_edges'])} & "
            f"${r['chance_f1_at_m']:.3f}$ & "
            f"${r['f1_at_m_mean']:.3f}\\pm{r['f1_at_m_std']:.3f}$ & "
            f"${r['f1_lift_over_chance_mean']:.3f}$ & "
            f"${r['chance_auc']:.3f}$ & "
            f"${r['auc_mean']:.3f}\\pm{r['auc_std']:.3f}$ & "
            f"${r['auc_lift_over_chance_mean']:.3f}$ \\\\"  # noqa: W605
        )


if __name__ == "__main__":
    main()
