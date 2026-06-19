from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def auc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank AUC with tie handling; avoids sklearn dependency."""
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    wins = 0.0
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / total)


def metrics_at_edge_count(df: pd.DataFrame) -> dict[str, float | int]:
    """Precision/recall/F1 at the planted edge count, plus AUC."""
    y = df["y_true"].astype(int).to_numpy()
    s = df["score"].astype(float).to_numpy()
    m = int(y.sum())
    if m <= 0 or m >= len(y):
        raise ValueError(
            f"Degenerate directed-edge truth: positives={m}, total={len(y)}. "
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

    return {
        "n_pairs": int(len(y)),
        "n_edges": int(m),
        "precision_at_m": float(precision),
        "recall_at_m": float(recall),
        "f1_at_m": float(f1),
        "auc": float(auc),
    }


def fmt(mean: float, std: float) -> str:
    return f"${mean:.3f}\\pm{std:.3f}$"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="CSV with benchmark,method,seed,src,dst,score")
    ap.add_argument("--truth", required=True, help="CSV with benchmark,seed,src,dst,y_true")
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
    for (benchmark, method, seed), g in df.groupby(["benchmark", "method", "seed"]):
        row = {"benchmark": benchmark, "method": method, "seed": int(seed)}
        row.update(metrics_at_edge_count(g))
        raw_rows.append(row)
    raw = pd.DataFrame(raw_rows)

    metric_cols = ["precision_at_m", "recall_at_m", "f1_at_m", "auc"]
    summary_rows = []
    for (benchmark, method), g in raw.groupby(["benchmark", "method"]):
        row = {
            "benchmark": benchmark,
            "method": method,
            "n_seeds": int(g["seed"].nunique()),
            "n_pairs": int(g["n_pairs"].iloc[0]),
            "n_edges": int(g["n_edges"].iloc[0]),
        }
        for col in metric_cols:
            row[col + "_mean"] = float(g[col].mean())
            row[col + "_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    raw.to_csv(outdir / "sparse_directed_edges_raw.csv", index=False)
    summary.to_csv(outdir / "sparse_directed_edges_summary.csv", index=False)

    print(summary.to_string(index=False))
    print("\nLaTeX rows:")
    for _, r in summary.iterrows():
        print(
            f"{r['benchmark']} & {int(r['n_pairs'])} & {int(r['n_edges'])} & "
            f"{fmt(r['precision_at_m_mean'], r['precision_at_m_std'])} & "
            f"{fmt(r['recall_at_m_mean'], r['recall_at_m_std'])} & "
            f"{fmt(r['f1_at_m_mean'], r['f1_at_m_std'])} & "
            f"{fmt(r['auc_mean'], r['auc_std'])} \\\\" 
        )


if __name__ == "__main__":
    main()
