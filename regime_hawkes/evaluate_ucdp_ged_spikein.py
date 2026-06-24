from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE_GROUP_COLS = ["method", "injection_strength", "coarsen_days"]
# Synthetic-P2 runs use a different schema (T_level, alpha_min_relative) and have
# no injection_strength/coarsen_days. Group on whatever axes are present.
OPTIONAL_GROUP_COLS = ["weak_receiver_frac", "episode_length", "n_episodes", "p2_fixed_active_rate",
                       "T_level", "alpha_min_relative"]

METRICS = [
    "member_f1", "receiver_f1", "hub_correct", "active_auc",
    "member_exact", "receiver_exact", "member_receiver_exact", "all_role_exact",
    "weak_receiver_recovered", "weak_receiver_rank", "weak_receiver_score",
    "gamma_gap_planted", "r_gamma_planted", "gamma_truth_corr",
    "gamma_active_minus_dormant", "r_gamma_oracle",
    "mark_escalation_lift", "active_occ", "loglik_final",
]
EXPOSURE = [
    "n_eff_planted", "signal_total", "signal_min_member_count",
    "signal_min_receiver_count", "signal_source_count", "alpha_min_proxy",
    "alpha_min_relative", "weak_receiver_share_realized", "receiver_directed_total",
    "expected_bg_active_windows", "signal_to_expected_active_bg_ratio",
    "nominal_lambda_delta_005", "lambda_T_delta_005",
    "orig_min_member_bg_count", "orig_mean_member_bg_count",
    "leftover_min_member_bg_count", "converted_min_member_count",
    "pi1T_planted", "pi1_planted", "injected_events_exposure_scale",
]

PRETTY = {
    "Proposed": "Proposed",
    "B1_static_gates": "B1 static gates",
    "B2_modular_hmm_hawkes": "B2 modular HMM-Hawkes",
    "B3_static_svd": "B3 static SVD",
}
METHOD_ORDER = ["B1_static_gates", "Proposed", "B2_modular_hmm_hawkes", "B3_static_svd"]
BASELINES = ["B1_static_gates", "B2_modular_hmm_hawkes", "B3_static_svd"]


def _ok(raw: pd.DataFrame) -> pd.DataFrame:
    return raw[raw.get("status", "ok").fillna("ok").astype(str).eq("ok")].copy()


def _num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _f(x, digits: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return "---"
    return "---" if not np.isfinite(v) else f"{v:.{digits}f}"


def _resolve_strength(df: pd.DataFrame, requested: float | None) -> float:
    strengths = sorted(_num(df["injection_strength"]).dropna().unique())
    if not strengths:
        raise ValueError("no injection_strength values present")
    if requested is None:
        return float(strengths[-1])
    arr = np.asarray(strengths, dtype=float)
    return float(arr[int(np.argmin(np.abs(arr - float(requested))))])


def _group_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in BASE_GROUP_COLS if c in df.columns]
    if not cols:
        cols = ["method"] if "method" in df.columns else []
    for c in OPTIONAL_GROUP_COLS:
        if c in df.columns and df[c].notna().any() and c not in cols:
            cols.append(c)
    return cols


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    df = _ok(raw)
    group_cols = _group_cols(df)
    cols = [c for c in (METRICS + EXPOSURE) if c in df.columns]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(g))
        for c in cols:
            vals = _num(g[c])
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def paired_delta(raw: pd.DataFrame, metric: str, strength: float, coarsen: float,
                 baseline: str, n_boot: int, seed: int) -> dict | None:
    df = _ok(raw)
    keys = [k for k in ["window_id", "seed"] if k in df.columns]
    if not keys or metric not in df.columns:
        return None

    def sel(m):
        return df[(df["method"].astype(str).eq(m))
                  & np.isclose(_num(df["injection_strength"]), strength)
                  & np.isclose(_num(df["coarsen_days"]), coarsen)][keys + [metric]]

    p, b = sel("Proposed"), sel(baseline)
    if p.empty or b.empty:
        return None
    merged = p.merge(b, on=keys, suffixes=("_p", "_b"))
    d = (_num(merged[f"{metric}_p"]) - _num(merged[f"{metric}_b"])).dropna().to_numpy()
    if d.size == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(int(n_boot))])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"baseline": baseline, "metric": metric, "n_pairs": int(d.size),
            "mean_delta": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def write_p1_latex(summary: pd.DataFrame, out_path: Path) -> None:
    df = summary[np.isclose(_num(summary["coarsen_days"]), 0.0)].copy()
    lines = [
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"Method & $\Neff$ & $\lambda_T$ & signal/bg & Member F1 & Receiver F1 & Weak recv. & Exact mem. & Exact recv. & $R_\gamma$ \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        g = df[df["method"].astype(str).eq(method)].sort_values("n_eff_planted_mean" if "n_eff_planted_mean" in df.columns else "injection_strength")
        if g.empty:
            continue
        for _, r in g.iterrows():
            n_eff = r.get("n_eff_planted_mean", r.get("signal_total_mean", np.nan))
            lam = r.get("lambda_T_delta_005_mean", r.get("nominal_lambda_delta_005_mean", np.nan))
            lines.append(
                f"{PRETTY.get(method, method)} & {_f(n_eff,0)} & {_f(lam,3)} & "
                f"{_f(r.get('signal_to_expected_active_bg_ratio_mean', np.nan),3)} & "
                f"{_f(r.get('member_f1_mean', np.nan),3)} & "
                f"{_f(r.get('receiver_f1_mean', np.nan),3)} & "
                f"{_f(r.get('weak_receiver_recovered_mean', np.nan),3)} & "
                f"{_f(r.get('member_exact_mean', np.nan),3)} & "
                f"{_f(r.get('receiver_exact_mean', np.nan),3)} & "
                f"{_f(r.get('r_gamma_planted_mean', np.nan),3)} \\")
        lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_p3_bands(raw: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Within-exposure hard-path R_gamma diagnostic only.

    No pooled table across N_eff is produced, because R_gamma is normalized by
    lambda_T and lambda_T changes with N_eff. This is a diagnostic, not a C6 proof.
    """
    df = _ok(raw)
    df = df[df["method"].astype(str).eq("Proposed")].copy()
    if "r_gamma_planted" not in df.columns or "n_eff_planted" not in df.columns:
        out_path.write_text("", encoding="utf-8")
        return pd.DataFrame()
    bins = [-np.inf, 0.5, 1.0, 1.5, np.inf]
    labels = ["$R_\\gamma<0.5$", "$0.5\\le R_\\gamma<1$", "$1\\le R_\\gamma<1.5$", "$R_\\gamma\\ge1.5$"]
    df["r_gamma_band"] = pd.cut(_num(df["r_gamma_planted"]), bins=bins, labels=labels)
    group_cols = ["n_eff_planted", "r_gamma_band"]
    if "weak_receiver_frac" in df.columns and _num(df["weak_receiver_frac"]).nunique() > 1:
        group_cols.insert(1, "weak_receiver_frac")
    rows = []
    for keys, g in df.groupby(group_cols, observed=False):
        if g.empty:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "n": int(len(g)),
            "r_gamma_mean": float(_num(g["r_gamma_planted"]).mean()),
            "member_f1_mean": float(_num(g.get("member_f1", np.nan)).mean()),
            "receiver_f1_mean": float(_num(g.get("receiver_f1", np.nan)).mean()),
            "weak_receiver_recovered_mean": float(_num(g.get("weak_receiver_recovered", np.nan)).mean()),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_path.with_suffix(".csv"), index=False)
    out_path.write_text("% See CSV: within-exposure hard-path R_gamma bands only.\n", encoding="utf-8")
    return out


def _p2_thresholds(frame: pd.DataFrame, level_col: str, alpha_col: str, metric: str,
                   recover_at: float = 0.9) -> pd.DataFrame:
    rows = []
    for lvl, g in frame.groupby(level_col):
        tmp = g[["pi1T_planted", metric, alpha_col, "n_eff_planted"]].copy()
        tmp[metric] = _num(tmp[metric])
        tmp[alpha_col] = _num(tmp[alpha_col])
        tmp["n_eff_planted"] = _num(tmp["n_eff_planted"])

        cell = (
            tmp.groupby("pi1T_planted", as_index=False)
            .agg(
                rec=(metric, "mean"),
                alpha=(alpha_col, "mean"),
                n_eff=("n_eff_planted", "mean"),
            )
            .sort_values("pi1T_planted")
        )
        alpha_mean = float(_num(g[alpha_col]).mean())
        hit = cell[cell["rec"] >= recover_at]
        if not hit.empty and np.isfinite(alpha_mean) and alpha_mean > 0:
            rows.append({"level": float(lvl), "alpha": alpha_mean,
                         "threshold_pi1T": float(hit.iloc[0]["pi1T_planted"]),
                         "threshold_n_eff": float(hit.iloc[0]["n_eff"]),
                         "censored": 0})
        else:
            rows.append({"level": float(lvl), "alpha": alpha_mean,
                         "threshold_pi1T": float("nan"),
                         "threshold_n_eff": float("nan"),
                         "censored": 1})
    return pd.DataFrame(rows)


def maybe_fit_p2(raw: pd.DataFrame, outdir: Path, n_boot: int = 1000, seed: int = 0,
                 recover_at: float = 0.9) -> pd.DataFrame:
    df = _ok(raw)
    df = df[df["method"].astype(str).eq("Proposed")].copy()
    if "pi1T_planted" not in df.columns or "weak_receiver_frac" not in df.columns:
        return pd.DataFrame()

    level_col = "weak_receiver_frac"
    # Use controlled relative link strength as alpha. Do NOT use realized weak
    # receiver count as alpha when episode length also changes, because that
    # count is exposure-confounded.
    alpha_col = "alpha_min_relative" if "alpha_min_relative" in df.columns else "weak_receiver_frac"
    metric = "weak_receiver_recovered" if "weak_receiver_recovered" in df.columns else ("receiver_exact" if "receiver_exact" in df.columns else "receiver_f1")
    seed_col = "seed" if "seed" in df.columns else None
    n_pi1T_levels = int(_num(df["pi1T_planted"]).nunique())

    base = _p2_thresholds(df, level_col, alpha_col, metric, recover_at=recover_at)
    fit = base[(base["censored"] == 0) & np.isfinite(base["alpha"]) & (base["alpha"] > 0)].copy()
    result = {
        "metric": metric,
        "level_col": level_col,
        "level_is_controlled": True,
        "alpha_col": alpha_col,
        "predicted_slope": -2.0,
        "n_levels": int(len(base)),
        "n_uncensored": int(len(fit)),
        "n_censored": int(base["censored"].sum()) if not base.empty else 0,
        "n_pi1T_levels": n_pi1T_levels,
        "slope": float("nan"),
        "intercept": float("nan"),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "n_boot_ok": 0,
        "covers_neg2": False,
        "note": "",
    }
    if n_pi1T_levels < 2:
        result["note"] = "degenerate: <2 pi1T levels; sweep --episode-lengths"
    if len(fit) >= 3:
        x = np.log(fit["alpha"].to_numpy(float))
        y = np.log(fit["threshold_pi1T"].to_numpy(float))
        slope, intercept = np.polyfit(x, y, 1)
        result["slope"], result["intercept"] = float(slope), float(intercept)
        if seed_col is not None and _num(df[seed_col]).nunique() > 2:
            rng = np.random.default_rng(seed)
            seeds_all = _num(df[seed_col]).dropna().unique()
            slopes = []
            for _ in range(int(n_boot)):
                samp = rng.choice(seeds_all, size=len(seeds_all), replace=True)
                bf = pd.concat([df[_num(df[seed_col]) == sv] for sv in samp], axis=0)
                tb = _p2_thresholds(bf, level_col, alpha_col, metric, recover_at=recover_at)
                tb = tb[(tb["censored"] == 0) & np.isfinite(tb["alpha"]) & (tb["alpha"] > 0)]
                if len(tb) >= 3:
                    sb, _ = np.polyfit(np.log(tb["alpha"].to_numpy(float)),
                                       np.log(tb["threshold_pi1T"].to_numpy(float)), 1)
                    slopes.append(float(sb))
            if len(slopes) >= 20:
                lo, hi = np.percentile(slopes, [2.5, 97.5])
                tol = 1e-6
                result.update(ci_lo=float(lo), ci_hi=float(hi), n_boot_ok=int(len(slopes)),
                              covers_neg2=bool(lo <= -2.0 + tol and hi >= -2.0 - tol))
                if (hi - lo) < 1e-6:
                    result["note"] = ("degenerate CI (zero width): recovery is too sharp for the "
                                      "seed count, so resampling never moves a threshold and the "
                                      "verdict reflects only the point estimate. Add seeds or a bar "
                                      "where some cells are partial.")
    elif not result["note"]:
        result["note"] = "need >=3 uncensored weak-link levels to fit a slope"

    base.to_csv(outdir / "ucdp_p2_threshold_points.csv", index=False)
    pd.DataFrame([result]).to_csv(outdir / "ucdp_p2_slope_fit.csv", index=False)
    base.attrs.update(result)
    return base


def main():
    p = argparse.ArgumentParser(description="Summarize theorem-facing UCDP GED spike-in outputs.")
    p.add_argument("--raw", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--headline-strength", type=float, default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--recover-at", type=float, default=0.9)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.raw)
    ok = _ok(raw)
    print(f"loaded {len(raw):,} rows; ok={len(ok):,}; errors={len(raw)-len(ok):,}")

    summary = summarize(raw)
    summary_path = outdir / "ucdp_ged_spikein_theorem_summary.csv"
    summary.to_csv(summary_path, index=False)

    is_ucdp = "injection_strength" in raw.columns and "coarsen_days" in raw.columns
    if is_ucdp:
        write_p1_latex(summary, outdir / "ucdp_p1_exposure_table.tex")
        p3 = write_p3_bands(raw, outdir / "ucdp_p3_within_exposure_bands.tex")
    else:
        p3 = pd.DataFrame()
    p2 = maybe_fit_p2(raw, outdir, n_boot=args.n_boot, recover_at=args.recover_at)

    if is_ucdp:
        headline = _resolve_strength(ok, args.headline_strength)
        delta_rows = []
        for metric in ["member_f1", "receiver_f1", "weak_receiver_recovered", "member_exact", "receiver_exact", "member_receiver_exact"]:
            if metric not in ok.columns:
                continue
            for base in BASELINES:
                r = paired_delta(raw, metric, headline, 0.0, base, args.n_boot, seed=0)
                if r is not None:
                    delta_rows.append(r)
        if delta_rows:
            ddf = pd.DataFrame(delta_rows)
            ddf.to_csv(outdir / "ucdp_theorem_paired_deltas.csv", index=False)
            print(f"\nProposed - baseline paired deltas at exact day, strength={headline:g}:")
            for _, r in ddf.iterrows():
                sig = "" if (r["ci_lo"] <= 0 <= r["ci_hi"]) else " *"
                print(f"  {r['metric']:>26s} vs {PRETTY.get(r['baseline'], r['baseline']):24s} "
                      f"Δ={r['mean_delta']:+.3f}  CI [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] n={int(r['n_pairs'])}{sig}")

    display_cols = [c for c in [
        "method", "injection_strength", "coarsen_days", "weak_receiver_frac", "episode_length", "n",
        "n_eff_planted_mean", "alpha_min_relative_mean", "alpha_min_proxy_mean",
        "pi1T_planted_mean", "member_f1_mean", "receiver_f1_mean",
        "weak_receiver_recovered_mean", "member_exact_mean", "receiver_exact_mean",
        "active_auc_mean", "r_gamma_planted_mean", "orig_min_member_bg_count_mean",
    ] if c in summary.columns]
    print("\n" + summary[display_cols].head(120).to_string(index=False))
    if not p3.empty:
        print("\nP3 within-exposure hard-path R_gamma bands written to CSV.")
    if not p2.empty:
        a = p2.attrs
        print("\nP2 quadratic weakest-link fit (predicted slope = -2):")
        print(f"  level={a.get('level_col')} alpha={a.get('alpha_col')} metric={a.get('metric')}  "
              f"levels={a.get('n_uncensored')}/{a.get('n_levels')} (censored={a.get('n_censored')}, "
              f"pi1T_levels={a.get('n_pi1T_levels')})")
        slope = a.get("slope")
        if slope is not None and np.isfinite(slope):
            ci_lo, ci_hi = a.get("ci_lo"), a.get("ci_hi")
            if ci_lo is not None and np.isfinite(ci_lo):
                verdict = "CONSISTENT with -2" if a.get("covers_neg2") else "INCONSISTENT with -2"
                print(f"  slope={slope:+.3f}  95% CI [{ci_lo:+.3f},{ci_hi:+.3f}]  "
                      f"({a.get('n_boot_ok')} boot)  -> {verdict}")
            else:
                print(f"  slope={slope:+.3f}  (no bootstrap CI: need >2 seeds)")
        if a.get("note"):
            print(f"  note: {a.get('note')}")


if __name__ == "__main__":
    main()
