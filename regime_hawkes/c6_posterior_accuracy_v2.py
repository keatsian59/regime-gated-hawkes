from __future__ import annotations

"""Reviewer-safe Condition C6 posterior-accuracy diagnostic (v2).

Place at regime_hawkes/c6_posterior_accuracy_v2.py and run from the repo root:

    python -m regime_hawkes.c6_posterior_accuracy_v2 --seeds 20 --workers 8

WHAT CHANGED vs run_c6_posterior_accuracy_mp.py
-----------------------------------------------
1. The reference posterior is the ORACLE POSTERIOR gamma_star = P(Z_b=1 | D, Theta*),
   computed by iterating the *same* E-step (run_estep) under the TRUE parameters to a
   fixed point -- NOT the birth-fraction / hard label. This is the object C6 bounds.
   The birth-fraction is retained only as a separate Bayes/calibration reference.
2. Label orientation is resolved ONCE per fit by a global flip/no-flip choice.
3. The headline statistic is the event-weighted mean gap; theorem-literal max, p95,
   Brier calibration, and the Bayes gap are reported alongside (not instead).
4. Recovery uses the SAME top-k evaluation as the main results (k = |S*|), so the
   y-axis matches Table 3 rather than the gate-thresholding 'posterior_ring' metric.
5. The centerpiece is a CONTROLLED POSTERIOR-CORRUPTION dose-response: gamma is clamped
   at a deliberately degraded oracle posterior and only the M-step runs, so recovery is
   a direct function of an *imposed* posterior gap (a manipulated cause, not a correlate).

NOTE: this file calls the repo's own functions (run_estep, run_mstep, evaluate, the
design-ablation helpers) and the EM driver from run_c6_posterior_accuracy_mp. It has been
syntax-checked but NOT executed here. Smoke-test a single seed before scaling out, and
verify the three VERIFY tags below match your package's signatures.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
PKG_DIR = SCRIPT_PATH.parent
REPO_ROOT = PKG_DIR.parent if PKG_DIR.name == "regime_hawkes" else PKG_DIR
for p in (REPO_ROOT, REPO_ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from regime_hawkes.config import SimConfig
from regime_hawkes.estep import run_estep
from regime_hawkes.mstep import MStepResult, run_mstep
from regime_hawkes.evaluate import evaluate
from regime_hawkes.simulate import simulate_regime_hawkes

import reviewer_upgrade_design_ablations as design

# Reuse the existing, validated EM driver + config + lambda convention so the
# "natural" full-model points are identical to the current ablation pipeline.
from regime_hawkes.run_c6_posterior_accuracy_mp import (
    RunOptions,
    run_em_variant_c6,
    _make_cfg,
    _event_bins,
    _effective_count,
    _lambda_scale,
)


# --------------------------------------------------------------------------------------
# Targets: oracle posterior (C6 reference) and birth-fraction (Bayes/calibration ref)
# --------------------------------------------------------------------------------------
def oracle_posterior(
    data: Any, interval_edges: np.ndarray, *, n_iters: int = 25, tol: float = 1e-5
) -> tuple[np.ndarray, Any]:
    """gamma_star = P(Z_b=1 | D, Theta*) via the same E-step under TRUE parameters.

    Iterates run_estep to a fixed point because the mean-field trace depends on the
    previous-iterate soft weights (gamma_prev_1). True A1 is data.A1 (already gated and
    stability-scaled by the simulator), so no recomputation from g/h/U/V is needed.
    """
    cfg = data.config
    B = len(interval_edges) - 1
    gamma_prev_1 = np.full(B, float(cfg.eta_on) / (float(cfg.eta_on) + float(cfg.eta_off)))
    last = None
    estep = None
    for _ in range(int(n_iters)):
        estep = run_estep(  # VERIFY: signature matches run_c6_posterior_accuracy_mp usage
            events=np.asarray(data.events),
            interval_edges=interval_edges,
            gamma_prev_1=gamma_prev_1,
            nu=data.nu,
            A0=data.A0,
            A1=data.A1,
            rho0=data.rho0,
            rho1=data.rho1,
            beta0=float(cfg.beta0),
            beta1=float(cfg.beta1),
            eta_on=float(cfg.eta_on),
            eta_off=float(cfg.eta_off),
        )
        g1 = np.asarray(estep.gamma[:, 1], dtype=float)
        if last is not None and float(np.max(np.abs(g1 - last))) < tol:
            gamma_prev_1 = g1
            break
        last = g1
        gamma_prev_1 = g1
    return np.asarray(estep.gamma, dtype=float), estep


def birth_fraction_target(data: Any, interval_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latent-truth reference (NOT C6): fraction of interval-b events born active."""
    B = len(interval_edges) - 1
    counts = np.zeros(B, dtype=int)
    active = np.zeros(B, dtype=float)
    events = np.asarray(data.events)
    if len(events):
        bins = _event_bins(events, interval_edges)
        z = np.asarray(data.Z_at_event, dtype=float)
        for b, zz in zip(bins, z):
            counts[int(b)] += 1
            active[int(b)] += float(zz)
    mask = counts > 0
    truth = np.zeros(B, dtype=float)
    truth[mask] = active[mask] / np.maximum(counts[mask], 1)
    return truth, mask, counts


def transition_interval_mask(data: Any, interval_edges: np.ndarray) -> np.ndarray:
    """Flag intervals that contain a CTMC switch (boundary-straddling)."""
    B = len(interval_edges) - 1
    flag = np.zeros(B, dtype=bool)
    for s, e, _ in data.Z_path:
        # a switch happens at e (end of a segment) for interior segments
        if 0.0 < e < float(data.config.T):
            b = int(np.searchsorted(interval_edges, e, side="right") - 1)
            if 0 <= b < B:
                flag[b] = True
    return flag


# --------------------------------------------------------------------------------------
# Orientation alignment + metric decomposition
# --------------------------------------------------------------------------------------
def align_orientation(gamma_hat1: np.ndarray, gamma_star1: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Resolve label-switching once: return gamma_hat1 or its flip 1-gamma_hat1,
    whichever is closer (min summed squared error) to gamma_star1 on event intervals."""
    a = gamma_hat1[mask]
    b = gamma_star1[mask]
    sse_same = float(np.sum((a - b) ** 2))
    sse_flip = float(np.sum(((1.0 - a) - b) ** 2))
    return gamma_hat1 if sse_same <= sse_flip else (1.0 - gamma_hat1)


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    s = float(w.sum())
    return float(np.sum(x * w) / s) if s > 0 else float("nan")


def decompose_metrics(
    gamma_hat: np.ndarray,
    gamma_star: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    counts: np.ndarray,
    lam: float,
) -> dict[str, float]:
    """Three references, one panel:
      C6 gap     = |gamma_hat - gamma_star|     (what the theorem bounds)
      Bayes gap  = |gamma_star - truth|         (irreducible; explains old saturation)
      calibration= Brier(gamma_hat, truth), Brier(gamma_star, truth)
    """
    gh = align_orientation(np.asarray(gamma_hat[:, 1], float), np.asarray(gamma_star[:, 1], float), mask)
    gs = np.asarray(gamma_star[:, 1], float)
    tr = np.asarray(truth, float)
    w = counts.astype(float)

    c6 = np.abs(gh[mask] - gs[mask])
    bayes = np.abs(gs[mask] - tr[mask])
    out = {
        "c6_gap_max": float(np.max(c6)) if c6.size else float("nan"),
        "c6_gap_p95": float(np.quantile(c6, 0.95)) if c6.size else float("nan"),
        "c6_gap_mean": float(np.mean(c6)) if c6.size else float("nan"),
        "c6_gap_wmean": _wmean(c6, w[mask]),
        "bayes_gap_wmean": _wmean(bayes, w[mask]),
        "brier_hat": _wmean((gh[mask] - tr[mask]) ** 2, w[mask]),
        "brier_star": _wmean((gs[mask] - tr[mask]) ** 2, w[mask]),
        "lambda_T": float(lam),
    }
    out["R_c6_wmean"] = float(out["c6_gap_wmean"] / lam) if np.isfinite(lam) and lam > 0 else float("nan")
    out["R_c6_max"] = float(out["c6_gap_max"] / lam) if np.isfinite(lam) and lam > 0 else float("nan")
    return out


# --------------------------------------------------------------------------------------
# Recovery, scored with the SAME top-k metric as the main results
# --------------------------------------------------------------------------------------
def evaluate_topk(data: Any, params: MStepResult, gamma: np.ndarray, interval_edges: np.ndarray) -> dict[str, float]:
    A1 = design.compute_A1(params.g, params.h, params.U, params.V)
    ev = evaluate(  # VERIFY: method='topk' is the same call your Table 3 uses
        gamma=gamma,
        interval_edges=interval_edges,
        Z_path=data.Z_path,
        true_ring=data.config.ring_actors,
        true_hub=data.config.hub_actor,
        method="topk",
        events=data.events,
        Z_at_event=data.Z_at_event,
        auc_target="active_event",
        g=params.g,
        h=params.h,
        active_matrix=A1,
    )
    return {
        "member_f1": float(ev.membership_f1),
        "hub_correct": float(bool(ev.hub_correct)),
        "field_auc": float(ev.field_auc),
    }


# --------------------------------------------------------------------------------------
# CONTROLLED POSTERIOR CORRUPTION (the causal centerpiece)
# --------------------------------------------------------------------------------------
def corrupt_posterior(gamma_star1: np.ndarray, eps: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Convex pull of the oracle posterior toward an uninformative/noisy target.
    mode='flatten' pulls toward 0.5 (decalibration); 'noise' pulls toward U(0,1)."""
    gs = np.asarray(gamma_star1, float)
    if mode == "flatten":
        target = np.full_like(gs, 0.5)
    elif mode == "noise":
        target = rng.uniform(0.0, 1.0, size=gs.shape)
    else:
        raise ValueError(f"unknown corruption mode {mode!r}")
    return np.clip((1.0 - eps) * gs + eps * target, 1e-3, 1.0 - 1e-3)


def recover_with_clamped_posterior(
    data: Any,
    gamma_clamped: np.ndarray,
    xi_star: Any,
    interval_edges: np.ndarray,
    opts: RunOptions,
    *,
    max_iters: int = 30,
) -> MStepResult:
    """Run ONLY the M-step with the regime posterior clamped at gamma_clamped (2-col).
    No E-step is run, so recovery is a pure function of the (corrupted) posterior. xi is
    held at the oracle value; gate recovery flows through the posterior-weighted active
    trace (gamma_prev_1), which is exactly the C6 channel."""
    cfg = data.config
    B = len(interval_edges) - 1
    init = design.create_initial_params(cfg, np.asarray(data.events), d_override=None, dense_active=False)
    params = design._enforce_variant_constraints(init, "full_model")
    g1 = np.asarray(gamma_clamped[:, 1], float)
    trace_w = design._mstep_trace_weights("full_model", g1, B)

    for it in range(1, int(max_iters) + 1):
        momentum = 0.5 if it <= 10 else 0.3
        scalar_momentum = 0.2 if it <= 10 else 0.1
        anneal = min(1.0, it / 10.0)
        proposed = run_mstep(  # VERIFY: arg list matches run_c6_posterior_accuracy_mp usage
            events=np.asarray(data.events),
            interval_edges=interval_edges,
            gamma=gamma_clamped,
            xi=xi_star,
            nu=params.nu, A0=params.A0, g=params.g, h=params.h, U=params.U, V=params.V,
            rho0=params.rho0, rho1=params.rho1, beta0=params.beta0, beta1=params.beta1,
            lambda_g=opts.lambda_g * anneal, lambda_h=opts.lambda_h * anneal,
            lambda_e=opts.lambda_e, lambda_0=opts.lambda_0,
            lr=opts.lr, n_inner_steps=opts.n_inner_steps,
            gamma_prev_1=trace_w, eta_on_prev=params.eta_on, eta_off_prev=params.eta_off,
        )
        params = MStepResult(
            nu=design._blend(params.nu, proposed.nu, momentum),
            A0=design._blend(params.A0, proposed.A0, momentum),
            g=design._blend(params.g, proposed.g, momentum),
            h=design._blend(params.h, proposed.h, momentum),
            U=design._blend(params.U, proposed.U, momentum),
            V=design._blend(params.V, proposed.V, momentum),
            rho0=design._project_rho_rows(design._blend(params.rho0, proposed.rho0, momentum)),
            rho1=design._project_rho_rows(design._blend(params.rho1, proposed.rho1, momentum)),
            beta0=float(scalar_momentum * params.beta0 + (1 - scalar_momentum) * proposed.beta0),
            beta1=float(scalar_momentum * params.beta1 + (1 - scalar_momentum) * proposed.beta1),
            eta_on=float(proposed.eta_on), eta_off=float(proposed.eta_off),
        )
        params = design._enforce_variant_constraints(params, "full_model")
        params = design._apply_joint_stability(params, threshold=opts.stability_threshold, target=opts.stability_target)
    return params


# --------------------------------------------------------------------------------------
# Tasks: one corruption sweep per seed (natural full-model point is eps=0)
# --------------------------------------------------------------------------------------
EPS_GRID = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)


def run_seed(benchmark: str, seed: int, opts: RunOptions, *, mode: str = "flatten") -> list[dict[str, Any]]:
    cfg = _make_cfg(benchmark, seed, opts)
    data = simulate_regime_hawkes(cfg)
    interval_edges = np.arange(0.0, float(cfg.T) + 1e-9, float(opts.delta_width))
    rng = np.random.default_rng(10_000 + seed)

    gamma_star, star_estep = oracle_posterior(data, interval_edges)
    truth, mask, counts = birth_fraction_target(data, interval_edges)
    neff = _effective_count(data)
    lam = _lambda_scale(cfg, neff, opts)
    xi_star = star_estep.xi  # VERIFY: estep exposes .xi (pairwise posteriors)

    rows: list[dict[str, Any]] = []
    for eps in EPS_GRID:
        gc1 = corrupt_posterior(gamma_star[:, 1], eps, mode, rng)
        gamma_clamped = np.column_stack([1.0 - gc1, gc1])
        params = recover_with_clamped_posterior(data, gamma_clamped, xi_star, interval_edges, opts)
        m = decompose_metrics(gamma_clamped, gamma_star, truth, mask, counts, lam)
        r = evaluate_topk(data, params, gamma_clamped, interval_edges)
        rows.append({
            "benchmark": benchmark, "seed": int(seed), "source": "corruption",
            "mode": mode, "eps": float(eps), "K": int(cfg.K), "k": int(len(cfg.ring_actors)),
            "T": float(cfg.T), "n_eff": int(neff), **m, **r,
        })

    # Natural full-model fit (real EM, eps implicitly 0) on the same data, for an anchor.
    init = design.create_initial_params(cfg, np.asarray(data.events), d_override=None, dense_active=False)
    nat = run_em_variant_c6(
        events=np.asarray(data.events), interval_edges=interval_edges, init_params=init,
        variant="full_model", max_iters=30, opts=opts,
    )
    m = decompose_metrics(nat.gamma, gamma_star, truth, mask, counts, lam)
    r = evaluate_topk(data, nat.params, nat.gamma, interval_edges)
    rows.append({
        "benchmark": benchmark, "seed": int(seed), "source": "natural_em",
        "mode": mode, "eps": float("nan"), "K": int(cfg.K), "k": int(len(cfg.ring_actors)),
        "T": float(cfg.T), "n_eff": int(neff), **m, **r,
    })
    return rows


# --------------------------------------------------------------------------------------
# One compact figure: recovery vs C6 gap / lambda_T
# --------------------------------------------------------------------------------------
def make_figure(rows: list[dict[str, Any]], out_pdf: Path, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)

    corr = [r for r in rows if r["source"] == "corruption"]
    nat = [r for r in rows if r["source"] == "natural_em"]

    # Left: recovery vs imposed C6 gap (event-weighted, in lambda_T units)
    x = np.array([r["R_c6_wmean"] for r in corr], float)
    yf = np.array([r["member_f1"] for r in corr], float)
    ya = np.array([r["field_auc"] for r in corr], float)
    sc = axL.scatter(x, yf, c=[r["eps"] for r in corr], cmap="viridis", s=26, label="F1 (top-k)")
    axL.scatter(x, ya, marker="x", c=[r["eps"] for r in corr], cmap="viridis", s=26, label="field AUC")
    if nat:
        axL.scatter([r["R_c6_wmean"] for r in nat], [r["member_f1"] for r in nat],
                    facecolors="none", edgecolors="k", s=70, label="natural EM (full)")
    axL.axvline(1.0, ls="--", lw=1.0, c="0.4")
    axL.set_xlabel(r"imposed posterior gap  $\overline{|\hat\gamma-\gamma^\star|}\,/\,\lambda_T$")
    axL.set_ylabel("recovery")
    axL.set_title("Controlled corruption: recovery vs C6 gap")
    axL.set_ylim(-0.02, 1.02)
    axL.legend(fontsize=7, loc="lower left")
    fig.colorbar(sc, ax=axL, label=r"corruption $\epsilon$")

    # Right: the decomposition that explains the old saturation
    axR.scatter([r["eps"] for r in corr], [r["c6_gap_wmean"] for r in corr], s=22, label=r"C6 gap (event-wt mean)")
    axR.scatter([r["eps"] for r in corr], [r["c6_gap_max"] for r in corr], s=22, marker="^", label="C6 gap (max)")
    if corr:
        axR.axhline(float(np.nanmean([r["bayes_gap_wmean"] for r in corr])), ls=":", c="0.4",
                    label="Bayes gap (irreducible)")
    axR.set_xlabel(r"corruption $\epsilon$")
    axR.set_ylabel("gap")
    axR.set_title("Why max saturates: literal-max vs event-weighted")
    axR.legend(fontsize=7, loc="best")

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="synthetic_scaled")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--mode", default="flatten", choices=["flatten", "noise"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", default="results/c6_v2")
    ap.add_argument("--delta-width", type=float, default=2.0)
    ap.add_argument("--delta-level", type=float, default=0.05)
    ap.add_argument("--lambda-c0", type=float, default=1.0)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    opts = RunOptions(delta_width=args.delta_width, delta_level=args.delta_level, lambda_c0=args.lambda_c0)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / "c6_v2_raw_checkpoint.csv"

    rows: list[dict[str, Any]] = []
    started = time.time()
    print(
        f"C6 v2 starting: benchmark={args.benchmark}, seeds={seeds[0]}..{seeds[-1]}, "
        f"workers={args.workers}, mode={args.mode}, outdir={outdir}",
        flush=True,
    )
    print(
        "Each seed runs 7 clamped-posterior corruption levels plus one natural-EM anchor. "
        "Checkpoint CSV is written after each completed seed.",
        flush=True,
    )

    def _record_seed(seed: int, seed_rows: list[dict[str, Any]], done: int) -> None:
        rows.extend(seed_rows)
        rows.sort(key=lambda r: (str(r["source"]), int(r["seed"]), float(r["eps"]) if r["eps"] == r["eps"] else -1.0))
        _write_csv(checkpoint, rows)
        elapsed = time.time() - started
        per_seed = elapsed / max(done, 1)
        eta = per_seed * max(len(seeds) - done, 0)
        print(
            f"[{done}/{len(seeds)}] seed={seed} done; rows={len(rows)}; "
            f"elapsed={elapsed/60:.1f} min; eta={eta/60:.1f} min; checkpoint={checkpoint}",
            flush=True,
        )

    if args.workers <= 1:
        for done, s in enumerate(seeds, start=1):
            print(f"starting seed={s}", flush=True)
            _record_seed(s, run_seed(args.benchmark, s, opts, mode=args.mode), done)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_seed, args.benchmark, s, opts, mode=args.mode): s for s in seeds}
            print(f"submitted {len(futs)} seed jobs", flush=True)
            done = 0
            for fut in as_completed(futs):
                seed = futs[fut]
                done += 1
                try:
                    seed_rows = fut.result()
                except Exception as e:
                    print(f"[{done}/{len(seeds)}] seed={seed} FAILED: {type(e).__name__}: {e}", flush=True)
                    raise
                _record_seed(seed, seed_rows, done)

    rows.sort(key=lambda r: (str(r["source"]), int(r["seed"]), float(r["eps"]) if r["eps"] == r["eps"] else -1.0))
    _write_csv(outdir / "c6_v2_raw.csv", rows)
    if not args.no_plot:
        print("making figure...", flush=True)
        make_figure(rows, outdir / "c6_v2_figure.pdf", outdir / "c6_v2_figure.png")
    print(f"wrote {outdir/'c6_v2_raw.csv'}", flush=True)


if __name__ == "__main__":
    main()
