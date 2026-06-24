from __future__ import annotations

"""Multiprocessing posterior-accuracy diagnostic for Condition C6.

Place this file under regime_hawkes/run_c6_posterior_accuracy_mp.py and run
from the repository root, e.g.

    python -m regime_hawkes.run_c6_posterior_accuracy_mp --benchmark all --seeds 20 --workers 8 --max-iters 30

The diagnostic measures, for event-containing birth intervals,

    Delta_gamma = max_b | gamma_hat_b(1) - gamma_star_b(1) |

where gamma_star is the oracle interval active probability computed from the
simulated birth-state labels. It also reports the theorem scale

    lambda_T = C0 * sqrt(log(K^2/delta_level) / N_eff)

and R_gamma = Delta_gamma / lambda_T.

The code is Windows multiprocessing safe: all worker entry points are top-level
and the parent process writes all CSV outputs.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Keep this before importing modules that may import jax/numpy BLAS.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_LOG_COMPILES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

for _name in ["jax", "jaxlib", "absl", "matplotlib", "PIL"]:
    _logger = logging.getLogger(_name)
    _logger.handlers.clear()
    _logger.setLevel(logging.CRITICAL)
    _logger.propagate = False

_root_logger = logging.getLogger()
_root_logger.handlers.clear()
_root_logger.setLevel(logging.CRITICAL)
_root_logger.propagate = False

SCRIPT_PATH = Path(__file__).resolve()
PKG_DIR = SCRIPT_PATH.parent
REPO_ROOT = PKG_DIR.parent if PKG_DIR.name == "regime_hawkes" else PKG_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from regime_hawkes.config import SimConfig
from regime_hawkes.estep import run_estep
from regime_hawkes.evaluate import evaluate
from regime_hawkes.mstep import MStepResult, run_mstep
from regime_hawkes.simulate import simulate_regime_hawkes

# Reuse the exact ablation helpers from the repo so this diagnostic runs the
# same D1-D4 logic as the existing ablation table.
try:
    import reviewer_upgrade_design_ablations as design
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import reviewer_upgrade_design_ablations.py. Run this from "
        "the repo root and keep that file at the repo root."
    ) from exc


@dataclass
class SimpleEMResult:
    gamma: np.ndarray
    params: MStepResult
    log_likelihood_trace: list[float]


@dataclass(frozen=True)
class RunOptions:
    delta_width: float = 2.0
    delta_level: float = 0.05
    lambda_c0: float = 1.0
    large_K: int = 30
    large_k: int = 10
    large_T: float = 900.0
    large_d: int = 4
    nu_base: float = 0.15
    alpha0_team: float = 0.01
    alpha1_max: float = 0.8
    alpha1_min: float = 0.35
    beta0: float = 1.0
    beta1: float = 3.0
    eta_on: float = 0.04
    eta_off: float = 0.30
    edge_rule: str = "hub_plus_cycle"
    lambda_g: float = 0.02
    lambda_h: float = 0.02
    lambda_e: float = 0.01
    lambda_0: float = 0.05
    lr: float = 0.01
    n_inner_steps: int = 10
    stability_threshold: float = 0.95
    stability_target: float = 0.80


VARIANT_LABELS = {
    "full_model": "Full model",
    "present_time_gating": "D1 present-time gating",
    "single_gate": "D2 single actor gate",
    "dense_active_block": "D3 dense active block",
    "shared_beta_rho": "D4 shared beta/rho",
}


def _make_cfg(benchmark: str, seed: int, opts: RunOptions) -> SimConfig:
    if benchmark == "synthetic_scaled":
        cfg = SimConfig(
            K=20,
            M=2,
            T=600.0,
            ring_actors=[0, 1, 2, 3],
            hub_actor=0,
            d=3,
            nu_base=opts.nu_base,
            alpha0_team=opts.alpha0_team,
            alpha1_max=opts.alpha1_max,
            alpha1_min=opts.alpha1_min,
            beta0=opts.beta0,
            beta1=opts.beta1,
            eta_on=0.06,
            eta_off=opts.eta_off,
            seed=int(seed),
        )
        setattr(cfg, "active_edge_rule", "dense_subgroup")
        return cfg

    if benchmark == "sparse_large":
        cfg = SimConfig(
            K=int(opts.large_K),
            M=2,
            T=float(opts.large_T),
            ring_actors=list(range(int(opts.large_k))),
            hub_actor=0,
            d=int(opts.large_d),
            nu_base=opts.nu_base,
            alpha0_team=opts.alpha0_team,
            alpha1_max=opts.alpha1_max,
            alpha1_min=opts.alpha1_min,
            beta0=opts.beta0,
            beta1=opts.beta1,
            eta_on=opts.eta_on,
            eta_off=opts.eta_off,
            seed=int(seed),
        )
        setattr(cfg, "active_edge_rule", str(opts.edge_rule))
        return cfg

    raise ValueError(f"Unknown benchmark={benchmark!r}")


def _benchmarks_from_arg(arg: str) -> list[str]:
    if arg == "synthetic":
        return ["synthetic_scaled"]
    if arg == "sparse":
        return ["sparse_large"]
    return ["synthetic_scaled", "sparse_large"]


def _variants_for_benchmark(benchmark: str) -> list[str]:
    if benchmark == "synthetic_scaled":
        return [
            "full_model",
            "present_time_gating",
            "single_gate",
            "dense_active_block",
            "shared_beta_rho",
        ]
    # The powered sparse-edge diagnostic is intended to test the fitted full
    # model's directed score, not every design ablation.
    if benchmark == "sparse_large":
        return ["full_model"]
    raise ValueError(benchmark)


def _enforce_variant_constraints(params: MStepResult, variant: str) -> MStepResult:
    # Start with the repo's existing D1-D3/shared-beta behavior.
    base_variant = "shared_beta" if variant == "shared_beta_rho" else variant
    params = design._enforce_variant_constraints(params, base_variant)

    if variant == "shared_beta_rho":
        b = 0.5 * (float(params.beta0) + float(params.beta1))
        rho_shared = design._project_rho_rows(0.5 * (np.asarray(params.rho0) + np.asarray(params.rho1)))
        params = MStepResult(
            nu=params.nu,
            A0=params.A0,
            g=params.g,
            h=params.h,
            U=params.U,
            V=params.V,
            rho0=rho_shared.copy(),
            rho1=rho_shared.copy(),
            beta0=b,
            beta1=b,
            eta_on=params.eta_on,
            eta_off=params.eta_off,
        )
    return params


def run_em_variant_c6(
    events: np.ndarray,
    interval_edges: np.ndarray,
    init_params: MStepResult,
    *,
    variant: str,
    max_iters: int,
    opts: RunOptions,
) -> SimpleEMResult:
    """Run the same ablation EM logic as the repo, adding D4 shared rho."""
    params = _enforce_variant_constraints(init_params, variant)
    B = len(interval_edges) - 1
    gamma_prev_1 = np.full(B, 0.15, dtype=float)
    gamma = np.column_stack([1.0 - gamma_prev_1, gamma_prev_1])
    ll_trace: list[float] = []
    prev_ll: float | None = None

    for it in range(1, int(max_iters) + 1):
        momentum = 0.5 if it <= 10 else 0.3
        scalar_momentum = 0.2 if it <= 10 else 0.1
        anneal = min(1.0, it / 10.0)
        lambda_g_eff = opts.lambda_g * anneal
        lambda_h_eff = opts.lambda_h * anneal
        lambda_e_eff = 0.0 if variant == "dense_active_block" else opts.lambda_e

        pi1_floor, gamma_floor = design._exploration_floors(it)
        gamma_prev_eff = np.clip(gamma_prev_1, gamma_floor, 1.0 - gamma_floor)
        eta_on_eff, eta_off_eff = design._enforce_min_stationary_active(
            params.eta_on, params.eta_off, pi1_floor
        )
        A1 = design.compute_A1(params.g, params.h, params.U, params.V)

        if variant == "present_time_gating":
            estep = design.run_estep_present_time(
                events=events,
                interval_edges=interval_edges,
                nu=params.nu,
                A0=params.A0,
                A1=A1,
                rho0=params.rho0,
                rho1=params.rho1,
                beta0=params.beta0,
                beta1=params.beta1,
                eta_on=eta_on_eff,
                eta_off=eta_off_eff,
            )
        else:
            estep = run_estep(
                events=events,
                interval_edges=interval_edges,
                gamma_prev_1=gamma_prev_eff,
                nu=params.nu,
                A0=params.A0,
                A1=A1,
                rho0=params.rho0,
                rho1=params.rho1,
                beta0=params.beta0,
                beta1=params.beta1,
                eta_on=eta_on_eff,
                eta_off=eta_off_eff,
            )

        gamma_prev_for_mstep = design._mstep_trace_weights(
            "present_time_gating" if variant == "present_time_gating" else variant,
            estep.gamma[:, 1],
            B,
        )

        proposed = run_mstep(
            events=events,
            interval_edges=interval_edges,
            gamma=estep.gamma,
            xi=estep.xi,
            nu=params.nu,
            A0=params.A0,
            g=params.g,
            h=params.h,
            U=params.U,
            V=params.V,
            rho0=params.rho0,
            rho1=params.rho1,
            beta0=params.beta0,
            beta1=params.beta1,
            lambda_g=lambda_g_eff,
            lambda_h=lambda_h_eff,
            lambda_e=lambda_e_eff,
            lambda_0=opts.lambda_0,
            lr=opts.lr,
            n_inner_steps=opts.n_inner_steps,
            gamma_prev_1=gamma_prev_for_mstep,
            eta_on_prev=params.eta_on,
            eta_off_prev=params.eta_off,
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
            beta0=float(scalar_momentum * params.beta0 + (1.0 - scalar_momentum) * proposed.beta0),
            beta1=float(scalar_momentum * params.beta1 + (1.0 - scalar_momentum) * proposed.beta1),
            eta_on=float(proposed.eta_on),
            eta_off=float(proposed.eta_off),
        )
        params = _enforce_variant_constraints(params, variant)
        params = design._apply_joint_stability(
            params,
            threshold=opts.stability_threshold,
            target=opts.stability_target,
        )

        gamma = estep.gamma
        ll = float(estep.log_likelihood)
        ll_trace.append(ll)
        gamma_prev_1 = np.clip(gamma[:, 1], gamma_floor, 1.0 - gamma_floor)

        if prev_ll is not None:
            rel = abs(ll - prev_ll) / (abs(prev_ll) + 1e-8)
            if rel < 1e-4:
                break
        prev_ll = ll

    final_pi1_floor, final_gamma_floor = design._exploration_floors(it if "it" in locals() else 1)
    eta_on_eff, eta_off_eff = design._enforce_min_stationary_active(
        params.eta_on, params.eta_off, final_pi1_floor
    )
    A1 = design.compute_A1(params.g, params.h, params.U, params.V)

    if variant == "present_time_gating":
        final_estep = design.run_estep_present_time(
            events=events,
            interval_edges=interval_edges,
            nu=params.nu,
            A0=params.A0,
            A1=A1,
            rho0=params.rho0,
            rho1=params.rho1,
            beta0=params.beta0,
            beta1=params.beta1,
            eta_on=eta_on_eff,
            eta_off=eta_off_eff,
        )
    else:
        final_estep = run_estep(
            events=events,
            interval_edges=interval_edges,
            gamma_prev_1=np.clip(gamma_prev_1, final_gamma_floor, 1.0 - final_gamma_floor),
            nu=params.nu,
            A0=params.A0,
            A1=A1,
            rho0=params.rho0,
            rho1=params.rho1,
            beta0=params.beta0,
            beta1=params.beta1,
            eta_on=eta_on_eff,
            eta_off=eta_off_eff,
        )

    return SimpleEMResult(
        gamma=np.asarray(final_estep.gamma, dtype=float),
        params=params,
        log_likelihood_trace=ll_trace,
    )


def _event_bins(events: np.ndarray, interval_edges: np.ndarray) -> np.ndarray:
    if len(events) == 0:
        return np.zeros(0, dtype=int)
    bins = np.searchsorted(interval_edges, events[:, 0], side="right") - 1
    return np.clip(bins.astype(int), 0, len(interval_edges) - 2)


def _oracle_interval_active_from_births(
    events: np.ndarray,
    z_at_event: np.ndarray,
    interval_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oracle active probability per interval and event-containing mask.

    gamma_star_b(1) is the fraction of events in interval b born in the active
    regime. On the fine grids used here this is normally 0/1; if a regime
    transition occurs inside an event-containing interval, the fractional value
    prevents a boundary artifact from dominating the diagnostic.
    """
    B = len(interval_edges) - 1
    counts = np.zeros(B, dtype=int)
    active = np.zeros(B, dtype=float)
    if len(events):
        bins = _event_bins(events, interval_edges)
        z = np.asarray(z_at_event, dtype=float)
        for b, zz in zip(bins, z):
            counts[int(b)] += 1
            active[int(b)] += float(zz)
    mask = counts > 0
    gamma_star = np.zeros(B, dtype=float)
    gamma_star[mask] = active[mask] / np.maximum(counts[mask], 1)
    return gamma_star, mask, counts


def _effective_count(data: Any) -> int:
    events = np.asarray(data.events)
    z = np.asarray(data.Z_at_event, dtype=int)
    ring = set(int(x) for x in data.config.ring_actors)
    if len(events) == 0:
        return 0
    actors = events[:, 1].astype(int)
    return int(np.sum((z == 1) & np.array([a in ring for a in actors], dtype=bool)))


def _lambda_scale(cfg: SimConfig, neff: int, opts: RunOptions) -> float:
    if neff <= 0:
        return float("inf")
    return float(opts.lambda_c0 * math.sqrt(math.log((cfg.K * cfg.K) / opts.delta_level) / neff))


def _posterior_gap_metrics(data: Any, result: SimpleEMResult, interval_edges: np.ndarray, opts: RunOptions) -> dict[str, Any]:
    gamma_hat = np.asarray(result.gamma[:, 1], dtype=float)
    gamma_star, event_mask, counts = _oracle_interval_active_from_births(
        np.asarray(data.events), np.asarray(data.Z_at_event), interval_edges
    )
    if gamma_hat.shape[0] != gamma_star.shape[0]:
        raise ValueError(f"gamma length {gamma_hat.shape[0]} != intervals {gamma_star.shape[0]}")

    diffs = np.abs(gamma_hat[event_mask] - gamma_star[event_mask])
    if diffs.size == 0:
        max_gap = p95_gap = mean_gap = float("nan")
    else:
        max_gap = float(np.max(diffs))
        p95_gap = float(np.quantile(diffs, 0.95))
        mean_gap = float(np.mean(diffs))

    neff = _effective_count(data)
    lam = _lambda_scale(data.config, neff, opts)
    return {
        "n_intervals": int(len(interval_edges) - 1),
        "n_event_intervals": int(np.sum(event_mask)),
        "n_events": int(len(data.events)),
        "n_eff": int(neff),
        "lambda_T_c0_1": float(lam),
        "delta_gamma_max": max_gap,
        "delta_gamma_p95": p95_gap,
        "delta_gamma_mean": mean_gap,
        "R_gamma_max": float(max_gap / lam) if np.isfinite(lam) and lam > 0 else float("nan"),
        "R_gamma_p95": float(p95_gap / lam) if np.isfinite(lam) and lam > 0 else float("nan"),
        "oracle_active_event_interval_frac": float(np.mean(gamma_star[event_mask])) if np.any(event_mask) else float("nan"),
    }


def _recovery_metrics(data: Any, result: SimpleEMResult, interval_edges: np.ndarray) -> dict[str, Any]:
    A1 = design.compute_A1(result.params.g, result.params.h, result.params.U, result.params.V)
    try:
        ev = evaluate(
            gamma=result.gamma,
            interval_edges=interval_edges,
            Z_path=data.Z_path,
            true_ring=data.config.ring_actors,
            true_hub=data.config.hub_actor,
            method="posterior_ring",
            events=data.events,
            Z_at_event=data.Z_at_event,
            auc_target="active_event",
            g=result.params.g,
            h=result.params.h,
            active_matrix=A1,
        )
    except TypeError:
        ev = evaluate(
            gamma=result.gamma,
            interval_edges=interval_edges,
            Z_path=data.Z_path,
            true_ring=data.config.ring_actors,
            true_hub=data.config.hub_actor,
            method="topk",
            events=data.events,
            Z_at_event=data.Z_at_event,
            auc_target="active_event",
            g=result.params.g,
            h=result.params.h,
        )
    return {
        "member_f1": float(ev.membership_f1),
        "member_precision": float(ev.membership_precision),
        "member_recall": float(ev.membership_recall),
        "hub_correct": float(bool(ev.hub_correct)),
        "field_auc": float(ev.field_auc),
        "predicted_ring": json.dumps([int(x) for x in ev.predicted_ring]),
    }


def _run_one_task(task: tuple[str, str, int, int, RunOptions]) -> dict[str, Any]:
    benchmark, variant, seed, max_iters, opts = task
    cfg = _make_cfg(benchmark, seed, opts)
    data = simulate_regime_hawkes(cfg)
    interval_edges = np.arange(0.0, float(cfg.T) + 1e-9, float(opts.delta_width))

    dense_active = variant == "dense_active_block"
    init = design.create_initial_params(
        cfg,
        np.asarray(data.events),
        d_override=int(cfg.K) if dense_active else None,
        dense_active=dense_active,
    )

    result = run_em_variant_c6(
        events=np.asarray(data.events),
        interval_edges=interval_edges,
        init_params=init,
        variant=variant,
        max_iters=int(max_iters),
        opts=opts,
    )

    gap = _posterior_gap_metrics(data, result, interval_edges, opts)
    rec = _recovery_metrics(data, result, interval_edges)

    return {
        "benchmark": benchmark,
        "variant": variant,
        "variant_label": VARIANT_LABELS.get(variant, variant),
        "seed": int(seed),
        "K": int(cfg.K),
        "k": int(len(cfg.ring_actors)),
        "T": float(cfg.T),
        "active_edge_rule": str(getattr(cfg, "active_edge_rule", "dense_subgroup")),
        "max_iters": int(max_iters),
        **gap,
        **rec,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((str(r["benchmark"]), str(r["variant"])), []).append(r)

    out: list[dict[str, Any]] = []
    for (benchmark, variant), gr in sorted(groups.items()):
        def arr(key: str) -> np.ndarray:
            return np.asarray([float(x[key]) for x in gr], dtype=float)
        f1 = arr("member_f1")
        auc = arr("field_auc")
        dmax = arr("delta_gamma_max")
        rp = arr("R_gamma_p95")
        rmax = arr("R_gamma_max")
        lam = arr("lambda_T_c0_1")
        neff = arr("n_eff")
        out.append({
            "benchmark": benchmark,
            "variant": variant,
            "variant_label": VARIANT_LABELS.get(variant, variant),
            "n_seeds": len(gr),
            "K": int(gr[0]["K"]),
            "k": int(gr[0]["k"]),
            "n_eff_mean": float(np.nanmean(neff)),
            "lambda_T_mean": float(np.nanmean(lam)),
            "delta_gamma_max_mean": float(np.nanmean(dmax)),
            "delta_gamma_max_std": float(np.nanstd(dmax, ddof=1)) if len(gr) > 1 else 0.0,
            "R_gamma_max_mean": float(np.nanmean(rmax)),
            "R_gamma_max_std": float(np.nanstd(rmax, ddof=1)) if len(gr) > 1 else 0.0,
            "R_gamma_p95_mean": float(np.nanmean(rp)),
            "member_f1_mean": float(np.nanmean(f1)),
            "member_f1_std": float(np.nanstd(f1, ddof=1)) if len(gr) > 1 else 0.0,
            "field_auc_mean": float(np.nanmean(auc)),
            "field_auc_std": float(np.nanstd(auc, ddof=1)) if len(gr) > 1 else 0.0,
            "hub_acc_mean": float(np.nanmean(arr("hub_correct"))),
        })
    return out


def _write_latex_rows(summary_rows: list[dict[str, Any]], path: Path) -> None:
    lines = []
    for r in summary_rows:
        lines.append(
            f"{r['benchmark']} & {r['variant_label']} & {r['n_seeds']} & "
            f"{r['n_eff_mean']:.1f} & {r['lambda_T_mean']:.3f} & "
            f"{r['delta_gamma_max_mean']:.3f} $\\pm$ {r['delta_gamma_max_std']:.3f} & "
            f"{r['R_gamma_max_mean']:.3f} $\\pm$ {r['R_gamma_max_std']:.3f} & "
            f"{r['member_f1_mean']:.3f} $\\pm$ {r['member_f1_std']:.3f} & "
            f"{r['field_auc_mean']:.3f} $\\pm$ {r['field_auc_std']:.3f} \\\\" 
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(rows: list[dict[str, Any]], out_png: Path, out_pdf: Path) -> None:
    # Optional plotting; keep import inside so CSV generation does not depend on matplotlib backend.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    markers = {
        "full_model": "o",
        "present_time_gating": "s",
        "single_gate": "^",
        "dense_active_block": "D",
        "shared_beta_rho": "x",
    }
    for variant in sorted(set(str(r["variant"]) for r in rows)):
        sub = [r for r in rows if str(r["variant"]) == variant]
        x = np.asarray([float(r["R_gamma_max"]) for r in sub])
        y = np.asarray([float(r["member_f1"]) for r in sub])
        ax.scatter(x, y, marker=markers.get(variant, "o"), label=VARIANT_LABELS.get(variant, variant), alpha=0.85)
    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Normalized posterior error $R_\gamma = \Delta_\gamma / \lambda_T$")
    ax.set_ylabel("Member-set F1")
    ax.set_title("Condition C6 posterior-accuracy diagnostic")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["all", "synthetic", "sparse"], default="all")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", default="results/c6_posterior_accuracy")
    ap.add_argument("--delta-width", type=float, default=2.0, help="interval width for E-step grid")
    ap.add_argument("--delta-level", type=float, default=0.05, help="theorem failure probability used in lambda_T")
    ap.add_argument("--lambda-c0", type=float, default=1.0, help="constant multiplier for lambda_T")
    ap.add_argument("--large-K", type=int, default=30)
    ap.add_argument("--large-k", type=int, default=10)
    ap.add_argument("--large-T", type=float, default=900.0)
    ap.add_argument("--large-d", type=int, default=4)
    ap.add_argument("--edge-rule", default="hub_plus_cycle", choices=["cycle", "hub_to_others", "hub_plus_cycle", "dense_subgroup"])
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    opts = RunOptions(
        delta_width=float(args.delta_width),
        delta_level=float(args.delta_level),
        lambda_c0=float(args.lambda_c0),
        large_K=int(args.large_K),
        large_k=int(args.large_k),
        large_T=float(args.large_T),
        large_d=int(args.large_d),
        edge_rule=str(args.edge_rule),
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    benchmarks = _benchmarks_from_arg(args.benchmark)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    tasks: list[tuple[str, str, int, int, RunOptions]] = []
    for benchmark in benchmarks:
        for variant in _variants_for_benchmark(benchmark):
            for seed in seeds:
                tasks.append((benchmark, variant, int(seed), int(args.max_iters), opts))

    print(f"Writing outputs to: {outdir.resolve()}")
    print(f"tasks={len(tasks)}, workers={args.workers}, max_iters={args.max_iters}")
    print(f"lambda_T convention: C0={opts.lambda_c0}, delta={opts.delta_level}")
    for benchmark in benchmarks:
        cfg0 = _make_cfg(benchmark, seeds[0], opts)
        print(f"{benchmark}: K={cfg0.K}, |S*|={len(cfg0.ring_actors)}, T={cfg0.T}, variants={_variants_for_benchmark(benchmark)}")

    rows: list[dict[str, Any]] = []
    if int(args.workers) <= 1:
        for task in tasks:
            r = _run_one_task(task)
            rows.append(r)
            print(
                f"[{r['benchmark']}/{r['variant']}] seed={r['seed']} "
                f"Delta={r['delta_gamma_max']:.3f} R={r['R_gamma_max']:.3f} "
                f"F1={r['member_f1']:.3f} AUC={r['field_auc']:.3f}"
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            future_to_task = {ex.submit(_run_one_task, task): task for task in tasks}
            for fut in as_completed(future_to_task):
                benchmark, variant, seed, *_ = future_to_task[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    print(f"[{benchmark}/{variant}] seed={seed} FAILED: {exc}")
                    raise
                rows.append(r)
                print(
                    f"[{r['benchmark']}/{r['variant']}] seed={r['seed']} "
                    f"Delta={r['delta_gamma_max']:.3f} R={r['R_gamma_max']:.3f} "
                    f"F1={r['member_f1']:.3f} AUC={r['field_auc']:.3f}"
                )

    rows.sort(key=lambda r: (str(r["benchmark"]), str(r["variant"]), int(r["seed"])))
    raw_path = outdir / "c6_posterior_accuracy_raw.csv"
    summary_path = outdir / "c6_posterior_accuracy_summary.csv"
    tex_path = outdir / "c6_posterior_accuracy_latex_rows.tex"
    _write_csv(raw_path, rows)
    summary_rows = _summarize(rows)
    _write_csv(summary_path, summary_rows)
    _write_latex_rows(summary_rows, tex_path)

    if not args.no_plot:
        _plot(rows, outdir / "c6_posterior_accuracy_plot.png", outdir / "c6_posterior_accuracy_plot.pdf")

    print("\nDone.")
    print(f"Raw:     {raw_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"LaTeX:   {tex_path.resolve()}")
    if not args.no_plot:
        print(f"Plot:    {(outdir / 'c6_posterior_accuracy_plot.pdf').resolve()}")
    print("\nPowerShell summary view:")
    print(f"Import-Csv .\\{summary_path} | Format-Table -AutoSize")


if __name__ == "__main__":
    main()
