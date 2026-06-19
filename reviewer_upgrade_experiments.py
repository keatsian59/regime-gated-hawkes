from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from regime_hawkes.config import SimConfig
from regime_hawkes.em import run_em
from regime_hawkes.evaluate import evaluate
from regime_hawkes.mstep import MStepResult
from regime_hawkes.simulate import simulate_regime_hawkes, summarize_simulation

# Optional baseline imports from flat files distributed with the reviewer code.
try:
    from regime_hawkes.run_baseline_static_hawkes import run_static_hawkes_baseline
except Exception as exc:  # pragma: no cover
    run_static_hawkes_baseline = None  # type: ignore[assignment]
    _STATIC_IMPORT_ERROR = exc
else:
    _STATIC_IMPORT_ERROR = None

try:
    from regime_hawkes.modular_hmm_hawkes_baseline import run_modular_hmm_hawkes_baseline
except Exception as exc:  # pragma: no cover
    run_modular_hmm_hawkes_baseline = None  # type: ignore[assignment]
    _MODULAR_IMPORT_ERROR = exc
else:
    _MODULAR_IMPORT_ERROR = None

try:
    from regime_hawkes.run_baseline_spectral import _fit_static_hawkes_full_matrix, _spectral_ring_detection, _degree_ring_detection
except Exception as exc:  # pragma: no cover
    _fit_static_hawkes_full_matrix = None  # type: ignore[assignment]
    _spectral_ring_detection = None  # type: ignore[assignment]
    _degree_ring_detection = None  # type: ignore[assignment]
    _SPECTRAL_IMPORT_ERROR = exc
else:
    _SPECTRAL_IMPORT_ERROR = None

# Optional, slow baseline. It is off by default and not needed for the main reviewer table.
try:
    from regime_hawkes.mmhp_pairwise_adapter import run_mmhp_pairwise_hawkes_adapter
except Exception:
    run_mmhp_pairwise_hawkes_adapter = None  # type: ignore[assignment]


# -----------------------------
# helpers borrowed from runner
# -----------------------------

def _softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def compute_A1(g: np.ndarray, h: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    inner = U @ V.T
    A1 = g[:, None] * h[None, :] * _softplus_np(inner)
    np.fill_diagonal(A1, 0.0)
    return A1


METHOD_ALIASES: dict[str, str] = {
    "proposed": "Proposed",
    "ours": "Proposed",
    "b1": "B1_static_gates",
    "static": "B1_static_gates",
    "static_gates": "B1_static_gates",
    "b2": "B2_modular_hmm_hawkes",
    "modular": "B2_modular_hmm_hawkes",
    "hmm": "B2_modular_hmm_hawkes",
    "modular_hmm_hawkes": "B2_modular_hmm_hawkes",
    "b3": "B3_static_svd",
    "svd": "B3_static_svd",
    "static_svd": "B3_static_svd",
    "b4": "B4_static_degree",
    "degree": "B4_static_degree",
    "b5": "B5_mmhp_pairwise",
    "mmhp": "B5_mmhp_pairwise",
}

ALL_METHOD_KEYS = [
    "Proposed",
    "B1_static_gates",
    "B2_modular_hmm_hawkes",
    "B3_static_svd",
    "B4_static_degree",
    "B5_mmhp_pairwise",
]

DEFAULT_REVIEWER_METHODS = {
    "Proposed",
    "B1_static_gates",
    "B2_modular_hmm_hawkes",
    "B3_static_svd",
}


def parse_method_selection(methods_arg: str | None, include_mmhp: bool = False) -> set[str]:
    if methods_arg is None or not methods_arg.strip():
        selected = set(DEFAULT_REVIEWER_METHODS)
        if include_mmhp:
            selected.add("B5_mmhp_pairwise")
        return selected

    raw_tokens = [tok.strip().lower() for tok in methods_arg.split(",") if tok.strip()]
    if not raw_tokens:
        raise ValueError("--methods was provided but no valid method names were found")

    selected: set[str] = set()
    for token in raw_tokens:
        if token == "all":
            selected.update(ALL_METHOD_KEYS)
            continue
        if token not in METHOD_ALIASES:
            allowed = ", ".join(sorted(set(METHOD_ALIASES)))
            raise ValueError(f"Unknown method '{token}'. Allowed values: {allowed}, all")
        selected.add(METHOD_ALIASES[token])

    return selected


def method_requested(selected_methods: set[str], method_name: str) -> bool:
    return method_name in selected_methods


def empirical_mark_init(events: np.ndarray, M: int, eps: float = 1.0) -> np.ndarray:
    counts = np.full((M, M), eps, dtype=float)
    if len(events) == 0:
        return counts / counts.sum(axis=1, keepdims=True)

    order = np.lexsort((events[:, 0], events[:, 1]))
    ev = events[order]
    last_mark_by_actor: dict[int, int] = {}
    for _, actor, mark in ev:
        actor = int(actor)
        mark = int(mark)
        if actor in last_mark_by_actor:
            counts[last_mark_by_actor[actor], mark] += 1.0
        last_mark_by_actor[actor] = mark
    return counts / counts.sum(axis=1, keepdims=True)


def rough_beta_init(events: np.ndarray) -> tuple[float, float]:
    import os
    beta0_env = os.environ.get("REGIME_HAWKES_BETA0_INIT")
    beta1_env = os.environ.get("REGIME_HAWKES_BETA1_INIT")
    if beta0_env is not None and beta1_env is not None:
        return float(beta0_env), float(beta1_env)
    if len(events) <= 1:
        return 1.0, 2.0
    t = np.sort(events[:, 0])
    dt = np.diff(t)
    dt = dt[dt > 1e-8]
    if len(dt) == 0:
        return 1.0, 2.0
    beta0 = float(np.clip(1.0 / np.median(dt), 0.1, 1.0))
    # Match the current synthetic runner: initialize the active decay well
    # above beta0 so EM does not get pinned near the too-small active timescale.
    beta1 = float(np.clip(1.0 / np.percentile(dt, 10), beta0 + 0.1, 8.0))
    return beta0, beta1


def create_initial_params(cfg: Any, events: np.ndarray) -> MStepResult:
    base_rate = max(len(events) / (cfg.K * cfg.M * cfg.T), 1e-3)
    fit_rng = np.random.default_rng(int(cfg.seed) + 1000)
    rho_init = empirical_mark_init(events, cfg.M)
    beta0_init, beta1_init = rough_beta_init(events)
    A0_init = fit_rng.uniform(0.0, 0.02, size=(cfg.K, cfg.K))
    np.fill_diagonal(A0_init, 0.0)
    return MStepResult(
        nu=np.full((cfg.K, cfg.M), base_rate),
        A0=A0_init,
        g=np.full(cfg.K, 0.1),
        h=np.full(cfg.K, 0.1),
        U=fit_rng.normal(scale=0.01, size=(cfg.K, cfg.d)),
        V=fit_rng.normal(scale=0.01, size=(cfg.K, cfg.d)),
        rho0=rho_init.copy(),
        rho1=rho_init.copy(),
        beta0=beta0_init,
        beta1=beta1_init,
        eta_on=0.03,
        eta_off=0.3,
    )


def default_config(seed: int = 7) -> SimConfig:
    """
    Reviewer-upgrade configuration: K=20 with a four-actor planted ring.

    This is the default configuration used for the baseline comparison and
    design-ablation runners.  It keeps the same coupling, decay, and CTMC
    parameters as the original synthetic setting while scaling the actor pool
    and active exposure to a moderate reviewer-facing stress test.
    """
    return SimConfig(
        K=20,
        M=2,
        T=600.0,
        ring_actors=[0, 1, 2, 3],
        hub_actor=0,
        d=3,
        nu_base=0.15,
        alpha0_team=0.01,
        alpha1_max=0.8,
        alpha1_min=0.35,
        beta0=1.0,
        beta1=3.0,
        eta_on=0.06,
        eta_off=0.3,
        seed=seed,
    )


def fit_once(
    cfg: SimConfig,
    *,
    interval_width: float = 2.0,
    max_iters: int = 50,
    lr: float = 0.01,
    lambda_g: float = 0.02,
    lambda_h: float = 0.02,
    lambda_e: float = 0.01,
    lambda_0: float = 0.05,
    n_inner_steps: int = 10,
) -> tuple[Any, Any, np.ndarray]:
    data = simulate_regime_hawkes(cfg)
    edges = np.arange(0.0, cfg.T + 1e-9, interval_width)
    init = create_initial_params(cfg, data.events)
    run_em_kwargs = dict(
        events=data.events,
        interval_edges=edges,
        init_params=init,
        max_iters=max_iters,
        lr=lr,
        lambda_g=lambda_g,
        lambda_h=lambda_h,
        lambda_e=lambda_e,
        lambda_0=lambda_0,
        n_inner_steps=n_inner_steps,
        verbose=False,
    )
    # Keep compatibility with the current EM signature while passing
    # the same stability settings used by the synthetic runner.
    sig = inspect.signature(run_em)
    if "stability_threshold" in sig.parameters:
        run_em_kwargs["stability_threshold"] = 0.95
    if "stability_target" in sig.parameters:
        run_em_kwargs["stability_target"] = 0.80
    result = run_em(**run_em_kwargs)
    return data, result, edges


def evaluate_result(data: Any, result: Any, edges: np.ndarray, method: str = "topk") -> Any:
    kwargs = dict(
        gamma=result.gamma,
        interval_edges=edges,
        Z_path=data.Z_path,
        g=result.params.g,
        h=result.params.h,
        true_ring=data.config.ring_actors,
        true_hub=data.config.hub_actor,
        method=method,
        events=data.events,
        Z_at_event=data.Z_at_event,
        auc_target="active_event",
    )
    if method == "posterior_ring":
        kwargs["active_matrix"] = compute_A1(result.params.g, result.params.h, result.params.U, result.params.V)
    return evaluate(**kwargs)


def active_event_truth(events: np.ndarray, z_at_event: np.ndarray, interval_edges: np.ndarray) -> np.ndarray:
    bn = len(interval_edges) - 1
    truth = np.zeros(bn, dtype=int)
    if events is None or z_at_event is None or len(events) == 0:
        return truth
    idx = np.searchsorted(interval_edges, np.asarray(events)[:, 0], side="right") - 1
    idx = np.clip(idx, 0, bn - 1)
    active_idx = idx[np.asarray(z_at_event, dtype=int) == 1]
    if len(active_idx):
        truth[np.unique(active_idx)] = 1
    return truth


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def membership_metrics(
    pred_ring: list[int] | set[int],
    truth_ring: list[int],
    hub_pred: int,
    true_hub: int,
) -> dict[str, float | int | str]:
    pred = set(int(x) for x in pred_ring)
    truth = set(int(x) for x in truth_ring)
    tp = len(pred & truth)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(truth), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "hub_correct": int(int(hub_pred) == int(true_hub)),
        "pred_ring": json.dumps(sorted(int(x) for x in pred)),
    }


def fit_proposed_from_data(
    data: Any,
    cfg: SimConfig,
    *,
    method: str = "topk",
    interval_width: float = 2.0,
    max_iters: int = 50,
    lr: float = 0.01,
    lambda_g: float = 0.02,
    lambda_h: float = 0.02,
    lambda_e: float = 0.01,
    lambda_0: float = 0.05,
    n_inner_steps: int = 10,
) -> tuple[Any, np.ndarray, Any]:
    edges = np.arange(0.0, cfg.T + 1e-9, interval_width)
    init = create_initial_params(cfg, data.events)
    run_em_kwargs = dict(
        events=data.events,
        interval_edges=edges,
        init_params=init,
        max_iters=max_iters,
        lr=lr,
        lambda_g=lambda_g,
        lambda_h=lambda_h,
        lambda_e=lambda_e,
        lambda_0=lambda_0,
        n_inner_steps=n_inner_steps,
        verbose=False,
    )
    sig = inspect.signature(run_em)
    if "stability_threshold" in sig.parameters:
        run_em_kwargs["stability_threshold"] = 0.95
    if "stability_target" in sig.parameters:
        run_em_kwargs["stability_target"] = 0.80
    result = run_em(**run_em_kwargs)
    ev = evaluate_result(data, result, edges, method=method)
    return result, edges, ev


def run_baseline_static_from_data(data: Any, cfg: SimConfig) -> dict[str, Any]:
    if run_static_hawkes_baseline is None:
        raise ImportError(f"Could not import run_static_hawkes_baseline: {_STATIC_IMPORT_ERROR}")
    edges = np.arange(0.0, cfg.T + 1e-9, 2.0)

    # Non-oracle initialization from the observed event stream only.
    init0 = create_initial_params(cfg, data.events)
    init = MStepResult(
        nu=np.array(init0.nu, copy=True),
        A0=np.array(init0.A0, copy=True),
        g=np.array(init0.g, copy=True),
        h=np.array(init0.h, copy=True),
        U=np.array(init0.U, copy=True),
        V=np.array(init0.V, copy=True),
        rho0=np.array(init0.rho0, copy=True),
        rho1=np.array(init0.rho1, copy=True),
        beta0=float(init0.beta0),
        beta1=float(init0.beta1),
        eta_on=0.5,
        eta_off=0.5,
    )
    params, ll = run_static_hawkes_baseline(
        events=data.events,
        interval_edges=edges,
        init_params=init,
        max_iters=30,
        lr=0.05,
        lambda_g=0.02,
        lambda_h=0.02,
        lambda_e=0.01,
        lambda_0=0.01,
        n_inner_steps=10,
        verbose=False,
    )
    B = len(edges) - 1
    gamma_uniform = np.column_stack([np.full(B, 0.5), np.full(B, 0.5)])
    ev = evaluate(
        gamma=gamma_uniform,
        interval_edges=edges,
        Z_path=data.Z_path,
        g=params.g,
        h=params.h,
        true_ring=cfg.ring_actors,
        true_hub=cfg.hub_actor,
        method="topk",
        events=data.events,
        Z_at_event=data.Z_at_event,
        auc_target="active_event",
    )
    return {
        "field_auc": float("nan"),
        "precision": float(ev.membership_precision),
        "recall": float(ev.membership_recall),
        "f1": float(ev.membership_f1),
        "hub_correct": int(bool(ev.hub_correct)),
        "pred_ring": json.dumps(ev.predicted_ring),
        "loglik_final": float(ll),
    }


def run_baseline_modular_from_data(data: Any, cfg: SimConfig) -> dict[str, Any]:
    if run_modular_hmm_hawkes_baseline is None:
        raise ImportError(f"Could not import run_modular_hmm_hawkes_baseline: {_MODULAR_IMPORT_ERROR}")
    edges = np.arange(0.0, cfg.T + 1e-9, 2.0)
    baseline = run_modular_hmm_hawkes_baseline(
        events=data.events,
        interval_edges=edges,
        true_ring=cfg.ring_actors,
        true_hub=cfg.hub_actor,
        Z_path=None,
        hmm_threshold=0.25,
        hmm_restarts=10,
        hmm_max_iters=100,
        hawkes_min_total_events=4,
        selection="topk",
        top_k=len(cfg.ring_actors),
        score_mode="max",
        seed=cfg.seed,
    )
    truth = active_event_truth(data.events, data.Z_at_event, edges)
    field_auc = roc_auc_binary(truth, baseline.hmm_gamma[:, 1])
    print(
        f"[B2 seed={cfg.seed}] "
        f"F1={baseline.membership_f1:.3f} "
        f"AUC={field_auc:.3f} "
        f"pred={baseline.predicted_ring} "
        f"hub_pred={baseline.hub_pred} "
        f"n_pair_fits={baseline.n_pair_fits} "
        f"active_fraction={baseline.active_fraction:.3f} "
        f"n_windows={len(baseline.active_windows)} "
        f"hmm_rates={baseline.hmm_rates}",
        flush=True,
    )
    order = np.argsort(baseline.member_scores)[::-1]
    ranks = {int(actor): int(np.where(order == actor)[0][0]) + 1 for actor in cfg.ring_actors}

    print(
        f"[B2 seed={cfg.seed}] "
        f"true_ring={cfg.ring_actors} "
        f"true_ranks={ranks} "
        f"top10={order[:10].tolist()} "
        f"top10_scores={baseline.member_scores[order[:10]].round(4).tolist()} "
        f"sender_true={baseline.sender_scores[cfg.ring_actors].round(4).tolist()} "
        f"receiver_true={baseline.receiver_scores[cfg.ring_actors].round(4).tolist()}",
        flush=True,
    )
    return {
        "field_auc": float(field_auc),
        "precision": float(baseline.membership_precision),
        "recall": float(baseline.membership_recall),
        "f1": float(baseline.membership_f1),
        "hub_correct": int(bool(baseline.hub_correct)),
        "pred_ring": json.dumps(baseline.predicted_ring),
        "loglik_final": float("nan"),
    }


def run_baseline_spectral_degree_from_data(
    data: Any,
    cfg: SimConfig,
    *,
    include_svd: bool = True,
    include_degree: bool = False,
) -> list[dict[str, Any]]:
    if _fit_static_hawkes_full_matrix is None or _spectral_ring_detection is None or _degree_ring_detection is None:
        raise ImportError(f"Could not import spectral baseline helpers: {_SPECTRAL_IMPORT_ERROR}")
    _nu_hat, A_hat, _rho_hat, _beta_hat = _fit_static_hawkes_full_matrix(
        events=data.events,
        K=cfg.K,
        M=cfg.M,
        T=cfg.T,
        beta_init=2.0,
        lr=0.005,
        n_iters=200,
        l1_penalty=0.01,
        verbose=False,
    )
    k_ring = len(cfg.ring_actors)
    ring_svd, hub_svd = _spectral_ring_detection(A_hat, k_ring)
    ring_deg, hub_deg = _degree_ring_detection(A_hat, k_ring)
    out: list[dict[str, Any]] = []
    if include_svd:
        out.append({
            "method": "B3_static_svd",
            **membership_metrics(ring_svd, cfg.ring_actors, hub_svd, cfg.hub_actor),
            "field_auc": float("nan"),
            "loglik_final": float("nan"),
        })
    if include_degree:
        out.append({
            "method": "B4_static_degree",
            **membership_metrics(ring_deg, cfg.ring_actors, hub_deg, cfg.hub_actor),
            "field_auc": float("nan"),
            "loglik_final": float("nan"),
        })
    return out


def run_mmhp_baseline_from_data(data: Any, cfg: SimConfig) -> dict[str, Any]:
    if run_mmhp_pairwise_hawkes_adapter is None:
        raise ImportError("mmhp_pairwise_adapter is not available")
    edges = np.arange(0.0, cfg.T + 1e-9, 2.0)
    baseline = run_mmhp_pairwise_hawkes_adapter(
        events=data.events,
        interval_edges=edges,
        true_ring=cfg.ring_actors,
        true_hub=cfg.hub_actor,
        Z_path=None,
        score_mode="geom",
        selection="topk",
        mmhp_threshold=0.5,
        hawkes_min_total_events=6,
        seed=cfg.seed,
        chains=4,
        iter_warmup=500,
        iter_sampling=500,
    )
    truth = active_event_truth(data.events, data.Z_at_event, edges)
    field_auc = roc_auc_binary(truth, baseline.gamma[:, 1])
    return {
        "field_auc": float(field_auc),
        "precision": float(baseline.membership_precision),
        "recall": float(baseline.membership_recall),
        "f1": float(baseline.membership_f1),
        "hub_correct": int(bool(baseline.hub_correct)),
        "pred_ring": json.dumps(baseline.predicted_ring),
        "loglik_final": float("nan"),
    }


# -----------------------------
# diagnostics and GOF
# -----------------------------

def _event_bins(events: np.ndarray, interval_edges: np.ndarray) -> np.ndarray:
    if len(events) == 0:
        return np.zeros(0, dtype=int)
    b = len(interval_edges) - 1
    idx = np.searchsorted(interval_edges, events[:, 0], side="right") - 1
    return np.clip(idx, 0, b - 1)


def compensator_residuals_superposed(events: np.ndarray, interval_edges: np.ndarray, result: Any) -> np.ndarray:
    """Approximate superposed-process time-rescaling residuals.

    We use the fitted posterior-active probability of the event's home interval
    as the birth-state weight for the active trace, matching the final
    event-time-gated approximation used by inference.
    """
    if len(events) == 0:
        return np.zeros(0, dtype=float)

    params = result.params
    K, M = params.nu.shape
    bins = _event_bins(events, interval_edges)
    event_active_w = result.gamma[bins, 1]

    B = np.zeros((K, M), dtype=float)
    A = np.zeros((K, M), dtype=float)
    t_prev = 0.0
    residuals: list[float] = []

    A1 = compute_A1(params.g, params.h, params.U, params.V)

    for n, (t, actor, mark) in enumerate(events):
        dt = float(t) - t_prev
        if dt < 0:
            raise ValueError("events must be time-sorted")

        routine_pre = params.A0.T @ (B @ params.rho0)
        active_pre = A1.T @ (A @ params.rho1)
        decay0 = (1.0 - np.exp(-params.beta0 * dt)) / max(params.beta0, 1e-12)
        decay1 = (1.0 - np.exp(-params.beta1 * dt)) / max(params.beta1, 1e-12)
        total_comp = (
            float(np.sum(params.nu)) * dt
            + float(np.sum(routine_pre)) * decay0
            + float(np.sum(active_pre)) * decay1
        )
        residuals.append(total_comp)

        B *= np.exp(-params.beta0 * dt)
        A *= np.exp(-params.beta1 * dt)
        i = int(actor)
        m = int(mark)
        B[i, m] += params.beta0
        A[i, m] += params.beta1 * float(event_active_w[n])
        t_prev = float(t)

    return np.asarray(residuals, dtype=float)


def ks_stat_exp1(residuals: np.ndarray) -> float:
    if residuals.size == 0:
        return float("nan")
    x = np.sort(np.asarray(residuals, dtype=float))
    n = len(x)
    ecdf = np.arange(1, n + 1) / n
    cdf = 1.0 - np.exp(-x)
    d_plus = np.max(ecdf - cdf)
    d_minus = np.max(cdf - np.arange(0, n) / n)
    return float(max(d_plus, d_minus))


def qq_plot_exp1(residuals: np.ndarray, out_path: Path) -> None:
    x = np.sort(np.asarray(residuals, dtype=float))
    n = len(x)
    u = (np.arange(1, n + 1) - 0.5) / max(n, 1)
    q_theory = -np.log(1.0 - u)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.scatter(q_theory, x, s=12)
    lo = min(float(np.min(q_theory)), float(np.min(x))) if n else 0.0
    hi = max(float(np.max(q_theory)), float(np.max(x))) if n else 1.0
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Exp(1) quantiles")
    ax.set_ylabel("Empirical rescaled residual quantiles")
    ax.set_title("Time-rescaling QQ plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def ks_plot_exp1(residuals: np.ndarray, out_path: Path) -> None:
    x = np.sort(np.asarray(residuals, dtype=float))
    n = len(x)
    ecdf = np.arange(1, n + 1) / max(n, 1)
    cdf = 1.0 - np.exp(-x)

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.step(x, ecdf, where="post", label="empirical CDF")
    ax.plot(x, cdf, label="Exp(1) CDF")
    ax.set_xlabel("rescaled residual")
    ax.set_ylabel("CDF")
    ax.set_title(f"Time-rescaling KS overlay (D={ks_stat_exp1(residuals):.3f})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# -----------------------------
# plotting for the paper
# -----------------------------

def plot_activation_timeline(data: Any, result: Any, interval_edges: np.ndarray, out_path: Path) -> None:
    mids = 0.5 * (interval_edges[:-1] + interval_edges[1:])
    gamma = np.asarray(result.gamma[:, 1], dtype=float)
    sd = np.sqrt(np.maximum(gamma * (1.0 - gamma), 0.0))
    lo = np.clip(gamma - sd, 0.0, 1.0)
    hi = np.clip(gamma + sd, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    for s, e, z in data.Z_path:
        if z == 1:
            ax.axvspan(float(s), float(e), alpha=0.12)
    ax.fill_between(mids, lo, hi, alpha=0.25, label="posterior ±1 SD")
    ax.plot(mids, gamma, linewidth=2.0, label=r"$\gamma_b(1)$")

    if len(data.events):
        active = data.Z_at_event.astype(int) == 1
        dormant = ~active
        ax.scatter(data.events[dormant, 0], np.full(np.sum(dormant), -0.03), marker="|", s=28, label="events born in z=0")
        ax.scatter(data.events[active, 0], np.full(np.sum(active), -0.07), marker="|", s=28, label="events born in z=1")

    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("time")
    ax.set_ylabel("active posterior")
    ax.set_title("Activation timeline, posterior band, and event births")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_parameter_recovery(rep_df: pd.DataFrame, cfg: SimConfig, out_path: Path) -> None:
    params = ["eta_on_hat", "eta_off_hat", "beta0_hat", "beta1_hat"]
    truths = {
        "eta_on_hat": cfg.eta_on,
        "eta_off_hat": cfg.eta_off,
        "beta0_hat": cfg.beta0,
        "beta1_hat": cfg.beta1,
    }
    display_names = {
        "eta_on_hat": "eta_on",
        "eta_off_hat": "eta_off",
        "beta0_hat": "beta0",
        "beta1_hat": "beta1",
    }

    # Flatten the layout aggressively for paper use: one horizontal row,
    # short height, tight inter-panel spacing, and tight output bounding box.
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 2.45))
    for ax, p in zip(axes, params):
        vals = rep_df[p].astype(float).to_numpy()
        ax.boxplot(vals, vert=True, widths=0.45)
        ax.axhline(truths[p], linestyle="--", color="red", linewidth=1.2)
        ax.set_title(display_names[p], fontsize=10, pad=3)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=8)
        ax.margins(x=0.08)

    fig.suptitle("Parameter recovery across replications", fontsize=12, y=1.02)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.13, top=0.83, wspace=0.22)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_gate_recovery(rep_df: pd.DataFrame, cfg: SimConfig, out_path: Path) -> None:
    K = cfg.K
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
    for a, prefix, truth in [(axes[0], "g", np.array([1 if j in cfg.ring_actors else 0 for j in range(K)])),
                             (axes[1], "h", np.array([1 if j in cfg.ring_actors else 0 for j in range(K)]))]:
        positions = np.arange(K)
        data = [rep_df[f"{prefix}{j}_hat"].astype(float).to_numpy() for j in range(K)]
        a.boxplot(data, positions=positions, widths=0.55)
        a.plot(positions + 1, truth, linestyle="", marker="_", markersize=18, color="red")
        a.set_xticks(positions + 1)
        a.set_xticklabels([str(j) for j in range(K)])
        a.set_xlabel("actor")
        a.set_title(f"{prefix} recovery")
    axes[0].set_ylabel("estimate")
    fig.suptitle("Gate recovery across replications (red ticks = planted truth)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_topology_heatmaps(data: Any, result: Any, out_path: Path) -> None:
    A1_hat = compute_A1(result.params.g, result.params.h, result.params.U, result.params.V)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    im0 = axes[0].imshow(data.A1)
    axes[0].set_title("True active coupling")
    axes[0].set_xlabel("receiver")
    axes[0].set_ylabel("sender")
    im1 = axes[1].imshow(A1_hat)
    axes[1].set_title("Recovered active coupling")
    axes[1].set_xlabel("receiver")
    fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_phase_diagram(
    grid_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    value_col: str,
    out_path: Path,
) -> None:
    pivot = grid_df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"{y:g}" for y in pivot.index])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Phase diagram: mean {value_col}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# -----------------------------
# replication suites
# -----------------------------

def render_baseline_replication_table(summary_df: pd.DataFrame, out_path: Path) -> None:
    order = [
        ("Proposed", "Proposed"),
        ("B1_static_gates", "B1: Static Hawkes + gates"),
        ("B2_modular_hmm_hawkes", "B2: Modular HMM + Hawkes"),
        ("B3_static_svd", "B3: Static Hawkes + SVD"),
        ("B4_static_degree", "B4: Static Hawkes + degree"),
        ("B5_mmhp_pairwise", "B5: MMHP pairwise"),
    ]

    def fmt_mean_std(m: float, s: float) -> str:
        if np.isnan(m):
            return "---"
        if np.isnan(s):
            return f"{m:.3f}"
        return f"{m:.3f} $\\pm$ {s:.3f}"

    lines = []
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Method & Precision & Recall & F1 & AUC \\")
    lines.append(r"\midrule")
    for key, label in order:
        if key not in summary_df.index:
            continue
        row = summary_df.loc[key]
        prec = fmt_mean_std(row["precision_mean"], row["precision_std"])
        rec = fmt_mean_std(row["recall_mean"], row["recall_std"])
        f1 = fmt_mean_std(row["f1_mean"], row["f1_std"])
        auc = fmt_mean_std(row["field_auc_mean"], row["field_auc_std"])
        lines.append(f"{label} & {prec} & {rec} & {f1} & {auc} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _shared_replication_metadata(seed: int, sim_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "events": int(sim_summary["total_events"]),
        "pi1_true": float(sim_summary["pi_1"]),
        "spec_true": float(sim_summary["spectral_radius"]),
    }

def run_replications(
    base_cfg: SimConfig,
    *,
    seeds: list[int],
    outdir: Path,
    method: str = "topk",
    methods: set[str] | None = None,
    include_mmhp: bool = False,
    interval_width: float = 2.0,
    max_iters: int = 30,
    lr: float = 0.01,
    lambda_g: float = 0.02,
    lambda_h: float = 0.02,
    lambda_e: float = 0.01,
    lambda_0: float = 0.05,
    n_inner_steps: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the reviewer replication suite on one simulated data set per seed.

    The same simulated stream is reused for the proposed model and every
    selected baseline, so the baseline table is a paired comparison rather than
    a mixture of different random draws. The default selection is exactly the
    reviewer-facing table: Proposed, B1, B2, and B3.

    This version also writes proposed_assignment_diagnostics.csv whenever the
    Proposed model is run. That long-format CSV supports the same rank /
    sender / receiver diagnostic figure used for B2.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    selected_methods = parse_method_selection(None, include_mmhp=include_mmhp) if methods is None else set(methods)
    if not selected_methods:
        raise ValueError("No methods selected")
    print(f"[info] running methods: {', '.join(sorted(selected_methods))}", flush=True)

    proposed_rows: list[dict[str, Any]] = []
    proposed_diag_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    exemplar: dict[str, Any] = {}

    for rep_idx, seed in enumerate(seeds):
        cfg = SimConfig(**{**asdict(base_cfg), "seed": int(seed)})
        data = simulate_regime_hawkes(cfg)
        sim_summary = summarize_simulation(data)
        shared_meta = _shared_replication_metadata(seed, sim_summary)
        print(
            f"[rep {rep_idx + 1}/{len(seeds)}] seed={seed} simulated "
            f"events={int(sim_summary['total_events'])} "
            f"(K={cfg.K}, M={cfg.M}, T={cfg.T:.1f})",
            flush=True,
        )

        if method_requested(selected_methods, "Proposed"):
            result, edges, ev = fit_proposed_from_data(
                data,
                cfg,
                method=method,
                interval_width=interval_width,
                max_iters=max_iters,
                lr=lr,
                lambda_g=lambda_g,
                lambda_h=lambda_h,
                lambda_e=lambda_e,
                lambda_0=lambda_0,
                n_inner_steps=n_inner_steps,
            )
            A1_hat = compute_A1(result.params.g, result.params.h, result.params.U, result.params.V)

            # Full-model assignment diagnostic.
            # A1_hat[i,j] is the recovered active coupling from sender i to receiver j,
            # matching the plotting convention used for topology heatmaps.
            sender_scores = np.sum(A1_hat, axis=1)
            receiver_scores = np.sum(A1_hat, axis=0)

            sender_norm = sender_scores / max(float(np.max(sender_scores)), 1e-12)
            receiver_norm = receiver_scores / max(float(np.max(receiver_scores)), 1e-12)

            # Match the B2 max-normalized diagnostic ranking so the figures are
            # comparable as assignment diagnostics. This is separate from the
            # paper's official evaluation call above.
            member_scores = np.maximum(sender_norm, receiver_norm)

            order = np.argsort(member_scores)[::-1]
            ranks = {
                int(actor): int(np.where(order == int(actor))[0][0]) + 1
                for actor in cfg.ring_actors
            }
            hub_pred_diag = int(np.argmax(sender_scores)) if len(sender_scores) else -1

            residuals = compensator_residuals_superposed(data.events, edges, result)
            proposed_row: dict[str, Any] = {
                **shared_meta,
                "field_auc": float(ev.field_auc),
                "precision": float(ev.membership_precision),
                "recall": float(ev.membership_recall),
                "f1": float(ev.membership_f1),
                "hub_correct": int(bool(ev.hub_correct)),
                "spec_hat": float(np.max(np.abs(np.linalg.eigvals(result.params.A0 + A1_hat)))),
                "eta_on_hat": float(result.params.eta_on),
                "eta_off_hat": float(result.params.eta_off),
                "beta0_hat": float(result.params.beta0),
                "beta1_hat": float(result.params.beta1),
                "loglik_final": float(result.log_likelihood_trace[-1]) if result.log_likelihood_trace else float("nan"),
                "ks_exp1": float(ks_stat_exp1(residuals)),
                "pred_ring": json.dumps(ev.predicted_ring),
                "diag_hub_pred": int(hub_pred_diag),
                "diag_hub_correct": int(hub_pred_diag == int(cfg.hub_actor)),
                "diag_top10": json.dumps([int(x) for x in order[:10]]),
                "diag_top10_scores": json.dumps([float(x) for x in np.round(member_scores[order[:10]], 6)]),
            }

            for j in range(cfg.K):
                proposed_row[f"g{j}_hat"] = float(result.params.g[j])
                proposed_row[f"h{j}_hat"] = float(result.params.h[j])
                proposed_row[f"sender{j}_score"] = float(sender_scores[j])
                proposed_row[f"receiver{j}_score"] = float(receiver_scores[j])
                proposed_row[f"member{j}_score"] = float(member_scores[j])
                proposed_row[f"rank_actor{j}"] = int(np.where(order == j)[0][0]) + 1

            for actor in cfg.ring_actors:
                actor_i = int(actor)
                proposed_diag_rows.append({
                    "seed": int(seed),
                    "actor": actor_i,
                    "rank": int(ranks[actor_i]),
                    "sender": float(sender_scores[actor_i]),
                    "receiver": float(receiver_scores[actor_i]),
                    "member_score": float(member_scores[actor_i]),
                    "g": float(result.params.g[actor_i]),
                    "h": float(result.params.h[actor_i]),
                    "selected_topk_diag": int(int(ranks[actor_i]) <= len(cfg.ring_actors)),
                    "true_hub": int(actor_i == int(cfg.hub_actor)),
                })

            print(
                f"[Full seed={cfg.seed}] "
                f"F1={float(ev.membership_f1):.3f} "
                f"AUC={float(ev.field_auc):.3f} "
                f"pred={ev.predicted_ring} "
                f"hub_pred_diag={hub_pred_diag} "
                f"true_ranks={ranks} "
                f"top10={order[:10].tolist()} "
                f"top10_scores={member_scores[order[:10]].round(4).tolist()} "
                f"sender_true={sender_scores[cfg.ring_actors].round(4).tolist()} "
                f"receiver_true={receiver_scores[cfg.ring_actors].round(4).tolist()}",
                flush=True,
            )

            proposed_rows.append(proposed_row)
            comparison_rows.append({"method": "Proposed", **proposed_row})
            if not exemplar:
                exemplar = {"cfg": cfg, "data": data, "result": result, "edges": edges, "residuals": residuals}
            print(
                f"    Proposed: F1={float(ev.membership_f1):.3f}, "
                f"hub={int(bool(ev.hub_correct))}, AUC={float(ev.field_auc):.3f}, "
                f"KS={float(proposed_row['ks_exp1']):.3f}",
                flush=True,
            )

        if method_requested(selected_methods, "B1_static_gates"):
            try:
                b1 = run_baseline_static_from_data(data, cfg)
            except Exception as exc:
                print(f"    B1_static_gates skipped: {exc}", flush=True)
            else:
                comparison_rows.append({
                    "method": "B1_static_gates",
                    **shared_meta,
                    **b1,
                    "spec_hat": float("nan"),
                    "eta_on_hat": float("nan"),
                    "eta_off_hat": float("nan"),
                    "beta0_hat": float("nan"),
                    "beta1_hat": float("nan"),
                    "ks_exp1": float("nan"),
                })
                print(f"    B1_static_gates: F1={b1['f1']:.3f}, hub={b1['hub_correct']}", flush=True)

        if method_requested(selected_methods, "B2_modular_hmm_hawkes"):
            try:
                b2 = run_baseline_modular_from_data(data, cfg)
            except Exception as exc:
                print(f"    B2_modular_hmm_hawkes skipped: {exc}", flush=True)
            else:
                comparison_rows.append({
                    "method": "B2_modular_hmm_hawkes",
                    **shared_meta,
                    **b2,
                    "spec_hat": float("nan"),
                    "eta_on_hat": float("nan"),
                    "eta_off_hat": float("nan"),
                    "beta0_hat": float("nan"),
                    "beta1_hat": float("nan"),
                    "ks_exp1": float("nan"),
                })
                print(
                    f"    B2_modular_hmm_hawkes: F1={b2['f1']:.3f}, "
                    f"hub={b2['hub_correct']}, AUC={b2['field_auc']:.3f}",
                    flush=True,
                )

        if method_requested(selected_methods, "B3_static_svd") or method_requested(selected_methods, "B4_static_degree"):
            try:
                b3_b4_rows = run_baseline_spectral_degree_from_data(
                    data,
                    cfg,
                    include_svd=method_requested(selected_methods, "B3_static_svd"),
                    include_degree=method_requested(selected_methods, "B4_static_degree"),
                )
            except Exception as exc:
                print(f"    B3/B4 static spectral baselines skipped: {exc}", flush=True)
            else:
                for b_row in b3_b4_rows:
                    comparison_rows.append({
                        **shared_meta,
                        **b_row,
                        "spec_hat": float("nan"),
                        "eta_on_hat": float("nan"),
                        "eta_off_hat": float("nan"),
                        "beta0_hat": float("nan"),
                        "beta1_hat": float("nan"),
                        "ks_exp1": float("nan"),
                    })
                    print(
                        f"    {b_row['method']}: F1={b_row['f1']:.3f}, hub={b_row['hub_correct']}",
                        flush=True,
                    )

        if include_mmhp or method_requested(selected_methods, "B5_mmhp_pairwise"):
            try:
                b5 = run_mmhp_baseline_from_data(data, cfg)
            except Exception as exc:
                print(f"    B5_mmhp_pairwise skipped: {exc}", flush=True)
            else:
                comparison_rows.append({
                    "method": "B5_mmhp_pairwise",
                    **shared_meta,
                    **b5,
                    "spec_hat": float("nan"),
                    "eta_on_hat": float("nan"),
                    "eta_off_hat": float("nan"),
                    "beta0_hat": float("nan"),
                    "beta1_hat": float("nan"),
                    "ks_exp1": float("nan"),
                })
                print(
                    f"    B5_mmhp_pairwise: F1={b5['f1']:.3f}, "
                    f"hub={b5['hub_correct']}, AUC={b5['field_auc']:.3f}",
                    flush=True,
                )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(outdir / "baseline_replications_raw.csv", index=False)

    if not comparison_df.empty:
        summary = (
            comparison_df.groupby("method")
            .agg(
                field_auc_mean=("field_auc", "mean"),
                field_auc_std=("field_auc", "std"),
                precision_mean=("precision", "mean"),
                precision_std=("precision", "std"),
                recall_mean=("recall", "mean"),
                recall_std=("recall", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
                hub_accuracy=("hub_correct", "mean"),
                n=("seed", "count"),
            )
            .reset_index()
        )
        summary = summary[[
            "method",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
            "f1_mean", "f1_std",
            "field_auc_mean", "field_auc_std",
            "hub_accuracy", "n",
        ]]
    else:
        summary = pd.DataFrame()
    summary.to_csv(outdir / "baseline_replications_summary.csv", index=False)
    if not summary.empty:
        render_baseline_replication_table(summary.set_index("method"), outdir / "baseline_replications_table.tex")

    rep_df = pd.DataFrame(proposed_rows)
    if not rep_df.empty:
        # Backward-compatible proposed-only diagnostics used by the existing paper figures.
        rep_df.to_csv(outdir / "replications_raw.csv", index=False)
        summary_df = rep_df[[
            "field_auc", "precision", "recall", "f1", "hub_correct", "events", "spec_hat", "ks_exp1",
            "eta_on_hat", "eta_off_hat", "beta0_hat", "beta1_hat",
        ]].agg(["mean", "std"])
        summary_df.to_csv(outdir / "replications_summary.csv")

    proposed_diag_df = pd.DataFrame(proposed_diag_rows)
    if not proposed_diag_df.empty:
        proposed_diag_df.to_csv(outdir / "proposed_assignment_diagnostics.csv", index=False)

    if exemplar:
        plot_activation_timeline(exemplar["data"], exemplar["result"], exemplar["edges"], outdir / "activation_timeline.png")
        if not rep_df.empty:
            plot_parameter_recovery(rep_df, exemplar["cfg"], outdir / "parameter_recovery.png")
            plot_gate_recovery(rep_df, exemplar["cfg"], outdir / "gate_recovery.png")
        plot_topology_heatmaps(exemplar["data"], exemplar["result"], outdir / "topology_heatmaps.png")
        qq_plot_exp1(exemplar["residuals"], outdir / "qq_rescaling.png")
        ks_plot_exp1(exemplar["residuals"], outdir / "ks_rescaling.png")

    return comparison_df, exemplar


def run_phase_grid(
    base_cfg: SimConfig,
    *,
    pi1_targets: list[float],
    T_values: list[float],
    seeds: list[int],
    outdir: Path,
    eta_off: float | None = None,
    method: str = "topk",
    fit_max_iters: int = 20,
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    eta_off_fixed = float(base_cfg.eta_off if eta_off is None else eta_off)

    total_jobs = len(pi1_targets) * len(T_values) * len(seeds)
    job_idx = 0
    for pi1 in pi1_targets:
        eta_on = eta_off_fixed * float(pi1) / max(1.0 - float(pi1), 1e-12)
        for T in T_values:
            for seed in seeds:
                job_idx += 1
                print(
                    f"[phase {job_idx}/{total_jobs}] pi1_target={float(pi1):.3f} T={float(T):.1f} seed={seed}",
                    flush=True,
                )
                cfg = SimConfig(**{**asdict(base_cfg), "seed": int(seed), "T": float(T), "eta_on": float(eta_on), "eta_off": eta_off_fixed})
                data, result, edges = fit_once(cfg, max_iters=fit_max_iters)
                ev = evaluate_result(data, result, edges, method=method)
                rows.append({
                    "pi1_target": float(pi1),
                    "T": float(T),
                    "seed": int(seed),
                    "f1": float(ev.membership_f1),
                    "field_auc": float(ev.field_auc),
                })
                print(
                    f"    done: F1={float(ev.membership_f1):.3f} AUC={float(ev.field_auc):.3f}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "phase_grid_raw.csv", index=False)
    plot_phase_diagram(df, x_col="pi1_target", y_col="T", value_col="f1", out_path=outdir / "phase_f1.png")
    plot_phase_diagram(df, x_col="pi1_target", y_col="T", value_col="field_auc", out_path=outdir / "phase_auc.png")
    return df


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Reviewer-upgrade experiments for the regime-switching Hawkes paper.")
    parser.add_argument("--outdir", type=str, default="reviewer_upgrade_outputs")
    parser.add_argument("--n-reps", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--method", type=str, default="topk", choices=["topk", "gap", "mixture", "posterior_ring"])
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help=(
            "Comma-separated model methods to run. Default: proposed,b1,b2,b3. "
            "Aliases: proposed, b1, b2, b3, b4, b5, svd, degree, mmhp, all."
        ),
    )
    parser.add_argument("--include-mmhp", action="store_true", help="Also run the optional slow MMHP pairwise baseline.")
    parser.add_argument("--only-proposed", action="store_true", help="Run only the proposed model.")
    parser.add_argument("--only-b1", action="store_true", help="Run only B1: static Hawkes + gates.")
    parser.add_argument("--only-b2", action="store_true", help="Run only B2: modular HMM + Hawkes.")
    parser.add_argument("--only-b3", action="store_true", help="Run only B3: static Hawkes + SVD.")
    parser.add_argument("--run-phase", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    cfg = default_config(seed=args.seed_start)
    seeds = list(range(args.seed_start, args.seed_start + args.n_reps))

    if args.only_proposed:
        selected_methods = {"Proposed"}
    elif args.only_b1:
        selected_methods = {"B1_static_gates"}
    elif args.only_b2:
        selected_methods = {"B2_modular_hmm_hawkes"}
    elif args.only_b3:
        selected_methods = {"B3_static_svd"}
    else:
        selected_methods = parse_method_selection(args.methods, include_mmhp=args.include_mmhp)

    print(
        f"Running reviewer_upgrade_experiments with n_reps={len(seeds)} seed_start={args.seed_start} "
        f"method={args.method} max_iters={args.max_iters} "
        f"model_methods={','.join(sorted(selected_methods))}",
        flush=True,
    )
    run_replications(
        cfg,
        seeds=seeds,
        outdir=outdir,
        method=args.method,
        methods=selected_methods,
        include_mmhp=args.include_mmhp,
        max_iters=args.max_iters,
    )
    print(f"Saved replication outputs to: {outdir}", flush=True)
    print(f"  baseline table: {outdir / 'baseline_replications_table.tex'}", flush=True)
    print(f"  baseline summary: {outdir / 'baseline_replications_summary.csv'}", flush=True)
    print(f"  baseline raw: {outdir / 'baseline_replications_raw.csv'}", flush=True)

    if args.run_phase:
        phase_dir = outdir / "phase_diagram"
        print(f"Running phase grid; outputs will be saved to: {phase_dir}", flush=True)
        run_phase_grid(
            cfg,
            pi1_targets=[0.05, 0.10, 0.15, 0.20],
            T_values=[250.0, 500.0, 750.0, 1000.0],
            seeds=seeds[: min(5, len(seeds))],
            outdir=phase_dir,
            fit_max_iters=min(args.max_iters, 20),
        )
        print(f"Saved phase-grid outputs to: {phase_dir}", flush=True)

if __name__ == "__main__":
    main()
