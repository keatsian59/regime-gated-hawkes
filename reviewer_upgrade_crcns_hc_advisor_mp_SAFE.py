# SAFE_V7_ADAPTIVE_INIT_AND_STABLE_ESTEP_RUNNER
from __future__ import annotations

"""Advisor-exact CRCNS hippocampal runner.

This runner is for the real-CA1 + additive spike-in design. It intentionally
DOES NOT inject internally. For spike-in rows it reads the already-generated
truth from crcns_additive_spikein.py: Z/is_injected columns, truth_json, and
active_intervals_csv. This avoids accidentally rerunning the synthetic/Hawkes
injection route.
"""

import argparse
import inspect
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import pandas as pd

import reviewer_upgrade_experiments as exp

def _clone_cfg_for_window(cfg: CRCNSConfig, *, T: float, seed: int | None = None) -> CRCNSConfig:
    vals = dict(vars(cfg))
    vals["T"] = float(T)
    if seed is not None:
        vals["seed"] = int(seed)
    return CRCNSConfig(**vals)


def _make_interval_edges(T: float, dt: float) -> np.ndarray:
    """Return interval edges that always include the final analysis time.

    np.arange(0, T, dt) silently drops the tail when T is not an exact
    multiple of dt. For point-process likelihoods that is fatal: events in
    (last_edge, T] get clipped/dropped into/out of the final interval. This
    helper appends T whenever needed so every in-window event has a legal bin.
    """
    T = float(T)
    dt = float(dt)
    if not np.isfinite(T) or T <= 0.0:
        return np.array([0.0, 1e-9], dtype=float)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt}")
    edges = np.arange(0.0, T, dt, dtype=float)
    if edges.size == 0 or edges[0] != 0.0:
        edges = np.insert(edges, 0, 0.0)
    if edges[-1] < T - 1e-12:
        edges = np.append(edges, T)
    elif edges[-1] > T + 1e-12:
        edges[-1] = T
    if edges.size < 2:
        edges = np.array([0.0, T], dtype=float)
    # Remove possible duplicates from tiny T or floating-point coincidence.
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([0.0, max(T, 1e-9)], dtype=float)
    return edges

from regime_hawkes.estep import run_estep


@dataclass
class CRCNSConfig:
    K: int
    M: int
    T: float
    ring_actors: list[int]
    hub_actor: int
    d: int = 3
    nu_base: float = 0.15
    alpha0_team: float = 0.01
    alpha1_max: float = 0.8
    alpha1_min: float = 0.35
    beta0: float = 1.0
    beta1: float = 3.0
    eta_on: float = 0.06
    eta_off: float = 0.30
    seed: int = 0


def _json_load_maybe(path: Any) -> dict[str, Any]:
    if path is None or (isinstance(path, float) and np.isnan(path)) or str(path).strip() == "":
        return {}
    p = Path(str(path))
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_ring(value: Any, K: int, ring_size: int = 4) -> list[int]:
    if value is not None and not (isinstance(value, float) and np.isnan(value)) and str(value).strip():
        try:
            parsed = json.loads(str(value))
            return [int(x) for x in parsed]
        except Exception:
            pass
    return list(range(min(int(ring_size), int(K))))


def _read_active_intervals(path: Any) -> list[tuple[float, float, int]] | None:
    if path is None or (isinstance(path, float) and np.isnan(path)) or str(path).strip() == "":
        return None
    p = Path(str(path))
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    # tolerate several possible column names
    start_col = "start" if "start" in df.columns else ("t_start" if "t_start" in df.columns else df.columns[0])
    end_col = "end" if "end" in df.columns else ("t_end" if "t_end" in df.columns else df.columns[1])
    return [(float(r[start_col]), float(r[end_col]), 1) for _, r in df.iterrows()]


def _make_event_array(df: pd.DataFrame) -> np.ndarray:
    needed = {"time", "actor", "mark"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"events CSV missing required columns: {sorted(missing)}")
    ev = df.sort_values("time")[["time", "actor", "mark"]].to_numpy(dtype=float)
    ev[:, 1] = ev[:, 1].astype(int)
    ev[:, 2] = ev[:, 2].astype(int)
    return ev




def _empirical_nu_matrix(events: np.ndarray, K: int, M: int, T: float, eps: float = 1e-6) -> np.ndarray:
    """Data-adaptive baseline intensity initialization.

    Real CA1 can have per-actor-per-mark rates near several spikes/sec.  The
    synthetic default nu_base=0.15 is far too small and forces EM to rescale
    intensities violently during the first M-step, which is exactly where the
    U/V and A1 overflows appear.  Initialize nu directly from observed counts.
    """
    T = max(float(T), eps)
    counts = np.zeros((int(K), int(M)), dtype=float)
    if events is not None and len(events):
        actors = np.asarray(events[:, 1], dtype=int)
        marks = np.asarray(events[:, 2], dtype=int)
        valid = (actors >= 0) & (actors < int(K)) & (marks >= 0) & (marks < int(M))
        np.add.at(counts, (actors[valid], marks[valid]), 1.0)
    nu = counts / T
    # A small floor avoids exact zeros for held-out windows and sparse units.
    # It should be much smaller than the empirical CA1 scale, not the synthetic
    # nu_base scale.
    positive = nu[nu > 0]
    floor = max(float(np.percentile(positive, 5)) * 0.05, eps) if positive.size else eps
    return np.maximum(nu, floor)


def _empirical_nu_base(events: np.ndarray, K: int, M: int, T: float) -> float:
    nu = _empirical_nu_matrix(events, K, M, T)
    return float(np.mean(nu))


def _cfg_with_empirical_nu_base(cfg: CRCNSConfig, events: np.ndarray) -> CRCNSConfig:
    base = _empirical_nu_base(events, int(cfg.K), int(cfg.M), float(cfg.T))
    return CRCNSConfig(
        K=int(cfg.K), M=int(cfg.M), T=float(cfg.T), ring_actors=list(cfg.ring_actors), hub_actor=int(cfg.hub_actor),
        d=int(cfg.d), nu_base=base, alpha0_team=float(cfg.alpha0_team),
        alpha1_max=float(cfg.alpha1_max), alpha1_min=float(cfg.alpha1_min), beta0=float(cfg.beta0), beta1=float(cfg.beta1),
        eta_on=float(cfg.eta_on), eta_off=float(cfg.eta_off), seed=int(cfg.seed),
    )


def _create_adaptive_initial_params(cfg: CRCNSConfig, events: np.ndarray) -> Any:
    """Call the paper init, then force nu to the empirical CA1 count rate."""
    cfg_emp = _cfg_with_empirical_nu_base(cfg, events)
    init = exp.create_initial_params(cfg_emp, events)
    init.nu = _empirical_nu_matrix(events, int(cfg.K), int(cfg.M), float(cfg.T))
    return init

def make_paper_data_from_manifest_row(row: pd.Series, *, seed: int, ring_size: int, time_start: float = 0.0, duration: float | None = None, max_events: int | None = None) -> tuple[Any, CRCNSConfig, dict[str, Any]]:
    df = pd.read_csv(row["path"]).sort_values("time").reset_index(drop=True)
    condition = str(row.get("condition", "real"))
    # Safety filters for real CRCNS sessions. Full hc-11 sessions can contain
    # millions of events, which is not viable for 20 parallel EM jobs.
    if time_start and float(time_start) > 0:
        df = df[df["time"] >= float(time_start)].copy()
    if duration is not None and float(duration) > 0:
        t0_filter = float(time_start) if time_start and float(time_start) > 0 else float(df["time"].min())
        df = df[df["time"] < t0_filter + float(duration)].copy()
    if max_events is not None and int(max_events) > 0 and len(df) > int(max_events):
        df = df.head(int(max_events)).copy()
    if df.empty:
        raise ValueError(f"No events left after filters: time_start={time_start}, duration={duration}, max_events={max_events}, path={row['path']}")
    # Re-zero each filtered analysis window. This keeps T small and avoids
    # treating pre-window silence as part of the likelihood.
    df["time"] = df["time"] - float(df["time"].min())
    df = df.sort_values("time").reset_index(drop=True)
    truth = _json_load_maybe(row.get("truth_json", None))

    K = int(df["actor"].max()) + 1
    M = int(df["mark"].max()) + 1
    T = float(df["time"].max())

    true_ring = [int(x) for x in truth.get("true_ring", _parse_ring(row.get("true_ring", None), K, ring_size))]
    true_hub = int(truth.get("true_hub", row.get("true_hub", true_ring[0] if true_ring else 0)))

    z_at_event = None
    if "Z" in df.columns:
        z_at_event = df["Z"].astype(int).to_numpy()
    elif "is_injected" in df.columns:
        z_at_event = df["is_injected"].astype(int).to_numpy()

    z_path = _read_active_intervals(row.get("active_intervals_csv", None))

    cfg = CRCNSConfig(K=K, M=M, T=T, ring_actors=true_ring, hub_actor=true_hub, seed=int(seed), nu_base=_empirical_nu_base(_make_event_array(df), K, M, T))
    data = SimpleNamespace(
        events=_make_event_array(df),
        Z_at_event=z_at_event,
        Z_path=z_path,
        config=cfg,
        source_events_csv=str(row["path"]),
        condition=condition,
    )
    meta = {
        "dataset": row.get("dataset", "crcns"),
        "session": row.get("session", Path(str(row["path"])).parent.name),
        "condition": condition,
        "alpha1_strength": float(row["alpha1_strength"]) if "alpha1_strength" in row and pd.notna(row["alpha1_strength"]) else np.nan,
        "events": int(len(df)),
        "K": K,
        "M": M,
        "T": T,
        "empirical_nu_base": float(cfg.nu_base),
        "mean_events_per_actor_mark_per_sec": float(cfg.nu_base),
        "ring_size": int(len(true_ring)),
        "hub_actor": true_hub,
        "true_ring": json.dumps(true_ring),
        "n_injected_events": int(np.sum(z_at_event)) if z_at_event is not None else 0,
        "hawkes_sampler_used": bool(truth.get("hawkes_sampler_used", False)) if truth else False,
    }
    return data, cfg, meta


def _active_event_truth(events: np.ndarray, z_at_event: np.ndarray | None, interval_edges: np.ndarray) -> np.ndarray:
    bn = len(interval_edges) - 1
    truth = np.zeros(bn, dtype=int)
    if z_at_event is None or len(events) == 0:
        return truth
    idx = np.searchsorted(interval_edges, np.asarray(events)[:, 0], side="right") - 1
    idx = np.clip(idx, 0, bn - 1)
    active_idx = idx[np.asarray(z_at_event, dtype=int) == 1]
    if len(active_idx):
        truth[np.unique(active_idx)] = 1
    return truth


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if hasattr(exp, "roc_auc_binary"):
        return float(exp.roc_auc_binary(y_true, scores))
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def _compute_A1(result: Any) -> np.ndarray:
    # Use a hardened local A1 construction even if reviewer_upgrade_experiments
    # exposes an older unclipped compute_A1 helper.
    p = result.params
    g = np.clip(np.nan_to_num(np.asarray(p.g, dtype=float), nan=0.0, posinf=5.0, neginf=0.0), 0.0, 5.0)
    h = np.clip(np.nan_to_num(np.asarray(p.h, dtype=float), nan=0.0, posinf=5.0, neginf=0.0), 0.0, 5.0)
    U = np.clip(np.nan_to_num(np.asarray(p.U, dtype=float), nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    V = np.clip(np.nan_to_num(np.asarray(p.V, dtype=float), nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    inner = np.clip(U @ V.T, -40.0, 40.0)
    sp = np.log1p(np.exp(-np.abs(inner))) + np.maximum(inner, 0.0)
    A1 = np.clip(g[:, None] * h[None, :] * sp, 0.0, 2.0)
    np.fill_diagonal(A1, 0.0)
    return A1


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if np.all(x == 0):
        return 0.0
    x = np.sort(np.maximum(x, 0.0))
    n = x.size
    return float((2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * np.sum(x))) - (n + 1.0) / n)


def _topk_membership_scores(member: np.ndarray, sender: np.ndarray, cfg: CRCNSConfig) -> dict[str, Any]:
    """Compute top-k membership/hub metrics only when spike-in truth exists."""
    k = max(len(cfg.ring_actors), 1)
    order = np.argsort(np.asarray(member, dtype=float))[::-1]
    pred_ring = [int(x) for x in order[:k]]
    hub_pred = int(np.argmax(sender)) if len(sender) else -1
    truth = set(int(x) for x in cfg.ring_actors)
    pred = set(pred_ring)
    tp = len(pred & truth)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(truth), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "hub_correct": int(hub_pred == int(cfg.hub_actor)),
        "pred_ring": json.dumps(pred_ring),
        "hub_pred": int(hub_pred),
    }



def _rho_channel_summary(rho0: Any, rho1: Any) -> dict[str, float]:
    """Flatten learned routine/active mark-transition matrices for CSV export."""
    out: dict[str, float] = {}
    r0 = np.asarray(rho0, dtype=float)
    r1 = np.asarray(rho1, dtype=float)

    if r0.ndim != 2 or r1.ndim != 2:
        return out

    M0 = min(r0.shape)
    M1 = min(r1.shape)
    M = min(M0, M1)

    for a in range(M):
        for b in range(M):
            out[f"rho0_{a}{b}"] = float(r0[a, b])
            out[f"rho1_{a}{b}"] = float(r1[a, b])
            out[f"rho_lift_{a}{b}"] = float(r1[a, b] - r0[a, b])

    if M >= 2:
        out["rho0_stress_same"] = float(r0[1, 1])
        out["rho1_stress_same"] = float(r1[1, 1])
        out["rho_lift_stress_same"] = float(r1[1, 1] - r0[1, 1])

    out["rho0_same_mark_mass"] = float(np.trace(r0[:M, :M]))
    out["rho1_same_mark_mass"] = float(np.trace(r1[:M, :M]))
    out["rho_lift_same_mark_mass"] = float(np.trace(r1[:M, :M]) - np.trace(r0[:M, :M]))

    return out


def _fit_proposed_no_eval(
    data: Any,
    cfg: CRCNSConfig,
    *,
    max_iters: int,
    dt: float,
    proposed_lr: float = 0.0025,
    proposed_lambda_g: float = 0.05,
    proposed_lambda_h: float = 0.05,
    proposed_lambda_e: float = 0.05,
    proposed_lambda_0: float = 0.10,
    proposed_n_inner_steps: int = 3,
    proposed_stability_target: float = 0.50,
    proposed_eta_prior_weight_s: float = 0.0,
    proposed_eta_prior_weight_pi: float = 0.0,
    proposed_eta_off_floor: float = 0.0,
    proposed_eta_pi1_target: float | None = None,
    proposed_eta_beta_prior_kappa: float = 0.0,
    proposed_eta_beta_prior_target: float = 0.5,
    proposed_eta_gamma_prior_shape: float = 2.0,
    proposed_eta_gamma_prior_rate: float = 1.0,
    proposed_eta_gamma_prior_weight: float = 0.0,
    proposed_pi1_init: float = 0.15,
) -> tuple[Any, np.ndarray]:
    """Fit Proposed without calling the synthetic evaluator.

    Real CA1 rows have no planted Z_path. reviewer_upgrade_experiments.fit_proposed_from_data
    always calls evaluate_result(), which expects Z_path and therefore crashes on raw
    real data. This helper runs the same EM fit but leaves evaluation to this
    advisor runner, where real rows get unsupervised diagnostics and spike-in rows
    get truth-based metrics.
    """
    edges = _make_interval_edges(float(cfg.T), float(dt))
    if len(edges) < 2:
        raise ValueError(f"Not enough interval edges for T={cfg.T}, dt={dt}")
    init = _create_adaptive_initial_params(cfg, data.events)
    kwargs = dict(
        events=data.events,
        interval_edges=edges,
        init_params=init,
        max_iters=int(max_iters),
        # Real CA1 is much denser than synthetic reviewer data.  These settings
        # keep the active block from exploding while still running the full 30 EM
        # iterations across 20 CPU workers.
        lr=float(proposed_lr),
        lambda_g=float(proposed_lambda_g),
        lambda_h=float(proposed_lambda_h),
        lambda_e=float(proposed_lambda_e),
        lambda_0=float(proposed_lambda_0),
        n_inner_steps=int(proposed_n_inner_steps),
        verbose=False,
    )
    sig = inspect.signature(exp.run_em)
    if "stability_threshold" in sig.parameters:
        kwargs["stability_threshold"] = 0.95
    if "stability_target" in sig.parameters:
        kwargs["stability_target"] = float(proposed_stability_target)
    if "eta_prior_weight_s" in sig.parameters:
        kwargs["eta_prior_weight_s"] = float(proposed_eta_prior_weight_s)
    if "eta_prior_weight_pi" in sig.parameters:
        kwargs["eta_prior_weight_pi"] = float(proposed_eta_prior_weight_pi)
    if "eta_off_floor" in sig.parameters:
        kwargs["eta_off_floor"] = float(proposed_eta_off_floor)
    if "eta_pi1_target" in sig.parameters and proposed_eta_pi1_target is not None:
        kwargs["eta_pi1_target"] = float(proposed_eta_pi1_target)
    if "eta_beta_prior_kappa" in sig.parameters:
        kwargs["eta_beta_prior_kappa"] = float(proposed_eta_beta_prior_kappa)
    if "eta_beta_prior_target" in sig.parameters:
        kwargs["eta_beta_prior_target"] = float(proposed_eta_beta_prior_target)
    if "eta_gamma_prior_shape" in sig.parameters:
        kwargs["eta_gamma_prior_shape"] = float(proposed_eta_gamma_prior_shape)
    if "eta_gamma_prior_rate" in sig.parameters:
        kwargs["eta_gamma_prior_rate"] = float(proposed_eta_gamma_prior_rate)
    if "eta_gamma_prior_weight" in sig.parameters:
        kwargs["eta_gamma_prior_weight"] = float(proposed_eta_gamma_prior_weight)
    if "pi1_init" in sig.parameters:
        kwargs["pi1_init"] = float(proposed_pi1_init)
    result = exp.run_em(**kwargs)
    return result, edges


def _proposed_row(
    data: Any,
    cfg: CRCNSConfig,
    meta: dict[str, Any],
    *,
    max_iters: int,
    selection_method: str,
    dt: float,
    proposed_lr: float = 0.0025,
    proposed_lambda_g: float = 0.05,
    proposed_lambda_h: float = 0.05,
    proposed_lambda_e: float = 0.05,
    proposed_lambda_0: float = 0.10,
    proposed_n_inner_steps: int = 3,
    proposed_stability_target: float = 0.50,
    proposed_eta_prior_weight_s: float = 0.0,
    proposed_eta_prior_weight_pi: float = 0.0,
    proposed_eta_off_floor: float = 0.0,
    proposed_eta_pi1_target: float | None = None,
    proposed_eta_beta_prior_kappa: float = 0.0,
    proposed_eta_beta_prior_target: float = 0.5,
    proposed_eta_gamma_prior_shape: float = 2.0,
    proposed_eta_gamma_prior_rate: float = 1.0,
    proposed_eta_gamma_prior_weight: float = 0.0,
    proposed_pi1_init: float = 0.15,
) -> dict[str, Any]:
    # Do not call exp.fit_proposed_from_data here. It invokes the synthetic evaluator,
    # which requires Z_path and is invalid for raw real CA1 rows.
    result, edges = _fit_proposed_no_eval(
        data,
        cfg,
        max_iters=max_iters,
        dt=dt,
        proposed_lr=proposed_lr,
        proposed_lambda_g=proposed_lambda_g,
        proposed_lambda_h=proposed_lambda_h,
        proposed_lambda_e=proposed_lambda_e,
        proposed_lambda_0=proposed_lambda_0,
        proposed_n_inner_steps=proposed_n_inner_steps,
        proposed_stability_target=proposed_stability_target,
        proposed_eta_prior_weight_s=proposed_eta_prior_weight_s,
        proposed_eta_prior_weight_pi=proposed_eta_prior_weight_pi,
        proposed_eta_off_floor=proposed_eta_off_floor,
        proposed_eta_pi1_target=proposed_eta_pi1_target,
        proposed_eta_beta_prior_kappa=proposed_eta_beta_prior_kappa,
        proposed_eta_beta_prior_target=proposed_eta_beta_prior_target,
        proposed_eta_gamma_prior_shape=proposed_eta_gamma_prior_shape,
        proposed_eta_gamma_prior_rate=proposed_eta_gamma_prior_rate,
        proposed_eta_gamma_prior_weight=proposed_eta_gamma_prior_weight,
        proposed_pi1_init=proposed_pi1_init,
    )
    A1 = _compute_A1(result)
    sender = np.sum(A1, axis=1)
    receiver = np.sum(A1, axis=0)
    member = np.maximum(sender / max(float(np.max(sender)), 1e-12), receiver / max(float(np.max(receiver)), 1e-12))

    has_truth = data.Z_at_event is not None and int(np.sum(data.Z_at_event)) > 0
    field_auc = np.nan
    metrics = {"precision": np.nan, "recall": np.nan, "f1": np.nan, "hub_correct": np.nan, "pred_ring": "[]", "hub_pred": np.nan}
    if has_truth:
        y = _active_event_truth(data.events, data.Z_at_event, edges)
        field_auc = _auc(y, result.gamma[:, 1]) if getattr(result, "gamma", None) is not None else np.nan
        metrics = _topk_membership_scores(member, sender, cfg)

    out = {
        "method": "Proposed",
        **meta,
        "field_auc": float(field_auc),
        **metrics,
        "loglik_final": float(result.log_likelihood_trace[-1]) if getattr(result, "log_likelihood_trace", None) else np.nan,
        "beta0_hat": float(result.params.beta0),
        "beta1_hat": float(result.params.beta1),
        "beta_gap_hat": float(abs(float(result.params.beta1) - float(result.params.beta0))),
        **_rho_channel_summary(result.params.rho0, result.params.rho1),
        "eta_on_hat": float(result.params.eta_on),
        "eta_off_hat": float(result.params.eta_off),
        "active_mean": float(np.mean(result.gamma[:, 1])) if getattr(result, "gamma", None) is not None else np.nan,
        "gate_gini": float(_gini(member)),
        "gate_effective_support": float((np.sum(member) ** 2) / max(float(np.sum(member ** 2)), 1e-12)),
        "proposed_lr": float(proposed_lr),
        "proposed_lambda_g": float(proposed_lambda_g),
        "proposed_lambda_h": float(proposed_lambda_h),
        "proposed_lambda_e": float(proposed_lambda_e),
        "proposed_lambda_0": float(proposed_lambda_0),
        "proposed_n_inner_steps": int(proposed_n_inner_steps),
        "proposed_stability_target": float(proposed_stability_target),
        "proposed_eta_prior_weight_s": float(proposed_eta_prior_weight_s),
        "proposed_eta_prior_weight_pi": float(proposed_eta_prior_weight_pi),
        "proposed_eta_off_floor": float(proposed_eta_off_floor),
        "proposed_eta_pi1_target": float(proposed_eta_pi1_target) if proposed_eta_pi1_target is not None else np.nan,
        "proposed_eta_beta_prior_kappa": float(proposed_eta_beta_prior_kappa),
        "proposed_eta_beta_prior_target": float(proposed_eta_beta_prior_target),
        "proposed_eta_gamma_prior_shape": float(proposed_eta_gamma_prior_shape),
        "proposed_eta_gamma_prior_rate": float(proposed_eta_gamma_prior_rate),
        "proposed_eta_gamma_prior_weight": float(proposed_eta_gamma_prior_weight),
        "proposed_pi1_init": float(proposed_pi1_init),
    }
    for j in range(cfg.K):
        out[f"g{j}_hat"] = float(result.params.g[j])
        out[f"h{j}_hat"] = float(result.params.h[j])
        out[f"sender{j}_score"] = float(sender[j])
        out[f"receiver{j}_score"] = float(receiver[j])
        out[f"member{j}_score"] = float(member[j])
    return out




def _clone_cfg_for_window(cfg: CRCNSConfig, *, T: float, seed: int | None = None) -> CRCNSConfig:
    return CRCNSConfig(
        K=int(cfg.K), M=int(cfg.M), T=float(T), ring_actors=list(cfg.ring_actors), hub_actor=int(cfg.hub_actor),
        d=int(cfg.d), nu_base=float(cfg.nu_base), alpha0_team=float(cfg.alpha0_team),
        alpha1_max=float(cfg.alpha1_max), alpha1_min=float(cfg.alpha1_min), beta0=float(cfg.beta0), beta1=float(cfg.beta1),
        eta_on=float(cfg.eta_on), eta_off=float(cfg.eta_off), seed=int(cfg.seed if seed is None else seed),
    )


def _split_train_test(data: Any, cfg: CRCNSConfig, *, heldout_frac: float) -> tuple[Any, CRCNSConfig, Any, CRCNSConfig, float]:
    """Contiguous time split for real-data held-out likelihood.

    We split by time, not random events, because random event splits break point-process
    history. Each side is re-zeroed and evaluated with reset history. This is a
    conservative held-out diagnostic for raw real CA1.
    """
    frac = float(heldout_frac)
    if frac <= 0.0 or frac >= 0.9:
        raise ValueError(f"heldout_frac must be in (0, 0.9); got {heldout_frac}")
    split_t = float(cfg.T) * (1.0 - frac)
    ev = np.asarray(data.events, dtype=float)
    train_mask = ev[:, 0] < split_t
    test_mask = ~train_mask
    train_ev = ev[train_mask].copy()
    test_ev = ev[test_mask].copy()
    if len(train_ev) < 10 or len(test_ev) < 10:
        raise ValueError(f"Held-out split too small: train={len(train_ev)} test={len(test_ev)} split_t={split_t:.3f}")
    test_ev[:, 0] -= split_t
    train_T = max(float(split_t), float(train_ev[:, 0].max()) + 1e-9)
    test_T = max(float(cfg.T - split_t), float(test_ev[:, 0].max()) + 1e-9)
    train_z = None
    test_z = None
    if getattr(data, "Z_at_event", None) is not None:
        z = np.asarray(data.Z_at_event, dtype=int)
        train_z = z[train_mask]
        test_z = z[test_mask]
    train_cfg = _clone_cfg_for_window(cfg, T=train_T, seed=int(cfg.seed))
    test_cfg = _clone_cfg_for_window(cfg, T=test_T, seed=int(cfg.seed))
    train_data = SimpleNamespace(events=train_ev, Z_at_event=train_z, Z_path=None, config=train_cfg, condition=getattr(data, "condition", "real"))
    test_data = SimpleNamespace(events=test_ev, Z_at_event=test_z, Z_path=None, config=test_cfg, condition=getattr(data, "condition", "real"))
    return train_data, train_cfg, test_data, test_cfg, split_t


def _normalize_rho(rho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rho = np.maximum(np.asarray(rho, dtype=float), eps)
    return rho / np.maximum(rho.sum(axis=1, keepdims=True), eps)


def _static_superposed_loglik(events: np.ndarray, *, T: float, K: int, M: int, nu: np.ndarray, A: np.ndarray, rho: np.ndarray, beta: float, eps: float = 1e-12) -> float:
    """Single-regime marked Hawkes log likelihood for held-out static baselines."""
    events = np.asarray(events, dtype=float)
    nu = np.asarray(nu, dtype=float).reshape(K, M)
    A = np.asarray(A, dtype=float).reshape(K, K)
    rho = _normalize_rho(np.asarray(rho, dtype=float).reshape(M, M))
    beta = float(max(beta, 1e-6))
    B = np.zeros((K, M), dtype=float)
    t_prev = 0.0
    ll = 0.0
    for t, actor, mark in events:
        t = float(t)
        if t < t_prev:
            raise ValueError("events must be sorted by time")
        dt = t - t_prev
        pre_exc = A.T @ (B @ rho)
        ll -= float(np.sum(nu)) * dt + float(np.sum(pre_exc)) * ((1.0 - np.exp(-beta * dt)) / beta)
        B *= np.exp(-beta * dt)
        i, m = int(actor), int(mark)
        exc_now = A.T @ (B @ rho)
        lam = max(float(nu[i, m] + exc_now[i, m]), eps)
        ll += float(np.log(lam))
        B[i, m] += beta
        t_prev = t
    tail = max(float(T) - t_prev, 0.0)
    if tail > 0:
        pre_exc = A.T @ (B @ rho)
        ll -= float(np.sum(nu)) * tail + float(np.sum(pre_exc)) * ((1.0 - np.exp(-beta * tail)) / beta)
    return float(ll)


def _proposed_heldout_row(
    data: Any,
    cfg: CRCNSConfig,
    meta: dict[str, Any],
    *,
    max_iters: int,
    dt: float,
    heldout_frac: float,
    proposed_lr: float = 0.0025,
    proposed_lambda_g: float = 0.05,
    proposed_lambda_h: float = 0.05,
    proposed_lambda_e: float = 0.05,
    proposed_lambda_0: float = 0.10,
    proposed_n_inner_steps: int = 3,
    proposed_stability_target: float = 0.50,
    proposed_eta_prior_weight_s: float = 0.0,
    proposed_eta_prior_weight_pi: float = 0.0,
    proposed_eta_off_floor: float = 0.0,
    proposed_eta_pi1_target: float | None = None,
    proposed_eta_beta_prior_kappa: float = 0.0,
    proposed_eta_beta_prior_target: float = 0.5,
    proposed_eta_gamma_prior_shape: float = 2.0,
    proposed_eta_gamma_prior_rate: float = 1.0,
    proposed_eta_gamma_prior_weight: float = 0.0,
    proposed_pi1_init: float = 0.15,
) -> dict[str, Any]:
    train_data, train_cfg, test_data, test_cfg, split_t = _split_train_test(data, cfg, heldout_frac=heldout_frac)
    result, train_edges = _fit_proposed_no_eval(
        train_data,
        train_cfg,
        max_iters=max_iters,
        dt=dt,
        proposed_lr=proposed_lr,
        proposed_lambda_g=proposed_lambda_g,
        proposed_lambda_h=proposed_lambda_h,
        proposed_lambda_e=proposed_lambda_e,
        proposed_lambda_0=proposed_lambda_0,
        proposed_n_inner_steps=proposed_n_inner_steps,
        proposed_stability_target=proposed_stability_target,
        proposed_eta_prior_weight_s=proposed_eta_prior_weight_s,
        proposed_eta_prior_weight_pi=proposed_eta_prior_weight_pi,
        proposed_eta_off_floor=proposed_eta_off_floor,
        proposed_eta_pi1_target=proposed_eta_pi1_target,
        proposed_eta_beta_prior_kappa=proposed_eta_beta_prior_kappa,
        proposed_eta_beta_prior_target=proposed_eta_beta_prior_target,
        proposed_eta_gamma_prior_shape=proposed_eta_gamma_prior_shape,
        proposed_eta_gamma_prior_rate=proposed_eta_gamma_prior_rate,
        proposed_eta_gamma_prior_weight=proposed_eta_gamma_prior_weight,
        proposed_pi1_init=proposed_pi1_init,
    )
    test_edges = _make_interval_edges(float(test_cfg.T), float(dt))
    if len(test_edges) < 2:
        test_edges = np.array([0.0, float(test_cfg.T)], dtype=float)
    A1 = _compute_A1(result)
    pi1 = float(result.params.eta_on / max(result.params.eta_on + result.params.eta_off, 1e-12))
    gamma_prev = np.full(len(test_edges) - 1, np.clip(pi1, 1e-4, 1.0 - 1e-4), dtype=float)
    est = run_estep(
        events=test_data.events,
        interval_edges=test_edges,
        gamma_prev_1=gamma_prev,
        nu=result.params.nu,
        A0=result.params.A0,
        A1=A1,
        rho0=result.params.rho0,
        rho1=result.params.rho1,
        beta0=result.params.beta0,
        beta1=result.params.beta1,
        eta_on=result.params.eta_on,
        eta_off=result.params.eta_off,
    )
    sender = np.sum(A1, axis=1)
    receiver = np.sum(A1, axis=0)
    member = np.maximum(sender / max(float(np.max(sender)), 1e-12), receiver / max(float(np.max(receiver)), 1e-12))
    heldout_ll = float(est.log_likelihood)
    out = {
        "method": "Proposed",
        **meta,
        "heldout_frac": float(heldout_frac),
        "heldout_split_t": float(split_t),
        "train_events": int(len(train_data.events)),
        "test_events": int(len(test_data.events)),
        "heldout_loglik": heldout_ll,
        "heldout_nll_per_event": float(-heldout_ll / max(len(test_data.events), 1)),
        "heldout_metric_type": "latent_ctmc_marginal_interval_loglik",
        "field_auc": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "hub_correct": np.nan,
        "loglik_final": float(result.log_likelihood_trace[-1]) if getattr(result, "log_likelihood_trace", None) else np.nan,
        "beta0_hat": float(result.params.beta0),
        "beta1_hat": float(result.params.beta1),
        "beta_gap_hat": float(abs(float(result.params.beta1) - float(result.params.beta0))),
        **_rho_channel_summary(result.params.rho0, result.params.rho1),
        "eta_on_hat": float(result.params.eta_on),
        "eta_off_hat": float(result.params.eta_off),
        "active_mean": float(np.mean(result.gamma[:, 1])) if getattr(result, "gamma", None) is not None else np.nan,
        "gate_gini": float(_gini(member)),
        "gate_effective_support": float((np.sum(member) ** 2) / max(float(np.sum(member ** 2)), 1e-12)),
        "proposed_lr": float(proposed_lr),
        "proposed_lambda_g": float(proposed_lambda_g),
        "proposed_lambda_h": float(proposed_lambda_h),
        "proposed_lambda_e": float(proposed_lambda_e),
        "proposed_lambda_0": float(proposed_lambda_0),
        "proposed_n_inner_steps": int(proposed_n_inner_steps),
        "proposed_stability_target": float(proposed_stability_target),
        "proposed_eta_prior_weight_s": float(proposed_eta_prior_weight_s),
        "proposed_eta_prior_weight_pi": float(proposed_eta_prior_weight_pi),
        "proposed_eta_off_floor": float(proposed_eta_off_floor),
        "proposed_eta_pi1_target": float(proposed_eta_pi1_target) if proposed_eta_pi1_target is not None else np.nan,
        "proposed_eta_beta_prior_kappa": float(proposed_eta_beta_prior_kappa),
        "proposed_eta_beta_prior_target": float(proposed_eta_beta_prior_target),
        "proposed_eta_gamma_prior_shape": float(proposed_eta_gamma_prior_shape),
        "proposed_eta_gamma_prior_rate": float(proposed_eta_gamma_prior_rate),
        "proposed_eta_gamma_prior_weight": float(proposed_eta_gamma_prior_weight),
        "proposed_pi1_init": float(proposed_pi1_init),
    }
    for j in range(cfg.K):
        out[f"g{j}_hat"] = float(result.params.g[j])
        out[f"h{j}_hat"] = float(result.params.h[j])
        out[f"sender{j}_score"] = float(sender[j])
        out[f"receiver{j}_score"] = float(receiver[j])
        out[f"member{j}_score"] = float(member[j])
    return out


def _fit_static_b1_params(train_data: Any, train_cfg: CRCNSConfig, *, max_iters: int) -> tuple[Any, float]:
    if exp.run_static_hawkes_baseline is None:
        raise ImportError(f"Could not import run_static_hawkes_baseline: {getattr(exp, '_STATIC_IMPORT_ERROR', None)}")
    edges = _make_interval_edges(float(train_cfg.T), 2.0)
    init0 = _create_adaptive_initial_params(train_cfg, train_data.events)
    params, ll = exp.run_static_hawkes_baseline(
        events=train_data.events,
        interval_edges=edges,
        init_params=init0,
        max_iters=min(int(max_iters), 30),
        lr=0.005,
        lambda_g=0.05,
        lambda_h=0.05,
        lambda_e=0.05,
        lambda_0=0.10,
        n_inner_steps=3,
        verbose=False,
    )
    return params, float(ll)


def _b1_heldout_row(data: Any, cfg: CRCNSConfig, meta: dict[str, Any], *, max_iters: int, heldout_frac: float) -> dict[str, Any]:
    train_data, train_cfg, test_data, test_cfg, split_t = _split_train_test(data, cfg, heldout_frac=heldout_frac)
    params, train_ll = _fit_static_b1_params(train_data, train_cfg, max_iters=max_iters)
    A1 = exp.compute_A1(params.g, params.h, params.U, params.V) if hasattr(exp, "compute_A1") else _compute_A1(SimpleNamespace(params=params))
    A = np.asarray(params.A0, dtype=float) + np.asarray(A1, dtype=float)
    rho = _normalize_rho(0.5 * (np.asarray(params.rho0, dtype=float) + np.asarray(params.rho1, dtype=float)))
    beta = float(0.5 * (float(params.beta0) + float(params.beta1)))
    ll = _static_superposed_loglik(test_data.events, T=test_cfg.T, K=cfg.K, M=cfg.M, nu=params.nu, A=A, rho=rho, beta=beta)
    return {
        "method": "B1_static_gates",
        **meta,
        "heldout_frac": float(heldout_frac),
        "heldout_split_t": float(split_t),
        "train_events": int(len(train_data.events)),
        "test_events": int(len(test_data.events)),
        "heldout_loglik": float(ll),
        "heldout_nll_per_event": float(-ll / max(len(test_data.events), 1)),
        "heldout_metric_type": "single_regime_static_marked_hawkes_loglik_from_static_gates",
        "field_auc": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "hub_correct": np.nan,
        "loglik_final": float(train_ll),
    }


def _b3_heldout_row(data: Any, cfg: CRCNSConfig, meta: dict[str, Any], *, max_iters: int, heldout_frac: float) -> dict[str, Any]:
    if exp._fit_static_hawkes_full_matrix is None:
        raise ImportError(f"Could not import spectral baseline fit helper: {getattr(exp, '_SPECTRAL_IMPORT_ERROR', None)}")
    train_data, train_cfg, test_data, test_cfg, split_t = _split_train_test(data, cfg, heldout_frac=heldout_frac)
    nu_hat, A_hat, rho_hat, beta_hat = exp._fit_static_hawkes_full_matrix(
        events=train_data.events,
        K=cfg.K,
        M=cfg.M,
        T=train_cfg.T,
        beta_init=2.0,
        lr=0.005,
        n_iters=min(max(50, int(max_iters) * 5), 200),
        l1_penalty=0.01,
        verbose=False,
    )
    ll = _static_superposed_loglik(test_data.events, T=test_cfg.T, K=cfg.K, M=cfg.M, nu=nu_hat, A=A_hat, rho=rho_hat, beta=float(beta_hat))
    return {
        "method": "B3_static_svd",
        **meta,
        "heldout_frac": float(heldout_frac),
        "heldout_split_t": float(split_t),
        "train_events": int(len(train_data.events)),
        "test_events": int(len(test_data.events)),
        "heldout_loglik": float(ll),
        "heldout_nll_per_event": float(-ll / max(len(test_data.events), 1)),
        "heldout_metric_type": "single_regime_full_matrix_marked_hawkes_loglik_before_svd_selection",
        "field_auc": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "hub_correct": np.nan,
        "loglik_final": np.nan,
    }


def _poisson_logpmf_counts(counts: np.ndarray, rate: float) -> np.ndarray:
    # log k! via cumulative sum table, no scipy dependency.
    counts = np.asarray(counts, dtype=int)
    max_k = int(np.max(counts)) if counts.size else 0
    logfac = np.zeros(max_k + 1, dtype=float)
    if max_k >= 1:
        logfac[1:] = np.cumsum(np.log(np.arange(1, max_k + 1)))
    lam = max(float(rate), 1e-9)
    return counts * np.log(lam) - lam - logfac[counts]


def _hmm_count_loglik(train_events: np.ndarray, test_events: np.ndarray, *, train_T: float, test_T: float, dt: float) -> tuple[float, dict[str, float]]:
    train_edges = _make_interval_edges(float(train_T), float(dt))
    test_edges = _make_interval_edges(float(test_T), float(dt))
    if len(train_edges) < 3:
        train_edges = np.linspace(0.0, float(train_T), 3)
    if len(test_edges) < 2:
        test_edges = np.array([0.0, float(test_T)], dtype=float)
    train_counts = np.histogram(train_events[:, 0], bins=train_edges)[0].astype(int)
    test_counts = np.histogram(test_events[:, 0], bins=test_edges)[0].astype(int)
    if train_counts.size < 4:
        rate = float(np.mean(train_counts)) if train_counts.size else 1e-3
        ll = float(np.sum(_poisson_logpmf_counts(test_counts, rate)))
        return ll, {"lambda0": rate, "lambda1": rate, "p00": 0.95, "p11": 0.95}
    thresh = float(np.quantile(train_counts, 0.75))
    z = (train_counts >= thresh).astype(int)
    low = train_counts[z == 0]
    high = train_counts[z == 1]
    lam0 = float(max(np.mean(low) if len(low) else np.mean(train_counts), 1e-6))
    lam1 = float(max(np.mean(high) if len(high) else np.mean(train_counts), lam0 + 1e-6))
    # Sticky transitions estimated from thresholded train sequence with smoothing.
    n00 = np.sum((z[:-1] == 0) & (z[1:] == 0)) + 1.0
    n01 = np.sum((z[:-1] == 0) & (z[1:] == 1)) + 1.0
    n11 = np.sum((z[:-1] == 1) & (z[1:] == 1)) + 1.0
    n10 = np.sum((z[:-1] == 1) & (z[1:] == 0)) + 1.0
    P = np.array([[n00 / (n00 + n01), n01 / (n00 + n01)], [n10 / (n10 + n11), n11 / (n10 + n11)]], dtype=float)
    pi = np.array([max(np.mean(z == 0), 1e-6), max(np.mean(z == 1), 1e-6)], dtype=float)
    pi = pi / pi.sum()
    logE = np.column_stack([_poisson_logpmf_counts(test_counts, lam0), _poisson_logpmf_counts(test_counts, lam1)])
    logP = np.log(np.clip(P, 1e-12, 1.0))
    alpha = np.log(pi + 1e-12) + logE[0]
    for b in range(1, len(test_counts)):
        prev = alpha[:, None] + logP
        m = np.max(prev, axis=0)
        alpha = logE[b] + m + np.log(np.sum(np.exp(prev - m), axis=0))
    m = float(np.max(alpha))
    ll = float(m + np.log(np.sum(np.exp(alpha - m))))
    return ll, {"lambda0": lam0, "lambda1": lam1, "p00": float(P[0,0]), "p11": float(P[1,1])}


def _b2_heldout_row(data: Any, cfg: CRCNSConfig, meta: dict[str, Any], *, heldout_frac: float, dt: float) -> dict[str, Any]:
    train_data, train_cfg, test_data, test_cfg, split_t = _split_train_test(data, cfg, heldout_frac=heldout_frac)
    ll, diag = _hmm_count_loglik(train_data.events, test_data.events, train_T=train_cfg.T, test_T=test_cfg.T, dt=dt)
    return {
        "method": "B2_modular_hmm_hawkes",
        **meta,
        "heldout_frac": float(heldout_frac),
        "heldout_split_t": float(split_t),
        "train_events": int(len(train_data.events)),
        "test_events": int(len(test_data.events)),
        "heldout_loglik": float(ll),
        "heldout_nll_per_event": float(-ll / max(len(test_data.events), 1)),
        "heldout_metric_type": "modular_hmm_interval_count_loglik_segmentation_stage",
        "field_auc": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "hub_correct": np.nan,
        "active_mean": float(diag.get("lambda1", np.nan) / max(diag.get("lambda0", 0.0) + diag.get("lambda1", 0.0), 1e-12)),
        "b2_lambda0": float(diag.get("lambda0", np.nan)),
        "b2_lambda1": float(diag.get("lambda1", np.nan)),
        "b2_p00": float(diag.get("p00", np.nan)),
        "b2_p11": float(diag.get("p11", np.nan)),
    }

def _run_one(payload: dict[str, Any]) -> list[dict[str, Any]]:
    row = pd.Series(payload["manifest_row"])
    seed = int(payload["seed"])
    methods = set(payload["methods"])
    data, cfg, meta = make_paper_data_from_manifest_row(row, seed=seed, ring_size=int(payload["ring_size"]), time_start=float(payload.get("time_start", 0.0) or 0.0), duration=payload.get("duration", None), max_events=payload.get("max_events", None))
    meta["seed"] = seed
    rows: list[dict[str, Any]] = []
    print(f"[{meta['dataset']}/{meta['session']}] condition={meta['condition']} alpha={meta['alpha1_strength']} seed={seed} events={meta['events']} K={meta['K']}", flush=True)

    heldout_frac = float(payload.get("heldout_frac", 0.0) or 0.0)
    no_truth = data.Z_at_event is None or int(np.sum(data.Z_at_event)) == 0
    use_heldout_real = heldout_frac > 0.0 and no_truth

    if "Proposed" in methods:
        if use_heldout_real:
            rows.append(_proposed_heldout_row(
                data,
                cfg,
                meta,
                max_iters=int(payload["max_iters"]),
                dt=float(payload.get("dt", 2.0)),
                heldout_frac=heldout_frac,
                proposed_lr=float(payload.get("proposed_lr", 0.0025)),
                proposed_lambda_g=float(payload.get("proposed_lambda_g", 0.05)),
                proposed_lambda_h=float(payload.get("proposed_lambda_h", 0.05)),
                proposed_lambda_e=float(payload.get("proposed_lambda_e", 0.05)),
                proposed_lambda_0=float(payload.get("proposed_lambda_0", 0.10)),
                proposed_n_inner_steps=int(payload.get("proposed_n_inner_steps", 3)),
                proposed_stability_target=float(payload.get("proposed_stability_target", 0.50)),
                proposed_eta_prior_weight_s=float(payload.get("proposed_eta_prior_weight_s", 0.0)),
                proposed_eta_prior_weight_pi=float(payload.get("proposed_eta_prior_weight_pi", 0.0)),
                proposed_eta_off_floor=float(payload.get("proposed_eta_off_floor", 0.0)),
                proposed_eta_pi1_target=(None if payload.get("proposed_eta_pi1_target", None) is None else float(payload.get("proposed_eta_pi1_target"))),
                proposed_eta_beta_prior_kappa=float(payload.get("proposed_eta_beta_prior_kappa", 0.0)),
                proposed_eta_beta_prior_target=float(payload.get("proposed_eta_beta_prior_target", 0.5)),
                proposed_eta_gamma_prior_shape=float(payload.get("proposed_eta_gamma_prior_shape", 2.0)),
                proposed_eta_gamma_prior_rate=float(payload.get("proposed_eta_gamma_prior_rate", 1.0)),
                proposed_eta_gamma_prior_weight=float(payload.get("proposed_eta_gamma_prior_weight", 0.0)),
                proposed_pi1_init=float(payload.get("proposed_pi1_init", 0.15)),
            ))
        else:
            rows.append(_proposed_row(
                data,
                cfg,
                meta,
                max_iters=int(payload["max_iters"]),
                selection_method=str(payload["selection_method"]),
                dt=float(payload.get("dt", 2.0)),
                proposed_lr=float(payload.get("proposed_lr", 0.0025)),
                proposed_lambda_g=float(payload.get("proposed_lambda_g", 0.05)),
                proposed_lambda_h=float(payload.get("proposed_lambda_h", 0.05)),
                proposed_lambda_e=float(payload.get("proposed_lambda_e", 0.05)),
                proposed_lambda_0=float(payload.get("proposed_lambda_0", 0.10)),
                proposed_n_inner_steps=int(payload.get("proposed_n_inner_steps", 3)),
                proposed_stability_target=float(payload.get("proposed_stability_target", 0.50)),
                proposed_eta_prior_weight_s=float(payload.get("proposed_eta_prior_weight_s", 0.0)),
                proposed_eta_prior_weight_pi=float(payload.get("proposed_eta_prior_weight_pi", 0.0)),
                proposed_eta_off_floor=float(payload.get("proposed_eta_off_floor", 0.0)),
                proposed_eta_pi1_target=(None if payload.get("proposed_eta_pi1_target", None) is None else float(payload.get("proposed_eta_pi1_target"))),
                proposed_eta_beta_prior_kappa=float(payload.get("proposed_eta_beta_prior_kappa", 0.0)),
                proposed_eta_beta_prior_target=float(payload.get("proposed_eta_beta_prior_target", 0.5)),
                proposed_eta_gamma_prior_shape=float(payload.get("proposed_eta_gamma_prior_shape", 2.0)),
                proposed_eta_gamma_prior_rate=float(payload.get("proposed_eta_gamma_prior_rate", 1.0)),
                proposed_eta_gamma_prior_weight=float(payload.get("proposed_eta_gamma_prior_weight", 0.0)),
                proposed_pi1_init=float(payload.get("proposed_pi1_init", 0.15)),
            ))

    if "B1_static_gates" in methods:
        try:
            if use_heldout_real:
                rows.append(_b1_heldout_row(data, cfg, meta, max_iters=int(payload["max_iters"]), heldout_frac=heldout_frac))
            else:
                b1 = exp.run_baseline_static_from_data(data, cfg)
                rows.append({"method": "B1_static_gates", **meta, **b1})
        except Exception as exc:
            print(f"    B1 skipped: {exc}", flush=True)

    if "B2_modular_hmm_hawkes" in methods:
        try:
            if use_heldout_real:
                rows.append(_b2_heldout_row(data, cfg, meta, heldout_frac=heldout_frac, dt=float(payload.get("dt", 2.0))))
            else:
                edges = _make_interval_edges(float(cfg.T), float(payload.get("dt", 2.0)))
                b2 = exp.run_modular_hmm_hawkes_baseline(
                    events=data.events,
                    interval_edges=edges,
                    true_ring=cfg.ring_actors,
                    true_hub=cfg.hub_actor,
                    Z_path=None,
                    selection="topk",
                    top_k=len(cfg.ring_actors),
                    seed=seed,
                )
                field_auc = np.nan
                if data.Z_at_event is not None and hasattr(b2, "hmm_gamma"):
                    field_auc = _auc(_active_event_truth(data.events, data.Z_at_event, edges), b2.hmm_gamma[:, 1])
                rows.append({
                    "method": "B2_modular_hmm_hawkes",
                    **meta,
                    "field_auc": float(field_auc),
                    "precision": float(getattr(b2, "membership_precision", np.nan)),
                    "recall": float(getattr(b2, "membership_recall", np.nan)),
                    "f1": float(getattr(b2, "membership_f1", np.nan)),
                    "hub_correct": int(bool(getattr(b2, "hub_correct", False))) if data.Z_at_event is not None else np.nan,
                    "active_mean": float(np.mean(b2.hmm_gamma[:, 1])) if hasattr(b2, "hmm_gamma") else np.nan,
                })
        except Exception as exc:
            print(f"    B2 skipped: {exc}", flush=True)

    if "B3_static_svd" in methods:
        try:
            if use_heldout_real:
                rows.append(_b3_heldout_row(data, cfg, meta, max_iters=int(payload["max_iters"]), heldout_frac=heldout_frac))
            else:
                b3_rows = exp.run_baseline_spectral_degree_from_data(data, cfg, include_svd=True, include_degree=False)
                for r in b3_rows:
                    rows.append({**meta, **r})
        except Exception as exc:
            print(f"    B3 skipped: {exc}", flush=True)
    return rows


def _selected_methods(text: str) -> set[str]:
    aliases = {
        "proposed": "Proposed",
        "b1": "B1_static_gates",
        "b2": "B2_modular_hmm_hawkes",
        "b3": "B3_static_svd",
    }
    out = set()
    for tok in [x.strip().lower() for x in text.split(",") if x.strip()]:
        if tok == "all":
            out.update(aliases.values())
        elif tok in aliases:
            out.add(aliases[tok])
        else:
            raise ValueError(f"Unknown method: {tok}")
    return out


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    if args.max_sessions:
        manifest = manifest.head(int(args.max_sessions)).copy()
    # If the manifest already has a seed column, use each row as a job; otherwise repeat n_reps seeds.
    methods = _selected_methods(args.methods)
    payloads = []

    def _base_payload(row_dict: dict[str, Any], seed: int) -> dict[str, Any]:
        return {
            "manifest_row": row_dict,
            "seed": int(seed),
            "methods": sorted(methods),
            "max_iters": args.max_iters,
            "selection_method": args.selection_method,
            "ring_size": args.ring_size,
            "dt": args.dt,
            "time_start": args.time_start,
            "duration": args.duration,
            "max_events": args.max_events,
            "heldout_frac": args.heldout_frac,
            "proposed_lr": args.proposed_lr,
            "proposed_lambda_g": args.proposed_lambda_g,
            "proposed_lambda_h": args.proposed_lambda_h,
            "proposed_lambda_e": args.proposed_lambda_e,
            "proposed_lambda_0": args.proposed_lambda_0,
            "proposed_n_inner_steps": args.proposed_n_inner_steps,
            "proposed_stability_target": args.proposed_stability_target,
            "proposed_eta_prior_weight_s": args.proposed_eta_prior_weight_s,
            "proposed_eta_prior_weight_pi": args.proposed_eta_prior_weight_pi,
            "proposed_eta_off_floor": args.proposed_eta_off_floor,
            "proposed_eta_pi1_target": args.proposed_eta_pi1_target,
            "proposed_eta_beta_prior_kappa": args.proposed_eta_beta_prior_kappa,
            "proposed_eta_beta_prior_target": args.proposed_eta_beta_prior_target,
            "proposed_eta_gamma_prior_shape": args.proposed_eta_gamma_prior_shape,
            "proposed_eta_gamma_prior_rate": args.proposed_eta_gamma_prior_rate,
            "proposed_eta_gamma_prior_weight": args.proposed_eta_gamma_prior_weight,
            "proposed_pi1_init": args.proposed_pi1_init,
        }

    if "seed" in manifest.columns and args.use_manifest_seeds:
        for _, row in manifest.iterrows():
            payloads.append(_base_payload(row.to_dict(), int(row["seed"])))
    else:
        seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.n_reps)))
        for _, row in manifest.iterrows():
            row_dict = row.to_dict()
            for seed in seeds:
                payloads.append(_base_payload(row_dict, seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nested = []
    error_rows = []

    def _payload_error_row(p: dict[str, Any], exc: BaseException) -> dict[str, Any]:
        mr = p.get("manifest_row", {})
        return {
            "dataset": mr.get("dataset", "crcns"),
            "session": mr.get("session", Path(str(mr.get("path", "unknown"))).parent.name),
            "condition": mr.get("condition", "real"),
            "seed": p.get("seed"),
            "methods": ",".join(p.get("methods", [])),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "path": mr.get("path", ""),
        }

    if int(args.workers) <= 1:
        for p in payloads:
            try:
                nested.append(_run_one(p))
            except Exception as exc:
                err = _payload_error_row(p, exc)
                print(f"[error] {err['session']} seed={err['seed']} {err['error_type']}: {err['error']}", flush=True)
                error_rows.append(err)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            future_to_payload = {ex.submit(_run_one, p): p for p in payloads}
            completed = 0
            for fut in as_completed(future_to_payload):
                p = future_to_payload[fut]
                completed += 1
                try:
                    nested.append(fut.result())
                except Exception as exc:
                    err = _payload_error_row(p, exc)
                    print(f"[error] {err['session']} seed={err['seed']} {err['error_type']}: {err['error']}", flush=True)
                    error_rows.append(err)
                # Checkpoint after every completed future so one bad job never loses a night.
                raw_ckpt = pd.DataFrame([r for block in nested for r in block])
                raw_ckpt.to_csv(outdir / "crcns_baseline_raw_checkpoint.csv", index=False)
                if error_rows:
                    pd.DataFrame(error_rows).to_csv(outdir / "crcns_error_rows.csv", index=False)
                print(f"[progress] completed {completed}/{len(payloads)} jobs; ok_rows={len(raw_ckpt)} errors={len(error_rows)}", flush=True)

    raw = pd.DataFrame([r for block in nested for r in block])
    raw.to_csv(outdir / "crcns_baseline_raw.csv", index=False)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(outdir / "crcns_error_rows.csv", index=False)
    if not raw.empty:
        group_cols = [c for c in ["dataset", "session", "condition", "alpha1_strength", "method"] if c in raw.columns]
        metric_cols = [c for c in ["field_auc", "precision", "recall", "f1", "hub_correct", "loglik_final", "active_mean", "beta0_hat", "beta1_hat", "beta_gap_hat", "gate_gini", "gate_effective_support", "heldout_loglik", "heldout_nll_per_event", "train_events", "test_events"] if c in raw.columns]
        metric_cols += [
            c for c in raw.columns
            if (
                c.startswith("rho0_") or c.startswith("rho1_") or c.startswith("rho_lift_")
            ) and c not in metric_cols
        ]
        summary = raw.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"])
        summary.to_csv(outdir / "crcns_baseline_summary.csv")
    print(f"[done] wrote outputs to {outdir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advisor-exact CRCNS Hawkes runner: real/null/additive-spikein manifests, no internal injection.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--methods", type=str, default="proposed,b1,b2,b3")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--n-reps", type=int, default=1)
    p.add_argument("--seed-start", type=int, default=7)
    p.add_argument("--use-manifest-seeds", action="store_true", help="Use the seed column in spike-in/null manifests instead of repeating n_reps.")
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument("--selection-method", type=str, default="topk", choices=["topk", "gap", "mixture", "posterior_ring"])
    p.add_argument("--ring-size", type=int, default=4)
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--dt", type=float, default=2.0, help="Interval width for proposed/B2 edges.")
    p.add_argument("--time-start", type=float, default=0.0, help="Start time within each event CSV before re-zeroing.")
    p.add_argument("--duration", type=float, default=None, help="Analysis-window duration in seconds. Strongly recommended for raw CRCNS.")
    p.add_argument("--max-events", type=int, default=None, help="Hard cap on events per job after time filtering.")
    p.add_argument("--heldout-frac", type=float, default=0.0, help="For raw no-truth real rows, fit on the first 1-frac of the window and report held-out likelihood on the final frac. Use 0.2 for §6.1 Proposed/B1/B2/B3 real comparison.")

    p.add_argument("--proposed-lr", type=float, default=0.0025, help="Learning rate for the Proposed EM active-block M-step.")
    p.add_argument("--proposed-lambda-g", type=float, default=0.05, help="Sender-gate L1 penalty for Proposed.")
    p.add_argument("--proposed-lambda-h", type=float, default=0.05, help="Receiver-gate L1 penalty for Proposed.")
    p.add_argument("--proposed-lambda-e", type=float, default=0.05, help="Embedding penalty for Proposed.")
    p.add_argument("--proposed-lambda-0", type=float, default=0.10, help="Routine/static-block penalty for Proposed.")
    p.add_argument("--proposed-n-inner-steps", type=int, default=3, help="Inner gradient steps per M-step for Proposed.")
    p.add_argument("--proposed-stability-target", type=float, default=0.50, help="Spectral-radius target used when the EM implementation exposes stability_target.")
    p.add_argument("--proposed-eta-prior-weight-s", type=float, default=0.0, help="Weak quadratic prior weight on CTMC total transition rate log scale.")
    p.add_argument("--proposed-eta-prior-weight-pi", type=float, default=0.0, help="Weak quadratic prior weight on CTMC stationary active probability logit.")
    p.add_argument("--proposed-eta-off-floor", type=float, default=0.0, help="Hard lower bound on eta_off during the Proposed CTMC M-step; diagnostic only.")
    p.add_argument("--proposed-eta-pi1-target", type=float, default=None, help="Legacy quadratic stationary-active target. Prefer --proposed-eta-beta-prior-target for paper runs.")
    p.add_argument("--proposed-eta-beta-prior-kappa", type=float, default=0.0, help="Beta prior concentration kappa for CTMC stationary active fraction pi1. Default 0 disables it.")
    p.add_argument("--proposed-eta-beta-prior-target", type=float, default=0.5, help="Beta prior target pi1* for CTMC stationary active fraction.")
    p.add_argument("--proposed-eta-gamma-prior-shape", type=float, default=2.0, help="Gamma prior shape for total CTMC switching rate s=eta_on+eta_off.")
    p.add_argument("--proposed-eta-gamma-prior-rate", type=float, default=1.0, help="Gamma prior rate for total CTMC switching rate s=eta_on+eta_off.")
    p.add_argument("--proposed-eta-gamma-prior-weight", type=float, default=0.0, help="Weight on Gamma prior for total CTMC switching rate. Use small values such as 1.0.")
    p.add_argument("--proposed-pi1-init", type=float, default=0.15, help="Initial active posterior mass used to initialize EM gamma_prev_1.")
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    run(parse_args())
