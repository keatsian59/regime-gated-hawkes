from __future__ import annotations

"""Synthetic P2 weakest-link runner for Theorem 2.

Place this file under ``regime_hawkes/run_synthetic_p2_weaklink_mp.py`` and run
from the repository root, for example:

    python -m regime_hawkes.run_synthetic_p2_weaklink_mp --seeds 20 --workers 8

Purpose
-------
This runner produces a raw checkpoint CSV that satisfies the same P2 contract as
``evaluate_ucdp_ged_spikein.py`` expects:

    method,status,seed,weak_receiver_frac,alpha_min_relative,alpha_min_proxy,
    pi1T_planted,n_eff_planted,weak_receiver_recovered,...

Unlike the UCDP spike-in, alpha_min here is a TRUE generative active-excitation
coefficient. One receiver is designated as the weak receiver; its hub->receiver
active coefficient is set to alpha_min_relative times the uniform coefficient,
and the remaining receiver coefficients are raised so the total hub->receiver
active mass stays fixed. This is the clean synthetic P2 test: vary true
alpha_min while keeping aggregate active coupling fixed, then sweep active
exposure pi1T and fit the log threshold slope.
"""

import argparse
import csv
import inspect
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Must be set before importing numerical/JAX-backed package modules.
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

from regime_hawkes.config import SimConfig
from regime_hawkes.em import run_em
from regime_hawkes.evaluate import evaluate
from regime_hawkes.mstep import MStepResult
from regime_hawkes.utils import compute_A1


@dataclass(frozen=True)
class P2Options:
    K: int = 20
    subgroup_size: int = 6
    d: int = 3
    M: int = 2
    nu_base: float = 0.10
    hub_rate: float = 0.5
    alpha0_team: float = 0.005
    active_mass: float = 0.80
    beta0: float = 1.0
    beta1: float = 3.0
    eta_on: float = 0.04
    eta_off: float = 0.30
    n_active_episodes: int = 4
    interval_width: float = 2.0
    lr: float = 0.01
    lambda_g: float = 0.02
    lambda_h: float = 0.02
    lambda_0: float = 0.05
    n_inner_steps: int = 10
    stability_threshold: float = 0.95
    stability_target: float = 0.80
    path_mode: str = "deterministic"


@dataclass
class SyntheticP2Data:
    events: np.ndarray
    Z_path: list[tuple[float, float, int]]
    Z_at_event: np.ndarray
    config: SimConfig
    nu: np.ndarray
    rho0: np.ndarray
    rho1: np.ndarray
    A0: np.ndarray
    A1: np.ndarray
    gates_g: np.ndarray
    gates_h: np.ndarray
    U: np.ndarray
    V: np.ndarray
    true_receivers: list[int]
    weak_receiver: int
    alpha_min_true: float
    alpha_uniform: float
    alpha_min_relative: float


def softplus_np(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x)
    out = np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0.0)
    if np.ndim(x) == 0:
        return float(out)
    return out


def state_at_time(path: list[tuple[float, float, int]], t: float) -> int:
    """Return the latent state at time t.

    In deterministic P2 mode, ``Z_path`` stores active windows only.  Times
    outside those windows are dormant.  The previous fallback returned the
    state of the last active block, which mislabeled every event outside an
    active window as active and made ``active_events_total == total_events``.
    """
    for s, e, z in path:
        if float(s) <= float(t) < float(e):
            return int(z)
    return 0


def deterministic_active_windows(T: float, eta_on: float, eta_off: float, n_episodes: int) -> list[tuple[float, float, int]]:
    """Create exact active windows with active time pi1*T.

    We use the stationary active fraction implied by eta_on/eta_off but make the
    design levels exact and shared across seeds. This keeps the evaluator's
    threshold grouping clean while preserving a fixed switching-rate design.
    """
    n = max(1, int(n_episodes))
    pi1 = float(eta_on) / max(float(eta_on) + float(eta_off), 1e-12)
    total_active = max(0.0, min(float(T), pi1 * float(T)))
    length = total_active / float(n)
    if length <= 0:
        return []
    centers = np.linspace(0.15 * float(T), 0.85 * float(T), n)
    windows: list[tuple[float, float, int]] = []
    last_end = 0.0
    for c in centers:
        s = max(0.0, float(c) - 0.5 * length)
        e = min(float(T), s + length)
        if s < last_end:
            s = last_end
            e = min(float(T), s + length)
        if e > s:
            windows.append((float(s), float(e), 1))
            last_end = float(e)
    return windows


def simulate_ctmc_path(T: float, eta_on: float, eta_off: float, rng: np.random.Generator) -> list[tuple[float, float, int]]:
    t = 0.0
    state = 0
    path: list[tuple[float, float, int]] = []
    while t < T:
        rate = eta_on if state == 0 else eta_off
        dt = rng.exponential(1.0 / max(rate, 1e-12))
        t_next = min(float(T), t + dt)
        path.append((float(t), float(t_next), int(state)))
        t = t_next
        if t < T:
            state = 1 - state
    return path


def active_time_from_path(path: list[tuple[float, float, int]]) -> float:
    return float(sum(float(e) - float(s) for s, e, z in path if int(z) == 1))


def empirical_mark_init(events: np.ndarray, M: int, eps: float = 1.0) -> np.ndarray:
    counts = np.full((M, M), eps, dtype=float)
    if len(events) == 0:
        return counts / counts.sum(axis=1, keepdims=True)
    order = np.lexsort((events[:, 0], events[:, 1]))
    ev = events[order]
    last_mark_by_actor: dict[int, int] = {}
    for _, actor, mark in ev:
        actor_i = int(actor)
        mark_i = int(mark)
        if actor_i in last_mark_by_actor:
            counts[last_mark_by_actor[actor_i], mark_i] += 1.0
        last_mark_by_actor[actor_i] = mark_i
    return counts / counts.sum(axis=1, keepdims=True)


def rough_beta_init(events: np.ndarray) -> tuple[float, float]:
    if len(events) <= 1:
        return 1.0, 2.0
    t = np.sort(events[:, 0])
    dt = np.diff(t)
    dt = dt[dt > 1e-8]
    if len(dt) == 0:
        return 1.0, 2.0
    beta0 = float(np.clip(1.0 / np.median(dt), 0.1, 1.0))
    beta1 = float(np.clip(1.0 / np.percentile(dt, 10), beta0 + 0.1, 8.0))
    return beta0, beta1


def set_f1(pred: list[int] | np.ndarray, truth: list[int] | np.ndarray) -> float:
    p = set(int(x) for x in pred)
    t = set(int(x) for x in truth)
    if not p and not t:
        return 1.0
    tp = len(p & t)
    prec = tp / max(len(p), 1)
    rec = tp / max(len(t), 1)
    return float(2.0 * prec * rec / max(prec + rec, 1e-12))


def receiver_f1_from_h(h: np.ndarray, true_receivers: list[int], hub: int) -> tuple[float, list[int]]:
    h = np.asarray(h, dtype=float)
    order = [int(x) for x in np.argsort(h)[::-1] if int(x) != int(hub)]
    pred = order[:len(true_receivers)]
    return set_f1(pred, true_receivers), pred


def exact_support_metrics(pred_ring, true_ring, pred_receivers, true_receivers, hub_pred: int, true_hub: int) -> dict[str, int]:
    member_exact = int(set(int(x) for x in pred_ring) == set(int(x) for x in true_ring))
    receiver_exact = int(set(int(x) for x in pred_receivers) == set(int(x) for x in true_receivers))
    hub_exact = int(int(hub_pred) == int(true_hub))
    return {
        "member_exact": member_exact,
        "receiver_exact": receiver_exact,
        "hub_exact": hub_exact,
        "member_receiver_exact": int(member_exact and receiver_exact),
        "all_role_exact": int(member_exact and receiver_exact and hub_exact),
    }


def active_event_truth(events: np.ndarray, z_at_event: np.ndarray, interval_edges: np.ndarray) -> np.ndarray:
    B = max(len(interval_edges) - 1, 0)
    truth = np.zeros(B, dtype=int)
    if len(events) == 0 or B <= 0:
        return truth
    idx = np.searchsorted(interval_edges, events[:, 0], side="right") - 1
    idx = np.clip(idx, 0, B - 1)
    active_idx = idx[np.asarray(z_at_event, dtype=int) == 1]
    if active_idx.size:
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


def build_p2_truth(cfg: SimConfig, opts: P2Options, alpha_min_relative: float, rng: np.random.Generator):
    K, M, d = int(cfg.K), int(cfg.M), int(cfg.d)
    subgroup = [int(x) for x in cfg.ring_actors]
    hub = int(cfg.hub_actor)
    receivers = [a for a in subgroup if a != hub]
    if len(receivers) < 2:
        raise ValueError("P2 requires at least two receivers so freed weak-link mass can be redistributed.")
    weak_receiver = int(receivers[0])

    nu = np.full((K, M), float(opts.nu_base), dtype=float)
    # The active block is a star hub->receivers with spectral radius 0: each hub
    # event sprays receivers exactly once, no self-sustaining chain. So the active-
    # born count is (hub events occurring inside active windows) x active_mass. If
    # the hub only fires at the shared background rate nu_base, almost no hub events
    # land in the ~pi1 fraction of active time, the active block is invisible, N_eff
    # collapses to background-in-window, and the alpha_min knob has no effect. We
    # therefore drive the hub at its own rate so the active cascade actually ignites.
    # This keeps role separation intact (hub is still sender-only via g) and does not
    # affect stability (star spectral radius stays 0).
    nu[hub, :] = float(getattr(opts, "hub_rate", opts.nu_base))
    rho0 = np.full((M, M), 1.0 / float(M), dtype=float)
    if M == 2:
        rho1 = np.array([[0.5, 1.5], [0.3, 0.7]], dtype=float)
        rho1 = rho1 / rho1.sum(axis=1, keepdims=True)
    else:
        rho1 = np.eye(M, dtype=float)

    A0 = np.zeros((K, K), dtype=float)
    team_size = int(getattr(cfg, "team_size", max(1, K)))
    for j in range(K):
        team_j = j // team_size
        for i in range(K):
            if i != j and i // team_size == team_j:
                A0[j, i] = float(opts.alpha0_team)

    # True P2 alpha coordinate: uniform coefficient under no weak link.
    n_recv = len(receivers)
    alpha_uniform = float(opts.active_mass) / float(n_recv)
    frac = float(alpha_min_relative)
    weak_alpha = max(0.0, frac * alpha_uniform)
    other_alpha = (float(opts.active_mass) - weak_alpha) / float(n_recv - 1)

    A1 = np.zeros((K, K), dtype=float)
    for r in receivers:
        A1[hub, int(r)] = weak_alpha if int(r) == weak_receiver else other_alpha

    # The estimator parameterization can represent this star through a sender gate
    # on the hub and receiver gates on the receivers.
    gates_g = np.zeros(K, dtype=float)
    gates_h = np.zeros(K, dtype=float)
    gates_g[hub] = 1.0
    for r in receivers:
        gates_h[int(r)] = A1[hub, int(r)]
    U = np.zeros((K, d), dtype=float)
    V = np.zeros((K, d), dtype=float)
    if d > 0:
        U[hub, 0] = 1.0
        for r in receivers:
            V[int(r), 0] = 1.0

    # Guard stability. Defaults are chosen not to trigger this, but if a user
    # raises active_mass aggressively, record truth after scaling.
    spec = float(np.max(np.linalg.eigvals(A0 + A1).real)) if K else 0.0
    if spec >= 0.98:
        scale = 0.98 / spec
        A0 *= scale
        A1 *= scale
        weak_alpha *= scale
        alpha_uniform *= scale
        gates_h *= scale

    return nu, rho0, rho1, A0, A1, gates_g, gates_h, U, V, receivers, weak_receiver, weak_alpha, alpha_uniform


def simulate_p2_data(cfg: SimConfig, opts: P2Options, alpha_min_relative: float) -> SyntheticP2Data:
    rng = np.random.default_rng(int(cfg.seed))
    nu, rho0, rho1, A0, A1, gates_g, gates_h, U, V, receivers, weak_receiver, weak_alpha, alpha_uniform = build_p2_truth(
        cfg, opts, alpha_min_relative, rng
    )
    if opts.path_mode == "ctmc":
        Z_path = simulate_ctmc_path(float(cfg.T), float(opts.eta_on), float(opts.eta_off), rng)
    else:
        Z_path = deterministic_active_windows(float(cfg.T), float(opts.eta_on), float(opts.eta_off), int(opts.n_active_episodes))

    K, M = int(cfg.K), int(cfg.M)
    t = 0.0
    B = np.zeros((K, M), dtype=float)
    A = np.zeros((K, M), dtype=float)
    events: list[tuple[float, int, int]] = []
    z_labels: list[int] = []

    max_jump = 0.0
    for src in range(K):
        for m_prev in range(M):
            inc0 = np.sum(A0[src][:, None] * rho0[m_prev][None, :] * float(opts.beta0))
            inc1 = np.sum(A1[src][:, None] * rho1[m_prev][None, :] * float(opts.beta1))
            max_jump = max(max_jump, float(inc0 + inc1))

    while t < float(cfg.T):
        routine = A0.T @ (B @ rho0)
        active = A1.T @ (A @ rho1)
        lam = nu + routine + active
        lam_sum = float(lam.sum())
        upper = lam_sum + max_jump + 1e-8
        if upper <= 0:
            break

        t_candidate = t + rng.exponential(1.0 / upper)
        if t_candidate > float(cfg.T):
            break

        dt = t_candidate - t
        B *= np.exp(-float(opts.beta0) * dt)
        A *= np.exp(-float(opts.beta1) * dt)

        routine_c = A0.T @ (B @ rho0)
        active_c = A1.T @ (A @ rho1)
        lam_c = nu + routine_c + active_c
        true_total = float(lam_c.sum())

        if true_total > 0 and rng.uniform() <= true_total / upper:
            flat = lam_c.reshape(-1)
            idx = int(rng.choice(flat.size, p=flat / flat.sum()))
            actor = idx // M
            mark = idx % M
            z_evt = state_at_time(Z_path, float(t_candidate))
            events.append((float(t_candidate), int(actor), int(mark)))
            z_labels.append(int(z_evt))
            B[actor, mark] += float(opts.beta0)
            if z_evt == 1:
                A[actor, mark] += float(opts.beta1)

        t = float(t_candidate)

    arr = np.array(events, dtype=float) if events else np.zeros((0, 3), dtype=float)
    z_arr = np.array(z_labels, dtype=int)
    return SyntheticP2Data(
        events=arr,
        Z_path=Z_path,
        Z_at_event=z_arr,
        config=cfg,
        nu=nu,
        rho0=rho0,
        rho1=rho1,
        A0=A0,
        A1=A1,
        gates_g=gates_g,
        gates_h=gates_h,
        U=U,
        V=V,
        true_receivers=[int(x) for x in receivers],
        weak_receiver=int(weak_receiver),
        alpha_min_true=float(weak_alpha),
        alpha_uniform=float(alpha_uniform),
        alpha_min_relative=float(weak_alpha / max(alpha_uniform, 1e-12)),
    )


def make_cfg(seed: int, T: float, opts: P2Options) -> SimConfig:
    cfg = SimConfig(
        K=int(opts.K),
        M=int(opts.M),
        T=float(T),
        ring_actors=list(range(int(opts.subgroup_size))),
        hub_actor=0,
        d=int(opts.d),
        nu_base=float(opts.nu_base),
        alpha0_team=float(opts.alpha0_team),
        alpha1_max=float(opts.active_mass),
        alpha1_min=float(opts.active_mass) / max(int(opts.subgroup_size) - 1, 1),
        beta0=float(opts.beta0),
        beta1=float(opts.beta1),
        eta_on=float(opts.eta_on),
        eta_off=float(opts.eta_off),
        seed=int(seed),
    )
    return cfg


def run_fit(data: SyntheticP2Data, opts: P2Options, max_iters: int, verbose: bool):
    cfg = data.config
    edges = np.arange(0.0, float(cfg.T) + float(opts.interval_width) + 1e-9, float(opts.interval_width))
    base_rate = max(float(len(data.events)) / max(float(cfg.K * cfg.M) * float(cfg.T), 1e-9), 1e-4)
    fit_rng = np.random.default_rng(int(cfg.seed) + 1000)
    rho_init = empirical_mark_init(data.events, int(cfg.M))
    beta0_init, beta1_init = rough_beta_init(data.events)
    A0_init = fit_rng.uniform(0.0, 0.02, size=(int(cfg.K), int(cfg.K)))
    np.fill_diagonal(A0_init, 0.0)
    init = MStepResult(
        nu=np.full((int(cfg.K), int(cfg.M)), base_rate, dtype=float),
        A0=A0_init,
        g=np.full(int(cfg.K), 0.1, dtype=float),
        h=np.full(int(cfg.K), 0.1, dtype=float),
        U=fit_rng.normal(scale=0.01, size=(int(cfg.K), int(cfg.d))),
        V=fit_rng.normal(scale=0.01, size=(int(cfg.K), int(cfg.d))),
        rho0=rho_init.copy(),
        rho1=rho_init.copy(),
        beta0=beta0_init,
        beta1=beta1_init,
        eta_on=0.03,
        eta_off=0.30,
    )
    kwargs = dict(
        events=data.events,
        interval_edges=edges,
        init_params=init,
        max_iters=int(max_iters),
        lr=float(opts.lr),
        lambda_g=float(opts.lambda_g),
        lambda_h=float(opts.lambda_h),
        lambda_0=float(opts.lambda_0),
        n_inner_steps=int(opts.n_inner_steps),
        verbose=bool(verbose),
        stability_threshold=float(opts.stability_threshold),
        stability_target=float(opts.stability_target),
    )
    sig = inspect.signature(run_em)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    result = run_em(**kwargs)
    return result, edges


def summarize_row(data: SyntheticP2Data, result, edges: np.ndarray, *, seed: int, T_level: float, alpha_level_requested: float) -> dict[str, Any]:
    cfg = data.config
    active_time = active_time_from_path(data.Z_path)
    subgroup = [int(x) for x in cfg.ring_actors]
    true_receivers = [int(x) for x in data.true_receivers]
    weak_receiver = int(data.weak_receiver)
    z = np.asarray(data.Z_at_event, dtype=int)
    actors = data.events[:, 1].astype(int) if len(data.events) else np.asarray([], dtype=int)
    active_mask = z == 1
    subgroup_mask = np.isin(actors, subgroup) if len(actors) else np.asarray([], dtype=bool)
    receiver_mask = np.isin(actors, true_receivers) if len(actors) else np.asarray([], dtype=bool)
    weak_mask = actors == weak_receiver if len(actors) else np.asarray([], dtype=bool)
    n_eff = int(np.sum(active_mask & subgroup_mask)) if len(actors) else 0
    weak_count = int(np.sum(active_mask & weak_mask)) if len(actors) else 0
    receiver_total = int(np.sum(active_mask & receiver_mask)) if len(actors) else 0

    ev = evaluate(
        gamma=result.gamma,
        interval_edges=edges,
        Z_path=data.Z_path,
        g=result.params.g,
        h=result.params.h,
        true_ring=cfg.ring_actors,
        true_hub=cfg.hub_actor,
        method="topk",
        events=data.events,
        Z_at_event=data.Z_at_event,
        auc_target="active_event",
    )
    A1_hat = compute_A1(result.params.g, result.params.h, result.params.U, result.params.V)
    sender_scores = A1_hat.sum(axis=1)
    hub_pred = int(np.argmax(sender_scores))
    receiver_f1, pred_receivers = receiver_f1_from_h(result.params.h, true_receivers, int(cfg.hub_actor))
    pred_receiver_order = [int(x) for x in np.argsort(np.asarray(result.params.h, dtype=float))[::-1] if int(x) != int(cfg.hub_actor)]
    weak_rank = int(pred_receiver_order.index(weak_receiver) + 1) if weak_receiver in pred_receiver_order else -1
    weak_score = float(np.asarray(result.params.h, dtype=float)[weak_receiver])
    exact = exact_support_metrics(ev.predicted_ring, cfg.ring_actors, pred_receivers, true_receivers, hub_pred, int(cfg.hub_actor))

    # P2 threshold metric hygiene.
    # A raw inclusion indicator, weak_receiver in top-|S_h*| receivers, can be 1
    # even when the whole fit is garbage.  But requiring exact recovery of every
    # receiver is too strict for the P2 weak-link question: the theorem-facing
    # event is whether the weak receiver is recovered in the active receiver gate
    # once the active member set and hub/source role are correctly identified.
    # We therefore expose weak_receiver_recovered as:
    #     weak receiver included AND member support exact AND hub/source exact.
    # receiver_exact is still emitted separately as a stricter diagnostic.
    weak_receiver_included = int(weak_receiver in set(pred_receivers))
    weak_link_role_recovered = int(
        weak_receiver_included
        and int(exact.get("member_exact", 0)) == 1
        and int(exact.get("hub_exact", 0)) == 1
    )

    truth_interval = active_event_truth(data.events, data.Z_at_event, edges)
    active_auc = float(ev.field_auc)
    if not np.isfinite(active_auc) and getattr(result, "gamma", None) is not None:
        active_auc = roc_auc_binary(truth_interval, np.asarray(result.gamma)[:, 1])

    return {
        "method": "Proposed",
        "method_key": "proposed",
        "status": "ok",
        "error": "",
        "dataset": "synthetic_p2_weaklink",
        "seed": int(seed),
        "T": float(cfg.T),
        "T_level": float(T_level),
        "K": int(cfg.K),
        "M": int(cfg.M),
        "subgroup_size": int(len(subgroup)),
        "true_subgroup_json": str(subgroup),
        "true_source_actor": int(cfg.hub_actor),
        "true_receivers_json": str(true_receivers),
        "weak_receiver": int(weak_receiver),
        "weak_receiver_frac": float(data.alpha_min_relative),
        "alpha_min_relative": float(data.alpha_min_relative),
        "alpha_min_relative_requested": float(alpha_level_requested),
        "alpha_min_true": float(data.alpha_min_true),
        "alpha_uniform_true": float(data.alpha_uniform),
        "alpha_min_proxy": int(weak_count),
        "weak_receiver_share_realized": float(weak_count / max(receiver_total, 1)),
        "receiver_directed_total": int(receiver_total),
        "pi1T_planted": float(active_time),
        "pi1_planted": float(active_time / max(float(cfg.T), 1e-9)),
        "active_time_realized": float(active_time),
        "n_eff_planted": int(n_eff),
        "signal_total": int(n_eff),
        "total_events": int(len(data.events)),
        "active_events_total": int(np.sum(active_mask)) if len(actors) else 0,
        "member_f1": float(ev.membership_f1),
        "precision": float(ev.membership_precision),
        "recall": float(ev.membership_recall),
        "hub_correct": int(bool(ev.hub_correct)),
        "hub_pred": int(hub_pred),
        "receiver_f1": float(receiver_f1),
        "weak_receiver_included": int(weak_receiver_included),
        "weak_link_role_recovered": int(weak_link_role_recovered),
        # The existing evaluator consumes this column.  Here it is stricter than
        # raw inclusion, but it does not require exact recovery of every strong
        # receiver, which would confound the weak-link threshold with unrelated
        # receiver-support noise.
        "weak_receiver_recovered": int(weak_link_role_recovered),
        "weak_receiver_rank": int(weak_rank),
        "weak_receiver_score": float(weak_score),
        "active_auc": float(active_auc),
        **exact,
        "pred_ring": str([int(x) for x in ev.predicted_ring]),
        "pred_receivers": str([int(x) for x in pred_receivers]),
        "loglik_final": float(result.log_likelihood_trace[-1]) if getattr(result, "log_likelihood_trace", None) else float("nan"),
        "beta0_hat": float(result.params.beta0),
        "beta1_hat": float(result.params.beta1),
        "active_occ": float(np.mean(np.asarray(result.gamma)[:, 1])) if getattr(result, "gamma", None) is not None else float("nan"),
    }


def run_one_task(task: dict[str, Any]) -> dict[str, Any]:
    seed = int(task["seed"])
    T = float(task["T"])
    alpha_rel = float(task["alpha_rel"])
    opts: P2Options = task["opts"]
    cfg = make_cfg(seed, T, opts)
    data = simulate_p2_data(cfg, opts, alpha_rel)
    result, edges = run_fit(data, opts, max_iters=int(task["max_iters"]), verbose=bool(task["verbose_em"]))
    return summarize_row(data, result, edges, seed=seed, T_level=T, alpha_level_requested=alpha_rel)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    keys = list(rows[0].keys())
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                keys = next(reader)
            except StopIteration:
                pass
    else:
        for r in rows[1:]:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic P2 true-alpha weakest-link threshold sweep.")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--verbose-em", action="store_true")
    ap.add_argument("--outdir", default="results/synthetic_p2_weaklink")
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--subgroup-size", type=int, default=6)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--T-levels", nargs="+", type=float, default=[150.0, 225.0, 300.0, 450.0, 600.0, 900.0])
    ap.add_argument("--alpha-min-relatives", nargs="+", type=float, default=[1.0, 0.70710678, 0.5, 0.35355339, 0.25])
    ap.add_argument("--nu-base", type=float, default=0.10)
    ap.add_argument("--hub-rate", type=float, default=0.5,
                    help="Background rate of the hub (driver) actor. Must be well above "
                         "--nu-base so hub events land inside active windows and the active "
                         "star ignites; otherwise the active block is invisible and N_eff "
                         "collapses to background-in-window. Working calibration: --nu-base 0.03 "
                         "--hub-rate 0.5 --active-mass 2.0 --eta-on 0.06 --beta1 1.5 gives N_eff "
                         "~140-370 over T 240-720 with the alpha_min knob biting cleanly.")
    ap.add_argument("--alpha0-team", type=float, default=0.005)
    ap.add_argument("--active-mass", type=float, default=0.80, help="fixed total hub->receiver active mass")
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--beta1", type=float, default=3.0)
    ap.add_argument("--eta-on", type=float, default=0.04)
    ap.add_argument("--eta-off", type=float, default=0.30)
    ap.add_argument("--n-active-episodes", type=int, default=4)
    ap.add_argument("--path-mode", choices=["deterministic", "ctmc"], default="deterministic")
    ap.add_argument("--interval-width", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lambda-g", type=float, default=0.02)
    ap.add_argument("--lambda-h", type=float, default=0.02)
    ap.add_argument("--lambda-0", type=float, default=0.05)
    ap.add_argument("--n-inner-steps", type=int, default=10)
    ap.add_argument("--stability-threshold", type=float, default=0.95)
    ap.add_argument("--stability-target", type=float, default=0.80)
    args = ap.parse_args()

    opts = P2Options(
        K=int(args.K), subgroup_size=int(args.subgroup_size), d=int(args.d), M=int(args.M),
        nu_base=float(args.nu_base), alpha0_team=float(args.alpha0_team), active_mass=float(args.active_mass),
        hub_rate=float(args.hub_rate),
        beta0=float(args.beta0), beta1=float(args.beta1), eta_on=float(args.eta_on), eta_off=float(args.eta_off),
        n_active_episodes=int(args.n_active_episodes), interval_width=float(args.interval_width),
        lr=float(args.lr), lambda_g=float(args.lambda_g), lambda_h=float(args.lambda_h), lambda_0=float(args.lambda_0),
        n_inner_steps=int(args.n_inner_steps), stability_threshold=float(args.stability_threshold),
        stability_target=float(args.stability_target), path_mode=str(args.path_mode),
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "synthetic_p2_raw_checkpoint.csv"

    done = set()
    if args.resume and raw_path.exists():
        try:
            import pandas as pd
            old = pd.read_csv(raw_path)
            ok = old[old.get("status", "ok").astype(str).eq("ok")]
            for _, r in ok.iterrows():
                done.add((int(r["seed"]), float(r["T_level"]), float(r["alpha_min_relative_requested"])))
        except Exception:
            done = set()

    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    tasks: list[dict[str, Any]] = []
    for seed in seeds:
        for T in args.T_levels:
            for alpha_rel in args.alpha_min_relatives:
                key = (int(seed), float(T), float(alpha_rel))
                if key in done:
                    continue
                tasks.append({
                    "seed": int(seed),
                    "T": float(T),
                    "alpha_rel": float(alpha_rel),
                    "opts": opts,
                    "max_iters": int(args.max_iters),
                    "verbose_em": bool(args.verbose_em),
                })

    pi1 = float(opts.eta_on) / max(float(opts.eta_on) + float(opts.eta_off), 1e-12)
    print(f"Synthetic P2 weak-link: {len(tasks)} jobs, workers={args.workers}, outdir={outdir}", flush=True)
    print(f"Fixed active mass={opts.active_mass}; uniform alpha={opts.active_mass / max(opts.subgroup_size - 1, 1):.6g}; pi1={pi1:.4f}", flush=True)
    print(f"Raw checkpoint: {raw_path}", flush=True)
    if not tasks:
        print("Nothing to run.")
        return

    if int(args.workers) <= 1:
        for idx, task in enumerate(tasks, 1):
            try:
                row = run_one_task(task)
            except Exception as exc:
                row = {
                    "method": "Proposed", "method_key": "proposed", "status": "error", "error": repr(exc),
                    "seed": int(task["seed"]), "T_level": float(task["T"]),
                    "weak_receiver_frac": float(task["alpha_rel"]),
                    "alpha_min_relative": float(task["alpha_rel"]),
                    "alpha_min_relative_requested": float(task["alpha_rel"]),
                    "pi1T_planted": float("nan"), "n_eff_planted": float("nan"),
                }
            write_rows(raw_path, [row])
            print(f"[{idx}/{len(tasks)}] {row.get('status')} seed={row.get('seed')} T={row.get('T_level')} alpha_rel={row.get('alpha_min_relative')}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = {ex.submit(run_one_task, task): task for task in tasks}
            for idx, fut in enumerate(as_completed(futs), 1):
                task = futs[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {
                        "method": "Proposed", "method_key": "proposed", "status": "error", "error": repr(exc),
                        "seed": int(task["seed"]), "T_level": float(task["T"]),
                        "weak_receiver_frac": float(task["alpha_rel"]),
                        "alpha_min_relative": float(task["alpha_rel"]),
                        "alpha_min_relative_requested": float(task["alpha_rel"]),
                        "pi1T_planted": float("nan"), "n_eff_planted": float("nan"),
                    }
                write_rows(raw_path, [row])
                print(f"[{idx}/{len(tasks)}] {row.get('status')} seed={row.get('seed')} T={row.get('T_level')} alpha_rel={row.get('alpha_min_relative')}", flush=True)

    print("Done.")
    print(f"Raw checkpoint: {raw_path}")
    print("Evaluate with your existing evaluator, e.g.:")
    print(
        "python -m regime_hawkes.evaluate_ucdp_ged_spikein `\n"
        f"  --raw {raw_path} `\n"
        f"  --outdir {outdir} `\n"
        "  --n-boot 2000"
    )


if __name__ == "__main__":
    main()
