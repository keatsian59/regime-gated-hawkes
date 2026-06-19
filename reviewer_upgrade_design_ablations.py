from __future__ import annotations

"""Replicated design-ablation runner for the regime-switching Hawkes paper.

This script fills the *design ablations* table, distinct from the baseline
comparison table and the phase-diagram sweep.

Variants:
    - full_model: event-time gating, split gates, low-rank active block,
      distinct beta0 and beta1.
    - present_time_gating: current-state gating ablation.
    - single_gate: enforce g == h after each M-step.
    - dense_active_block: remove the low-rank bottleneck by setting d = K,
      fixing U = I_K, and letting V carry one free parameter per ordered pair.
    - shared_beta: enforce beta0 == beta1 after each M-step.

Outputs:
    - design_ablation_raw.csv
    - design_ablation_summary.csv
    - design_ablation_table.tex
    - design_ablation_barplot.png
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
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
from regime_hawkes.evaluate import evaluate
from regime_hawkes.estep import EStepResult, ctmc_transition_matrix, forward_backward_logspace, run_estep
from regime_hawkes.likelihood import event_bins
from regime_hawkes.mstep import MStepResult, run_mstep
from regime_hawkes.simulate import simulate_regime_hawkes
from regime_hawkes.traces import traces_at_interval_starts


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def compute_A1(g: np.ndarray, h: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    inner = U @ V.T
    A1 = g[:, None] * h[None, :] * _softplus_np(inner)
    np.fill_diagonal(A1, 0.0)
    return A1


def spectral_radius(matrix: np.ndarray) -> float:
    vals = np.linalg.eigvals(np.asarray(matrix, dtype=float))
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def _blend(old: np.ndarray, new: np.ndarray, momentum: float) -> np.ndarray:
    return momentum * old + (1.0 - momentum) * new


def _project_rho_rows(rho: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    rho = np.maximum(np.asarray(rho, dtype=float), eps)
    row_sums = np.maximum(np.sum(rho, axis=1, keepdims=True), eps)
    return rho / row_sums


def _linear_schedule(it: int, warm_iters: int, start: float, end: float) -> float:
    if warm_iters <= 1:
        return float(end)
    a = min(max((it - 1) / float(warm_iters - 1), 0.0), 1.0)
    return float((1.0 - a) * start + a * end)


def _exploration_floors(it: int, warm_iters: int = 12) -> tuple[float, float]:
    pi1_floor = _linear_schedule(it, warm_iters, start=0.05, end=0.005)
    gamma_floor = _linear_schedule(it, warm_iters, start=0.05, end=0.001)
    return pi1_floor, gamma_floor


def _enforce_min_stationary_active(eta_on: float, eta_off: float, pi1_floor: float) -> tuple[float, float]:
    eta_on = float(eta_on)
    eta_off = float(eta_off)
    pi1_floor = float(np.clip(pi1_floor, 1e-6, 1.0 - 1e-6))
    min_eta_on = pi1_floor / (1.0 - pi1_floor) * max(eta_off, 1e-12)
    return max(eta_on, min_eta_on), eta_off


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
    if len(events) <= 1:
        return 1.0, 2.0
    t = np.sort(events[:, 0])
    dt = np.diff(t)
    dt = dt[dt > 1e-8]
    if len(dt) == 0:
        return 1.0, 2.0
    beta0 = float(np.clip(1.0 / np.median(dt), 0.1, 1.0))
    # Match the current synthetic runner's active-timescale warm start.
    beta1 = float(np.clip(1.0 / np.percentile(dt, 10), beta0 + 0.1, 8.0))
    return beta0, beta1


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

def create_initial_params(cfg: Any, events: np.ndarray, *, d_override: int | None = None, dense_active: bool = False) -> MStepResult:
    d = int(d_override) if d_override is not None else int(cfg.d)
    base_rate = max(len(events) / (cfg.K * cfg.M * cfg.T), 1e-3)
    fit_rng = np.random.default_rng(int(cfg.seed) + 1000 + 17 * d)
    rho_init = empirical_mark_init(events, cfg.M)
    beta0_init, beta1_init = rough_beta_init(events)
    A0_init = fit_rng.uniform(0.0, 0.02, size=(cfg.K, cfg.K))
    np.fill_diagonal(A0_init, 0.0)

    if dense_active:
        U = np.eye(cfg.K, dtype=float)
        V = fit_rng.normal(scale=0.01, size=(cfg.K, cfg.K))
    else:
        U = fit_rng.normal(scale=0.01, size=(cfg.K, d))
        V = fit_rng.normal(scale=0.01, size=(cfg.K, d))

    return MStepResult(
        nu=np.full((cfg.K, cfg.M), base_rate),
        A0=A0_init,
        g=np.full(cfg.K, 0.1),
        h=np.full(cfg.K, 0.1),
        U=U,
        V=V,
        rho0=rho_init.copy(),
        rho1=rho_init.copy(),
        beta0=beta0_init,
        beta1=beta1_init,
        eta_on=0.03,
        eta_off=0.3,
    )


@dataclass
class SimpleEMResult:
    gamma: np.ndarray
    params: MStepResult
    log_likelihood_trace: list[float]


# ---------------------------------------------------------------------
# Approximate goodness-of-fit via time-rescaling
# ---------------------------------------------------------------------

def _event_bins(events: np.ndarray, interval_edges: np.ndarray) -> np.ndarray:
    if len(events) == 0:
        return np.zeros(0, dtype=int)
    b = len(interval_edges) - 1
    idx = np.searchsorted(events[:, 0], interval_edges, side="right")
    raise RuntimeError("unused helper")


def compensator_residuals_event_time(events: np.ndarray, interval_edges: np.ndarray, result: SimpleEMResult) -> np.ndarray:
    if len(events) == 0:
        return np.zeros(0, dtype=float)

    params = result.params
    K, M = params.nu.shape
    bins = event_bins(events, interval_edges)
    event_active_w = result.gamma[bins, 1]

    B = np.zeros((K, M), dtype=float)
    A = np.zeros((K, M), dtype=float)
    t_prev = 0.0
    residuals: list[float] = []
    A1 = compute_A1(params.g, params.h, params.U, params.V)

    for n, (t, actor, mark) in enumerate(events):
        dt = float(t) - t_prev
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


def compensator_residuals_present_time(events: np.ndarray, interval_edges: np.ndarray, result: SimpleEMResult) -> np.ndarray:
    if len(events) == 0:
        return np.zeros(0, dtype=float)

    params = result.params
    K, M = params.nu.shape
    B = np.zeros((K, M), dtype=float)
    C = np.zeros((K, M), dtype=float)
    t_prev = 0.0
    residuals: list[float] = []
    A1 = compute_A1(params.g, params.h, params.U, params.V)

    times = np.asarray(events[:, 0], dtype=float)
    bins = np.searchsorted(interval_edges, times, side="right") - 1
    bins = np.clip(bins, 0, len(interval_edges) - 2)
    interval_gamma = result.gamma[bins, 1]

    for n, (t, actor, mark) in enumerate(events):
        dt = float(t) - t_prev
        routine_pre = params.A0.T @ (B @ params.rho0)
        active_pre = A1.T @ (C @ params.rho1)
        decay0 = (1.0 - np.exp(-params.beta0 * dt)) / max(params.beta0, 1e-12)
        decay1 = (1.0 - np.exp(-params.beta1 * dt)) / max(params.beta1, 1e-12)
        total_comp = (
            float(np.sum(params.nu)) * dt
            + float(np.sum(routine_pre)) * decay0
            + float(interval_gamma[n]) * float(np.sum(active_pre)) * decay1
        )
        residuals.append(total_comp)

        B *= np.exp(-params.beta0 * dt)
        C *= np.exp(-params.beta1 * dt)
        i = int(actor)
        m = int(mark)
        B[i, m] += params.beta0
        C[i, m] += params.beta1
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


# ---------------------------------------------------------------------
# Present-time gating E-step
# ---------------------------------------------------------------------

def build_interval_emissions_present_time(
    events: np.ndarray,
    interval_edges: np.ndarray,
    nu: np.ndarray,
    A0: np.ndarray,
    A1: np.ndarray,
    rho0: np.ndarray,
    rho1: np.ndarray,
    beta0: float,
    beta1: float,
    eps: float = 1e-9,
) -> np.ndarray:
    """Current-state-gated interval emissions.

    Under z_b = 0, active excitation is off for the entire interval.
    Under z_b = 1, all past events contribute through the active kernel,
    regardless of birth state.
    """
    K, M = nu.shape
    Bn = len(interval_edges) - 1
    ell = np.zeros((Bn, 2), dtype=float)
    evt_bins = event_bins(events, interval_edges)
    start_B, start_C = traces_at_interval_starts(
        events, interval_edges, K, M, beta0, beta1, np.ones(len(events), dtype=float)
    )

    for b in range(Bn):
        t_start = float(interval_edges[b])
        t_end = float(interval_edges[b + 1])
        delta = t_end - t_start
        idx = np.where(evt_bins == b)[0]

        B_pre = start_B[b].copy()
        C_pre = start_C[b].copy()
        B_within = np.zeros((K, M), dtype=float)
        C_within = np.zeros((K, M), dtype=float)
        t_prev = t_start
        event_log0 = 0.0
        event_log1 = 0.0

        for n in idx:
            t_n = float(events[n, 0])
            i_n, m_n = int(events[n, 1]), int(events[n, 2])
            dt = t_n - t_prev

            B_pre *= np.exp(-beta0 * dt)
            C_pre *= np.exp(-beta1 * dt)
            B_within *= np.exp(-beta0 * dt)
            C_within *= np.exp(-beta1 * dt)

            routine_n = A0.T @ ((B_pre + B_within) @ rho0)
            active_all = A1.T @ ((C_pre + C_within) @ rho1)

            lam0 = max(nu[i_n, m_n] + routine_n[i_n, m_n], eps)
            lam1 = max(nu[i_n, m_n] + routine_n[i_n, m_n] + active_all[i_n, m_n], eps)

            event_log0 += np.log(lam0)
            event_log1 += np.log(lam1)

            B_within[i_n, m_n] += beta0
            C_within[i_n, m_n] += beta1
            t_prev = t_n

        routine_comp = A0.T @ (start_B[b] @ rho0)
        active_comp = A1.T @ (start_C[b] @ rho1)
        decay0 = (1.0 - np.exp(-beta0 * delta)) / max(beta0, 1e-12)
        decay1 = (1.0 - np.exp(-beta1 * delta)) / max(beta1, 1e-12)
        base_comp = np.sum(nu) * delta

        comp0 = base_comp + np.sum(routine_comp) * decay0
        comp1 = base_comp + np.sum(routine_comp) * decay0 + np.sum(active_comp) * decay1

        for n in idx:
            t_n = float(events[n, 0])
            i_n, m_n = int(events[n, 1]), int(events[n, 2])
            remaining = t_end - t_n

            impulse0 = np.zeros((K, M), dtype=float)
            impulse0[i_n, m_n] = beta0
            routine_inc = A0.T @ (impulse0 @ rho0)
            comp0 += np.sum(routine_inc) * (1.0 - np.exp(-beta0 * remaining)) / max(beta0, 1e-12)
            comp1 += np.sum(routine_inc) * (1.0 - np.exp(-beta0 * remaining)) / max(beta0, 1e-12)

            impulse1 = np.zeros((K, M), dtype=float)
            impulse1[i_n, m_n] = beta1
            active_inc = A1.T @ (impulse1 @ rho1)
            comp1 += np.sum(active_inc) * (1.0 - np.exp(-beta1 * remaining)) / max(beta1, 1e-12)

        ell[b, 0] = event_log0 - comp0
        ell[b, 1] = event_log1 - comp1

    return ell


def run_estep_present_time(
    events: np.ndarray,
    interval_edges: np.ndarray,
    nu: np.ndarray,
    A0: np.ndarray,
    A1: np.ndarray,
    rho0: np.ndarray,
    rho1: np.ndarray,
    beta0: float,
    beta1: float,
    eta_on: float,
    eta_off: float,
) -> EStepResult:
    ell = build_interval_emissions_present_time(
        events=events,
        interval_edges=interval_edges,
        nu=nu,
        A0=A0,
        A1=A1,
        rho0=rho0,
        rho1=rho1,
        beta0=beta0,
        beta1=beta1,
    )
    delta = float(np.mean(np.diff(interval_edges)))
    P = ctmc_transition_matrix(eta_on, eta_off, delta)
    s = eta_on + eta_off
    pi = np.array([eta_off / s, eta_on / s], dtype=float)
    return forward_backward_logspace(ell, P, pi=pi)


# ---------------------------------------------------------------------
# Variant-specific EM wrappers
# ---------------------------------------------------------------------

def _apply_joint_stability(params: MStepResult, threshold: float = 0.95, target: float = 0.80) -> MStepResult:
    A1 = compute_A1(params.g, params.h, params.U, params.V)
    spec = spectral_radius(params.A0 + A1)
    if spec <= threshold:
        return params

    scale = float(np.clip(target / max(spec, 1e-12), 0.0, 1.0))
    params = MStepResult(
        nu=params.nu,
        A0=params.A0 * scale,
        g=params.g * np.sqrt(scale),
        h=params.h * np.sqrt(scale),
        U=params.U,
        V=params.V,
        rho0=params.rho0,
        rho1=params.rho1,
        beta0=params.beta0,
        beta1=params.beta1,
        eta_on=params.eta_on,
        eta_off=params.eta_off,
    )
    return params


def _enforce_variant_constraints(params: MStepResult, variant: str) -> MStepResult:
    if variant == "single_gate":
        avg = 0.5 * (params.g + params.h)
        params = MStepResult(
            nu=params.nu, A0=params.A0, g=avg.copy(), h=avg.copy(), U=params.U, V=params.V,
            rho0=params.rho0, rho1=params.rho1, beta0=params.beta0, beta1=params.beta1,
            eta_on=params.eta_on, eta_off=params.eta_off,
        )
    elif variant == "dense_active_block":
        K = len(params.g)
        params = MStepResult(
            nu=params.nu, A0=params.A0, g=params.g, h=params.h, U=np.eye(K, dtype=float), V=params.V,
            rho0=params.rho0, rho1=params.rho1, beta0=params.beta0, beta1=params.beta1,
            eta_on=params.eta_on, eta_off=params.eta_off,
        )
    elif variant == "shared_beta":
        b = 0.5 * (float(params.beta0) + float(params.beta1))
        params = MStepResult(
            nu=params.nu, A0=params.A0, g=params.g, h=params.h, U=params.U, V=params.V,
            rho0=params.rho0, rho1=params.rho1, beta0=b, beta1=b,
            eta_on=params.eta_on, eta_off=params.eta_off,
        )
    return params


def _mstep_trace_weights(
    variant: str,
    gamma_active: np.ndarray,
    B: int,
) -> np.ndarray:
    """Trace weights fed into the shared M-step backend.

    For the present-time gating ablation, active traces in the M-step should not
    be birth-time weighted. Every past event remains eligible to contribute when
    the current interval is active, so we pass all-ones weights here.
    """
    if variant == "present_time_gating":
        return np.ones(B, dtype=float)
    return np.asarray(gamma_active, dtype=float)



def run_em_variant(
    events: np.ndarray,
    interval_edges: np.ndarray,
    init_params: MStepResult,
    *,
    variant: str,
    max_iters: int = 50,
    tol: float = 1e-4,
    lambda_g: float = 0.02,
    lambda_h: float = 0.02,
    lambda_e: float = 0.01,
    lambda_0: float = 0.05,
    lr: float = 0.01,
    n_inner_steps: int = 10,
    stability_threshold: float = 0.95,
    stability_target: float = 0.80,
) -> SimpleEMResult:
    params = _enforce_variant_constraints(init_params, variant)
    B = len(interval_edges) - 1
    gamma_prev_1 = np.full(B, 0.15, dtype=float)
    gamma = np.column_stack([1.0 - gamma_prev_1, gamma_prev_1])
    ll_trace: list[float] = []
    prev_ll: float | None = None

    for it in range(1, max_iters + 1):
        momentum = 0.5 if it <= 10 else 0.3
        scalar_momentum = 0.2 if it <= 10 else 0.1
        anneal = min(1.0, it / 10.0)
        lambda_g_eff = lambda_g * anneal
        lambda_h_eff = lambda_h * anneal
        lambda_e_eff = 0.0 if variant == "dense_active_block" else lambda_e

        pi1_floor, gamma_floor = _exploration_floors(it)
        gamma_prev_eff = np.clip(gamma_prev_1, gamma_floor, 1.0 - gamma_floor)
        eta_on_eff, eta_off_eff = _enforce_min_stationary_active(params.eta_on, params.eta_off, pi1_floor)
        A1 = compute_A1(params.g, params.h, params.U, params.V)

        if variant == "present_time_gating":
            estep = run_estep_present_time(
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

        gamma_prev_for_mstep = _mstep_trace_weights(variant, estep.gamma[:, 1], B)

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
            lambda_0=lambda_0,
            lr=lr,
            n_inner_steps=n_inner_steps,
            gamma_prev_1=gamma_prev_for_mstep,
            eta_on_prev=params.eta_on,
            eta_off_prev=params.eta_off,
        )

        params = MStepResult(
            nu=_blend(params.nu, proposed.nu, momentum),
            A0=_blend(params.A0, proposed.A0, momentum),
            g=_blend(params.g, proposed.g, momentum),
            h=_blend(params.h, proposed.h, momentum),
            U=_blend(params.U, proposed.U, momentum),
            V=_blend(params.V, proposed.V, momentum),
            rho0=_project_rho_rows(_blend(params.rho0, proposed.rho0, momentum)),
            rho1=_project_rho_rows(_blend(params.rho1, proposed.rho1, momentum)),
            beta0=float(scalar_momentum * params.beta0 + (1.0 - scalar_momentum) * proposed.beta0),
            beta1=float(scalar_momentum * params.beta1 + (1.0 - scalar_momentum) * proposed.beta1),
            eta_on=float(proposed.eta_on),
            eta_off=float(proposed.eta_off),
        )
        params = _enforce_variant_constraints(params, variant)
        params = _apply_joint_stability(params, threshold=stability_threshold, target=stability_target)

        gamma = estep.gamma
        ll = float(estep.log_likelihood)
        ll_trace.append(ll)
        gamma_prev_1 = np.clip(gamma[:, 1], gamma_floor, 1.0 - gamma_floor)

        if prev_ll is not None:
            rel = abs(ll - prev_ll) / (abs(prev_ll) + 1e-8)
            if rel < tol:
                break
        prev_ll = ll

    # Final synchronized E-step
    final_pi1_floor, final_gamma_floor = _exploration_floors(it if 'it' in locals() else 1)
    eta_on_eff, eta_off_eff = _enforce_min_stationary_active(params.eta_on, params.eta_off, final_pi1_floor)
    A1 = compute_A1(params.g, params.h, params.U, params.V)
    if variant == "present_time_gating":
        final_estep = run_estep_present_time(
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

    return SimpleEMResult(gamma=final_estep.gamma, params=params, log_likelihood_trace=ll_trace)


# ---------------------------------------------------------------------
# Evaluation and table rendering
# ---------------------------------------------------------------------

def evaluate_variant(data: Any, result: SimpleEMResult, edges: np.ndarray, *, variant: str) -> dict[str, Any]:
    ev = evaluate(
        gamma=result.gamma,
        interval_edges=edges,
        Z_path=data.Z_path,
        g=result.params.g,
        h=result.params.h,
        true_ring=data.config.ring_actors,
        true_hub=data.config.hub_actor,
        method="topk",
        events=data.events,
        Z_at_event=data.Z_at_event,
        auc_target="active_event",
    )
    if variant == "present_time_gating":
        resid = compensator_residuals_present_time(data.events, edges, result)
    else:
        resid = compensator_residuals_event_time(data.events, edges, result)

    return {
        "precision": float(ev.membership_precision),
        "recall": float(ev.membership_recall),
        "f1": float(ev.membership_f1),
        "hub_acc": float(ev.hub_correct),
        "field_auc": float(ev.field_auc),
        "ks": float(ks_stat_exp1(resid)),
        "pred_ring": json.dumps(ev.predicted_ring),
    }


def render_design_ablation_table(summary_df: pd.DataFrame, out_path: Path) -> None:
    order = [
        ("full_model", "Full model"),
        ("present_time_gating", "Present-time gating"),
        ("single_gate", "Single gate per actor"),
        ("dense_active_block", "Dense active block"),
        ("shared_beta", r"Shared $\beta_0=\beta_1$"),
    ]

    def fmt_ms(m: float, s: float) -> str:
        if np.isnan(m):
            return "---"
        if np.isnan(s):
            return f"{m:.3f}"
        return f"{m:.3f} $\\pm$ {s:.3f}"

    def fmt_mean(m: float) -> str:
        return "---" if np.isnan(m) else f"{m:.3f}"

    lines = []
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Model variant & F1 & Hub acc. & AUC & KS $\downarrow$ \\")
    lines.append(r"\midrule")
    for key, label in order:
        if key not in summary_df.index:
            continue
        row = summary_df.loc[key]
        lines.append(
            f"{label} & {fmt_ms(row['f1_mean'], row['f1_std'])} & {fmt_mean(row['hub_acc_mean'])} & "
            f"{fmt_ms(row['field_auc_mean'], row['field_auc_std'])} & {fmt_ms(row['ks_mean'], row['ks_std'])} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def render_barplot(summary_df: pd.DataFrame, out_path: Path) -> None:
    order = ["full_model", "present_time_gating", "single_gate", "dense_active_block", "shared_beta"]
    labels = ["Full", "Present-time", "Single gate", "Dense", r"Shared $\beta$"]
    df = summary_df.reindex(order)
    xs = np.arange(len(order))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    axes[0].bar(xs, df["f1_mean"].values, yerr=df["f1_std"].values, capsize=3)
    axes[0].set_title("F1")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")

    axes[1].bar(xs, df["field_auc_mean"].values, yerr=df["field_auc_std"].values, capsize=3)
    axes[1].set_title("Interval active-event AUC")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")

    axes[2].bar(xs, df["ks_mean"].values, yerr=df["ks_std"].values, capsize=3)
    axes[2].set_title(r"KS statistic $\downarrow$")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")

    # constrained_layout handles spacing robustly on Windows/matplotlib.
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main replicated runner
# ---------------------------------------------------------------------

def run_design_ablations(base_cfg: SimConfig, *, seeds: list[int], outdir: Path, max_iters: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    edges = np.arange(0.0, base_cfg.T + 1e-9, 2.0)
    rows: list[dict[str, Any]] = []

    variants = [
        "full_model",
        "present_time_gating",
        "single_gate",
        "dense_active_block",
        "shared_beta",
    ]

    for rep_idx, seed in enumerate(seeds):
        cfg = SimConfig(**{**asdict(base_cfg), "seed": int(seed)})
        data = simulate_regime_hawkes(cfg)
        print(f"[rep {rep_idx + 1}/{len(seeds)}] seed={seed} events={len(data.events)}")

        for variant in variants:
            if variant == "dense_active_block":
                init = create_initial_params(cfg, data.events, d_override=cfg.K, dense_active=True)
            else:
                init = create_initial_params(cfg, data.events)

            result = run_em_variant(
                events=data.events,
                interval_edges=edges,
                init_params=init,
                variant=variant,
                max_iters=max_iters,
                tol=1e-4,
                lambda_g=0.02,
                lambda_h=0.02,
                lambda_e=0.01,
                lambda_0=0.05,
                lr=0.01,
                n_inner_steps=10,
                stability_threshold=0.95,
                stability_target=0.80,
            )
            metrics = evaluate_variant(data, result, edges, variant=variant)
            rows.append({
                "rep": rep_idx,
                "seed": int(seed),
                "variant": variant,
                "final_ll": float(result.log_likelihood_trace[-1]) if result.log_likelihood_trace else float("nan"),
                **metrics,
            })
            print(
                f"    {variant}: F1={metrics['f1']:.3f}, hub={metrics['hub_acc']:.3f}, "
                f"AUC={metrics['field_auc']:.3f}, KS={metrics['ks']:.3f}"
            )

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / "design_ablation_raw.csv", index=False)

    summary = (
        raw.groupby("variant")
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            hub_acc_mean=("hub_acc", "mean"),
            hub_acc_std=("hub_acc", "std"),
            field_auc_mean=("field_auc", "mean"),
            field_auc_std=("field_auc", "std"),
            ks_mean=("ks", "mean"),
            ks_std=("ks", "std"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "design_ablation_summary.csv", index=False)
    summary_idx = summary.set_index("variant")

    render_design_ablation_table(summary_idx, outdir / "design_ablation_table.tex")
    render_barplot(summary_idx, outdir / "design_ablation_barplot.png")
    return raw, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("reviewer_upgrade_outputs"))
    p.add_argument("--n-reps", type=int, default=20)
    p.add_argument("--max-iters", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.n_reps)))
    base_cfg = default_config(seed=int(args.seed_start))
    raw, summary = run_design_ablations(base_cfg, seeds=seeds, outdir=args.outdir, max_iters=args.max_iters)
    print("\nSaved:")
    print(f"  {args.outdir / 'design_ablation_raw.csv'}")
    print(f"  {args.outdir / 'design_ablation_summary.csv'}")
    print(f"  {args.outdir / 'design_ablation_table.tex'}")
    print(f"  {args.outdir / 'design_ablation_barplot.png'}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
