"""M-step for regime-switching Hawkes EM.

This version updates all parameter blocks promised by the paper:
- baselines nu
- routine coupling A0
- active gates/embeddings (g, h, U, V)
- mark-transition matrices rho0, rho1
- decay rates beta0, beta1
- CTMC rates eta_on, eta_off

The gate/embedding block still uses the precomputed-feature surrogate for speed.
The routine/mark/decay blocks are updated by projected generalized-EM ascent on
that same surrogate using fresh emission recomputation.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Callable

import numpy as np

from regime_hawkes.likelihood import build_interval_emissions, event_bins
from regime_hawkes.traces import compute_event_traces, traces_at_interval_starts

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except ModuleNotFoundError:
    _HAS_JAX = False


logger = logging.getLogger(__name__)


def _mean_or_nan(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.mean(arr)) if arr.size else float('nan')


def _nnz(x: np.ndarray, tol: float = 1e-12) -> int:
    return int(np.sum(np.abs(np.asarray(x, dtype=float)) > tol))


@dataclass
class MStepResult:
    nu: np.ndarray
    A0: np.ndarray
    g: np.ndarray
    h: np.ndarray
    U: np.ndarray
    V: np.ndarray
    rho0: np.ndarray
    rho1: np.ndarray
    beta0: float
    beta1: float
    eta_on: float
    eta_off: float


# ── basic helpers ───────────────────────────────────────────────────
def _softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _compute_A1_np(g: np.ndarray, h: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    inner = U @ V.T
    A1 = g[:, None] * h[None, :] * _softplus_np(inner)
    np.fill_diagonal(A1, 0.0)
    return A1


def _project_nonnegative(x: np.ndarray, max_value: float | None = None) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    if max_value is not None:
        x = np.minimum(x, max_value)
    return x


def _project_rho(rho: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    rho = np.maximum(np.asarray(rho, dtype=float), eps)
    row_sums = np.sum(rho, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, eps)
    return rho / row_sums


def _stabilize_eta(eta_on: float, eta_off: float) -> tuple[float, float]:
    raw_eta_on, raw_eta_off = float(eta_on), float(eta_off)
    eta_on = float(np.clip(eta_on, 1e-4, 5.0))
    eta_off = float(np.clip(eta_off, 1e-4, 5.0))
    logger.debug(
        "Stabilized eta: raw=(%.6f, %.6f) clipped=(%.6f, %.6f) active_prior=%.4f",
        raw_eta_on,
        raw_eta_off,
        eta_on,
        eta_off,
        eta_on / max(eta_on + eta_off, 1e-12),
    )
    return eta_on, eta_off


def _ctmc_transition(eta_on: float, eta_off: float, delta: float) -> np.ndarray:
    s = eta_on + eta_off
    e = np.exp(-s * delta)
    p00 = (eta_off + eta_on * e) / s
    p11 = (eta_on + eta_off * e) / s
    return np.array([[p00, 1 - p00], [1 - p11, p11]], dtype=float)


# ── feature precomputation for gate/embedding block ────────────────
def precompute_features(
    events, interval_edges, gamma, gamma_prev_1,
    nu, A0, rho0, rho1, beta0, beta1,
) -> dict[str, Any]:
    """Extract features that are fixed when updating (g, h, U, V)."""
    K, M = nu.shape
    N = len(events)
    Bn = len(interval_edges) - 1
    logger.debug(
        "Precompute features start: events=%d intervals=%d K=%d M=%d avg_active=%.4f",
        N,
        Bn,
        K,
        M,
        _mean_or_nan(gamma[:, 1]) if np.asarray(gamma).ndim == 2 else _mean_or_nan(gamma),
    )

    bins = event_bins(events, interval_edges)
    evt_weights = gamma_prev_1[bins] if N > 0 else np.zeros(0)
    start_B, start_A = traces_at_interval_starts(
        events, interval_edges, K, M, beta0, beta1, evt_weights,
    )

    routine_base = np.zeros(N, dtype=float)
    active_feat_z0 = np.zeros((N, K), dtype=float)
    active_feat_z1 = np.zeros((N, K), dtype=float)
    actor_idx = np.zeros(N, dtype=int)
    comp_routine = np.zeros(Bn, dtype=float)
    comp_feat_pre = np.zeros((Bn, K), dtype=float)
    comp_feat_within = np.zeros(N, dtype=float)

    for b in range(Bn):
        t_start = float(interval_edges[b])
        t_end = float(interval_edges[b + 1])
        delta = t_end - t_start

        B_pre = start_B[b].copy()
        A_pre = start_A[b].copy()
        B_within = np.zeros((K, M), dtype=float)
        A_within_z1 = np.zeros((K, M), dtype=float)

        idx = np.where(bins == b)[0]
        t_prev = t_start

        for n in idx:
            t_n = float(events[n, 0])
            i_n, m_n = int(events[n, 1]), int(events[n, 2])
            dt = t_n - t_prev

            B_pre *= np.exp(-beta0 * dt) # rate of decay
            A_pre *= np.exp(-beta1 * dt)
            B_within *= np.exp(-beta0 * dt)
            A_within_z1 *= np.exp(-beta1 * dt)

            routine_n = A0.T @ ((B_pre + B_within) @ rho0)
            routine_base[n] = nu[i_n, m_n] + routine_n[i_n, m_n]
            active_feat_z0[n] = (A_pre @ rho1)[:, m_n]
            active_feat_z1[n] = ((A_pre + A_within_z1) @ rho1)[:, m_n]
            actor_idx[n] = i_n

            remaining = t_end - t_n
            comp_feat_within[n] = (
                float(np.sum(rho1[m_n, :]))
                * (1.0 - np.exp(-beta1 * remaining))
                / beta1
            )

            B_within[i_n, m_n] += beta0
            A_within_z1[i_n, m_n] += beta1
            t_prev = t_n

        decay0 = (1.0 - np.exp(-beta0 * delta)) / beta0
        decay1 = (1.0 - np.exp(-beta1 * delta)) / beta1
        comp_routine[b] = np.sum(nu) * delta + np.sum(A0.T @ (start_B[b] @ rho0)) * decay0

        for n in idx:
            t_n = float(events[n, 0])
            i_n, m_n = int(events[n, 1]), int(events[n, 2])
            remaining = t_end - t_n
            impulse = np.zeros((K, M), dtype=float)
            impulse[i_n, m_n] = beta0
            comp_routine[b] += (
                np.sum(A0.T @ (impulse @ rho0))
                * (1.0 - np.exp(-beta0 * remaining))
                / beta0
            )

        comp_feat_pre[b] = np.sum(start_A[b] @ rho1, axis=1) * decay1

    features = dict(
        routine_base=routine_base,
        active_feat_z0=active_feat_z0,
        active_feat_z1=active_feat_z1,
        actor_idx=actor_idx,
        bins=bins,
        comp_routine=comp_routine,
        comp_feat_pre=comp_feat_pre,
        comp_feat_within=comp_feat_within,
    )
    logger.debug(
        "Precompute features done: routine_base_mean=%.4f active_feat_shapes=(%s, %s) comp_routine_mean=%.4f",
        _mean_or_nan(routine_base),
        active_feat_z0.shape,
        active_feat_z1.shape,
        _mean_or_nan(comp_routine),
    )
    return features


# ── JAX gate/embedding surrogate ───────────────────────────────────
if _HAS_JAX:

    @jax.jit
    def _jax_Q(
        g, h, U, V,
        routine_base, feat_z0, feat_z1, actor_idx, bins,
        comp_routine, comp_feat_pre, comp_feat_within,
        gamma, xi,
        eta_on, eta_off, delta_ref,
    ):
        K = g.shape[0]
        eps = 1e-9

        routine_base = jnp.nan_to_num(routine_base, nan=eps, posinf=1e12, neginf=eps)
        feat_z0 = jnp.nan_to_num(feat_z0, nan=0.0, posinf=1e12, neginf=-1e12)
        feat_z1 = jnp.nan_to_num(feat_z1, nan=0.0, posinf=1e12, neginf=-1e12)
        comp_routine = jnp.nan_to_num(comp_routine, nan=0.0, posinf=1e12, neginf=0.0)
        comp_feat_pre = jnp.nan_to_num(comp_feat_pre, nan=0.0, posinf=1e12, neginf=-1e12)
        comp_feat_within = jnp.nan_to_num(comp_feat_within, nan=0.0, posinf=1e12, neginf=0.0)

        inner = jnp.clip(jnp.einsum("jd,id->ji", U, V), -40.0, 40.0)
        A1 = g[:, None] * h[None, :] * jax.nn.softplus(inner)
        A1 = jnp.nan_to_num(A1, nan=0.0, posinf=1e12, neginf=0.0)
        A1 = A1 * (1.0 - jnp.eye(K))

        A1_at_recv = A1[:, actor_idx]
        active_z0 = jnp.sum(A1_at_recv.T * feat_z0, axis=1)
        active_z1 = jnp.sum(A1_at_recv.T * feat_z1, axis=1)

        lam0 = jnp.maximum(routine_base + active_z0, eps)
        lam1 = jnp.maximum(routine_base + active_z1, eps)

        g0 = gamma[bins, 0]
        g1 = gamma[bins, 1]
        q_event = jnp.sum(g0 * jnp.log(lam0) + g1 * jnp.log(lam1))

        out_strength = A1.sum(axis=1)
        comp_active_pre = comp_feat_pre @ out_strength
        comp_within_z1 = jnp.zeros(comp_routine.shape[0]).at[bins].add(
            comp_feat_within * out_strength[actor_idx]
        )

        comp_z0 = comp_routine + comp_active_pre
        comp_z1 = comp_routine + comp_active_pre + comp_within_z1
        q_comp = jnp.sum(gamma[:, 0] * comp_z0 + gamma[:, 1] * comp_z1)

        s = eta_on + eta_off
        e = jnp.exp(-s * delta_ref)
        p00 = (eta_off + eta_on * e) / s
        p11 = (eta_on + eta_off * e) / s
        P = jnp.array([[p00, 1 - p00], [1 - p11, p11]])
        logP = jnp.log(jnp.clip(P, eps, 1.0))
        q_ctmc = jnp.sum(xi * logP[None, :, :])
        return q_event - q_comp + q_ctmc

    _grad_Q = jax.jit(jax.grad(_jax_Q, argnums=(0, 1, 2, 3)))


# ── baseline closed form ───────────────────────────────────────────
def _baseline_closed_form(
    events, interval_edges, gamma, gamma_prev_1,
    nu, A0, A1, rho0, rho1, beta0, beta1, T, eps=1e-9,
):
    K, M = nu.shape
    logger.debug(
        "Baseline refresh start: events=%d K=%d M=%d T=%.4f",
        len(events),
        K,
        M,
        T,
    )
    bins = event_bins(events, interval_edges)
    weights = gamma_prev_1[bins] if len(events) else np.zeros(0)
    pre_B, pre_A = compute_event_traces(events, K, M, beta0, beta1, weights)

    numer = np.zeros_like(nu)
    for n, (_, actor, mark) in enumerate(events):
        b = bins[n]
        i, m = int(actor), int(mark)
        routine = A0.T @ (pre_B[n] @ rho0)
        active = A1.T @ (pre_A[n] @ rho1)
        lam0 = max(nu[i, m] + routine[i, m], eps)
        lam1 = max(nu[i, m] + routine[i, m] + active[i, m], eps)
        numer[i, m] += gamma[b, 0] * nu[i, m] / lam0 + gamma[b, 1] * nu[i, m] / lam1
    nu_out = np.maximum(numer / max(T, eps), eps)
    logger.debug(
        "Baseline refresh done: nu_mean=%.6f nu_min=%.6f nu_max=%.6f",
        float(np.mean(nu_out)),
        float(np.min(nu_out)),
        float(np.max(nu_out)),
    )
    return nu_out


# ── surrogate objective for full Hawkes blocks ─────────────────────
def _surrogate_q_full(
    *,
    events, interval_edges, gamma_prev_1, gamma, xi,
    nu, A0, A1, rho0, rho1, beta0, beta1,
    eta_on, eta_off, delta_ref, lambda_0,
) -> float:
    eps = 1e-9
    try:
        ell = build_interval_emissions(
            events=events,
            interval_edges=interval_edges,
            gamma_prev_1=gamma_prev_1,
            nu=nu,
            A0=A0,
            A1=A1,
            rho0=rho0,
            rho1=rho1,
            beta0=beta0,
            beta1=beta1,
            eps=eps,
        )
    except Exception as exc:
        logger.debug("Surrogate objective failed during emission build: %s", exc)
        return float("-inf")
    if not np.all(np.isfinite(ell)):
        logger.debug("Surrogate objective received non-finite emission matrix")
        return float("-inf")

    q = float(np.sum(gamma * ell))
    if xi is not None and len(xi):
        P = np.clip(_ctmc_transition(eta_on, eta_off, delta_ref), eps, 1.0)
        q += float(np.sum(xi * np.log(P[None, :, :])))
    q -= float(lambda_0 * np.sum(np.asarray(A0, dtype=float) ** 2))
    return q


# ── numerical optimization helpers for nuisance blocks ─────────────
def _forward_diff_grad_positive(vec: np.ndarray, f: Callable[[np.ndarray], float], rel_eps: float = 1e-4) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    grad = np.zeros_like(vec)
    base = f(vec.copy())
    if not np.isfinite(base):
        return grad

    it = np.nditer(vec, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = float(vec[idx])
        h = rel_eps * max(1.0, abs(orig))
        vec[idx] = orig + h
        fp = f(vec.copy())
        vec[idx] = orig
        if np.isfinite(fp):
            grad[idx] = (fp - base) / h
        it.iternext()
    return grad


def _projected_ascent_block(
    x: np.ndarray,
    objective: Callable[[np.ndarray], float],
    projector: Callable[[np.ndarray], np.ndarray],
    step: float,
    rel_eps: float = 1e-4,
    max_backtracks: int = 6,
    label: str = "block",
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    base = objective(x.copy())
    if not np.isfinite(base):
        logger.debug("Projected ascent [%s]: base objective not finite; returning projected input", label)
        return projector(x)
    grad = _forward_diff_grad_positive(x.copy(), objective, rel_eps=rel_eps)
    grad_norm = float(np.linalg.norm(grad.ravel()))
    if not np.all(np.isfinite(grad)) or grad_norm == 0.0:
        logger.debug("Projected ascent [%s]: gradient unusable (norm=%.4f); returning projected input", label, grad_norm)
        return projector(x)

    cur_step = float(step)
    best = projector(x)
    for backtrack in range(max_backtracks):
        cand = projector(x + cur_step * grad)
        val = objective(cand)
        if np.isfinite(val) and val >= base:
            logger.debug(
                "Projected ascent [%s]: accepted step=%.6f backtracks=%d base=%.6f new=%.6f grad_norm=%.4f",
                label,
                cur_step,
                backtrack,
                base,
                val,
                grad_norm,
            )
            return cand
        cur_step *= 0.5
    logger.debug(
        "Projected ascent [%s]: no improving step found after %d backtracks (base=%.6f grad_norm=%.4f)",
        label,
        max_backtracks,
        base,
        grad_norm,
    )
    return best


def _sigmoid(x: float | np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return float(np.log(p) - np.log1p(-p))


def _eta_transition_objective(
    theta: np.ndarray,
    xi: np.ndarray,
    delta_ref: float,
    eps: float = 1e-12,
    log_s_ref: float | None = None,
    logit_pi1_ref: float | None = None,
    prior_weight_s: float = 0.0,
    prior_weight_pi: float = 0.0,
    eta_beta_prior_kappa: float = 0.0,
    eta_beta_prior_target: float = 0.5,
    eta_gamma_prior_shape: float = 2.0,
    eta_gamma_prior_rate: float = 1.0,
    eta_gamma_prior_weight: float = 0.0,
) -> float:
    log_s = float(np.clip(theta[0], np.log(1e-6), np.log(50.0)))
    logit_pi1 = float(np.clip(theta[1], -20.0, 20.0))
    s = float(np.exp(log_s))
    pi1 = float(_sigmoid(logit_pi1))
    eta_on = pi1 * s
    eta_off = (1.0 - pi1) * s
    P = np.clip(_ctmc_transition(eta_on, eta_off, delta_ref), eps, 1.0)
    q = float(np.sum(np.asarray(xi, dtype=float) * np.log(P[None, :, :])))

    # Legacy quadratic priors, kept default-off for backward compatibility.
    if prior_weight_s > 0.0 and log_s_ref is not None:
        q -= 0.5 * float(prior_weight_s) * (log_s - float(log_s_ref)) ** 2
    if prior_weight_pi > 0.0 and logit_pi1_ref is not None:
        q -= 0.5 * float(prior_weight_pi) * (logit_pi1 - float(logit_pi1_ref)) ** 2

    # Paper-defensible CTMC prior in the (pi1, s) parameterization.
    # pi1 ~ Beta(kappa*pi*, kappa*(1-pi*)) and s ~ Gamma(shape, rate).
    # Defaults are off, preserving all previous experiments unless CLI flags
    # activate the prior.  This regularizes the stationary active fraction
    # directly rather than trying to bound eta_on or eta_off separately.
    if eta_beta_prior_kappa is not None and float(eta_beta_prior_kappa) > 0.0:
        kappa = float(eta_beta_prior_kappa)
        target = float(np.clip(eta_beta_prior_target, 1e-4, 1.0 - 1e-4))
        alpha = max(kappa * target, eps)
        beta = max(kappa * (1.0 - target), eps)
        pi_safe = float(np.clip(pi1, eps, 1.0 - eps))
        q += float((alpha - 1.0) * np.log(pi_safe) + (beta - 1.0) * np.log(1.0 - pi_safe))

    if eta_gamma_prior_weight is not None and float(eta_gamma_prior_weight) > 0.0:
        shape = max(float(eta_gamma_prior_shape), eps)
        rate = max(float(eta_gamma_prior_rate), eps)
        q += float(eta_gamma_prior_weight) * float((shape - 1.0) * log_s - rate * s)
    return q

def _eta_exact_mstep(
    gamma: np.ndarray,
    xi: np.ndarray,
    interval_edges: np.ndarray,
    eta_on_init: float | None = None,
    eta_off_init: float | None = None,
    prior_weight_s: float = 0.0,
    prior_weight_pi: float = 0.0,
    eta_off_floor: float = 0.0,
    eta_pi1_target: float | None = None,
    eta_beta_prior_kappa: float = 0.0,
    eta_beta_prior_target: float = 0.5,
    eta_gamma_prior_shape: float = 2.0,
    eta_gamma_prior_rate: float = 1.0,
    eta_gamma_prior_weight: float = 0.0,
) -> tuple[float, float]:
    deltas = np.diff(interval_edges)
    delta_ref = float(np.mean(deltas))
    eps = 1e-9

    T0 = float(np.sum(np.asarray(gamma[:, 0], dtype=float) * deltas))
    T1 = float(np.sum(np.asarray(gamma[:, 1], dtype=float) * deltas))
    N01 = float(np.sum(np.asarray(xi[:, 0, 1], dtype=float))) if len(xi) else 0.0
    N10 = float(np.sum(np.asarray(xi[:, 1, 0], dtype=float))) if len(xi) else 0.0
    active_occ = float(T1 / max(T0 + T1, eps))

    if eta_on_init is None:
        eta_on_init = max(N01 / max(T0, eps), 1e-3)
    if eta_off_init is None:
        eta_off_init = max(N10 / max(T1, eps), 1e-3)

    s_init = max(float(eta_on_init) + float(eta_off_init), 1e-4)
    pi1_init = np.clip(float(eta_on_init) / s_init, max(active_occ * 0.5, 1e-4), min(max(active_occ * 1.5, 1e-3), 1.0 - 1e-4))

    theta = np.array([np.log(s_init), _logit(pi1_init)], dtype=float)
    log_s_ref = float(np.log(s_init))
    # Default legacy behavior anchors the pi prior to the current posterior
    # occupancy.  For real CRCNS diagnostics, an external stationary-active
    # target can be supplied to prevent active-all self-reinforcement.
    if eta_pi1_target is not None and np.isfinite(float(eta_pi1_target)):
        pi_ref = float(np.clip(float(eta_pi1_target), 1e-4, 1.0 - 1e-4))
    else:
        pi_ref = float(np.clip(active_occ, 1e-4, 1.0 - 1e-4))
    logit_pi1_ref = _logit(pi_ref)

    def projector(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x[0] = np.clip(x[0], np.log(1e-6), np.log(50.0))
        x[1] = np.clip(x[1], -20.0, 20.0)
        return x

    if len(xi):
        objective = lambda x: _eta_transition_objective(
            x,
            xi=np.asarray(xi, dtype=float),
            delta_ref=delta_ref,
            log_s_ref=log_s_ref,
            logit_pi1_ref=logit_pi1_ref,
            prior_weight_s=prior_weight_s,
            prior_weight_pi=prior_weight_pi,
            eta_beta_prior_kappa=eta_beta_prior_kappa,
            eta_beta_prior_target=eta_beta_prior_target,
            eta_gamma_prior_shape=eta_gamma_prior_shape,
            eta_gamma_prior_rate=eta_gamma_prior_rate,
            eta_gamma_prior_weight=eta_gamma_prior_weight,
        )
        for _ in range(12):
            theta = _projected_ascent_block(
                theta,
                objective=objective,
                projector=projector,
                step=0.5,
                rel_eps=5e-3,
                max_backtracks=8,
                label='eta',
            )

    s = float(np.exp(theta[0]))
    pi1 = float(_sigmoid(theta[1]))
    eta_on = pi1 * s
    eta_off = (1.0 - pi1) * s
    if eta_off_floor is not None and float(eta_off_floor) > 0.0:
        eta_off = max(float(eta_off), float(eta_off_floor))
    logger.debug(
        "Exact CTMC eta update: T0=%.4f T1=%.4f N01=%.4f N10=%.4f delta=%.4f occ=%.4f -> (eta_on=%.6f, eta_off=%.6f)",
        T0,
        T1,
        N01,
        N10,
        delta_ref,
        active_occ,
        eta_on,
        eta_off,
    )
    return _stabilize_eta(eta_on, eta_off)


def _sanitize_gradients(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.nan_to_num(np.asarray(a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0) for a in arrays)


def _clip_by_global_norm(arrays: list[np.ndarray], max_norm: float = 50.0) -> tuple[list[np.ndarray], float]:
    sq_norm = 0.0
    for arr in arrays:
        arr_np = np.asarray(arr, dtype=float)
        sq_norm += float(np.sum(arr_np * arr_np))
    norm = float(np.sqrt(sq_norm))
    if (not np.isfinite(norm)) or norm == 0.0 or norm <= max_norm:
        return arrays, norm
    scale = max_norm / norm
    return [np.asarray(arr, dtype=float) * scale for arr in arrays], norm


# ── main M-step ────────────────────────────────────────────────────
def run_mstep(
    events, interval_edges, gamma, xi,
    nu, A0, g, h, U, V, rho0, rho1, beta0, beta1,
    lambda_g=0.1, lambda_h=0.1, lambda_e=0.01, lambda_0=0.01,
    lr=0.01, n_inner_steps=5, gamma_prev_1=None,
    nuisance_lr: float | None = None,
    n_nuisance_steps: int = 1,
    eta_on_prev: float | None = None,
    eta_off_prev: float | None = None,
    eta_prior_weight_s: float = 0.0,
    eta_prior_weight_pi: float = 0.0,
    eta_off_floor: float = 0.0,
    eta_pi1_target: float | None = None,
    eta_beta_prior_kappa: float = 0.0,
    eta_beta_prior_target: float = 0.5,
    eta_gamma_prior_shape: float = 2.0,
    eta_gamma_prior_rate: float = 1.0,
    eta_gamma_prior_weight: float = 0.0,
) -> MStepResult:
    logger.debug(
        "M-step start: events=%d intervals=%d lr=%.4f inner_steps=%d nuisance_steps=%d",
        len(events),
        max(len(interval_edges) - 1, 0),
        lr,
        n_inner_steps,
        n_nuisance_steps,
    )
    if gamma_prev_1 is None:
        gamma_prev_1 = gamma[:, 1]
    if nuisance_lr is None:
        nuisance_lr = 0.25 * lr

    eta_on, eta_off = _eta_exact_mstep(
        gamma,
        xi,
        interval_edges,
        eta_on_init=eta_on_prev,
        eta_off_init=eta_off_prev,
        prior_weight_s=eta_prior_weight_s,
        prior_weight_pi=eta_prior_weight_pi,
        eta_off_floor=eta_off_floor,
        eta_pi1_target=eta_pi1_target,
        eta_beta_prior_kappa=eta_beta_prior_kappa,
        eta_beta_prior_target=eta_beta_prior_target,
        eta_gamma_prior_shape=eta_gamma_prior_shape,
        eta_gamma_prior_rate=eta_gamma_prior_rate,
        eta_gamma_prior_weight=eta_gamma_prior_weight,
    )
    delta_ref = float(np.mean(np.diff(interval_edges)))
    T = float(interval_edges[-1] - interval_edges[0])
    logger.debug(
        "M-step timing summary: delta_ref=%.4f T=%.4f avg_active=%.4f",
        delta_ref,
        T,
        _mean_or_nan(gamma[:, 1]),
    )

    # initial baseline refresh under current Hawkes blocks
    A1_cur = _compute_A1_np(g, h, U, V)
    nu_new = _baseline_closed_form(
        events, interval_edges, gamma, gamma_prev_1,
        nu, A0, A1_cur, rho0, rho1, beta0, beta1, T,
    )
    logger.debug(
        "Initial active matrix from current gates: A1 nnz=%d mean=%.6f",
        _nnz(A1_cur),
        _mean_or_nan(A1_cur),
    )

    feats = precompute_features(
        events, interval_edges, gamma, gamma_prev_1,
        nu_new, A0, rho0, rho1, beta0, beta1,
    )
    feats["gamma"] = gamma
    feats["xi"] = xi

    g_new = np.array(g, dtype=np.float64)
    h_new = np.array(h, dtype=np.float64)
    U_new = np.array(U, dtype=np.float64)
    V_new = np.array(V, dtype=np.float64)

    # ── block 1: active gates / embeddings ─────────────────────────
    if _HAS_JAX:
        j_rb = jnp.array(feats["routine_base"])
        j_f0 = jnp.array(feats["active_feat_z0"])
        j_f1 = jnp.array(feats["active_feat_z1"])
        j_ai = jnp.array(feats["actor_idx"], dtype=jnp.int32)
        j_bins = jnp.array(feats["bins"], dtype=jnp.int32)
        j_cr = jnp.array(feats["comp_routine"])
        j_cp = jnp.array(feats["comp_feat_pre"])
        j_cw = jnp.array(feats["comp_feat_within"])
        j_gamma = jnp.array(gamma)
        j_xi = jnp.array(xi)

        current_q = float(_jax_Q(
            jnp.array(g_new), jnp.array(h_new), jnp.array(U_new), jnp.array(V_new),
            j_rb, j_f0, j_f1, j_ai, j_bins,
            j_cr, j_cp, j_cw,
            j_gamma, j_xi,
            eta_on, eta_off, delta_ref,
        ))

        logger.debug("Active block (JAX) start: q=%.6f", current_q)

        for inner_step in range(n_inner_steps):
            dg, dh, dU, dV = _grad_Q(
                jnp.array(g_new), jnp.array(h_new), jnp.array(U_new), jnp.array(V_new),
                j_rb, j_f0, j_f1, j_ai, j_bins,
                j_cr, j_cp, j_cw,
                j_gamma, j_xi,
                eta_on, eta_off, delta_ref,
            )
            dg, dh, dU, dV = _sanitize_gradients(dg, dh, dU, dV)
            (dg, dh, dU, dV), grad_norm = _clip_by_global_norm([dg, dh, dU, dV], max_norm=50.0)
            if grad_norm == 0.0:
                logger.debug("Active block (JAX) step %d: zero gradient norm; stopping", inner_step + 1)
                break

            step = float(lr)
            accepted = False
            logger.debug("Active block (JAX) step %d: grad_norm=%.4f", inner_step + 1, grad_norm)
            for _bt in range(6):
                cand_g = np.maximum(0.0, g_new + step * dg - step * lambda_g)
                cand_h = np.maximum(0.0, h_new + step * dh - step * lambda_h)

                U_plus = U_new + step * dU
                cand_U = np.array(U_new, copy=True)
                for j in range(U_plus.shape[0]):
                    nrm = np.linalg.norm(U_plus[j])
                    cand_U[j] = U_plus[j] * max(0.0, 1.0 - step * lambda_e / (nrm + 1e-12))

                V_plus = V_new + step * dV
                cand_V = np.array(V_new, copy=True)
                for j in range(V_plus.shape[0]):
                    nrm = np.linalg.norm(V_plus[j])
                    cand_V[j] = V_plus[j] * max(0.0, 1.0 - step * lambda_e / (nrm + 1e-12))

                cand_q = float(_jax_Q(
                    jnp.array(cand_g), jnp.array(cand_h), jnp.array(cand_U), jnp.array(cand_V),
                    j_rb, j_f0, j_f1, j_ai, j_bins,
                    j_cr, j_cp, j_cw,
                    j_gamma, j_xi,
                    eta_on, eta_off, delta_ref,
                ))
                if np.isfinite(cand_q) and (not np.isfinite(current_q) or cand_q >= current_q - 1e-8):
                    g_new, h_new, U_new, V_new = cand_g, cand_h, cand_U, cand_V
                    current_q = cand_q
                    accepted = True
                    logger.debug(
                        "Active block (JAX) step %d: accepted step=%.6f backtracks=%d q=%.6f nnz(g,h)=(%d,%d)",
                        inner_step + 1,
                        step,
                        _bt,
                        cand_q,
                        _nnz(cand_g),
                        _nnz(cand_h),
                    )
                    break
                step *= 0.5

            if not accepted:
                logger.debug("Active block (JAX) step %d: no acceptable update found; stopping", inner_step + 1)
                break
    else:
        def _fallback_q(gv, hv, Uv, Vv):
            A1 = _compute_A1_np(gv, hv, Uv, Vv)
            return _surrogate_q_full(
                events=events, interval_edges=interval_edges,
                gamma_prev_1=gamma_prev_1, gamma=gamma, xi=xi,
                nu=nu_new, A0=A0, A1=A1, rho0=rho0, rho1=rho1,
                beta0=beta0, beta1=beta1,
                eta_on=eta_on, eta_off=eta_off, delta_ref=delta_ref, lambda_0=lambda_0,
            )

        logger.debug("Active block (finite diff) start")
        for inner_step in range(n_inner_steps):
            dg = _forward_diff_grad_positive(g_new, lambda x: _fallback_q(x, h_new, U_new, V_new))
            dh = _forward_diff_grad_positive(h_new, lambda x: _fallback_q(g_new, x, U_new, V_new))
            dU = _forward_diff_grad_positive(U_new, lambda x: _fallback_q(g_new, h_new, x, V_new))
            dV = _forward_diff_grad_positive(V_new, lambda x: _fallback_q(g_new, h_new, U_new, x))

            logger.debug(
                "Active block (finite diff) step %d: grad_norms=(%.4f, %.4f, %.4f, %.4f)",
                inner_step + 1,
                float(np.linalg.norm(dg)),
                float(np.linalg.norm(dh)),
                float(np.linalg.norm(dU)),
                float(np.linalg.norm(dV)),
            )

            g_new = np.maximum(0.0, g_new + lr * dg - lr * lambda_g)
            h_new = np.maximum(0.0, h_new + lr * dh - lr * lambda_h)

            U_plus = U_new + lr * dU
            for j in range(U_plus.shape[0]):
                nrm = np.linalg.norm(U_plus[j])
                U_new[j] = U_plus[j] * max(0.0, 1.0 - lr * lambda_e / (nrm + 1e-12))

            V_plus = V_new + lr * dV
            for j in range(V_plus.shape[0]):
                nrm = np.linalg.norm(V_plus[j])
                V_new[j] = V_plus[j] * max(0.0, 1.0 - lr * lambda_e / (nrm + 1e-12))

    logger.debug(
        "Active block done: nnz(g,h)=(%d,%d) ||U||=%.4f ||V||=%.4f",
        _nnz(g_new),
        _nnz(h_new),
        float(np.linalg.norm(U_new)),
        float(np.linalg.norm(V_new)),
    )

    # ── block 2: routine coupling, mark weights, decays ────────────
    A1_new = _compute_A1_np(g_new, h_new, U_new, V_new)
    A0_new = np.array(A0, dtype=float)
    rho0_new = _project_rho(np.array(rho0, dtype=float))
    rho1_new = _project_rho(np.array(rho1, dtype=float))
    beta0_new = float(beta0)
    beta1_new = float(beta1)

    def _objective_current(A0v, rho0v, rho1v, beta0v, beta1v):
        return _surrogate_q_full(
            events=events,
            interval_edges=interval_edges,
            gamma_prev_1=gamma_prev_1,
            gamma=gamma,
            xi=xi,
            nu=nu_new,
            A0=A0v,
            A1=A1_new,
            rho0=rho0v,
            rho1=rho1v,
            beta0=beta0v,
            beta1=beta1v,
            eta_on=eta_on,
            eta_off=eta_off,
            delta_ref=delta_ref,
            lambda_0=lambda_0,
        )

    logger.debug(
        "Nuisance block start: A0 nnz=%d beta=(%.4f, %.4f)",
        _nnz(A0_new),
        beta0_new,
        beta1_new,
    )
    # CRCNS real-data horizon runs are dominated by finite-difference A0/rho
    # nuisance updates: for K=30, A0 has 900 coordinates, and each coordinate
    # perturbation rebuilds full interval emissions.  When enabled, this fast
    # path freezes A0/rho at their current projected values and still updates
    # beta via the existing low-dimensional objective.  This preserves the
    # active gate/embedding block, baseline refresh, beta separation, and CTMC
    # update, while removing the O(K^2) finite-difference bottleneck.
    fast_nuisance = str(os.environ.get("CRCNS_FAST_MSTEP_NUISANCE", "0")).lower() in {"1", "true", "yes", "on"}
    if fast_nuisance:
        logger.info(
            "CRCNS_FAST_MSTEP_NUISANCE enabled: skipping finite-difference A0/rho nuisance updates; beta update retained."
        )
    for nuisance_step in range(max(int(n_nuisance_steps), 0)):
        logger.debug(
            "Nuisance block iteration %d: entering with beta=(%.4f, %.4f) rho_row_sums=(%s, %s)",
            nuisance_step + 1,
            beta0_new,
            beta1_new,
            np.array2string(np.sum(rho0_new, axis=1), precision=4, suppress_small=True),
            np.array2string(np.sum(rho1_new, axis=1), precision=4, suppress_small=True),
        )
        if fast_nuisance:
            A0_new = _project_nonnegative(A0_new, max_value=2.0)
            rho0_new = _project_rho(rho0_new)
            rho1_new = _project_rho(rho1_new)
        else:
            A0_new = _projected_ascent_block(
                A0_new,
                lambda x: _objective_current(x, rho0_new, rho1_new, beta0_new, beta1_new),
                lambda x: _project_nonnegative(x, max_value=2.0),
                step=nuisance_lr,
                rel_eps=5e-4,
                label="A0",
            )
            rho0_new = _projected_ascent_block(
                rho0_new,
                lambda x: _objective_current(A0_new, _project_rho(x), rho1_new, beta0_new, beta1_new),
                _project_rho,
                step=nuisance_lr,
                rel_eps=5e-4,
                label="rho0",
            )
            rho1_new = _projected_ascent_block(
                rho1_new,
                lambda x: _objective_current(A0_new, rho0_new, _project_rho(x), beta0_new, beta1_new),
                _project_rho,
                step=nuisance_lr,
                rel_eps=5e-4,
                label="rho1",
            )
        # ── beta decay rates: separate iterative update ──────────────
        # Beta parameters live on a very different scale than A0 or rho
        # entries.  The generic nuisance_lr (derived from the gate lr) is
        # far too small: 0.5 * 0.0025 = 0.00125 moves beta by at most
        # ~0.06 per EM iteration, which is insufficient for convergence.
        # We use a dedicated step size and iterate within each nuisance
        # step so beta can actually track the likelihood surface.
        beta_min = float(os.environ.get("REGIME_HAWKES_BETA_MIN", "1e-3"))
        beta_max = float(os.environ.get("REGIME_HAWKES_BETA_MAX", "20.0"))

        beta_obj = lambda x: _objective_current(
            A0_new, rho0_new, rho1_new,
            float(np.clip(x[0], beta_min, beta_max)),
            float(np.clip(x[1], beta_min, beta_max)),
        )
        # Enforce beta1 >= RATIO * beta0.  The active kernel must decay
        # meaningfully faster than routine — this is the operational form
        # of identifiability condition (c) in Theorem 1.
        #
        # An additive gap (beta1 >= beta0 + 0.5) allows beta0 to climb to
        # 2.0 with beta1 = 2.5, making the kernels nearly collinear and
        # collapsing regime separation.  A multiplicative constraint
        # prevents this: beta0 = 2.0 would force beta1 >= 4.0, creating
        # a steep compensator penalty that pushes beta0 back down.
        _BETA_RATIO = 2.0
        def beta_proj(x):
            x = np.asarray(x, dtype=float)
            x[0] = np.clip(x[0], beta_min, beta_max)
            x[1] = np.clip(x[1], beta_min, beta_max)
            if x[1] < _BETA_RATIO * x[0]:
                # Project onto the constraint boundary beta1 = RATIO * beta0
                # while preserving the geometric mean (minimizes distortion).
                geo = np.sqrt(x[0] * x[1])
                x[0] = geo / np.sqrt(_BETA_RATIO)
                x[1] = geo * np.sqrt(_BETA_RATIO)
                x[0] = np.clip(x[0], beta_min, beta_max)
                x[1] = np.clip(x[1], _BETA_RATIO * x[0], beta_max)
            return x
        beta_vec = np.array([beta0_new, beta1_new], dtype=float)
        for _beta_step in range(5):
            beta_vec = _projected_ascent_block(
                beta_vec,
                beta_obj,
                beta_proj,
                step=0.5,
                rel_eps=0.01,
                label="beta",
            )
        beta0_new = float(beta_vec[0])
        beta1_new = float(beta_vec[1])
        logger.debug(
            "Nuisance block iteration %d done: A0 nnz=%d beta=(%.4f, %.4f)",
            nuisance_step + 1,
            _nnz(A0_new),
            beta0_new,
            beta1_new,
        )

    logger.debug(
        "Nuisance block done: A1 nnz=%d A0 nnz=%d beta=(%.4f, %.4f)",
        _nnz(A1_new),
        _nnz(A0_new),
        beta0_new,
        beta1_new,
    )

    # refresh baseline under updated nuisance blocks
    nu_final = _baseline_closed_form(
        events, interval_edges, gamma, gamma_prev_1,
        nu_new, A0_new, A1_new, rho0_new, rho1_new, beta0_new, beta1_new, T,
    )

    logger.debug(
        "M-step done: nu_mean=%.6f eta=(%.6f, %.6f) nnz(A0,A1)=(%d,%d)",
        _mean_or_nan(nu_final),
        eta_on,
        eta_off,
        _nnz(A0_new),
        _nnz(A1_new),
    )
    return MStepResult(
        nu=nu_final,
        A0=A0_new,
        g=g_new,
        h=h_new,
        U=U_new,
        V=V_new,
        rho0=rho0_new,
        rho1=rho1_new,
        beta0=beta0_new,
        beta1=beta1_new,
        eta_on=float(eta_on),
        eta_off=float(eta_off),
    )
