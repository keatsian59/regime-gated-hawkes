from __future__ import annotations

from dataclasses import dataclass

import os
import numpy as np

from regime_hawkes.estep import run_estep
from regime_hawkes.mstep import MStepResult, run_mstep
from regime_hawkes.utils import compute_A1, spectral_radius


@dataclass
class IterSummary:
    iteration: int
    log_likelihood: float
    active_frac: float
    nonzero_gates: int
    spectral_radius: float
    beta0: float
    beta1: float
    eta_on: float
    eta_off: float
    a0_mean: float
    a0_max: float
    rho0_diag_mean: float
    rho1_diag_mean: float


@dataclass
class EMResult:
    gamma: np.ndarray
    params: MStepResult
    log_likelihood_trace: list[float]
    gate_trace: list[np.ndarray]
    g_trace: list[np.ndarray]
    h_trace: list[np.ndarray]
    a1_trace: list[np.ndarray]
    gamma_trace: list[np.ndarray]
    iter_summaries: list[IterSummary]


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
    # Mild continuation: keep a small amount of active mass alive in early
    # iterations when gates haven't separated yet.  Decay to near-zero so
    # they don't inflate active attribution for non-ring actors later on.
    pi1_floor = _linear_schedule(it, warm_iters, start=0.05, end=0.005)
    gamma_floor = _linear_schedule(it, warm_iters, start=0.05, end=0.001)
    return pi1_floor, gamma_floor


def _enforce_min_stationary_active(eta_on: float, eta_off: float, pi1_floor: float) -> tuple[float, float]:
    eta_on = float(eta_on)
    eta_off = float(eta_off)
    pi1_floor = float(np.clip(pi1_floor, 1e-6, 1.0 - 1e-6))
    min_eta_on = pi1_floor / (1.0 - pi1_floor) * max(eta_off, 1e-12)
    return max(eta_on, min_eta_on), eta_off


def _joint_stability_project(
    params: MStepResult,
    threshold: float = 0.95,
    target: float = 0.80,
) -> tuple[MStepResult, np.ndarray, float, float]:
    """Project the *joint* branching matrix A0 + A1 back into the stable region.

    The previous code only had a commented-out rescaling of A1. That does not fix
    runs where the routine block A0 is the part that grows. Here we rescale both
    blocks together whenever the joint spectral radius crosses ``threshold``.

    Returns:
        projected_params, projected_A1, spectral_radius_after, scale_factor
    """
    A1 = compute_A1(params.g, params.h, params.U, params.V)
    joint = np.asarray(params.A0, dtype=float) + np.asarray(A1, dtype=float)
    spec = float(spectral_radius(joint))
    if (not np.isfinite(spec)) or spec <= threshold:
        return params, A1, spec, 1.0

    scale = float(np.clip(target / max(spec, 1e-12), 0.0, 1.0))
    if scale >= 1.0:
        return params, A1, spec, 1.0

    # Default behavior scales both A0 and A1, preserving prior results.
    # For the fast CRCNS nuisance-update diagnostic, A0 is not fully refreshed;
    # repeatedly scaling it causes routine-regime collapse.  The env flag keeps
    # A0 fixed and stabilizes by shrinking the active gates only.
    preserve_a0 = os.environ.get("CRCNS_STABILITY_PRESERVE_A0", "0").strip().lower() in {"1", "true", "yes"}
    gate_scale = float(np.sqrt(scale))
    A0_projected = np.asarray(params.A0, dtype=float) if preserve_a0 else np.asarray(params.A0, dtype=float) * scale
    projected = MStepResult(
        nu=params.nu,
        A0=A0_projected,
        g=np.asarray(params.g, dtype=float) * gate_scale,
        h=np.asarray(params.h, dtype=float) * gate_scale,
        U=params.U,
        V=params.V,
        rho0=params.rho0,
        rho1=params.rho1,
        beta0=params.beta0,
        beta1=params.beta1,
        eta_on=params.eta_on,
        eta_off=params.eta_off,
    )
    A1_proj = compute_A1(projected.g, projected.h, projected.U, projected.V)
    spec_proj = float(spectral_radius(projected.A0 + A1_proj))
    return projected, A1_proj, spec_proj, scale


def run_em(
    events: np.ndarray,
    interval_edges: np.ndarray,
    init_params: MStepResult,
    max_iters: int = 50,
    tol: float = 1e-4,
    lambda_g: float = 0.1,
    lambda_h: float = 0.1,
    lambda_e: float = 0.01,
    lambda_0: float = 0.01,
    lr: float = 0.01,
    n_inner_steps: int = 5,
    verbose: bool = True,
    stability_threshold: float = 0.95,
    stability_target: float = 0.80,
    eta_prior_weight_s: float = 0.0,
    eta_prior_weight_pi: float = 0.0,
    eta_off_floor: float = 0.0,
    eta_pi1_target: float | None = None,
    eta_beta_prior_kappa: float = 0.0,
    eta_beta_prior_target: float = 0.5,
    eta_gamma_prior_shape: float = 2.0,
    eta_gamma_prior_rate: float = 1.0,
    eta_gamma_prior_weight: float = 0.0,
    pi1_init: float = 0.15,
) -> EMResult:
    params = init_params
    pi1_init = float(np.clip(pi1_init, 1e-4, 1.0 - 1e-4))
    gamma_prev_1 = np.full(len(interval_edges) - 1, pi1_init)
    ll_trace: list[float] = []
    gate_trace: list[np.ndarray] = []
    g_trace: list[np.ndarray] = []
    h_trace: list[np.ndarray] = []
    a1_trace: list[np.ndarray] = []
    gamma_trace: list[np.ndarray] = []
    iter_summaries: list[IterSummary] = []
    gamma = np.column_stack([1.0 - gamma_prev_1, gamma_prev_1])

    prev_ll = None
    for it in range(1, max_iters + 1):
        momentum = 0.5 if it <= 10 else 0.3
        # Scalar parameters (beta) learn from gradient ascent on a noisy
        # finite-diff landscape with sparse active evidence — they need
        # lower damping than the high-dimensional blocks (gates, embeddings)
        # to let their M-step proposals accumulate.
        scalar_momentum = 0.2 if it <= 10 else 0.1
        anneal = min(1.0, it / 10.0)
        lambda_g_eff = lambda_g * anneal
        lambda_h_eff = lambda_h * anneal

        pi1_floor, gamma_floor = _exploration_floors(it)
        gamma_prev_eff = np.clip(gamma_prev_1, gamma_floor, 1.0 - gamma_floor)
        eta_on_eff, eta_off_eff = _enforce_min_stationary_active(params.eta_on, params.eta_off, pi1_floor)

        A1 = compute_A1(params.g, params.h, params.U, params.V)
        e = run_estep(
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
        gamma = e.gamma

        proposed = run_mstep(
            events=events,
            interval_edges=interval_edges,
            gamma=e.gamma,
            xi=e.xi,
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
            lambda_e=lambda_e,
            lambda_0=lambda_0,
            lr=lr,
            n_inner_steps=n_inner_steps,
            gamma_prev_1=e.gamma[:, 1],
            eta_on_prev=params.eta_on,
            eta_off_prev=params.eta_off,
            eta_prior_weight_s=float(eta_prior_weight_s),
            eta_prior_weight_pi=float(eta_prior_weight_pi),
            eta_off_floor=float(eta_off_floor),
            eta_pi1_target=eta_pi1_target,
            eta_beta_prior_kappa=float(eta_beta_prior_kappa),
            eta_beta_prior_target=float(eta_beta_prior_target),
            eta_gamma_prior_shape=float(eta_gamma_prior_shape),
            eta_gamma_prior_rate=float(eta_gamma_prior_rate),
            eta_gamma_prior_weight=float(eta_gamma_prior_weight),
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
            # CTMC rates come from the exact expected transition-likelihood
            # M-step under the same transition matrix used in the E-step.
            # We still avoid momentum here because low eta can suppress active
            # mass and create a self-reinforcing downward spiral.
            eta_on=float(proposed.eta_on),
            eta_off=float(proposed.eta_off),
        )

        for name, arr in {
            "g": params.g,
            "h": params.h,
            "U": params.U,
            "V": params.V,
            "A0": params.A0,
            "rho0": params.rho0,
            "rho1": params.rho1,
        }.items():
            if not np.all(np.isfinite(arr)):
                raise FloatingPointError(f"{name} became non-finite before stability projection")

        params, A1_new, spec, stability_scale = _joint_stability_project(
            params,
            threshold=stability_threshold,
            target=stability_target,
        )

        ll = e.log_likelihood
        ll_trace.append(ll)
        gate_trace.append(params.g.copy())
        g_trace.append(params.g.copy())
        h_trace.append(params.h.copy())
        a1_trace.append(A1_new.copy())
        gamma_trace.append(gamma.copy())

        summary = IterSummary(
            iteration=it,
            log_likelihood=float(ll),
            active_frac=float(gamma[:, 1].mean()),
            nonzero_gates=int(np.sum(params.g > 0.01)),
            spectral_radius=float(spec),
            beta0=float(params.beta0),
            beta1=float(params.beta1),
            eta_on=float(params.eta_on),
            eta_off=float(params.eta_off),
            a0_mean=float(np.mean(params.A0)),
            a0_max=float(np.max(params.A0)),
            rho0_diag_mean=float(np.mean(np.diag(params.rho0))),
            rho1_diag_mean=float(np.mean(np.diag(params.rho1))),
        )
        iter_summaries.append(summary)

        if verbose:
            proj_msg = "" if stability_scale >= 1.0 else f", stability_scale={stability_scale:.3f}"
            print(
                f"iter {it}: ll={summary.log_likelihood:.3f}, "
                f"active_frac={summary.active_frac:.3f}, "
                f"nonzero_gates={summary.nonzero_gates}, "
                f"spectral_radius={summary.spectral_radius:.3f}, "
                f"beta0={summary.beta0:.3f}, beta1={summary.beta1:.3f}, "
                f"eta_on={summary.eta_on:.3f}, eta_off={summary.eta_off:.3f}, "
                f"A0(mean/max)=({summary.a0_mean:.4f}/{summary.a0_max:.4f}), "
                f"rho_diag(mean)=({summary.rho0_diag_mean:.3f}/{summary.rho1_diag_mean:.3f}), "
                f"lambda_g_eff={lambda_g_eff:.4f}, momentum={momentum:.2f}{proj_msg}"
            )

        if prev_ll is not None:
            rel = abs(ll - prev_ll) / (abs(prev_ll) + 1e-8)
            if rel < tol:
                break
        prev_ll = ll
        gamma_prev_1 = np.clip(gamma[:, 1], gamma_floor, 1.0 - gamma_floor)

    # Final synchronized E-step so returned gamma is aligned with the tempered
    # EM fixed point used during fitting.
    final_pi1_floor, final_gamma_floor = _exploration_floors(it if 'it' in locals() else 1)
    A1_final = compute_A1(params.g, params.h, params.U, params.V)
    eta_on_eff, eta_off_eff = _enforce_min_stationary_active(params.eta_on, params.eta_off, final_pi1_floor)
    e_final = run_estep(
        events=events,
        interval_edges=interval_edges,
        gamma_prev_1=np.clip(gamma[:, 1], final_gamma_floor, 1.0 - final_gamma_floor),
        nu=params.nu,
        A0=params.A0,
        A1=A1_final,
        rho0=params.rho0,
        rho1=params.rho1,
        beta0=params.beta0,
        beta1=params.beta1,
        eta_on=eta_on_eff,
        eta_off=eta_off_eff,
    )
    gamma = e_final.gamma

    if gamma_trace:
        gamma_trace[-1] = gamma.copy()
        a1_trace[-1] = A1_final.copy()
    else:
        gamma_trace.append(gamma.copy())
        g_trace.append(params.g.copy())
        h_trace.append(params.h.copy())
        a1_trace.append(A1_final.copy())
        gate_trace.append(params.g.copy())

    return EMResult(
        gamma=gamma,
        params=params,
        log_likelihood_trace=ll_trace,
        gate_trace=gate_trace,
        g_trace=g_trace,
        h_trace=h_trace,
        a1_trace=a1_trace,
        gamma_trace=gamma_trace,
        iter_summaries=iter_summaries,
    )
