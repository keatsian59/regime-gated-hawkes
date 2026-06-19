"""Baseline 1: Static Hawkes + group lasso.

This is the proposed model with Z ≡ 1 everywhere (no regime switching).
The EM reduces to penalized MLE on a single-regime marked Hawkes with
the same gate/embedding parameterization.

Ablates: the latent regime field.
Keeps: factored gates, directed low-rank topology, mark structure.
"""
from __future__ import annotations

import numpy as np

from regime_hawkes.config import SimConfig
from regime_hawkes.evaluate import evaluate
from regime_hawkes.mstep import MStepResult, run_mstep, _baseline_closed_form, _stabilize_eta
from regime_hawkes.simulate import simulate_regime_hawkes, summarize_simulation
from regime_hawkes.utils import compute_A1, spectral_radius
from regime_hawkes.likelihood import build_interval_emissions, event_bins
from regime_hawkes.traces import compute_event_traces


def _static_mstep(
    events, interval_edges, nu, A0, g, h, U, V,
    rho0, rho1, beta0, beta1,
    lambda_g, lambda_h, lambda_e, lambda_0,
    lr, n_inner_steps,
):
    """M-step with gamma fixed to all-active."""
    B = len(interval_edges) - 1
    gamma_all_active = np.zeros((B, 2))
    gamma_all_active[:, 1] = 1.0  # Z ≡ 1
    xi_all_active = np.zeros((B - 1, 2, 2))
    xi_all_active[:, 1, 1] = 1.0  # always stay in state 1

    gamma_prev_1 = np.ones(B)  # all events are "active"

    return run_mstep(
        events=events,
        interval_edges=interval_edges,
        gamma=gamma_all_active,
        xi=xi_all_active,
        nu=nu, A0=A0, g=g, h=h, U=U, V=V,
        rho0=rho0, rho1=rho1, beta0=beta0, beta1=beta1,
        lambda_g=lambda_g, lambda_h=lambda_h,
        lambda_e=lambda_e, lambda_0=lambda_0,
        lr=lr, n_inner_steps=n_inner_steps,
        gamma_prev_1=gamma_prev_1,
    )


def run_static_hawkes_baseline(
    events, interval_edges, init_params,
    max_iters=30, lr=0.05, lambda_g=0.02, lambda_h=0.02,
    lambda_e=0.01, lambda_0=0.01, n_inner_steps=10,
    tol=1e-4, verbose=True,
):
    """Fit the model with Z ≡ 1 (no regime switching)."""
    params = init_params
    B = len(interval_edges) - 1
    prev_ll = None

    for it in range(1, max_iters + 1):
        momentum = 0.5 if it <= 10 else 0.3
        anneal = min(1.0, it / 10.0)
        lg_eff = lambda_g * anneal
        lh_eff = lambda_h * anneal

        # Single "E-step": gamma is fixed at all-active
        # Compute log-likelihood under Z≡1 for monitoring
        A1 = compute_A1(params.g, params.h, params.U, params.V)
        ell = build_interval_emissions(
            events=events, interval_edges=interval_edges,
            gamma_prev_1=np.ones(B),
            nu=params.nu, A0=params.A0, A1=A1,
            rho0=params.rho0, rho1=params.rho1,
            beta0=params.beta0, beta1=params.beta1,
        )
        ll = float(np.sum(ell[:, 1]))  # Z≡1, so use column 1

        # M-step
        proposed = _static_mstep(
            events, interval_edges,
            params.nu, params.A0, params.g, params.h,
            params.U, params.V,
            params.rho0, params.rho1,
            params.beta0, params.beta1,
            lg_eff, lh_eff, lambda_e, lambda_0,
            lr, n_inner_steps,
        )

        # Blend
        params = MStepResult(
            nu=momentum * params.nu + (1 - momentum) * proposed.nu,
            A0=momentum * params.A0 + (1 - momentum) * proposed.A0,
            g=momentum * params.g + (1 - momentum) * proposed.g,
            h=momentum * params.h + (1 - momentum) * proposed.h,
            U=momentum * params.U + (1 - momentum) * proposed.U,
            V=momentum * params.V + (1 - momentum) * proposed.V,
            rho0=params.rho0, rho1=params.rho1,
            beta0=params.beta0, beta1=params.beta1,
            eta_on=0.5, eta_off=0.5,  # irrelevant, Z≡1
        )

        # Stability
        A1_new = compute_A1(params.g, params.h, params.U, params.V)
        spec = spectral_radius(params.A0 + A1_new)
        if spec > 0.95:
            scale = 0.8 / spec
            params.g *= np.sqrt(scale)
            params.h *= np.sqrt(scale)
            spec = spectral_radius(params.A0 + compute_A1(params.g, params.h, params.U, params.V))

        if verbose:
            print(f"  static iter {it}: ll={ll:.1f}, spec={spec:.3f}, "
                  f"nz_gates={np.sum(params.g > 0.01)}")

        if prev_ll is not None and abs(ll - prev_ll) / (abs(prev_ll) + 1e-8) < tol:
            break
        prev_ll = ll

    return params, ll


def main():
    cfg = SimConfig(
        K=10, M=2, T=500.0, ring_actors=[0, 1, 2], hub_actor=0, d=3,
        nu_base=0.15, alpha0_team=0.01, alpha1_max=0.8, alpha1_min=0.35,
        beta0=1.0, beta1=3.0, eta_on=0.04, eta_off=0.3, seed=7,
    )
    data = simulate_regime_hawkes(cfg)
    summary = summarize_simulation(data)
    print("=== BASELINE 1: Static Hawkes + Group Lasso (Z ≡ 1) ===\n")
    for k, v in summary.items():
        print(f"{k}: {v}")

    edges = np.arange(0.0, cfg.T + 1e-9, 2.0)
    base_rate = summary["total_events"] / (cfg.K * cfg.M * cfg.T)
    rng = np.random.default_rng(cfg.seed)

    init = MStepResult(
        nu=np.full((cfg.K, cfg.M), max(base_rate, 1e-3)),
        A0=np.array(data.A0, copy=True),
        g=np.full(cfg.K, 0.1),
        h=np.full(cfg.K, 0.1),
        U=rng.normal(scale=0.01, size=(cfg.K, cfg.d)),
        V=rng.normal(scale=0.01, size=(cfg.K, cfg.d)),
        rho0=np.array(data.rho0, copy=True),
        rho1=np.array(data.rho1, copy=True),
        beta0=float(cfg.beta0),
        beta1=float(cfg.beta1),
        eta_on=0.5,
        eta_off=0.5,
    )

    print("\nFitting static Hawkes (no regime switching)...")
    params, final_ll = run_static_hawkes_baseline(
        events=data.events, interval_edges=edges, init_params=init,
        max_iters=30, lr=0.05, lambda_g=0.02, lambda_h=0.02,
        n_inner_steps=10, verbose=True,
    )

    # Evaluate with both methods
    # For field AUC: static model has no gamma, so use uniform 0.5
    B = len(edges) - 1
    gamma_uniform = np.column_stack([np.full(B, 0.5), np.full(B, 0.5)])

    for method in ["topk", "gap"]:
        ev = evaluate(
            gamma=gamma_uniform,
            interval_edges=edges,
            Z_path=data.Z_path,
            g=params.g, h=params.h,
            true_ring=cfg.ring_actors,
            true_hub=cfg.hub_actor,
            method=method,
        )
        print(f"\n--- {method.upper()} ---")
        print(f"Field AUC: N/A (no regime inference)")
        print(f"Membership P/R/F1: {ev.membership_precision:.3f} / "
              f"{ev.membership_recall:.3f} / {ev.membership_f1:.3f}")
        print(f"Hub correct: {ev.hub_correct}")
        print(f"Predicted ring: {ev.predicted_ring}")

    # Gate ranking
    g = params.g
    order = np.argsort(g)[::-1]
    print("\n  Actor | True role    | Gate g_j   | Gate h_j   | Rank")
    print("  ------|------------- |------------|------------|-----")
    for rank, j in enumerate(order, 1):
        role = "**hub**" if j == cfg.hub_actor else (
            "**ring**" if j in cfg.ring_actors else "non-ring"
        )
        print(f"  {j:5d} | {role:12s} | {g[j]:10.4f} | {params.h[j]:10.4f} | {rank}")

    print(f"\nFinal log-likelihood: {final_ll:.1f}")


if __name__ == "__main__":
    main()
