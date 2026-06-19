from __future__ import annotations

import numpy as np
from modular_hmm_hawkes_baseline import run_modular_hmm_hawkes_baseline
from regime_hawkes.config import SimConfig
from regime_hawkes.em import run_em
from regime_hawkes.evaluate import evaluate
from regime_hawkes.mstep import MStepResult
from regime_hawkes.plots import plot_convergence, plot_event_raster, plot_gamma_timeline, plot_gates
from regime_hawkes.simulate import simulate_regime_hawkes, summarize_simulation


def main():
    cfg = SimConfig(
        K=10, M=2, T=500.0, ring_actors=[0, 1, 2], hub_actor=0, d=3,
        nu_base=0.15, alpha0_team=0.01, alpha1_max=0.8, alpha1_min=0.35,
        beta0=1.0, beta1=3.0, eta_on=0.04, eta_off=0.3, seed=7,
    )
    data = simulate_regime_hawkes(cfg)
    summary = summarize_simulation(data)
    for k, v in summary.items():
        print(f"{k}: {v}")

    edges = np.arange(0.0, cfg.T + 1e-9, 2)

    # --- Strong modular baseline: Poisson HMM + segmented pairwise Hawkes ---
    baseline = run_modular_hmm_hawkes_baseline(
        events=data.events,
        interval_edges=edges,
        true_ring=cfg.ring_actors,
        true_hub=cfg.hub_actor,
        Z_path=data.Z_path,
        selection="topk",      # strongest if true ring size is known in synthetic eval
        score_mode="geom",     # strongest actor score in this setup
        hmm_threshold=0.5,
        hmm_restarts=20,
        hmm_max_iters=200,
        hawkes_min_total_events=6,
        seed=cfg.seed,
    )

    print("\n--- MODULAR HMM + HAWKES baseline ---")
    print(f"Field AUC: {baseline.field_auc:.3f}")
    print(
        f"Membership P/R/F1: "
        f"{baseline.membership_precision:.3f} / "
        f"{baseline.membership_recall:.3f} / "
        f"{baseline.membership_f1:.3f}"
    )
    print(f"Hub correct: {baseline.hub_correct}")
    print(f"Predicted ring: {baseline.predicted_ring}")
    print(f"Predicted hub: {baseline.hub_pred}")
    print(f"HMM active fraction: {baseline.active_fraction:.3f}")
    print(f"HMM rates: {baseline.hmm_rates}")
    print(f"HMM transition matrix:\n{baseline.hmm_transition}")
    print(f"Number of pairwise Hawkes fits: {baseline.n_pair_fits}")

    print("\nBaseline sender scores:", baseline.sender_scores)
    print("Baseline receiver scores:", baseline.receiver_scores)
    print("Baseline member scores:", baseline.member_scores)

if __name__ == "__main__":
    main()
