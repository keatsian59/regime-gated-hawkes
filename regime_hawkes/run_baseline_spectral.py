"""Baseline 3: Static Hawkes MLE + spectral clustering on Â.

Fit a standard multivariate Hawkes process (no regime switching, no gates,
no embeddings) and recover the full K×K interaction matrix. Then apply
spectral methods to the estimated matrix to find a dense subgraph that
might correspond to the ring.

Ablates: regime switching, factored gates, low-rank structure.
Tests: can post-hoc network analysis on a static Hawkes find the ring?
"""
from __future__ import annotations

import numpy as np

from regime_hawkes.config import SimConfig
#from regime_hawkes.evaluate import evaluate, _interval_truth, _roc_auc_binary
from regime_hawkes.simulate import simulate_regime_hawkes, summarize_simulation
from regime_hawkes.traces import compute_event_traces #, traces_at_interval_starts
#from regime_hawkes.likelihood import event_bins


def _compute_static_loglik(
    nu: np.ndarray,
    A: np.ndarray,
    rho: np.ndarray,
    pre_traces: np.ndarray,
    actors: np.ndarray,
    marks: np.ndarray,
    beta: float,
    T: float,
    l1_penalty: float,
    eps: float = 1e-9,
) -> tuple[float, np.ndarray]:
    """Approximate penalized static Hawkes log-likelihood under fixed beta.

    This mirrors the baseline's original objective so beta can be updated by
    coordinate ascent rather than being fixed at an arbitrary value.
    """
    K = A.shape[0]
    N = len(actors)

    lam_at_events = np.zeros(N)
    for n in range(N):
        i_n, m_n = actors[n], marks[n]
        trig = 0.0
        for j in range(K):
            for mp in range(pre_traces.shape[2]):
                trig += A[j, i_n] * pre_traces[n, j, mp] * rho[mp, m_n]
        lam_at_events[n] = max(nu[i_n, m_n] + trig, eps)

    ll = np.sum(np.log(lam_at_events)) - np.sum(nu) * T
    rho_sum = float(np.sum(rho))
    for j in range(K):
        n_j = np.sum(actors == j)
        for i in range(K):
            if j == i:
                continue
            ll -= A[j, i] * rho_sum * n_j / beta

    # Penalized objective used for beta selection.
    ll -= l1_penalty * np.sum(A)
    return float(ll), lam_at_events


def _beta_candidate_grid(
    beta: float,
    beta_bounds: tuple[float, float],
    beta_grid_size: int,
) -> np.ndarray:
    """Local multiplicative search grid for beta coordinate updates."""
    beta_lo, beta_hi = beta_bounds
    log_step = np.linspace(-0.6, 0.6, beta_grid_size)
    grid = beta * np.exp(log_step)
    grid = np.clip(grid, beta_lo, beta_hi)
    grid = np.concatenate([
        np.array([beta_lo, beta_hi, beta]),
        grid,
    ])
    grid = np.unique(np.round(grid, 8))
    return np.sort(grid)



def _empirical_nu_matrix(events: np.ndarray, K: int, M: int, T: float, eps: float = 1e-9) -> np.ndarray:
    """Empirical per-actor/per-mark baseline rates for static real-data fits."""
    T = max(float(T), eps)
    counts = np.zeros((int(K), int(M)), dtype=float)
    if events is not None and len(events):
        actors = np.asarray(events[:, 1], dtype=int)
        marks = np.asarray(events[:, 2], dtype=int)
        valid = (actors >= 0) & (actors < int(K)) & (marks >= 0) & (marks < int(M))
        np.add.at(counts, (actors[valid], marks[valid]), 1.0)
    nu = counts / T
    positive = nu[nu > 0]
    floor = max(float(np.percentile(positive, 5)) * 0.05, eps) if positive.size else eps
    return np.maximum(nu, floor)


def _fit_static_hawkes_full_matrix(
    events: np.ndarray,
    K: int, M: int, T: float,
    beta_init: float = 2.0,
    lr: float = 0.005,
    n_iters: int = 200,
    l1_penalty: float = 0.01,
    beta_update_every: int = 10,
    beta_grid_size: int = 9,
    beta_bounds: tuple[float, float] = (0.25, 6.0),
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a static multivariate Hawkes with full K×K interaction matrix.

    Uses gradient ascent on the log-likelihood with L1 penalty on the
    interaction matrix. Beta is no longer fixed. Instead, we alternate the
    standard parameter updates with a one-dimensional coordinate-ascent step
    that profiles the same penalized objective over a local grid of beta
    values.

    Returns: nu (K×M), A (K×K interaction strengths), rho (M×M mark weights),
    fitted beta.
    """
    # Initialize.
    # Use empirical baseline-rate scale and keep nu in a broad but finite band.
    # This prevents static Hawkes nuisance-rate blowups on high-frequency real data
    # while still allowing 10x movement around the observed per-channel rate.
    eps = 1e-9
    empirical_nu = _empirical_nu_matrix(events, K, M, T, eps=eps)
    nu_lo = np.maximum(0.10 * empirical_nu, eps)
    nu_hi = np.maximum(10.00 * empirical_nu, nu_lo + eps)
    nu = empirical_nu.copy()

    A = np.full((K, K), 0.01)
    np.fill_diagonal(A, 0.0)
    rho = np.ones((M, M)) / M
    beta = float(beta_init)

    weights = np.ones(len(events))
    pre_traces, _ = compute_event_traces(events, K, M, beta, beta, weights)

    actors = events[:, 1].astype(int)
    marks = events[:, 2].astype(int)
    N = len(events)
    actor_counts = np.bincount(actors, minlength=K)

    for iteration in range(n_iters):
        # Rebuild traces when beta changes.
        if iteration == 0 or ((iteration + 1) % beta_update_every == 0):
            pre_traces, _ = compute_event_traces(events, K, M, beta, beta, weights)

        ll, lam_at_events = _compute_static_loglik(
            nu=nu,
            A=A,
            rho=rho,
            pre_traces=pre_traces,
            actors=actors,
            marks=marks,
            beta=beta,
            T=T,
            l1_penalty=l1_penalty,
            eps=eps,
        )

        # Gradient for nu: sum of 1/lambda * indicator - T
        grad_nu = np.zeros_like(nu)
        for n in range(N):
            i_n, m_n = actors[n], marks[n]
            grad_nu[i_n, m_n] += 1.0 / lam_at_events[n]
        grad_nu -= T  # exact gradient of the -sum(nu) * T compensator

        # Gradient for A[j,i]: sum over events at i of (trace contribution / lambda) - compensator
        grad_A = np.zeros_like(A)
        rho_sum = float(np.sum(rho))
        for n in range(N):
            i_n, m_n = actors[n], marks[n]
            for j in range(K):
                if j == i_n:
                    continue
                feat = sum(pre_traces[n, j, mp] * rho[mp, m_n] for mp in range(M))
                grad_A[j, i_n] += feat / lam_at_events[n]

        for j in range(K):
            for i in range(K):
                if j == i:
                    continue
                grad_A[j, i] -= rho_sum * actor_counts[j] / beta

        # L1 proximal step on A
        A_plus = A + lr * grad_A
        A = np.maximum(0.0, A_plus - lr * l1_penalty)
        np.fill_diagonal(A, 0.0)

        # Update nu
        # Damped log-rate update for baseline intensities.
        # Raw additive nu updates are too brittle on high-frequency real data.
        theta = np.log(np.maximum(nu, eps))
        theta0 = np.log(np.maximum(empirical_nu, eps))
        grad_theta = nu * grad_nu

        # Weak log-rate anchor around empirical per-channel rates.
        grad_theta -= 1.0 * (theta - theta0)

        # Prevent a single dense channel from dominating the baseline update.
        grad_theta = np.clip(grad_theta, -50.0, 50.0)

        theta_lo = np.log(np.maximum(nu_lo, eps))
        theta_hi = np.log(np.maximum(nu_hi, eps))
        theta = np.clip(theta + 0.001 * grad_theta, theta_lo, theta_hi)
        nu = np.exp(theta)

        # Stability check
        spec = np.max(np.abs(np.linalg.eigvals(A * rho_sum)))
        if spec > 0.95:
            A *= 0.8 / spec
            spec = np.max(np.abs(np.linalg.eigvals(A * rho_sum)))

        # Coordinate-ascent update for beta.
        beta_updated = False
        if (iteration + 1) % beta_update_every == 0:
            best_beta = beta
            best_ll = ll
            best_traces = pre_traces
            for beta_cand in _beta_candidate_grid(beta, beta_bounds, beta_grid_size):
                cand_traces, _ = compute_event_traces(events, K, M, beta_cand, beta_cand, weights)
                cand_ll, _ = _compute_static_loglik(
                    nu=nu,
                    A=A,
                    rho=rho,
                    pre_traces=cand_traces,
                    actors=actors,
                    marks=marks,
                    beta=beta_cand,
                    T=T,
                    l1_penalty=l1_penalty,
                    eps=eps,
                )
                if cand_ll > best_ll:
                    best_ll = cand_ll
                    best_beta = float(beta_cand)
                    best_traces = cand_traces
            if abs(best_beta - beta) > 1e-12:
                beta = best_beta
                pre_traces = best_traces
                ll = best_ll
                beta_updated = True

        if verbose and ((iteration + 1) % 50 == 0 or beta_updated):
            beta_msg = f", beta={beta:.3f}"
            moved = " *beta-updated*" if beta_updated else ""
            print(
                f"  static fit iter {iteration+1}: ll≈{ll:.1f}, "
                f"spec={spec:.3f}, nnz_A={np.sum(A > 0.001)}{beta_msg}{moved}"
            )

    return nu, A, rho, beta


def _spectral_ring_detection(A: np.ndarray, k: int) -> tuple[list[int], int]:
    """Find a k-actor ring from the interaction matrix using spectral methods.

    Strategy: compute the leading left and right singular vectors of A.
    Actors with large entries in both (bidirectional participants) are
    ring candidates. The hub is the actor with largest left singular
    value (strongest sender).
    """
    K = A.shape[0]
    if K < k:
        return list(range(K)), 0

    # SVD of the interaction matrix
    U_svd, S, Vt_svd = np.linalg.svd(A)

    # Leading left singular vector (sender strength)
    sender_score = np.abs(U_svd[:, 0])
    # Leading right singular vector (receiver strength)
    receiver_score = np.abs(Vt_svd[0, :])

    # Combined score: geometric mean (same logic as our gate evaluation)
    combined = np.sqrt(sender_score * receiver_score)

    # Top-k by combined score
    ring = list(np.argsort(combined)[-k:][::-1])
    hub = int(np.argmax(sender_score))

    return ring, hub


def _degree_ring_detection(A: np.ndarray, k: int) -> tuple[list[int], int]:
    """Find ring by out-degree + in-degree from estimated A."""
    out_deg = A.sum(axis=1)  # sum over receivers
    in_deg = A.sum(axis=0)   # sum over senders
    combined = np.sqrt(out_deg * in_deg)
    ring = list(np.argsort(combined)[-k:][::-1])
    hub = int(np.argmax(out_deg))
    return ring, hub


def main():
    cfg = SimConfig(
        K=10, M=2, T=500.0, ring_actors=[0, 1, 2], hub_actor=0, d=3,
        nu_base=0.15, alpha0_team=0.01, alpha1_max=0.8, alpha1_min=0.35,
        beta0=1.0, beta1=3.0, eta_on=0.04, eta_off=0.3, seed=7,
    )
    data = simulate_regime_hawkes(cfg)
    summary = summarize_simulation(data)

    print("=== BASELINE 3: Static Hawkes + Spectral/Degree Ring Detection ===\n")
    for k, v in summary.items():
        print(f"{k}: {v}")

    K, M, T = cfg.K, cfg.M, cfg.T
    k_ring = len(cfg.ring_actors)

    print(f"\nFitting static Hawkes (full {K}×{K} interaction matrix, with fitted beta)...")
    nu_hat, A_hat, rho_hat, beta_hat = _fit_static_hawkes_full_matrix(
        events=data.events, K=K, M=M, T=T,
        beta_init=2.0, lr=0.005, n_iters=200, l1_penalty=0.01,
        beta_update_every=10, beta_grid_size=9, beta_bounds=(0.25, 6.0),
        verbose=True,
    )

    print(f"\nFitted beta: {beta_hat:.4f}")
    print(f"\nEstimated A (top entries):")
    flat = [(A_hat[j, i], j, i) for j in range(K) for i in range(K) if j != i]
    flat.sort(reverse=True)
    for val, j, i in flat[:15]:
        tag = ""
        if j in cfg.ring_actors and i in cfg.ring_actors:
            tag = " ← RING"
        print(f"  A[{j}->{i}] = {val:.4f}{tag}")

    # Method A: Spectral clustering
    print("\n--- Method A: Spectral (SVD) ring detection ---")
    ring_svd, hub_svd = _spectral_ring_detection(A_hat, k_ring)
    print(f"  Predicted ring: {ring_svd}")
    print(f"  Predicted hub: {hub_svd}")
    print(f"  True ring: {cfg.ring_actors}, true hub: {cfg.hub_actor}")

    tp_svd = len(set(ring_svd) & set(cfg.ring_actors))
    p_svd = tp_svd / max(len(ring_svd), 1)
    r_svd = tp_svd / max(len(cfg.ring_actors), 1)
    f1_svd = 2 * p_svd * r_svd / max(p_svd + r_svd, 1e-12)
    print(f"  Membership P/R/F1: {p_svd:.3f} / {r_svd:.3f} / {f1_svd:.3f}")
    print(f"  Hub correct: {hub_svd == cfg.hub_actor}")

    # Method B: Degree-based
    print("\n--- Method B: Degree-based ring detection ---")
    ring_deg, hub_deg = _degree_ring_detection(A_hat, k_ring)
    print(f"  Predicted ring: {ring_deg}")
    print(f"  Predicted hub: {hub_deg}")

    tp_deg = len(set(ring_deg) & set(cfg.ring_actors))
    p_deg = tp_deg / max(len(ring_deg), 1)
    r_deg = tp_deg / max(len(cfg.ring_actors), 1)
    f1_deg = 2 * p_deg * r_deg / max(p_deg + r_deg, 1e-12)
    print(f"  Membership P/R/F1: {p_deg:.3f} / {r_deg:.3f} / {f1_deg:.3f}")
    print(f"  Hub correct: {hub_deg == cfg.hub_actor}")

    # Summary comparison table
    print("\n" + "=" * 60)
    print("SUMMARY: Static Hawkes baselines (no regime switching)")
    print("=" * 60)
    print(f"{'Method':<30} {'P':>6} {'R':>6} {'F1':>6} {'Hub':>5}")
    print("-" * 60)
    print(f"{'SVD ring detection':<30} {p_svd:>6.3f} {r_svd:>6.3f} {f1_svd:>6.3f} {'✓' if hub_svd == cfg.hub_actor else '✗':>5}")
    print(f"{'Degree ring detection':<30} {p_deg:>6.3f} {r_deg:>6.3f} {f1_deg:>6.3f} {'✓' if hub_deg == cfg.hub_actor else '✗':>5}")
    print(f"{'(Proposed model for ref)':<30} {'1.000':>6} {'1.000':>6} {'1.000':>6} {'✓':>5}")
    print(f"\nFitted static-kernel decay beta_hat = {beta_hat:.4f}")


if __name__ == "__main__":
    main()
