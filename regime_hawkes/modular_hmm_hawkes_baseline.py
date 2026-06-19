from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

try:
    from hawkes.likelihood import hawkes_log_likelihood
except ImportError:  # flat-file fallback
    from likelihood import hawkes_log_likelihood

try:
    from hmm.fb_poisson import forward_backward_poisson_hmm
except ImportError:  # flat-file fallback
    from fb_poisson import forward_backward_poisson_hmm


@dataclass(frozen=True)
class SegmentedHawkesFitResult:
    mu: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    log_likelihood: float
    success: bool
    message: str
    n_iter: int


@dataclass(frozen=True)
class PoissonHMMFitResult:
    transition: np.ndarray
    rates: np.ndarray
    init_probs: np.ndarray
    gamma: np.ndarray
    xi: np.ndarray
    log_likelihood: float
    n_iter: int
    converged: bool


@dataclass(frozen=True)
class ModularBaselineResult:
    field_auc: float
    membership_precision: float
    membership_recall: float
    membership_f1: float
    hub_correct: bool
    predicted_ring: list[int]
    hub_pred: int
    hmm_gamma: np.ndarray
    hmm_transition: np.ndarray
    hmm_rates: np.ndarray
    active_windows: list[tuple[float, float]]
    pair_integrated: np.ndarray
    sender_scores: np.ndarray
    receiver_scores: np.ndarray
    member_scores: np.ndarray
    n_pair_fits: int
    active_fraction: float


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------

def _interval_truth(Z_path: list[tuple[float, float, int]], interval_edges: np.ndarray) -> np.ndarray:
    mids = 0.5 * (interval_edges[:-1] + interval_edges[1:])
    truth = np.zeros(len(mids), dtype=int)
    for b, t in enumerate(mids):
        for s, e, z in Z_path:
            if s <= t < e:
                truth[b] = int(z)
                break
    return truth


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
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


def _active_event_truth(events: np.ndarray, z_at_event: np.ndarray, interval_edges: np.ndarray) -> np.ndarray:
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


def _topk_membership(score: np.ndarray, k: int) -> set[int]:
    if len(score) == 0 or k <= 0:
        return set()
    k_eff = min(int(k), len(score))
    idx = np.argsort(score)[-k_eff:]
    return set(idx.tolist())


def _gap_membership(score: np.ndarray, min_k: int = 1) -> set[int]:
    order = np.argsort(score)[::-1]
    sorted_scores = score[order]
    if len(sorted_scores) <= 1:
        return set(order[:min_k].tolist())
    gaps = sorted_scores[:-1] - sorted_scores[1:]
    best_split = int(np.argmax(gaps)) + 1
    best_split = max(best_split, min_k)
    return set(order[:best_split].tolist())


def _combine_actor_scores(sender: np.ndarray, receiver: np.ndarray, mode: str = "geom") -> np.ndarray:
    s = np.asarray(sender, dtype=float)
    r = np.asarray(receiver, dtype=float)

    if mode == "sum":
        return s + r
    if mode == "max":
        s_norm = s / max(float(np.max(s)), 1e-12)
        r_norm = r / max(float(np.max(r)), 1e-12)
        return np.maximum(s_norm, r_norm)
    if mode == "geom":
        s_norm = s / max(float(np.max(s)), 1e-12)
        r_norm = r / max(float(np.max(r)), 1e-12)
        return np.sqrt(s_norm * r_norm)
    raise ValueError(f"Unknown score mode: {mode}")


# -------------------------------------------------------------------
# Poisson HMM EM
# -------------------------------------------------------------------

def _reorder_hmm_states(
    transition: np.ndarray,
    rates: np.ndarray,
    init_probs: np.ndarray,
    gamma: np.ndarray,
    xi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(rates)
    transition = transition[order][:, order]
    rates = rates[order]
    init_probs = init_probs[order]
    gamma = gamma[:, order]
    if xi.size:
        xi = xi[:, order][:, :, order]
    return transition, rates, init_probs, gamma, xi



def fit_poisson_hmm_em(
    y: np.ndarray,
    n_restarts: int = 10,
    max_iters: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
) -> PoissonHMMFitResult:
    """Fit a 2-state Poisson HMM by EM and keep the best restart.

    State 0 is reordered to be the lower-rate state, state 1 the higher-rate state.
    """
    y = np.asarray(y, dtype=int)
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if np.any(y < 0):
        raise ValueError("y must be nonnegative")

    rng = np.random.default_rng(seed)
    y_mean = max(float(np.mean(y)), 1e-3)
    y_q25 = float(np.quantile(y, 0.25))
    y_q75 = float(np.quantile(y, 0.75))

    best: PoissonHMMFitResult | None = None

    for restart in range(n_restarts):
        if restart == 0:
            low = max(0.5 * y_mean, 1e-3)
            high = max(1.5 * y_mean, low + 1e-3)
            transition = np.array([[0.95, 0.05], [0.10, 0.90]], dtype=float)
            rates = np.array([low, high], dtype=float)
            init_probs = np.array([0.9, 0.1], dtype=float)
        else:
            low = max(y_q25 + 0.2 * y_mean * rng.uniform(-1.0, 1.0), 1e-3)
            high = max(y_q75 + 0.2 * y_mean * rng.uniform(-1.0, 1.0), low + 1e-3)
            p01 = rng.uniform(0.01, 0.20)
            p10 = rng.uniform(0.05, 0.40)
            transition = np.array([[1.0 - p01, p01], [p10, 1.0 - p10]], dtype=float)
            rates = np.array([low, high], dtype=float)
            init_probs = np.array([0.9, 0.1], dtype=float)

        prev_ll: float | None = None
        converged = False
        out: dict[str, Any] | None = None
        n_iter = 0

        for it in range(1, max_iters + 1):
            out = forward_backward_poisson_hmm(y, transition, rates, init_probs)
            ll = float(out["log_likelihood"])
            gamma = np.asarray(out["gamma"], dtype=float)
            xi = np.asarray(out["xi"], dtype=float)

            init_probs = np.clip(gamma[0], 1e-12, 1.0)
            init_probs = init_probs / init_probs.sum()

            denom = np.maximum(np.sum(gamma[:-1], axis=0), 1e-12)
            transition = np.sum(xi, axis=0) / denom[:, None]
            transition = np.clip(transition, 1e-12, 1.0)
            transition = transition / transition.sum(axis=1, keepdims=True)

            rates = np.sum(gamma * y[:, None], axis=0) / np.maximum(np.sum(gamma, axis=0), 1e-12)
            rates = np.clip(rates, 1e-3, None)

            n_iter = it
            if prev_ll is not None:
                rel = abs(ll - prev_ll) / (abs(prev_ll) + 1e-12)
                if rel < tol:
                    converged = True
                    break
            prev_ll = ll

        if out is None:
            continue

        transition, rates, init_probs, gamma, xi = _reorder_hmm_states(
            transition=np.asarray(transition, dtype=float),
            rates=np.asarray(rates, dtype=float),
            init_probs=np.asarray(init_probs, dtype=float),
            gamma=np.asarray(out["gamma"], dtype=float),
            xi=np.asarray(out["xi"], dtype=float),
        )
        ll = float(out["log_likelihood"])

        result = PoissonHMMFitResult(
            transition=transition,
            rates=rates,
            init_probs=init_probs,
            gamma=gamma,
            xi=xi,
            log_likelihood=ll,
            n_iter=n_iter,
            converged=converged,
        )
        if best is None or result.log_likelihood > best.log_likelihood:
            best = result

    if best is None:
        raise RuntimeError("Poisson HMM fitting failed across all restarts")
    return best


# -------------------------------------------------------------------
# Generic segmented Hawkes MLE
# -------------------------------------------------------------------

def _pack_hawkes(mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.concatenate([mu.ravel(), alpha.ravel(), beta.ravel()])



def _unpack_hawkes(x: np.ndarray, n_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu_end = n_dim
    alpha_end = mu_end + n_dim * n_dim
    mu = x[:mu_end]
    alpha = x[mu_end:alpha_end].reshape(n_dim, n_dim)
    beta = x[alpha_end:alpha_end + n_dim * n_dim].reshape(n_dim, n_dim)
    return mu, alpha, beta



def segmented_hawkes_log_likelihood(
    segments: list[tuple[np.ndarray, np.ndarray, float]],
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    total = 0.0
    for times, marks, T in segments:
        total += hawkes_log_likelihood(times=times, marks=marks, T=T, mu=mu, alpha=alpha, beta=beta)
    return float(total)



def fit_hawkes_segments_mle(
    segments: list[tuple[np.ndarray, np.ndarray, float]],
    n_dim: int,
    bounds_mu: tuple[float, float] = (1e-8, 10.0),
    bounds_alpha: tuple[float, float] = (1e-8, 10.0),
    bounds_beta: tuple[float, float] = (1e-4, 20.0),
    method: str = "L-BFGS-B",
) -> SegmentedHawkesFitResult:
    """Fit a single marked Hawkes model shared across multiple active windows.

    When n_dim = 2 * M, dimensions 0:(M-1) correspond to actor_i by true mark label,
    and dimensions M:(2M-1) correspond to actor_j by true mark label.
    """
    if not segments:
        raise ValueError("segments must be non-empty")
    if n_dim <= 0:
        raise ValueError("n_dim must be positive")

    total_T = float(sum(T for _, _, T in segments))
    if total_T <= 0:
        raise ValueError("Total observation time across segments must be positive")

    counts = np.zeros(n_dim, dtype=float)
    for _, marks, _ in segments:
        if len(marks):
            counts += np.bincount(marks, minlength=n_dim)

    init_mu = np.maximum(counts / total_T, 1e-3)
    init_alpha = np.full((n_dim, n_dim), 0.05, dtype=float)
    np.fill_diagonal(init_alpha, 0.02)
    init_beta = np.full((n_dim, n_dim), 1.0, dtype=float)

    x0 = _pack_hawkes(init_mu, init_alpha, init_beta)
    bounds = [bounds_mu] * n_dim + [bounds_alpha] * (n_dim * n_dim) + [bounds_beta] * (n_dim * n_dim)

    def objective(x: np.ndarray) -> float:
        mu, alpha, beta = _unpack_hawkes(x, n_dim=n_dim)
        ll = segmented_hawkes_log_likelihood(segments, mu, alpha, beta)
        return -ll if np.isfinite(ll) else 1e20

    res = minimize(objective, x0=x0, method=method, bounds=bounds)
    mu_hat, alpha_hat, beta_hat = _unpack_hawkes(res.x, n_dim=n_dim)

    return SegmentedHawkesFitResult(
        mu=mu_hat,
        alpha=alpha_hat,
        beta=beta_hat,
        log_likelihood=-float(res.fun),
        success=bool(res.success),
        message=str(res.message),
        n_iter=int(res.nit) if hasattr(res, "nit") else -1,
    )


# -------------------------------------------------------------------
# Event preprocessing for the modular pipeline
# -------------------------------------------------------------------

def counts_per_interval(events: np.ndarray, interval_edges: np.ndarray) -> np.ndarray:
    """Count all events in each interval (tau_{b-1}, tau_b]."""
    times = np.asarray(events[:, 0], dtype=float)
    bins = np.searchsorted(interval_edges, times, side="right") - 1
    valid = (bins >= 0) & (bins < len(interval_edges) - 1)
    counts = np.bincount(bins[valid], minlength=len(interval_edges) - 1)
    return counts.astype(int)



def active_windows_from_gamma(
    gamma_active: np.ndarray,
    interval_edges: np.ndarray,
    threshold: float = 0.5,
    min_active_bins: int = 1,
) -> list[tuple[float, float]]:
    active = np.asarray(gamma_active, dtype=float) >= float(threshold)
    if active.sum() < min_active_bins:
        if len(active):
            idx = int(np.argmax(gamma_active))
            active[idx] = True

    windows: list[tuple[float, float]] = []
    b = 0
    B = len(active)
    while b < B:
        if not active[b]:
            b += 1
            continue
        start = float(interval_edges[b])
        while b + 1 < B and active[b + 1]:
            b += 1
        end = float(interval_edges[b + 1])
        windows.append((start, end))
        b += 1
    return windows



def pair_segments_from_windows(
    events: np.ndarray,
    actor_i: int,
    actor_j: int,
    windows: list[tuple[float, float]],
    use_marks: bool = True,
    n_mark_labels: int | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], int]:
    """Create independent Hawkes segments for one actor pair.

    If use_marks is False, this reproduces the original 2D baseline:
      dimension 0 = actor_i, dimension 1 = actor_j.

    If use_marks is True, each actor-mark combination gets its own dimension:
      0, ..., M-1       = actor_i with true mark labels 0, ..., M-1
      M, ..., 2M-1      = actor_j with true mark labels 0, ..., M-1

    The segment clock is reset to 0 at the start of each window.
    """
    segments: list[tuple[np.ndarray, np.ndarray, float]] = []
    if not windows:
        return segments, (2 if not use_marks else 0)

    times = np.asarray(events[:, 0], dtype=float)
    actors = np.asarray(events[:, 1], dtype=int)
    true_marks = np.asarray(events[:, 2], dtype=int)

    if use_marks:
        if np.any(true_marks < 0):
            raise ValueError("mark labels must be nonnegative integers")
        M = int(np.max(true_marks)) + 1 if n_mark_labels is None else int(n_mark_labels)
        if M <= 0:
            raise ValueError("n_mark_labels must be positive")
        n_dim = 2 * M
    else:
        M = 1
        n_dim = 2

    for s, e in windows:
        mask = (times > s) & (times <= e) & ((actors == actor_i) | (actors == actor_j))
        seg_events = events[mask]
        if len(seg_events):
            order = np.argsort(seg_events[:, 0])
            seg_events = seg_events[order]
            seg_times = np.asarray(seg_events[:, 0] - s, dtype=float)
            seg_actors = np.asarray(seg_events[:, 1], dtype=int)
            if use_marks:
                seg_true_marks = np.asarray(seg_events[:, 2], dtype=int)
                if np.any(seg_true_marks >= M):
                    raise ValueError("Observed mark label exceeds n_mark_labels - 1")
                seg_marks = np.where(seg_actors == actor_i, seg_true_marks, M + seg_true_marks).astype(int)
            else:
                seg_marks = np.where(seg_actors == actor_i, 0, 1).astype(int)
        else:
            seg_times = np.zeros(0, dtype=float)
            seg_marks = np.zeros(0, dtype=int)
        segments.append((seg_times, seg_marks, float(e - s)))
    return segments, n_dim


# -------------------------------------------------------------------
# Main modular baseline
# -------------------------------------------------------------------

def run_modular_hmm_hawkes_baseline(
    events: np.ndarray,
    interval_edges: np.ndarray,
    true_ring: list[int],
    true_hub: int,
    Z_path: list[tuple[float, float, int]] | None = None,
    Z_at_event: np.ndarray | None = None,
    auc_target: str = "interval_state",
    hmm_threshold: float = 0.5,
    hmm_restarts: int = 10,
    hmm_max_iters: int = 200,
    hawkes_min_total_events: int = 4,
    selection: str = "topk",
    top_k: int | None = None,
    score_mode: str = "geom",
    use_marks: bool = True,
    n_mark_labels: int | None = None,
    seed: int = 0,
) -> ModularBaselineResult:
    """Run the modular baseline: Poisson HMM + segmented pairwise Hawkes.

    The default now preserves the true mark labels inside each pairwise Hawkes fit.
    That makes the pairwise stage mark-aware while keeping the pipeline modular.
    The HMM segmentation remains unchanged and is still uninformed by network structure.
    """
    events = np.asarray(events, dtype=float)
    if events.ndim != 2 or events.shape[1] < 3:
        raise ValueError("events must be an (N,3) array with columns (time, actor, mark)")

    K = int(np.max(events[:, 1])) + 1 if len(events) else 0
    if K <= 0:
        raise ValueError("No actors found in events")

    if use_marks:
        observed_marks = np.asarray(events[:, 2], dtype=int)
        if np.any(observed_marks < 0):
            raise ValueError("mark labels must be nonnegative integers")
        M = int(np.max(observed_marks)) + 1 if n_mark_labels is None else int(n_mark_labels)
        if M <= 0:
            raise ValueError("n_mark_labels must be positive")
    else:
        M = 1

    counts = counts_per_interval(events, interval_edges)
    hmm = fit_poisson_hmm_em(
        y=counts,
        n_restarts=hmm_restarts,
        max_iters=hmm_max_iters,
        seed=seed,
    )
    gamma_active = np.asarray(hmm.gamma[:, 1], dtype=float)
    windows = active_windows_from_gamma(gamma_active, interval_edges, threshold=hmm_threshold)

    pair_integrated = np.zeros((K, K), dtype=float)
    n_pair_fits = 0

    for i in range(K):
        for j in range(i + 1, K):
            segments, n_dim = pair_segments_from_windows(
                events,
                i,
                j,
                windows,
                use_marks=use_marks,
                n_mark_labels=M if use_marks else None,
            )
            total_events = int(sum(len(times) for times, _, _ in segments))
            counts_by_dim = np.zeros(n_dim, dtype=int)
            for _, marks, _ in segments:
                if len(marks):
                    counts_by_dim += np.bincount(marks, minlength=n_dim)

            if total_events < hawkes_min_total_events:
                continue

            if use_marks:
                counts_i = int(np.sum(counts_by_dim[:M]))
                counts_j = int(np.sum(counts_by_dim[M:]))
                if counts_i == 0 or counts_j == 0:
                    continue
            else:
                if counts_by_dim[0] == 0 or counts_by_dim[1] == 0:
                    continue

            fit = fit_hawkes_segments_mle(segments, n_dim=n_dim)
            #if not np.isfinite(fit.log_likelihood):
            #    continue
            if (not fit.success) or (not np.isfinite(fit.log_likelihood)):
                continue

            integrated = np.maximum(0.0, fit.alpha / np.maximum(fit.beta, 1e-12))
            if use_marks:
                # rows = targets, cols = sources
                # actor_i -> actor_j means source dims for actor_i and target dims for actor_j
                pair_integrated[i, j] = float(np.sum(integrated[M:, :M]))
                pair_integrated[j, i] = float(np.sum(integrated[:M, M:]))
            else:
                pair_integrated[i, j] = float(integrated[1, 0])
                pair_integrated[j, i] = float(integrated[0, 1])
            n_pair_fits += 1

    sender_scores = np.sum(pair_integrated, axis=1)
    receiver_scores = np.sum(pair_integrated, axis=0)
    member_scores = _combine_actor_scores(sender_scores, receiver_scores, mode=score_mode)

    truth = set(true_ring)
    if top_k is None:
        top_k = len(truth)

    if selection == "gap":
        pred = _gap_membership(member_scores, min_k=1)
    else:
        pred = _topk_membership(member_scores, top_k)

    tp = len(pred & truth)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(truth), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    hub_pred = int(np.argmax(sender_scores)) if len(sender_scores) else -1
    hub_correct = hub_pred == int(true_hub)

    if auc_target == "active_event" and Z_at_event is not None:
        z_true = _active_event_truth(events, Z_at_event, interval_edges)
        field_auc = _roc_auc_binary(z_true, gamma_active)
    elif Z_path is not None:
        z_true = _interval_truth(Z_path, interval_edges)
        field_auc = _roc_auc_binary(z_true, gamma_active)
    else:
        field_auc = float("nan")

    return ModularBaselineResult(
        field_auc=float(field_auc),
        membership_precision=float(precision),
        membership_recall=float(recall),
        membership_f1=float(f1),
        hub_correct=bool(hub_correct),
        predicted_ring=sorted(pred),
        hub_pred=hub_pred,
        hmm_gamma=np.asarray(hmm.gamma, dtype=float),
        hmm_transition=np.asarray(hmm.transition, dtype=float),
        hmm_rates=np.asarray(hmm.rates, dtype=float),
        active_windows=windows,
        pair_integrated=pair_integrated,
        sender_scores=sender_scores,
        receiver_scores=receiver_scores,
        member_scores=member_scores,
        n_pair_fits=n_pair_fits,
        active_fraction=float(np.mean(gamma_active)),
    )
