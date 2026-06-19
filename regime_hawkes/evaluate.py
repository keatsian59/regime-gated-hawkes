"""Evaluation metrics for regime-switching Hawkes recovery."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalResult:
    field_auc: float
    membership_precision: float
    membership_recall: float
    membership_f1: float
    hub_correct: bool
    gate_values: np.ndarray
    predicted_ring: list[int]
    method: str  # "topk" or "gap"

import numpy as np


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -50.0, 50.0))
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_matrix(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return A.copy()
    A = np.maximum(A, 0.0)
    A = A.copy()
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        np.fill_diagonal(A, 0.0)
    return A / max(float(np.max(A)), 1e-12)


def _safe_logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return np.log(p / (1.0 - p))


def _posterior_ring_membership(
    g: np.ndarray,
    h: np.ndarray,
    active_matrix: np.ndarray,
    seed_idx: int | None = None,
    posterior_threshold: float = 0.5,
    beta_gate: float = 1.0,
    beta_affinity: float = 2.0,
    bias: float = 0.0,
    min_members: int = 1,
    max_members: int | None = None,
) -> set[int]:
    """Grow a ring by posterior-style conditional inclusion.

    Membership is evaluated relative to the ring built so far.

    Prior term:
        combined gate support score_j = sqrt(g_norm[j] * h_norm[j])

    Evidence term:
        compare actor j's average active coupling to the current ring
        against its average active coupling to actors outside the ring.

    Inclusion log-odds:
        logit P(j in ring | current ring) =
            bias
            + beta_gate * logit(score_j)
            + beta_affinity * log((ring_affinity_j + eps) / (background_affinity_j + eps))

    Greedy rule:
        start from a seed actor, then repeatedly add the remaining actor with the
        largest posterior inclusion probability, provided it exceeds the threshold.
    """
    g_norm = _normalize_gate(g)
    h_norm = _normalize_gate(h)
    gate_score = np.sqrt(g_norm * h_norm)

    A = _normalize_matrix(active_matrix)
    n = len(gate_score)
    if n == 0:
        return set()

    if max_members is None:
        max_members = n

    if seed_idx is None:
        # hub-like seed
        seed_idx = int(np.argmax(g)) if len(g) else 0

    S = {seed_idx}
    remaining = set(range(n)) - S
    eps = 1e-12
    global_bg = float(np.mean(A)) if A.size else eps

    while remaining and len(S) < max_members:
        S_list = sorted(S)
        best_j = None
        best_post = -1.0

        for j in remaining:
            # affinity to current ring
            ring_in = float(np.mean(A[S_list, j])) if S_list else 0.0
            ring_out = float(np.mean(A[j, S_list])) if S_list else 0.0
            ring_aff = 0.5 * (ring_in + ring_out)

            # affinity to outside actors
            outside = sorted(list(remaining - {j}))
            if outside:
                bg_in = float(np.mean(A[outside, j]))
                bg_out = float(np.mean(A[j, outside]))
                bg_aff = 0.5 * (bg_in + bg_out)
            else:
                bg_aff = global_bg

            log_odds = (
                bias
                + beta_gate * _safe_logit(gate_score[j])
                + beta_affinity * np.log((ring_aff + eps) / (bg_aff + eps))
            )
            post = _sigmoid(log_odds)

            if post > best_post:
                best_post = post
                best_j = j

        if best_j is None:
            break

        if best_post >= posterior_threshold or len(S) < int(min_members):
            S.add(best_j)
            remaining.remove(best_j)
        else:
            break

    return S

# ── AUC ────────────────────────────────────────────────────────────
def _interval_truth_midpoint(Z_path, interval_edges):
    mids = 0.5 * (interval_edges[:-1] + interval_edges[1:])
    truth = np.zeros(len(mids), dtype=int)
    for b, t in enumerate(mids):
        for s, e, z in Z_path:
            if s <= t < e:
                truth[b] = z
                break
    return truth


def _interval_truth_active_event(events, Z_at_event, interval_edges):
    """Binary interval labels aligned with event-time gating.

    An interval is labeled active iff it contains at least one event that was
    born while the latent field was active. This matches the event-time-gated
    model more closely than midpoint state labels.
    """
    bn = len(interval_edges) - 1
    truth = np.zeros(bn, dtype=int)
    if events is None or Z_at_event is None or len(events) == 0:
        return truth

    idx = np.searchsorted(interval_edges, np.asarray(events)[:, 0], side="right") - 1
    idx = np.clip(idx, 0, bn - 1)
    active_idx = idx[np.asarray(Z_at_event, dtype=int) == 1]
    if len(active_idx):
        truth[np.unique(active_idx)] = 1
    return truth


def _build_auc_truth(
    interval_edges,
    Z_path,
    events=None,
    Z_at_event=None,
    auc_target: str = "active_event",
):
    if auc_target == "active_event":
        if events is not None and Z_at_event is not None:
            return _interval_truth_active_event(events, Z_at_event, interval_edges)
        return _interval_truth_midpoint(Z_path, interval_edges)
    if auc_target == "midpoint":
        return _interval_truth_midpoint(Z_path, interval_edges)
    raise ValueError(f"Unknown auc_target: {auc_target}")


def _roc_auc_binary(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


# ── Membership selection ───────────────────────────────────────────
def _normalize_gate(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    return x / max(float(np.max(x)), 1e-12)



def _topk_membership(g, h, k):
    """Select top-k actors by combined sender/receiver support."""
    if len(g) == 0 or k <= 0:
        return set()
    g_norm = _normalize_gate(g)
    h_norm = _normalize_gate(h)
    score = np.sqrt(g_norm * h_norm)
    k_eff = min(int(k), len(score))
    idx = np.argsort(score)[-k_eff:]
    return set(idx.tolist())



def _largest_gap_cutoff(x: np.ndarray, min_k: int = 1, use_log: bool = False) -> set[int]:
    """Select indices above the largest descending gap.

    When ``use_log`` is True, the gap is computed on log-scale normalized values,
    which makes the cutoff target the separation between supported actors and the
    numerical noise floor rather than the separation between a dominant hub and
    the rest of the ring.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return set()

    order = np.argsort(x)[::-1]
    sorted_scores = x[order]
    if len(sorted_scores) <= 1:
        split = min(max(int(min_k), 1), len(sorted_scores))
        return set(order[:split].tolist())

    gap_scores = np.log(np.maximum(sorted_scores, 1e-12)) if use_log else sorted_scores
    gaps = gap_scores[:-1] - gap_scores[1:]
    best_split = int(np.argmax(gaps)) + 1
    best_split = max(best_split, int(min_k))
    best_split = min(best_split, len(sorted_scores))
    return set(order[:best_split].tolist())

def _mixture_membership(
    g: np.ndarray,
    h: np.ndarray,
    min_k: int = 1,
    max_iter: int = 200,
    tol: float = 1e-6,
    posterior_threshold: float = 0.8, # high threshold
) -> set[int]:
    """Estimate membership from a 2-component Gaussian mixture on log combined support.

    The combined support score is
        score_j = sqrt(g_norm[j] * h_norm[j]),
    where g_norm and h_norm are max-normalized sender and receiver gates.

    We fit a 2-component 1D Gaussian mixture to log(score_j + eps). The component
    with the larger mean is treated as the supported-actor component, and actors
    are selected if their posterior probability of belonging to that component
    exceeds ``posterior_threshold``.

    Args:
        g, h: sender and receiver gate vectors.
        min_k: ensure at least this many actors are returned by falling back to the
               top combined scores if the posterior rule selects too few.
        max_iter: max EM iterations for the 1D mixture.
        tol: convergence tolerance on log-likelihood.
        posterior_threshold: posterior cutoff for assigning an actor to the
                             supported component.

    Returns:
        Set of predicted ring-member indices.
    """
    g_norm = _normalize_gate(g)
    h_norm = _normalize_gate(h)
    score = np.sqrt(g_norm * h_norm)

    if score.size == 0:
        return set()

    eps = 1e-12
    x = np.log(np.maximum(score, eps))

    # Degenerate small cases
    if x.size == 1:
        return {0}

    # Robust initialization from lower/upper halves
    order = np.argsort(x)
    mid = max(1, len(x) // 2)
    lo = x[order[:mid]]
    hi = x[order[mid:]]

    mu = np.array([
        float(np.mean(lo)) if lo.size else float(np.min(x)),
        float(np.mean(hi)) if hi.size else float(np.max(x)),
    ])
    var = np.array([
        float(np.var(lo)) + 1e-4 if lo.size else 1e-2,
        float(np.var(hi)) + 1e-4 if hi.size else 1e-2,
    ])
    pi = np.array([0.5, 0.5])

    def _log_normal_pdf(xv, mean, variance):
        variance = max(float(variance), 1e-6)
        return -0.5 * (
            np.log(2.0 * np.pi * variance) + ((xv - mean) ** 2) / variance
        )

    prev_ll = -np.inf
    for _ in range(max_iter):
        # E-step
        log_r0 = np.log(max(pi[0], 1e-8)) + _log_normal_pdf(x, mu[0], var[0])
        log_r1 = np.log(max(pi[1], 1e-8)) + _log_normal_pdf(x, mu[1], var[1])

        m = np.maximum(log_r0, log_r1)
        denom = m + np.log(np.exp(log_r0 - m) + np.exp(log_r1 - m))
        r0 = np.exp(log_r0 - denom)
        r1 = np.exp(log_r1 - denom)

        # M-step
        N0 = max(float(np.sum(r0)), 1e-8)
        N1 = max(float(np.sum(r1)), 1e-8)

        pi[0] = N0 / len(x)
        pi[1] = N1 / len(x)

        mu[0] = float(np.sum(r0 * x) / N0)
        mu[1] = float(np.sum(r1 * x) / N1)

        var[0] = float(np.sum(r0 * (x - mu[0]) ** 2) / N0) + 1e-6
        var[1] = float(np.sum(r1 * (x - mu[1]) ** 2) / N1) + 1e-6

        ll = float(np.sum(denom))
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # Supported component = one with larger mean log-score
    signal_comp = int(np.argmax(mu))
    signal_post = r1 if signal_comp == 1 else r0

    pred = {i for i, p in enumerate(signal_post) if p >= posterior_threshold}

    # Ensure at least min_k actors selected
    if len(pred) < int(min_k):
        k_eff = min(max(int(min_k), 1), len(score))
        top_idx = np.argsort(score)[-k_eff:]
        pred = set(top_idx.tolist())

    return pred

def _gap_membership(g, h, min_k=1):
    """Estimate membership from combined sender/receiver support."""
    g_norm = _normalize_gate(g)
    h_norm = _normalize_gate(h)
    score = np.sqrt(g_norm * h_norm)
    return _largest_gap_cutoff(score, min_k=min_k, use_log=True)
'''
def _gap_membership(g, h, min_k=1):
    """Estimate membership from sender and receiver supports separately.

    We first normalize sender and receiver gates separately. We then apply the
    gap rule to each role on a log scale, which better isolates the split
    between supported actors and the near-zero tail. Final membership is the
    intersection of the sender-supported and receiver-supported sets. If that
    intersection is empty, we fall back to the union so the selector remains
    well-defined in highly asymmetric settings.
    """
    g_norm = _normalize_gate(g)
    h_norm = _normalize_gate(h)

    sender_set = _largest_gap_cutoff(g_norm, min_k=min_k, use_log=True)
    receiver_set = _largest_gap_cutoff(h_norm, min_k=min_k, use_log=True)

    pred = sender_set & receiver_set
    if pred:
        return pred
    return sender_set | receiver_set
'''

# ── Main evaluate ──────────────────────────────────────────────────
def evaluate(
    gamma: np.ndarray,
    interval_edges: np.ndarray,
    Z_path: list,
    g: np.ndarray,
    h: np.ndarray,
    true_ring: list[int],
    true_hub: int,
    top_k: int | None = None,
    method: str = "posterior_ring",
    active_matrix: np.ndarray | None = None,
    events: np.ndarray | None = None,
    Z_at_event: np.ndarray | None = None,
    auc_target: str = "active_event",
) -> EvalResult:
    """Evaluate ring recovery.

    Args:
        method:
            "topk"           -> oracle top-k on combined gate support
            "gap"            -> separate sender/receiver gap rule
            "posterior_ring" -> greedy conditional ring-growth selector
                                using active_matrix
        auc_target:
            "active_event"   -> interval is positive if it contains at least
                                one active-born event
            "midpoint"       -> interval truth from midpoint CTMC state
    """
    z_true = _build_auc_truth(
        interval_edges=interval_edges,
        Z_path=Z_path,
        events=events,
        Z_at_event=Z_at_event,
        auc_target=auc_target,
    )
    auc = _roc_auc_binary(z_true, gamma[:, 1])

    truth = set(true_ring)

    if method == "topk":
        if top_k is None:
            top_k = len(truth)
        pred = _topk_membership(g, h, top_k)
    elif method == "mixture":
        pred = _mixture_membership(g, h, min_k=1)
    elif method == "gap":
        pred = _gap_membership(g, h, min_k=1)
    elif method == "posterior_ring":
        if active_matrix is None:
            raise ValueError(
                "method='posterior_ring' requires active_matrix, e.g. "
                "the fitted alpha^(1) matrix."
            )
        pred = _posterior_ring_membership(
            g=g,
            h=h,
            active_matrix=active_matrix,
            seed_idx=int(np.argmax(g)) if len(g) else None,
            posterior_threshold=0.8,
            beta_gate=1.0,
            beta_affinity=2.0,
            bias=0.0,
            min_members=1,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    tp = len(pred & truth)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(truth), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    hub_pred = int(np.argmax(g)) if len(g) else -1

    return EvalResult(
        field_auc=float(auc),
        membership_precision=float(precision),
        membership_recall=float(recall),
        membership_f1=float(f1),
        hub_correct=hub_pred == true_hub,
        gate_values=np.array(g, dtype=float),
        predicted_ring=sorted(pred),
        method=method,
    )