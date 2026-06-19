from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np

from regime_hawkes.traces import traces_at_interval_starts

logger = logging.getLogger(__name__)

FORENSIC_TOP_N = 8
FORENSIC_MAX_INTERVALS = 10
FORENSIC_MAX_EVENTS = 10
FORENSIC_TOP_ENTRIES = 5
FORENSIC_DELTA_POS_THRESH = 2.5
FORENSIC_DELTA_NEG_THRESH = -2.5
FORENSIC_ACTIVE_COMP_RATIO_THRESH = 1.0
FORENSIC_VERBOSE_EVENTS = False


def _format_interval_triplets(indices: np.ndarray, counts: np.ndarray, delta_ell: np.ndarray, n: int = 5) -> str:
    if len(indices) == 0:
        return "[]"
    out = []
    for b in indices[:n]:
        out.append(f"(b={int(b)}, events={int(counts[b])}, delta_ell={float(delta_ell[b]):.4f})")
    return "[" + ", ".join(out) + "]"


def _format_top_entries(arr: np.ndarray, top_k: int = FORENSIC_TOP_ENTRIES) -> str:
    flat = np.asarray(arr, dtype=float).ravel()
    if flat.size == 0 or np.allclose(flat, 0.0):
        return "[]"
    top_idx = np.argsort(flat)[::-1]
    out = []
    added = 0
    for idx in top_idx:
        val = float(flat[idx])
        if val <= 0.0:
            break
        actor, mark = np.unravel_index(int(idx), arr.shape)
        out.append(f"(actor={actor}, mark={mark}, val={val:.4f})")
        added += 1
        if added >= top_k:
            break
    return "[" + ", ".join(out) + "]"


def _format_actor_mark_counter(counter: Counter[tuple[int, int]], top_k: int = FORENSIC_TOP_ENTRIES) -> str:
    if not counter:
        return "[]"
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[:top_k]
    return "[" + ", ".join(
        f"(actor={actor}, mark={mark}, count={count})" for (actor, mark), count in items
    ) + "]"


def _format_event_details(event_details: list[dict[str, float]], max_events: int = FORENSIC_MAX_EVENTS) -> str:
    if not event_details:
        return "[]"
    shown = event_details[:max_events]
    parts = []
    for d in shown:
        parts.append(
            "(dt={dt:.4f}, actor={actor}, mark={mark}, lam0={lam0:.4f}, lam1={lam1:.4f}, "
            "routine={routine:.4f}, active_pre={active_pre:.4f}, active_within={active_within:.4f})".format(**d)
        )
    if len(event_details) > max_events:
        parts.append(f"... +{len(event_details) - max_events} more")
    return "[" + "; ".join(parts) + "]"


def _format_weighted_items(items: list[tuple[tuple[int, ...], float]], label: str) -> str:
    if not items:
        return f"{label}=[]"
    out = []
    for key, val in items[:FORENSIC_TOP_ENTRIES]:
        if len(key) == 1:
            out.append(f"({key[0]}:{val:.4f})")
        elif len(key) == 2:
            out.append(f"({key[0]}->{key[1]}:{val:.4f})")
        elif len(key) == 4:
            out.append(f"(({key[0]},{key[1]})->({key[2]},{key[3]}):{val:.4f})")
        else:
            out.append(f"({key}:{val:.4f})")
    return f"{label}=[" + ", ".join(out) + "]"


def _format_mark_routes(items: list[tuple[tuple[int, int], float]]) -> str:
    if not items:
        return "mark_routes=[]"
    out = [f"({src}->{dst}:{val:.4f})" for (src, dst), val in items[:FORENSIC_TOP_ENTRIES]]
    return "mark_routes=[" + ", ".join(out) + "]"


def _select_forensic_intervals(
    interval_event_counts: np.ndarray,
    delta_ell: np.ndarray,
    comp_shared: np.ndarray,
    comp_within_z1: np.ndarray,
    event_log1: np.ndarray,
    max_intervals: int = FORENSIC_MAX_INTERVALS,
) -> list[tuple[int, str, float]]:
    nonempty = np.where(interval_event_counts > 0)[0]
    if len(nonempty) == 0:
        return []

    candidates: list[tuple[float, int, str]] = []
    abs_event1 = np.maximum(np.abs(event_log1), 1e-9)
    active_ratio = comp_within_z1 / abs_event1

    for b in nonempty:
        if delta_ell[b] >= FORENSIC_DELTA_POS_THRESH:
            candidates.append((float(delta_ell[b]), int(b), "strong_active"))
        #if delta_ell[b] <= FORENSIC_DELTA_NEG_THRESH:
        #   candidates.append((float(-delta_ell[b]), int(b), "strong_dormant"))
        if active_ratio[b] >= FORENSIC_ACTIVE_COMP_RATIO_THRESH:
            candidates.append((float(active_ratio[b]), int(b), "active_compensator_large"))

    if not candidates:
        top_sep = nonempty[np.argsort(np.abs(delta_ell[nonempty]))[::-1][: min(max_intervals, len(nonempty))]]
        return [(int(b), "top_state_sep", float(delta_ell[b])) for b in top_sep]

    selected: list[tuple[int, str, float]] = []
    seen: set[int] = set()
    for score, b, reason in sorted(candidates, key=lambda x: x[0], reverse=True):
        if b in seen:
            continue
        selected.append((b, reason, score))
        seen.add(b)
        if len(selected) >= max_intervals:
            break
    return selected


def _log_interval_forensics(
    *,
    b: int,
    reason: str,
    score: float,
    interval_edges: np.ndarray,
    interval_event_counts: np.ndarray,
    ell: np.ndarray,
    gamma_prev_1: np.ndarray,
    event_log0: np.ndarray,
    event_log1: np.ndarray,
    comp_shared: np.ndarray,
    comp_within_routine: np.ndarray,
    comp_within_z1: np.ndarray,
    start_B: np.ndarray,
    start_A: np.ndarray,
    actor_mark_counters: list[Counter[tuple[int, int]]],
    event_details_by_interval: list[list[dict[str, float]]],
    active_comp_by_actor_mark: list[defaultdict[tuple[int, int], float]],
    sender_strength_by_interval: list[defaultdict[int, float]],
    receiver_strength_by_interval: list[defaultdict[int, float]],
    edge_strength_by_interval: list[defaultdict[tuple[int, int], float]],
    mark_route_strength_by_interval: list[defaultdict[tuple[int, int], float]],
    actor_mark_edge_strength_by_interval: list[defaultdict[tuple[int, int, int, int], float]],
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    delta = float(ell[b, 1] - ell[b, 0])
    abs_event1 = max(abs(float(event_log1[b])), 1e-9)
    shared_ratio = float(comp_shared[b]) / abs_event1
    active_ratio = float(comp_within_z1[b]) / abs_event1
    start = float(interval_edges[b])
    end = float(interval_edges[b + 1])

    comp_items = sorted(active_comp_by_actor_mark[b].items(), key=lambda kv: kv[1], reverse=True)
    comp_str = "[]"
    if comp_items:
        comp_str = "[" + ", ".join(
            f"(actor={actor}, mark={mark}, comp={val:.4f})"
            for (actor, mark), val in comp_items[:FORENSIC_TOP_ENTRIES]
        ) + "]"

    sender_items = sorted(sender_strength_by_interval[b].items(), key=lambda kv: kv[1], reverse=True)
    receiver_items = sorted(receiver_strength_by_interval[b].items(), key=lambda kv: kv[1], reverse=True)
    edge_items = sorted(edge_strength_by_interval[b].items(), key=lambda kv: kv[1], reverse=True)
    route_items = sorted(mark_route_strength_by_interval[b].items(), key=lambda kv: kv[1], reverse=True)
    actor_mark_edge_items = sorted(actor_mark_edge_strength_by_interval[b].items(), key=lambda kv: kv[1], reverse=True)

    logger.debug(
        "Interval forensic [%s]: b=%d score=%.4f range=(%.4f, %.4f] events=%d ell=(%.4f, %.4f) delta_ell=%.4f gamma_prev=%.4f",
        reason,
        b,
        score,
        start,
        end,
        int(interval_event_counts[b]),
        float(ell[b, 0]),
        float(ell[b, 1]),
        delta,
        float(gamma_prev_1[b]),
    )
    logger.debug(
        "Interval forensic [%s]: event_logs=(%.4f, %.4f) compensators shared=%.4f within_routine=%.4f within_z1=%.4f ratios shared=%.4f active=%.4f",
        reason,
        float(event_log0[b]),
        float(event_log1[b]),
        float(comp_shared[b]),
        float(comp_within_routine[b]),
        float(comp_within_z1[b]),
        shared_ratio,
        active_ratio,
    )
    '''logger.debug(
        "Interval forensic [%s]: actor-mark counts %s",
        reason,
        _format_actor_mark_counter(actor_mark_counters[b]),
    )
    logger.debug(
        "Interval forensic [%s]: start_B top entries %s",
        reason,
        _format_top_entries(start_B[b]),
    )
    logger.debug(
        "Interval forensic [%s]: start_A top entries %s",
        reason,
        _format_top_entries(start_A[b]),
    )
    logger.debug(
        "Interval forensic [%s]: graph summary %s | %s | %s | %s | overpredictors=%s",
        reason,
        _format_weighted_items([((actor,), val) for actor, val in sender_items], "senders"),
        _format_weighted_items([((actor,), val) for actor, val in receiver_items], "receivers"),
        _format_weighted_items(edge_items, "actor_edges"),
        _format_mark_routes(route_items),
        comp_str,
    )'''
    logger.debug(
        "Interval forensic [%s]: fine-grain edges %s",
        reason,
        _format_weighted_items(actor_mark_edge_items, "actor_mark_edges"),
    )
    if FORENSIC_VERBOSE_EVENTS:
        logger.debug(
            "Interval forensic [%s]: event details %s",
            reason,
            _format_event_details(event_details_by_interval[b]),
        )
        logger.debug(
            "Interval forensic [%s]: active compensator by actor-mark %s",
            reason,
            comp_str,
        )


def _diagnostic_interval_summary(
    *,
    interval_edges: np.ndarray,
    gamma_prev_1: np.ndarray,
    interval_event_counts: np.ndarray,
    event_log0: np.ndarray,
    event_log1: np.ndarray,
    comp_shared: np.ndarray,
    comp_within_routine: np.ndarray,
    comp_within_z1: np.ndarray,
    ell: np.ndarray,
    start_B: np.ndarray,
    start_A: np.ndarray,
    actor_mark_counters: list[Counter[tuple[int, int]]],
    event_details_by_interval: list[list[dict[str, float]]],
    active_comp_by_actor_mark: list[defaultdict[tuple[int, int], float]],
    sender_strength_by_interval: list[defaultdict[int, float]],
    receiver_strength_by_interval: list[defaultdict[int, float]],
    edge_strength_by_interval: list[defaultdict[tuple[int, int], float]],
    mark_route_strength_by_interval: list[defaultdict[tuple[int, int], float]],
    actor_mark_edge_strength_by_interval: list[defaultdict[tuple[int, int, int, int], float]],
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    bn = len(interval_event_counts)
    if bn == 0:
        logger.debug("Emission diagnostics: no intervals")
        return

    delta_ell = ell[:, 1] - ell[:, 0]
    nonempty = np.where(interval_event_counts > 0)[0]
    active_favor = np.where(delta_ell > 0)[0]
    dormant_favor = np.where(delta_ell < 0)[0]

    logger.debug(
        "Emission diagnostics: nonempty=%d/%d total_events=%d events_per_interval[min=%d mean=%.2f max=%d]",
        len(nonempty),
        bn,
        int(np.sum(interval_event_counts)),
        int(np.min(interval_event_counts)),
        float(np.mean(interval_event_counts)),
        int(np.max(interval_event_counts)),
    )

    logger.debug(
        "Emission diagnostics: delta_ell=ell1-ell0[min=%.4f mean=%.4f max=%.4f] active_favor=%d dormant_favor=%d tied=%d",
        float(np.min(delta_ell)),
        float(np.mean(delta_ell)),
        float(np.max(delta_ell)),
        len(active_favor),
        len(dormant_favor),
        int(np.sum(np.isclose(delta_ell, 0.0))),
    )

    logger.debug(
        "Emission diagnostics: event_log0 sum=%.4f event_log1 sum=%.4f shared_comp sum=%.4f within_routine sum=%.4f within_z1 sum=%.4f",
        float(np.sum(event_log0)),
        float(np.sum(event_log1)),
        float(np.sum(comp_shared)),
        float(np.sum(comp_within_routine)),
        float(np.sum(comp_within_z1)),
    )

    if len(nonempty):
        logger.debug(
            "Emission diagnostics: nonempty intervals preview %s",
            _format_interval_triplets(nonempty, interval_event_counts, delta_ell, n=FORENSIC_TOP_N),
        )

        by_events = nonempty[np.argsort(interval_event_counts[nonempty])[::-1]]
        logger.debug(
            "Emission diagnostics: top burst intervals %s",
            _format_interval_triplets(by_events, interval_event_counts, delta_ell, n=FORENSIC_TOP_N),
        )

        by_abs_sep = nonempty[np.argsort(np.abs(delta_ell[nonempty]))[::-1]]
        logger.debug(
            "Emission diagnostics: top state-separating intervals %s",
            _format_interval_triplets(by_abs_sep, interval_event_counts, delta_ell, n=FORENSIC_TOP_N),
        )

    if len(active_favor):
        active_rank = active_favor[np.argsort(delta_ell[active_favor])[::-1]]
        logger.debug(
            "Emission diagnostics: strongest active-favoring intervals %s",
            _format_interval_triplets(active_rank, interval_event_counts, delta_ell, n=FORENSIC_TOP_N),
        )

    if len(dormant_favor):
        dormant_rank = dormant_favor[np.argsort(delta_ell[dormant_favor])]
        logger.debug(
            "Emission diagnostics: strongest dormant-favoring intervals %s",
            _format_interval_triplets(dormant_rank, interval_event_counts, delta_ell, n=FORENSIC_TOP_N),
        )

    if len(nonempty):
        shared_ratio = comp_shared[nonempty] / np.maximum(np.abs(event_log1[nonempty]), 1e-9)
        within_ratio = comp_within_z1[nonempty] / np.maximum(np.abs(event_log1[nonempty]), 1e-9)
        logger.debug(
            "Emission diagnostics: nonempty compensator ratios shared/|event_log1|[min=%.4f mean=%.4f max=%.4f] active_within/|event_log1|[min=%.4f mean=%.4f max=%.4f]",
            float(np.min(shared_ratio)),
            float(np.mean(shared_ratio)),
            float(np.max(shared_ratio)),
            float(np.min(within_ratio)),
            float(np.mean(within_ratio)),
            float(np.max(within_ratio)),
        )

    forensic_intervals = _select_forensic_intervals(
        interval_event_counts=interval_event_counts,
        delta_ell=delta_ell,
        comp_shared=comp_shared,
        comp_within_z1=comp_within_z1,
        event_log1=event_log1,
        max_intervals=FORENSIC_MAX_INTERVALS,
    )
    if forensic_intervals:
        logger.debug(
            "Emission diagnostics: interval forensics selected %s",
            "[" + ", ".join(f"(b={b}, reason={reason}, score={score:.4f})" for b, reason, score in forensic_intervals) + "]",
        )
    for b, reason, score in forensic_intervals:
        _log_interval_forensics(
            b=b,
            reason=reason,
            score=score,
            interval_edges=interval_edges,
            interval_event_counts=interval_event_counts,
            ell=ell,
            gamma_prev_1=gamma_prev_1,
            event_log0=event_log0,
            event_log1=event_log1,
            comp_shared=comp_shared,
            comp_within_routine=comp_within_routine,
            comp_within_z1=comp_within_z1,
            start_B=start_B,
            start_A=start_A,
            actor_mark_counters=actor_mark_counters,
            event_details_by_interval=event_details_by_interval,
            active_comp_by_actor_mark=active_comp_by_actor_mark,
            sender_strength_by_interval=sender_strength_by_interval,
            receiver_strength_by_interval=receiver_strength_by_interval,
            edge_strength_by_interval=edge_strength_by_interval,
            mark_route_strength_by_interval=mark_route_strength_by_interval,
            actor_mark_edge_strength_by_interval=actor_mark_edge_strength_by_interval,
        )




def _prepare_events_for_interval_edges(
    events: np.ndarray,
    interval_edges: np.ndarray,
    k: int,
    m: int,
) -> tuple[np.ndarray, int]:
    """Return finite, in-window, time-sorted events for emission evaluation.

    This is deliberately defensive for CRCNS/real-data runs.  The old
    event_bins() helper clipped events outside the interval grid into the first
    or last interval.  If an event with t > interval_edges[-1] is clipped into
    the final interval, the within-interval compensator sees
        remaining = t_end - t_n < 0,
    which turns 1 - exp(-beta * remaining) into a huge negative number.  The
    log-likelihood then subtracts that negative compensator and creates the
    million-to-1e18 positive held-out emissions we observed.  We must drop
    out-of-window events before binning and keep event order monotone.
    """
    arr = np.asarray(events, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float), 0
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"events must be an N x >=3 array, got shape {arr.shape}")

    arr = arr[:, :3].copy()
    n0 = len(arr)
    lo = float(interval_edges[0])
    hi = float(interval_edges[-1])
    finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    in_time = (arr[:, 0] >= lo) & (arr[:, 0] <= hi)
    in_actor = (arr[:, 1] >= 0) & (arr[:, 1] < int(k))
    in_mark = (arr[:, 2] >= 0) & (arr[:, 2] < int(m))
    keep = finite & in_time & in_actor & in_mark
    dropped = int(n0 - int(np.sum(keep)))
    arr = arr[keep]
    if len(arr):
        # Sort by time, then actor, then mark so each interval loop has dt >= 0.
        order = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
        arr = arr[order]
    return arr, dropped

def event_bins(events: np.ndarray, interval_edges: np.ndarray) -> np.ndarray:
    if len(events) == 0:
        logger.debug("event_bins: no events to assign")
        return np.zeros(0, dtype=int)
    events = np.asarray(events, dtype=float)
    b = len(interval_edges) - 1
    idx = np.searchsorted(interval_edges, events[:, 0], side="right") - 1
    outside = int(np.sum((idx < 0) | (idx >= b)))
    if outside:
        logger.debug(
            "event_bins: %d/%d events outside interval grid [%.6f, %.6f]; clipping for legacy caller",
            outside,
            len(events),
            float(interval_edges[0]),
            float(interval_edges[-1]),
        )
    bins = np.clip(idx, 0, b - 1)
    logger.debug("event_bins: assigned %d events across %d intervals", len(events), b)
    return bins


def build_interval_emissions(
    events: np.ndarray,
    interval_edges: np.ndarray,
    gamma_prev_1: np.ndarray,
    nu: np.ndarray,
    A0: np.ndarray,
    A1: np.ndarray,
    rho0: np.ndarray,
    rho1: np.ndarray,
    beta0: float,
    beta1: float,
    eps: float = 1e-9,
) -> np.ndarray:
    """Per-interval log-likelihood contributions for z_b in {0,1}.

    z_b is shared by all events in interval I_b=(tau_{b-1}, tau_b].
    Past active traces are approximated with gamma_prev_1 for the intervals in
    which source events occurred. Within-interval active offspring are included
    only under z_b=1, while routine offspring are included under both states.
    """
    k, m = nu.shape
    bn = len(interval_edges) - 1
    raw_event_count = len(events)
    events, dropped_events = _prepare_events_for_interval_edges(events, interval_edges, k, m)
    ell = np.zeros((bn, 2), dtype=float)

    if dropped_events:
        logger.debug(
            "build_interval_emissions: dropped %d/%d out-of-window/nonfinite actor/mark events before binning; grid=[%.6f, %.6f]",
            dropped_events,
            raw_event_count,
            float(interval_edges[0]),
            float(interval_edges[-1]),
        )
        if __import__("os").environ.get("CRCNS_DEBUG_LIKELIHOOD") == "1":
            print(
                "[LIKDBG filter]",
                "raw_events=", int(raw_event_count),
                "kept=", int(len(events)),
                "dropped=", int(dropped_events),
                "grid0=", float(interval_edges[0]),
                "grid1=", float(interval_edges[-1]),
                flush=True,
            )

    logger.debug(
        "Build interval emissions start: raw_events=%d kept_events=%d intervals=%d K=%d M=%d beta0=%.4f beta1=%.4f",
        raw_event_count,
        len(events),
        bn,
        k,
        m,
        beta0,
        beta1,
    )

    interval_event_counts = np.zeros(bn, dtype=int)
    event_log0_by_interval = np.zeros(bn, dtype=float)
    event_log1_by_interval = np.zeros(bn, dtype=float)
    comp_shared_by_interval = np.zeros(bn, dtype=float)
    comp_within_routine_by_interval = np.zeros(bn, dtype=float)
    comp_within_z1_by_interval = np.zeros(bn, dtype=float)
    actor_mark_counters: list[Counter[tuple[int, int]]] = [Counter() for _ in range(bn)]
    event_details_by_interval: list[list[dict[str, float]]] = [[] for _ in range(bn)]
    active_comp_by_actor_mark: list[defaultdict[tuple[int, int], float]] = [defaultdict(float) for _ in range(bn)]
    sender_strength_by_interval: list[defaultdict[int, float]] = [defaultdict(float) for _ in range(bn)]
    receiver_strength_by_interval: list[defaultdict[int, float]] = [defaultdict(float) for _ in range(bn)]
    edge_strength_by_interval: list[defaultdict[tuple[int, int], float]] = [defaultdict(float) for _ in range(bn)]
    mark_route_strength_by_interval: list[defaultdict[tuple[int, int], float]] = [defaultdict(float) for _ in range(bn)]
    actor_mark_edge_strength_by_interval: list[defaultdict[tuple[int, int, int, int], float]] = [defaultdict(float) for _ in range(bn)]

    evt_bins = event_bins(events, interval_edges)
    evt_weights = gamma_prev_1[evt_bins] if len(events) else np.zeros(0, dtype=float)
    start_B, start_A = traces_at_interval_starts(
        events, interval_edges, k, m, beta0, beta1, evt_weights
    )
    logger.debug(
        "Interval start traces ready: start_B shape=%s start_A shape=%s",
        start_B.shape,
        start_A.shape,
    )

    for b in range(bn):
        t_start = float(interval_edges[b])
        t_end = float(interval_edges[b + 1])
        delta = t_end - t_start

        B_pre = start_B[b].copy()
        A_pre = start_A[b].copy()
        B_within = np.zeros((k, m), dtype=float)
        A_within_z1 = np.zeros((k, m), dtype=float)

        idx = np.where(evt_bins == b)[0]
        interval_event_counts[b] = len(idx)
        t_prev = t_start

        event_log0 = 0.0
        event_log1 = 0.0

        for n in idx:
            t_n = float(events[n, 0])
            i_n, m_n = int(events[n, 1]), int(events[n, 2])
            dt = t_n - t_prev
            if dt < -1e-9:
                raise ValueError(
                    f"events must be sorted within interval {b}; got negative dt={dt} at t={t_n}, previous={t_prev}"
                )
            dt = max(float(dt), 0.0)

            B_pre *= np.exp(-beta0 * dt)
            A_pre *= np.exp(-beta1 * dt)
            B_within *= np.exp(-beta0 * dt)
            A_within_z1 *= np.exp(-beta1 * dt)

            routine_n = A0.T @ ((B_pre + B_within) @ rho0)
            active_pre_full = A1.T @ (A_pre @ rho1)
            active_z1_full = A1.T @ ((A_pre + A_within_z1) @ rho1)
            active_within_full = active_z1_full - active_pre_full

            lam0 = max(nu[i_n, m_n] + routine_n[i_n, m_n] + active_pre_full[i_n, m_n], eps)
            lam1 = max(nu[i_n, m_n] + routine_n[i_n, m_n] + active_z1_full[i_n, m_n], eps)

            log_lam0 = np.log(lam0)
            log_lam1 = np.log(lam1)
            ell[b, 0] += log_lam0
            ell[b, 1] += log_lam1
            event_log0 += log_lam0
            event_log1 += log_lam1

            actor_mark_counters[b][(i_n, m_n)] += 1
            event_details_by_interval[b].append(
                {
                    "dt": float(t_n - t_start),
                    "actor": i_n,
                    "mark": m_n,
                    "lam0": float(lam0),
                    "lam1": float(lam1),
                    "routine": float(routine_n[i_n, m_n]),
                    "active_pre": float(active_pre_full[i_n, m_n]),
                    "active_within": float(active_within_full[i_n, m_n]),
                }
            )

            # Aggregate within-interval active graph contributions by sender, receiver, edge, and mark route.
            mixed_within = A_within_z1 @ rho1  # K x M
            if np.any(mixed_within):
                sender_contrib = A1[:, i_n] * mixed_within[:, m_n]
                for j, val in enumerate(sender_contrib):
                    val = float(val)
                    if val <= 0.0:
                        continue
                    sender_strength_by_interval[b][j] += val
                    receiver_strength_by_interval[b][i_n] += val
                    edge_strength_by_interval[b][(j, i_n)] += val
                for src_mark in range(m):
                    route_vec = A1[:, i_n] * A_within_z1[:, src_mark] * rho1[src_mark, m_n]
                    for j, val in enumerate(route_vec):
                        val = float(val)
                        if val <= 0.0:
                            continue
                        mark_route_strength_by_interval[b][(src_mark, m_n)] += val
                        actor_mark_edge_strength_by_interval[b][(j, src_mark, i_n, m_n)] += val

            B_within[i_n, m_n] += beta0
            A_within_z1[i_n, m_n] += beta1
            t_prev = t_n

        routine_comp = A0.T @ (start_B[b] @ rho0)
        active_comp = A1.T @ (start_A[b] @ rho1)
        if __import__("os").environ.get("CRCNS_DEBUG_LIKELIHOOD") == "1" and (b < 3 or b == bn - 1):
            try:
                print(
                    "[LIKDBG interval]",
                    "b=", int(b),
                    "n_idx=", int(len(idx)) if "idx" in locals() else -1,
                    "nu_sum=", float(np.nansum(nu)) if "nu" in locals() else float("nan"),
                    "active_pre_full_sum=", float(np.nansum(active_pre_full)) if "active_pre_full" in locals() else float("nan"),
                    "active_z1_full_sum=", float(np.nansum(active_z1_full)) if "active_z1_full" in locals() else float("nan"),
                    "active_within_full_sum=", float(np.nansum(active_within_full)) if "active_within_full" in locals() else float("nan"),
                    "active_comp_sum=", float(np.nansum(active_comp)),
                    "active_comp_max=", float(np.nanmax(active_comp)) if np.size(active_comp) else float("nan"),
                    flush=True,
                )
            except Exception as _crcns_dbg_exc:
                print("[LIKDBG interval failed]", repr(_crcns_dbg_exc), flush=True)
        decay0 = (1.0 - np.exp(-beta0 * delta)) / beta0
        decay1 = (1.0 - np.exp(-beta1 * delta)) / beta1
        base_comp = np.sum(nu) * delta
        comp_shared = base_comp + np.sum(routine_comp) * decay0 + np.sum(active_comp) * decay1

        comp_within_routine = 0.0
        comp_within_z1 = 0.0
        for n in idx:
            t_n = float(events[n, 0])
            i_n, m_n = int(events[n, 1]), int(events[n, 2])
            remaining = max(float(t_end - t_n), 0.0)

            impulse0 = np.zeros((k, m), dtype=float)
            impulse0[i_n, m_n] = beta0
            routine_inc = A0.T @ (impulse0 @ rho0)
            comp_within_routine += np.sum(routine_inc) * (1.0 - np.exp(-beta0 * remaining)) / beta0

            impulse1 = np.zeros((k, m), dtype=float)
            impulse1[i_n, m_n] = beta1
            active_inc = A1.T @ (impulse1 @ rho1)
            active_contrib = np.sum(active_inc) * (1.0 - np.exp(-beta1 * remaining)) / beta1
            comp_within_z1 += active_contrib
            active_comp_by_actor_mark[b][(i_n, m_n)] += float(active_contrib)

        ell[b, 0] -= comp_shared + comp_within_routine
        ell[b, 1] -= comp_shared + comp_within_routine + comp_within_z1

        event_log0_by_interval[b] = event_log0
        event_log1_by_interval[b] = event_log1
        comp_shared_by_interval[b] = comp_shared
        comp_within_routine_by_interval[b] = comp_within_routine
        comp_within_z1_by_interval[b] = comp_within_z1

    _diagnostic_interval_summary(
        interval_edges=interval_edges,
        gamma_prev_1=gamma_prev_1,
        interval_event_counts=interval_event_counts,
        event_log0=event_log0_by_interval,
        event_log1=event_log1_by_interval,
        comp_shared=comp_shared_by_interval,
        comp_within_routine=comp_within_routine_by_interval,
        comp_within_z1=comp_within_z1_by_interval,
        ell=ell,
        start_B=start_B,
        start_A=start_A,
        actor_mark_counters=actor_mark_counters,
        event_details_by_interval=event_details_by_interval,
        active_comp_by_actor_mark=active_comp_by_actor_mark,
        sender_strength_by_interval=sender_strength_by_interval,
        receiver_strength_by_interval=receiver_strength_by_interval,
        edge_strength_by_interval=edge_strength_by_interval,
        mark_route_strength_by_interval=mark_route_strength_by_interval,
        actor_mark_edge_strength_by_interval=actor_mark_edge_strength_by_interval,
    )

    logger.debug(
        "Build interval emissions done: total=(%.4f, %.4f)",
        float(np.sum(ell[:, 0])) if len(ell) else 0.0,
        float(np.sum(ell[:, 1])) if len(ell) else 0.0,
    )
    if __import__("os").environ.get("CRCNS_DEBUG_LIKELIHOOD") == "1":
        try:
            _ell_dbg = np.asarray(ell, dtype=float)
            print(
                "[LIKDBG return]",
                "shape=", _ell_dbg.shape,
                "finite=", bool(np.all(np.isfinite(_ell_dbg))),
                "min=", float(np.nanmin(_ell_dbg)) if _ell_dbg.size else float("nan"),
                "max=", float(np.nanmax(_ell_dbg)) if _ell_dbg.size else float("nan"),
                "sum0=", float(np.nansum(_ell_dbg[:,0])) if _ell_dbg.ndim == 2 and _ell_dbg.shape[1] > 0 else float("nan"),
                "sum1=", float(np.nansum(_ell_dbg[:,1])) if _ell_dbg.ndim == 2 and _ell_dbg.shape[1] > 1 else float("nan"),
                "argmax=", tuple(int(x) for x in np.unravel_index(np.nanargmax(_ell_dbg), _ell_dbg.shape)) if _ell_dbg.size else (),
                "dropped_events=", int(dropped_events),
                flush=True,
            )
        except Exception as _crcns_dbg_exc:
            print("[LIKDBG return failed]", repr(_crcns_dbg_exc), flush=True)
    return ell
