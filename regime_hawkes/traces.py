from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _preview_trace_rows(name: str, arr: np.ndarray, n_rows: int = 5, precision: int = 4) -> None:
    """Log the first few rows of a trace tensor after flattening (K, M) -> K*M."""
    if arr.size == 0:
        logger.debug("%s preview: empty", name)
        return
    rows = min(n_rows, arr.shape[0])
    flat = arr[:rows].reshape(rows, -1)
    preview = np.array2string(flat, precision=precision, suppress_small=True, max_line_width=200)
    logger.debug("%s preview first %d rows (flattened KxM):\n%s", name, rows, preview)


def _trace_mass_summary(name: str, arr: np.ndarray, precision: int = 4) -> None:
    """Compact summary of trace mass over intervals/events."""
    if arr.size == 0:
        logger.debug("%s summary: empty", name)
        return
    flat = arr.reshape(arr.shape[0], -1)
    row_sums = flat.sum(axis=1)
    row_max = flat.max(axis=1)
    logger.debug(
        "%s summary: row_sum[min=%.4f mean=%.4f max=%.4f] row_max[min=%.4f mean=%.4f max=%.4f]",
        name,
        float(row_sums.min()),
        float(row_sums.mean()),
        float(row_sums.max()),
        float(row_max.min()),
        float(row_max.mean()),
        float(row_max.max()),
    )


def compute_event_traces(
    events: np.ndarray,
    K: int,
    M: int,
    beta0: float,
    beta1: float,
    active_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return routine/active traces right before each event."""
    N = len(events)
    logger.debug(
        "compute_event_traces start: beta0=%.4f beta1=%.4f active_weight[min=%.4f mean=%.4f max=%.4f]",
        beta0,
        beta1,
        float(np.min(active_weights)) if len(active_weights) else 0.0,
        float(np.mean(active_weights)) if len(active_weights) else 0.0,
        float(np.max(active_weights)) if len(active_weights) else 0.0,
    )
    pre_B = np.zeros((N, K, M), dtype=float)
    pre_A = np.zeros((N, K, M), dtype=float)
    B = np.zeros((K, M), dtype=float)
    A = np.zeros((K, M), dtype=float)
    t_prev = 0.0

    for n, (t, actor, mark) in enumerate(events):
        dt = float(t) - t_prev
        if dt < 0:
            raise ValueError("events must be sorted by time")
        B *= np.exp(-beta0 * dt)
        A *= np.exp(-beta1 * dt)
        pre_B[n] = B
        pre_A[n] = A
        i = int(actor)
        m = int(mark)
        B[i, m] += beta0
        A[i, m] += beta1 * float(active_weights[n])
        t_prev = float(t)

    _trace_mass_summary("pre_B", pre_B)
    _trace_mass_summary("pre_A", pre_A)
    _preview_trace_rows("pre_B", pre_B)
    _preview_trace_rows("pre_A", pre_A)
    logger.debug(
        "compute_event_traces done: final_norms=(%.4f, %.4f)",
        float(np.linalg.norm(B)),
        float(np.linalg.norm(A)),
    )
    return pre_B, pre_A


def traces_at_interval_starts(
    events: np.ndarray,
    interval_edges: np.ndarray,
    K: int,
    M: int,
    beta0: float,
    beta1: float,
    active_weights: np.ndarray, # gamma(0) and gamma(1)
) -> tuple[np.ndarray, np.ndarray]:
    """Return traces at each interval start b (shape B x K x M)."""
    Bn = len(interval_edges) - 1
    logger.debug(
        "traces_at_interval_starts start: beta0=%.4f beta1=%.4f interval_width_mean=%.4f active_weight[min=%.4f mean=%.4f max=%.4f]",
        beta0,
        beta1,
        float(np.mean(np.diff(interval_edges))) if Bn > 0 else 0.0,
        float(np.min(active_weights)) if len(active_weights) else 0.0,
        float(np.mean(active_weights)) if len(active_weights) else 0.0,
        float(np.max(active_weights)) if len(active_weights) else 0.0,
    )
    out_B = np.zeros((Bn, K, M), dtype=float)
    out_A = np.zeros((Bn, K, M), dtype=float)
    B = np.zeros((K, M), dtype=float)
    A = np.zeros((K, M), dtype=float)

    n = 0
    t_prev = 0.0
    for b in range(Bn):
        start = float(interval_edges[b])
        while n < len(events) and float(events[n, 0]) < start:
            t, actor, mark = events[n]
            dt_n = float(t) - t_prev
            B *= np.exp(-beta0 * dt_n)
            A *= np.exp(-beta1 * dt_n)
            i, m = int(actor), int(mark)
            B[i, m] += beta0
            A[i, m] += beta1 * float(active_weights[n])
            t_prev = float(t)
            n += 1

        dt = start - t_prev
        B *= np.exp(-beta0 * dt)
        A *= np.exp(-beta1 * dt)
        t_prev = start
        out_B[b] = B
        out_A[b] = A

    _trace_mass_summary("out_B", out_B)
    _trace_mass_summary("out_A", out_A)
    _preview_trace_rows("out_B", out_B)
    _preview_trace_rows("out_A", out_A)
    logger.debug(
        "traces_at_interval_starts done: consumed_events=%d/%d last_trace_norms=(%.4f, %.4f)",
        n,
        len(events),
        float(np.linalg.norm(B)),
        float(np.linalg.norm(A)),
    )
    return out_B, out_A
