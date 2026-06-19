from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import csv

import numpy as np

from regime_hawkes.config import SimConfig


@dataclass
class SimulatedData:
    events: np.ndarray
    Z_path: list[tuple[float, float, int]]
    Z_at_event: np.ndarray
    config: SimConfig
    nu: np.ndarray
    rho0: np.ndarray
    rho1: np.ndarray
    A0: np.ndarray
    A1: np.ndarray
    gates_g: np.ndarray
    gates_h: np.ndarray
    U: np.ndarray
    V: np.ndarray
    # Added metadata only. These are derived from A1 after stability scaling
    # and do not affect simulation, RNG state, likelihoods, or fitted results.
    active_adjacency_true: np.ndarray | None = None
    active_edges_true: list[tuple[int, int]] | None = None
    active_edge_rule: str = "dense_subgroup"


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def simulate_ctmc_path(T: float, eta_on: float, eta_off: float, rng: np.random.Generator) -> list[tuple[float, float, int]]:
    t = 0.0
    state = 0
    path: list[tuple[float, float, int]] = []
    while t < T:
        rate = eta_on if state == 0 else eta_off
        dt = rng.exponential(1.0 / rate)
        t_next = min(T, t + dt)
        path.append((t, t_next, state))
        t = t_next
        if t < T:
            state = 1 - state
    return path


def state_at_time(path: list[tuple[float, float, int]], t: float) -> int:
    for s, e, z in path:
        if s <= t < e:
            return z
    return path[-1][2]


def _active_edge_mask(config: SimConfig) -> np.ndarray:
    """Return the planted directed active-edge mask.

    The default, ``dense_subgroup``, exactly preserves the original synthetic
    data-generating process used by the completed experiments: every ordered
    non-self pair inside ``ring_actors`` has positive active coupling.

    The sparse alternatives are for new directed-edge diagnostics only.
    """
    K = int(config.K)
    S = [int(x) for x in config.ring_actors]
    hub = int(config.hub_actor)
    rule = str(getattr(config, "active_edge_rule", "dense_subgroup"))

    mask = np.zeros((K, K), dtype=bool)

    if rule == "dense_subgroup":
        for j in S:
            for i in S:
                if i != j:
                    mask[j, i] = True

    elif rule == "cycle":
        for j, i in zip(S, S[1:] + S[:1]):
            if i != j:
                mask[j, i] = True

    elif rule == "hub_to_others":
        for i in S:
            if i != hub:
                mask[hub, i] = True

    elif rule == "hub_plus_cycle":
        for i in S:
            if i != hub:
                mask[hub, i] = True
        for j, i in zip(S, S[1:] + S[:1]):
            if i != j:
                mask[j, i] = True

    else:
        raise ValueError(
            f"Unknown active_edge_rule={rule!r}; expected one of "
            "'dense_subgroup', 'cycle', 'hub_to_others', 'hub_plus_cycle'."
        )

    np.fill_diagonal(mask, False)
    return mask


def active_edges_from_adjacency(A: np.ndarray, tol: float = 1e-12) -> list[tuple[int, int]]:
    """List directed non-self edges from a square adjacency/weight matrix."""
    A = np.asarray(A, dtype=float)
    K = A.shape[0]
    return [
        (int(j), int(i))
        for j in range(K)
        for i in range(K)
        if j != i and float(A[j, i]) > tol
    ]


def directed_edge_pairs(subgroup: Sequence[int]) -> list[tuple[int, int]]:
    """Ordered non-self pairs inside a planted subgroup.

    These are the candidate pairs for sparse directed-edge diagnostics.
    The function deliberately evaluates only within ``S^star x S^star``
    because the diagnostic is intended to separate directed-edge recovery
    from the harder member-selection problem.
    """
    S = [int(x) for x in subgroup]
    return [(j, i) for j in S for i in S if i != j]


def directed_edge_truth_rows(
    data: SimulatedData,
    *,
    benchmark: str,
    seed: int,
    subgroup: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Return truth rows for ordered within-subgroup directed-edge recovery.

    Each row has columns ``benchmark, seed, src, dst, y_true``. The truth is
    taken from ``data.active_adjacency_true``, which is derived from the planted
    active matrix after stability scaling. For sparse diagnostics, make sure the
    simulation was generated with ``active_edge_rule`` set to ``cycle``,
    ``hub_to_others``, or ``hub_plus_cycle``.
    """
    if data.active_adjacency_true is None:
        raise ValueError("SimulatedData.active_adjacency_true is missing.")
    if subgroup is None:
        subgroup = data.config.ring_actors

    rows: list[dict[str, Any]] = []
    for src, dst in directed_edge_pairs(subgroup):
        rows.append(
            {
                "benchmark": benchmark,
                "seed": int(seed),
                "src": int(src),
                "dst": int(dst),
                "y_true": int(data.active_adjacency_true[src, dst]),
            }
        )
    return rows


def directed_edge_score_rows(
    *,
    benchmark: str,
    method: str,
    seed: int,
    subgroup: Sequence[int],
    fitted_g: np.ndarray,
    fitted_h: np.ndarray,
    fitted_U: np.ndarray,
    fitted_V: np.ndarray,
) -> list[dict[str, Any]]:
    """Return score rows for ordered within-subgroup directed-edge recovery.

    Scores use the gauge-invariant active pair score
    ``g_j h_i softplus(u_j^T v_i)`` rather than raw gates or embeddings.
    Each row has columns ``benchmark, method, seed, src, dst, score``.
    """
    g = np.asarray(fitted_g, dtype=float)
    h = np.asarray(fitted_h, dtype=float)
    U = np.asarray(fitted_U, dtype=float)
    V = np.asarray(fitted_V, dtype=float)

    rows: list[dict[str, Any]] = []
    for src, dst in directed_edge_pairs(subgroup):
        score = float(g[src] * h[dst] * softplus(np.dot(U[src], V[dst])))
        rows.append(
            {
                "benchmark": benchmark,
                "method": method,
                "seed": int(seed),
                "src": int(src),
                "dst": int(dst),
                "score": score,
            }
        )
    return rows


def append_directed_edge_diagnostic_rows(
    *,
    score_path: str | Path,
    truth_path: str | Path,
    data: SimulatedData,
    benchmark: str,
    seed: int,
    subgroup: Sequence[int] | None = None,
    fitted_g: np.ndarray,
    fitted_h: np.ndarray,
    fitted_U: np.ndarray,
    fitted_V: np.ndarray,
    method: str = "Proposed",
) -> None:
    """Append truth and fitted-score rows for the sparse directed diagnostic.

    Call this once per completed model fit. It creates the CSV files and writes
    headers if they do not exist.
    """
    if subgroup is None:
        subgroup = data.config.ring_actors

    score_path = Path(score_path)
    truth_path = Path(truth_path)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)

    truth_rows = directed_edge_truth_rows(
        data, benchmark=benchmark, seed=seed, subgroup=subgroup
    )
    score_rows = directed_edge_score_rows(
        benchmark=benchmark,
        method=method,
        seed=seed,
        subgroup=subgroup,
        fitted_g=fitted_g,
        fitted_h=fitted_h,
        fitted_U=fitted_U,
        fitted_V=fitted_V,
    )

    def _append_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    _append_rows(
        score_path,
        ["benchmark", "method", "seed", "src", "dst", "score"],
        score_rows,
    )
    _append_rows(
        truth_path,
        ["benchmark", "seed", "src", "dst", "y_true"],
        truth_rows,
    )


def summarize_active_edge_rule(config: SimConfig) -> dict[str, Any]:
    """Summarize the planted within-subgroup directed-edge mask for a config."""
    mask = _active_edge_mask(config)
    pairs = directed_edge_pairs(config.ring_actors)
    positives = [(j, i) for (j, i) in pairs if bool(mask[j, i])]
    negatives = [(j, i) for (j, i) in pairs if not bool(mask[j, i])]
    return {
        "active_edge_rule": str(getattr(config, "active_edge_rule", "dense_subgroup")),
        "subgroup": [int(x) for x in config.ring_actors],
        "n_pairs": int(len(pairs)),
        "n_edges": int(len(positives)),
        "n_non_edges": int(len(negatives)),
        "edges": positives,
        "non_edges": negatives,
    }


def _build_ground_truth(config: SimConfig, rng: np.random.Generator):
    K, M, d = config.K, config.M, config.d
    nu = np.full((K, M), config.nu_base)

    rho0 = np.full((M, M), 1.0 / M)
    if M == 2:
        rho1 = np.array([[0.5, 1.5], [0.3, 0.7]], dtype=float)
        rho1 = rho1 / rho1.sum(axis=1, keepdims=True)
    else:
        rho1 = np.eye(M, dtype=float)

    A0 = np.zeros((K, K), dtype=float)
    for j in range(K):
        team_j = j // config.team_size
        for i in range(K):
            if i == j:
                continue
            if i // config.team_size == team_j:
                A0[j, i] = config.alpha0_team

    g = np.zeros(K, dtype=float)
    h = np.zeros(K, dtype=float)
    g[config.ring_actors] = 1.0
    h[config.ring_actors] = 1.0

    U = rng.normal(scale=0.2, size=(K, d))
    V = rng.normal(scale=0.2, size=(K, d))
    A1 = np.zeros((K, K), dtype=float)
    active_mask = _active_edge_mask(config)
    for j in range(K):
        for i in range(K):
            if not active_mask[j, i]:
                continue
            if j == config.hub_actor:
                base = config.alpha1_max
            else:
                base = config.alpha1_min
            A1[j, i] = base * softplus(np.dot(U[j], V[i])) / softplus(1.0)

    spec = max(np.linalg.eigvals(A0 + A1).real)
    if spec >= 0.98:
        scale = 0.98 / spec
        A0 *= scale
        A1 *= scale

    return nu, rho0, rho1, A0, A1, g, h, U, V


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def simulate_regime_hawkes(config: SimConfig) -> SimulatedData:
    rng = np.random.default_rng(config.seed)
    nu, rho0, rho1, A0, A1, g, h, U, V = _build_ground_truth(config, rng)
    Z_path = simulate_ctmc_path(config.T, config.eta_on, config.eta_off, rng)

    K, M = config.K, config.M
    t = 0.0
    B = np.zeros((K, M), dtype=float)
    A = np.zeros((K, M), dtype=float)
    events: list[tuple[float, int, int]] = []
    z_labels: list[int] = []

    max_jump = 0.0
    for src in range(K):
        for m_prev in range(M):
            inc0 = np.sum(A0[src][:, None] * rho0[m_prev][None, :] * config.beta0)
            inc1 = np.sum(A1[src][:, None] * rho1[m_prev][None, :] * config.beta1)
            max_jump = max(max_jump, inc0 + inc1)

    while t < config.T:
        routine = A0.T @ (B @ rho0)
        active = A1.T @ (A @ rho1)
        lam = nu + routine + active
        lam_sum = float(lam.sum())
        upper = lam_sum + max_jump + 1e-8

        t_candidate = t + rng.exponential(1.0 / upper)
        if t_candidate > config.T:
            break

        dt = t_candidate - t
        B *= np.exp(-config.beta0 * dt)
        A *= np.exp(-config.beta1 * dt)

        routine_c = A0.T @ (B @ rho0)
        active_c = A1.T @ (A @ rho1)
        lam_c = nu + routine_c + active_c
        true_total = float(lam_c.sum())

        if rng.uniform() <= true_total / upper:
            flat = lam_c.reshape(-1)
            idx = int(rng.choice(flat.size, p=flat / flat.sum()))
            actor = idx // M
            mark = idx % M
            z_evt = state_at_time(Z_path, t_candidate)
            events.append((t_candidate, actor, mark))
            z_labels.append(z_evt)
            B[actor, mark] += config.beta0
            if z_evt == 1:
                A[actor, mark] += config.beta1

        t = t_candidate

    arr = np.array(events, dtype=float) if events else np.zeros((0, 3), dtype=float)
    z_arr = np.array(z_labels, dtype=int)
    return SimulatedData(
        events=arr,
        Z_path=Z_path,
        Z_at_event=z_arr,
        config=config,
        nu=nu,
        rho0=rho0,
        rho1=rho1,
        A0=A0,
        A1=A1,
        gates_g=g,
        gates_h=h,
        U=U,
        V=V,
        active_adjacency_true=(A1 > 1e-12).astype(int),
        active_edges_true=active_edges_from_adjacency(A1),
        active_edge_rule=str(getattr(config, "active_edge_rule", "dense_subgroup")),
    )


def summarize_simulation(data: SimulatedData) -> dict[str, np.ndarray | float | int]:
    events = data.events
    K = data.config.K
    M = data.config.M
    by_actor = np.bincount(events[:, 1].astype(int), minlength=K) if len(events) else np.zeros(K, dtype=int)
    by_mark = np.bincount(events[:, 2].astype(int), minlength=M) if len(events) else np.zeros(M, dtype=int)
    active_intervals = [p for p in data.Z_path if p[2] == 1]
    active_time = float(sum(e - s for s, e, _ in active_intervals))
    return {
        "total_events": int(len(events)),
        "events_per_actor": by_actor,
        "events_per_mark": by_mark,
        "active_episodes": len(active_intervals),
        "active_time": active_time,
        "pi_1": active_time / data.config.T,
        "spectral_radius": spectral_radius(data.A0 + data.A1),
    }
