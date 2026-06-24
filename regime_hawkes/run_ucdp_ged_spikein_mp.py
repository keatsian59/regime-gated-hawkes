from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os

# Prevent BLAS/OpenMP thread oversubscription under ProcessPoolExecutor.
# This must run before importing numpy/pandas. Users can override by setting
# these variables in the shell before invoking the script.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from regime_hawkes.config import SimConfig
from regime_hawkes.em import run_em
from regime_hawkes.evaluate import evaluate
from regime_hawkes.mstep import MStepResult
from regime_hawkes.utils import compute_A1

try:
    from regime_hawkes.run_baseline_static_hawkes import run_static_hawkes_baseline
except Exception as exc:  # pragma: no cover
    run_static_hawkes_baseline = None
    STATIC_IMPORT_ERROR = exc
else:
    STATIC_IMPORT_ERROR = None

try:
    from regime_hawkes.modular_hmm_hawkes_baseline import run_modular_hmm_hawkes_baseline
except Exception as exc:  # pragma: no cover
    run_modular_hmm_hawkes_baseline = None
    MODULAR_IMPORT_ERROR = exc
else:
    MODULAR_IMPORT_ERROR = None

try:
    from regime_hawkes.run_baseline_spectral import _fit_static_hawkes_full_matrix, _spectral_ring_detection
except Exception as exc:  # pragma: no cover
    _fit_static_hawkes_full_matrix = None
    _spectral_ring_detection = None
    SPECTRAL_IMPORT_ERROR = exc
else:
    SPECTRAL_IMPORT_ERROR = None


@dataclass
class UCDPData:
    events: np.ndarray
    Z_path: list[tuple[float, float, int]]
    Z_at_event: np.ndarray
    config: SimConfig
    stream_names: list[str]
    injected_mask: np.ndarray
    true_receivers: list[int]
    weak_receiver: int
    background_count: int
    injected_count: int


def softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def empirical_mark_init(events: np.ndarray, M: int, eps: float = 1.0) -> np.ndarray:
    counts = np.full((M, M), eps, dtype=float)
    if len(events) <= 1:
        return counts / counts.sum(axis=1, keepdims=True)

    # Sort by actor, then time. Consecutive rows with the same actor define
    # within-actor mark transitions. Vectorized np.add.at replaces a Python loop
    # over all events, which is noticeable when called once per fit.
    ev = events[np.lexsort((events[:, 0], events[:, 1]))]
    actors = ev[:, 1].astype(int, copy=False)
    marks = ev[:, 2].astype(int, copy=False)
    same_actor = actors[1:] == actors[:-1]
    if np.any(same_actor):
        prev_marks = marks[:-1][same_actor]
        next_marks = marks[1:][same_actor]
        np.add.at(counts, (prev_marks, next_marks), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def rough_beta_init(events: np.ndarray) -> tuple[float, float]:
    if len(events) <= 1:
        return 0.2, 0.8
    t = np.sort(events[:, 0])
    dt = np.diff(t)
    dt = dt[dt > 1e-8]
    if len(dt) == 0:
        return 0.2, 0.8
    # GED is daily/windowed, so keep a broad but stable day-scale initialization.
    beta0 = float(np.clip(1.0 / np.percentile(dt, 75), 0.03, 1.0))
    beta1 = float(np.clip(1.0 / np.percentile(dt, 25), beta0 + 0.05, 4.0))
    return beta0, beta1


def create_initial_params(cfg: SimConfig, events: np.ndarray, seed: int) -> MStepResult:
    rng = np.random.default_rng(int(seed) + 1000)
    T = max(float(cfg.T), 1e-6)
    base_rate = max(float(len(events)) / (cfg.K * cfg.M * T), 1e-5)
    rho_init = empirical_mark_init(events, cfg.M)
    beta0, beta1 = rough_beta_init(events)
    A0 = rng.uniform(0.0, 0.01, size=(cfg.K, cfg.K))
    np.fill_diagonal(A0, 0.0)
    return MStepResult(
        nu=np.full((cfg.K, cfg.M), base_rate, dtype=float),
        A0=A0,
        g=np.full(cfg.K, 0.1, dtype=float),
        h=np.full(cfg.K, 0.1, dtype=float),
        U=rng.normal(scale=0.01, size=(cfg.K, cfg.d)),
        V=rng.normal(scale=0.01, size=(cfg.K, cfg.d)),
        rho0=rho_init.copy(),
        rho1=rho_init.copy(),
        beta0=beta0,
        beta1=beta1,
        eta_on=0.02,
        eta_off=0.20,
    )


def _fatality_bin(best: pd.Series) -> np.ndarray:
    b = pd.to_numeric(best, errors="coerce").fillna(1).to_numpy(dtype=float)
    # 0: one fatality / low; 1: 2--4; 2: 5+
    return np.where(b >= 5, 2, np.where(b >= 2, 1, 0)).astype(int)


def _type_mark(type_of_violence: pd.Series) -> np.ndarray:
    t = pd.to_numeric(type_of_violence, errors="coerce").fillna(1).astype(int).to_numpy()
    vals = sorted(set(int(x) for x in t))
    remap = {v: i for i, v in enumerate(vals)}
    return np.asarray([remap[int(x)] for x in t], dtype=int)


def load_candidate_windows(path: str | None, max_windows: int, window_ranks: list[int] | None) -> list[dict]:
    if not path:
        return []
    df = pd.read_csv(path)
    if "rank" not in df.columns:
        df = df.reset_index().rename(columns={"index": "rank"})
    if window_ranks:
        df = df[df["rank"].astype(int).isin([int(x) for x in window_ranks])]
    else:
        df = df.head(int(max_windows))
    return df.to_dict("records")


def _select_window_events_uncached(
    events_csv: str,
    *,
    candidate: dict | None,
    country: str | None,
    region: str | None,
    conflict_name: str | None,
    window_start: str | None,
    window_end: str | None,
    top_k: int,
    mark_mode: str,
) -> tuple[pd.DataFrame, list[str], float, str]:
    df = pd.read_csv(events_csv, low_memory=False)
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df = df.dropna(subset=["date_start", "stream"])

    label_bits: list[str] = []
    if candidate is not None:
        if candidate.get("group_by") and candidate.get("group") and candidate.get("group_by") != "all":
            col = str(candidate["group_by"])
            if col in df.columns:
                df = df[df[col].astype(str) == str(candidate["group"])]
                label_bits.append(f"{col}={candidate['group']}")
        window_start = str(candidate.get("window_start", window_start))
        window_end = str(candidate.get("window_end", window_end))
    if country and "country" in df.columns:
        df = df[df["country"].astype(str) == str(country)]
        label_bits.append(f"country={country}")
    if region and "region" in df.columns:
        df = df[df["region"].astype(str) == str(region)]
        label_bits.append(f"region={region}")
    if conflict_name and "conflict_name" in df.columns:
        df = df[df["conflict_name"].astype(str) == str(conflict_name)]
        label_bits.append(f"conflict={conflict_name}")
    if window_start:
        ws = pd.Timestamp(window_start)
        df = df[df["date_start"] >= ws]
    else:
        ws = df["date_start"].min()
    if window_end:
        we = pd.Timestamp(window_end)
        df = df[df["date_start"] < we]
    else:
        we = df["date_start"].max() + pd.Timedelta(days=1)
    if df.empty:
        raise ValueError("Selected UCDP window has no events.")

    stream_counts = df["stream"].astype(str).value_counts()
    stream_names = stream_counts.head(int(top_k)).index.tolist()
    df = df[df["stream"].astype(str).isin(stream_names)].copy()
    if df.empty:
        raise ValueError("Top-K stream selection has no events.")
    t0 = pd.Timestamp(ws)
    df["base_time"] = (df["date_start"] - t0).dt.total_seconds() / 86400.0
    if mark_mode == "fatality_bin":
        if "best" not in df.columns:
            raise ValueError("mark-mode fatality_bin requires column 'best'.")
        df["mark"] = _fatality_bin(df["best"])
        M = 3
    elif mark_mode == "type_of_violence":
        if "type_of_violence" not in df.columns:
            raise ValueError("mark-mode type_of_violence requires column 'type_of_violence'.")
        df["mark"] = _type_mark(df["type_of_violence"])
        M = int(df["mark"].max()) + 1
    else:
        raise ValueError(f"Unknown mark_mode {mark_mode!r}")

    stream_to_actor = {s: i for i, s in enumerate(stream_names)}
    df["actor"] = df["stream"].astype(str).map(stream_to_actor).astype(int)
    T = float(max((pd.Timestamp(we) - t0).total_seconds() / 86400.0, df["base_time"].max() + 1.0, 1.0))
    label = ";".join(label_bits) if label_bits else "ucdp_window"
    df.attrs["M"] = M
    df.attrs["T"] = T
    df.attrs["window_start"] = str(pd.Timestamp(ws).date())
    df.attrs["window_end"] = str(pd.Timestamp(we).date())
    return df, stream_names, T, label


def _candidate_cache_key(candidate: dict | None) -> str:
    if candidate is None:
        return ""
    return json.dumps(candidate, sort_keys=True, default=str)


@lru_cache(maxsize=64)
def _select_window_events_cached(
    events_csv: str,
    candidate_key: str,
    country: str,
    region: str,
    conflict_name: str,
    window_start: str,
    window_end: str,
    top_k: int,
    mark_mode: str,
) -> tuple[pd.DataFrame, tuple[str, ...], float, str]:
    candidate = json.loads(candidate_key) if candidate_key else None
    df, stream_names, T, label = _select_window_events_uncached(
        events_csv,
        candidate=candidate,
        country=country or None,
        region=region or None,
        conflict_name=conflict_name or None,
        window_start=window_start or None,
        window_end=window_end or None,
        top_k=int(top_k),
        mark_mode=str(mark_mode),
    )
    # Make accidental mutation fail loudly inside a worker; callers only read df.
    # Pandas does not provide full immutable frames, but a shallow copy on return
    # protects attrs/list mutation while keeping the cached arrays shared.
    return df, tuple(stream_names), float(T), str(label)


def select_window_events(
    events_csv: str,
    *,
    candidate: dict | None,
    country: str | None,
    region: str | None,
    conflict_name: str | None,
    window_start: str | None,
    window_end: str | None,
    top_k: int,
    mark_mode: str,
) -> tuple[pd.DataFrame, list[str], float, str]:
    df, stream_names, T, label = _select_window_events_cached(
        str(events_csv),
        _candidate_cache_key(candidate),
        str(country or ""),
        str(region or ""),
        str(conflict_name or ""),
        str(window_start or ""),
        str(window_end or ""),
        int(top_k),
        str(mark_mode),
    )
    out = df.copy(deep=False)
    out.attrs = dict(df.attrs)
    return out, list(stream_names), float(T), str(label)


def _active_windows(T: float, n_episodes: int, episode_length: float, rng: np.random.Generator) -> list[tuple[float, float, int]]:
    if n_episodes <= 0:
        return [(0.0, float(T), 0)]
    margin = max(episode_length, 1.0)
    lo, hi = margin, max(float(T) - margin, margin + 1.0)
    if hi <= lo:
        starts = np.linspace(0.1 * T, 0.8 * T, n_episodes)
    else:
        starts = np.sort(rng.uniform(lo, hi, size=n_episodes))
    active = []
    for s in starts:
        active.append((float(s), float(min(T, s + episode_length)), 1))
    # Keep only active blocks in Z_path; evaluation uses Z_at_event for active-event AUC.
    return active


def _apply_time_observation_model(t: np.ndarray, T: float, coarsen_days: float, rng: np.random.Generator, jitter: float, grid: float = 1.0) -> np.ndarray:
    # GED's native temporal resolution is the day (`grid`). Background AND injected
    # events pass through this identical model, so the only thing distinguishing them
    # is coordination, never timestamp granularity. Anything at or below the native
    # grid is the exact-day regime; coarser settings only reveal the enclosing bin.
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    if coarsen_days and coarsen_days > grid:
        c = float(coarsen_days)
        base = np.floor(t / c) * c
        # Only the coarse bin is observed; impute uniformly within it.
        obs = base + rng.uniform(0.0, c, size=t.shape)
    else:
        # Exact-day: snap to the daily grid, then add a small shared jitter purely
        # to break ties for the continuous-time kernel (applied to all events alike).
        base = np.floor(t / grid) * grid
        obs = base + (rng.uniform(0.0, float(jitter), size=t.shape) if (jitter and jitter > 0) else 0.0)
    return np.clip(obs, 0.0, max(T - 1e-6, 0.0))


def _select_planted_subgroup(
    *, n_streams: int, subgroup_size: int, source_actor: int,
    mode: str, rng: np.random.Generator,
) -> tuple[list[int], int, list[int]]:
    """Choose the planted member streams and the source actor.

    mode='random' (default) draws the subgroup uniformly from the streams that
    are actually present, so member and source identity are independent of
    background volume -- recovery then reflects coordination, not a "pick the
    busiest streams" prior. mode='topk' reproduces the old volume-ranked
    behaviour and is kept only for ablation.
    """
    size = max(1, min(int(subgroup_size), int(n_streams)))
    if str(mode) == "topk":
        subgroup = list(range(size))
        source = int(source_actor) if int(source_actor) in subgroup else subgroup[0]
    else:
        subgroup = sorted(int(x) for x in rng.choice(int(n_streams), size=size, replace=False))
        source = int(rng.choice(subgroup))
    receivers = [a for a in subgroup if a != source]
    return subgroup, source, receivers


def _stable_seed(base_seed: int, label: str, *values: object) -> int:
    """Deterministic 32-bit seed independent of Python's randomized hash()."""
    acc = (int(base_seed) + 0x9E3779B9) & 0xFFFFFFFF
    text = str(label) + "|" + "|".join(str(v) for v in values)
    for b in text.encode("utf-8"):
        acc = (1664525 * acc + int(b) + 1013904223) & 0xFFFFFFFF
    return int(acc)


def _stable_permutation(values: np.ndarray, base_seed: int, label: str, *values_for_seed: object) -> np.ndarray:
    """Return a deterministic random ordering; prefixes are nested across sizes."""
    arr = np.asarray(values, dtype=int)
    if arr.size <= 1:
        return arr.copy()
    rng = np.random.default_rng(_stable_seed(base_seed, label, *values_for_seed))
    priority = rng.random(arr.size)
    return arr[np.argsort(priority, kind="mergesort")]


def _in_active_windows(t: np.ndarray, active_windows: list[tuple[float, float, int]]) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    mask = np.zeros(t.shape, dtype=bool)
    for s_win, e_win, z_win in active_windows:
        if int(z_win) == 1:
            mask |= (t >= float(s_win)) & (t < float(e_win))
    return mask


def _generate_nested_injections(
    *,
    n_inj_total: int,
    windows: list[tuple[float, float, int]],
    source_actor: int,
    true_receivers: list[int],
    M: int,
    seed: int,
    weak_receiver: int = -1,
    weak_receiver_frac: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate planted events so strength sweeps are nested by event index.

    P2 weak-link knob: ``weak_receiver_frac`` scales one designated receiver's
    receiver-directed mass relative to the uniform receiver share. Freed mass is
    redistributed to other receivers. The total number of planted events is
    controlled outside this function, so the weak-link knob changes the weakest
    link's relative strength without directly changing aggregate exposure.
    """
    n = int(max(0, n_inj_total))
    inj_t = np.empty(n, dtype=float)
    inj_actor = np.empty(n, dtype=int)
    inj_mark = np.empty(n, dtype=int)
    if n == 0:
        return inj_t, inj_actor, inj_mark
    if not windows:
        raise ValueError("_generate_nested_injections requires at least one active window")
    rec_arr = np.asarray(true_receivers, dtype=int) if true_receivers else np.asarray([int(source_actor)], dtype=int)

    recv_probs = None
    w_frac = float(weak_receiver_frac)
    if rec_arr.size > 1 and int(weak_receiver) in set(int(a) for a in rec_arr) and w_frac != 1.0:
        widx = int(np.where(rec_arr == int(weak_receiver))[0][0])
        base = 1.0 / float(rec_arr.size)
        weak_p = float(min(max(w_frac * base, 0.0), 1.0))
        rest = (1.0 - weak_p) / float(rec_arr.size - 1)
        recv_probs = np.full(rec_arr.size, rest, dtype=float)
        recv_probs[widx] = weak_p
        recv_probs = recv_probs / float(recv_probs.sum())

    for j in range(n):
        s_win, e_win, _ = windows[int(j) % len(windows)]
        length = max(float(e_win) - float(s_win), 1.0)
        rng = np.random.default_rng(_stable_seed(seed, "inj_event", j))
        is_source = bool(rng.random() < 0.35)
        if is_source:
            inj_t[j] = float(s_win) + float(rng.beta(1.5, 5.0)) * length
            inj_actor[j] = int(source_actor)
            if M >= 3:
                inj_mark[j] = int(rng.choice(np.array([0, 1], dtype=int), p=[0.75, 0.25]))
            else:
                inj_mark[j] = int(rng.integers(0, M))
        else:
            inj_t[j] = float(s_win) + float(rng.beta(3.0, 2.0)) * length
            if recv_probs is None:
                inj_actor[j] = int(rng.choice(rec_arr))
            else:
                inj_actor[j] = int(rng.choice(rec_arr, p=recv_probs))
            if M >= 3:
                inj_mark[j] = int(rng.choice(np.array([1, 2], dtype=int), p=[0.35, 0.65]))
            else:
                inj_mark[j] = int(rng.integers(0, M))
    return inj_t, inj_actor, inj_mark


def build_spikein_data(
    events_csv: str,
    *,
    candidate: dict | None,
    country: str | None,
    region: str | None,
    conflict_name: str | None,
    window_start: str | None,
    window_end: str | None,
    top_k: int,
    subgroup_size: int,
    source_actor: int,
    subgroup_mode: str = "random",
    volume_mode: str = "neutral",
    mark_mode: str,
    injection_strength: float,
    base_injected_events: int,
    n_episodes: int,
    episode_length: float,
    weak_receiver_frac: float = 1.0,
    p2_fixed_active_rate: bool = False,
    p2_active_rate_reference_length: float | None = None,
    coarsen_days: float,
    jitter: float,
    seed: int,
    d: int,
) -> tuple[UCDPData, dict]:
    # Use separate deterministic RNG streams for each design component.  This is
    # critical for exposure sweeps: for a fixed seed, the window, subgroup,
    # source, receivers, and active windows must be identical across strengths.
    # Only the number of planted/converted active events changes.
    seed = int(seed)
    df, stream_names, T, label = select_window_events(
        events_csv,
        candidate=candidate,
        country=country,
        region=region,
        conflict_name=conflict_name,
        window_start=window_start,
        window_end=window_end,
        top_k=top_k,
        mark_mode=mark_mode,
    )
    M = int(df.attrs["M"])
    K_present = int(len(stream_names))

    # Original real background, before any volume-neutral conversion. These
    # strength-invariant counts are capacity / background diagnostics. They are
    # NOT the exposure covariate in the theorem; the planted active counts below
    # are the exposure proxy.
    orig_bg_t_true = df["base_time"].to_numpy(dtype=float)
    orig_bg_actor = df["actor"].to_numpy(dtype=int)
    orig_bg_mark = df["mark"].to_numpy(dtype=int)
    orig_bg_count = int(len(orig_bg_t_true))
    orig_counts_by_actor = (
        np.bincount(orig_bg_actor, minlength=K_present).astype(int)
        if orig_bg_count
        else np.zeros(K_present, dtype=int)
    )

    subgroup_rng = np.random.default_rng(_stable_seed(seed, "subgroup"))
    subgroup, source_actor, true_receivers = _select_planted_subgroup(
        n_streams=K_present,
        subgroup_size=int(subgroup_size),
        source_actor=int(source_actor),
        mode=str(subgroup_mode),
        rng=subgroup_rng,
    )
    subgroup_arr = np.asarray(subgroup, dtype=int)
    receivers_arr = np.asarray(true_receivers, dtype=int)

    window_rng = np.random.default_rng(_stable_seed(seed, "active_windows"))
    active_windows = _active_windows(T, int(n_episodes), float(episode_length), window_rng)
    windows = active_windows if active_windows else [(0.25 * T, 0.25 * T + episode_length, 1)]
    active_window_days = float(sum(max(float(e) - float(s), 0.0) for s, e, z in active_windows if int(z) == 1))

    # P2 weak link: deterministic designated receiver for this seed/design.
    weak_receiver = int(true_receivers[0]) if true_receivers else int(source_actor)

    # Base count is the reference count at the reference active duration.
    # For a P2 exposure-time sweep, use --p2-fixed-active-rate so longer
    # active windows contain proportionally more planted active-born events.
    # Without this flag, legacy/P1 behavior is preserved exactly.
    base_n_inj_total = float(injection_strength) * int(base_injected_events)
    active_rate_reference_length = (
        float(p2_active_rate_reference_length)
        if p2_active_rate_reference_length is not None
        else float(episode_length)
    )
    active_rate_reference_days = max(float(n_episodes) * active_rate_reference_length, 1e-9)
    if bool(p2_fixed_active_rate):
        exposure_scale = active_window_days / active_rate_reference_days
    else:
        exposure_scale = 1.0
    n_inj_total = int(max(1, round(base_n_inj_total * exposure_scale)))

    inj_t, inj_actor, inj_mark = _generate_nested_injections(
        n_inj_total=n_inj_total,
        windows=windows,
        source_actor=int(source_actor),
        true_receivers=true_receivers,
        M=int(M),
        seed=seed,
        weak_receiver=int(weak_receiver),
        weak_receiver_frac=float(weak_receiver_frac),
    )
    inj_arr_true = np.column_stack([inj_t, inj_actor, inj_mark]).astype(float)

    # Realized planted active exposure. In this spike-in all injected events are
    # born from planted subgroup actors, so signal_total is the empirical proxy
    # for N_eff. Receiver/source splits are reported separately.
    signal_counts_by_actor = (
        np.bincount(inj_actor.astype(int), minlength=K_present).astype(int)
        if len(inj_actor)
        else np.zeros(K_present, dtype=int)
    )
    signal_member_counts = signal_counts_by_actor[subgroup_arr]
    signal_receiver_counts = signal_counts_by_actor[receivers_arr] if receivers_arr.size else np.asarray([], dtype=int)
    signal_total = int(signal_member_counts.sum()) if signal_member_counts.size else 0
    signal_min_member_count = int(signal_member_counts.min()) if signal_member_counts.size else 0
    signal_mean_member_count = float(signal_member_counts.mean()) if signal_member_counts.size else 0.0
    signal_source_count = int(signal_counts_by_actor[int(source_actor)]) if K_present else 0
    signal_min_receiver_count = int(signal_receiver_counts.min()) if signal_receiver_counts.size else 0
    signal_mean_receiver_count = float(signal_receiver_counts.mean()) if signal_receiver_counts.size else 0.0
    alpha_min_proxy = int(signal_counts_by_actor[int(weak_receiver)]) if K_present else 0
    receiver_directed_total = int(signal_receiver_counts.sum()) if signal_receiver_counts.size else 0
    weak_receiver_share_realized = float(alpha_min_proxy) / max(float(receiver_directed_total), 1.0)
    weak_receiver_share_uniform = 1.0 / max(float(len(true_receivers)), 1.0)
    alpha_min_relative = float(weak_receiver_frac)

    orig_member_counts = orig_counts_by_actor[subgroup_arr] if subgroup_arr.size else np.asarray([], dtype=int)
    orig_receiver_counts = orig_counts_by_actor[receivers_arr] if receivers_arr.size else np.asarray([], dtype=int)
    orig_min_member_bg_count = int(orig_member_counts.min()) if orig_member_counts.size else 0
    orig_mean_member_bg_count = float(orig_member_counts.mean()) if orig_member_counts.size else 0.0
    orig_source_bg_count = int(orig_counts_by_actor[int(source_actor)]) if K_present else 0
    orig_min_receiver_bg_count = int(orig_receiver_counts.min()) if orig_receiver_counts.size else 0
    orig_mean_receiver_bg_count = float(orig_receiver_counts.mean()) if orig_receiver_counts.size else 0.0

    active_bg_mask_orig = _in_active_windows(orig_bg_t_true, active_windows)
    expected_bg_active_windows = float(orig_bg_count) / max(float(T), 1e-9) * active_window_days
    orig_bg_active_windows = int(active_bg_mask_orig.sum())
    orig_member_active_counts = []
    for a in subgroup:
        orig_member_active_counts.append(int(np.sum((orig_bg_actor == int(a)) & active_bg_mask_orig)))

    # Volume-neutral spike-in. We remove the same number of real background
    # events from each planted actor as that actor receives planted events.  The
    # removal ordering is fixed per seed/actor, so strength sweeps are nested:
    # low-strength removals are a prefix of high-strength removals. Original
    # background counts above remain strength-invariant diagnostics; leftover
    # counts below are explicitly labeled as post-conversion leftovers.
    bg_t_true = orig_bg_t_true.copy()
    bg_actor = orig_bg_actor.copy()
    bg_mark = orig_bg_mark.copy()
    removed_mask = np.zeros(orig_bg_count, dtype=bool)
    converted_counts_by_actor = np.zeros(K_present, dtype=int)
    if str(volume_mode) == "neutral" and len(inj_arr_true):
        active_bg_mask = active_bg_mask_orig
        for actor in np.unique(inj_actor.astype(int)):
            actor = int(actor)
            n_remove = int(signal_counts_by_actor[actor]) if actor < len(signal_counts_by_actor) else 0
            if n_remove <= 0:
                continue
            actor_idx = np.where(orig_bg_actor == actor)[0]
            outside_idx = actor_idx[~active_bg_mask[actor_idx]]
            inside_idx = actor_idx[active_bg_mask[actor_idx]]
            ordered_outside = _stable_permutation(outside_idx, seed, "remove_outside", actor)
            ordered_inside = _stable_permutation(inside_idx, seed, "remove_inside", actor)
            ordered_candidates = np.concatenate([ordered_outside, ordered_inside])
            n_take = min(int(n_remove), int(ordered_candidates.size))
            if n_take > 0:
                selected = ordered_candidates[:n_take]
                removed_mask[selected] = True
                converted_counts_by_actor[actor] = int(n_take)
        bg_t_true = orig_bg_t_true[~removed_mask]
        bg_actor = orig_bg_actor[~removed_mask]
        bg_mark = orig_bg_mark[~removed_mask]

    removed_background_events = int(removed_mask.sum())
    converted_member_counts = converted_counts_by_actor[subgroup_arr] if subgroup_arr.size else np.asarray([], dtype=int)
    converted_receiver_counts = converted_counts_by_actor[receivers_arr] if receivers_arr.size else np.asarray([], dtype=int)
    converted_min_member_count = int(converted_member_counts.min()) if converted_member_counts.size else 0
    converted_mean_member_count = float(converted_member_counts.mean()) if converted_member_counts.size else 0.0
    converted_min_receiver_count = int(converted_receiver_counts.min()) if converted_receiver_counts.size else 0
    converted_mean_receiver_count = float(converted_receiver_counts.mean()) if converted_receiver_counts.size else 0.0
    conversion_cap_hit = int(removed_background_events < len(inj_arr_true))

    leftover_counts_by_actor = (
        np.bincount(bg_actor, minlength=K_present).astype(int)
        if len(bg_actor)
        else np.zeros(K_present, dtype=int)
    )
    leftover_member_counts = leftover_counts_by_actor[subgroup_arr] if subgroup_arr.size else np.asarray([], dtype=int)
    leftover_receiver_counts = leftover_counts_by_actor[receivers_arr] if receivers_arr.size else np.asarray([], dtype=int)
    leftover_min_member_bg_count = int(leftover_member_counts.min()) if leftover_member_counts.size else 0
    leftover_mean_member_bg_count = float(leftover_member_counts.mean()) if leftover_member_counts.size else 0.0
    leftover_source_bg_count = int(leftover_counts_by_actor[int(source_actor)]) if K_present else 0
    leftover_min_receiver_bg_count = int(leftover_receiver_counts.min()) if leftover_receiver_counts.size else 0

    # One observation model for background AND injected events. The GED background
    # is day-quantized; planted events are snapped to the same daily grid, so the
    # exact-day condition cannot be won via timestamp granularity.
    bg_true = np.column_stack([bg_t_true, bg_actor, bg_mark]).astype(float)
    all_true = np.vstack([bg_true, inj_arr_true]) if len(inj_arr_true) else bg_true.copy()
    injected_mask = np.concatenate([
        np.zeros(len(bg_true), dtype=bool), np.ones(len(inj_arr_true), dtype=bool)
    ])
    all_events = all_true.copy()
    obs_rng = np.random.default_rng(_stable_seed(seed, "observation", float(coarsen_days), float(jitter)))
    all_events[:, 0] = _apply_time_observation_model(all_true[:, 0], T, coarsen_days, obs_rng, jitter)

    z_at_event = injected_mask.astype(int)
    order = np.lexsort((all_events[:, 2], all_events[:, 1], all_events[:, 0]))
    all_events = all_events[order]
    injected_mask = injected_mask[order]
    z_at_event = z_at_event[order]
    bg_events_count = int(len(bg_true))
    inj_events_count = int(len(inj_arr_true))

    log_factor_delta_005 = float(math.log((float(max(K_present, 1)) ** 2) / 0.05))
    nominal_lambda_delta_005 = float(math.sqrt(log_factor_delta_005 / max(float(signal_total), 1.0)))
    nominal_lambda_min_member_delta_005 = float(math.sqrt(log_factor_delta_005 / max(float(signal_min_member_count), 1.0)))
    nominal_lambda_min_receiver_delta_005 = float(math.sqrt(log_factor_delta_005 / max(float(signal_min_receiver_count), 1.0)))
    signal_to_expected_active_bg_ratio = float(signal_total) / max(float(expected_bg_active_windows), 1e-9)
    signal_to_observed_active_bg_ratio = float(signal_total) / max(float(orig_bg_active_windows), 1e-9)

    cfg = SimConfig(
        K=int(top_k),
        M=int(M),
        T=float(T),
        ring_actors=[int(x) for x in subgroup],
        hub_actor=int(source_actor),
        d=int(d),
        seed=int(seed),
    )
    data = UCDPData(
        events=all_events,
        Z_path=active_windows,
        Z_at_event=z_at_event,
        config=cfg,
        stream_names=stream_names,
        injected_mask=injected_mask,
        true_receivers=true_receivers,
        weak_receiver=int(weak_receiver),
        background_count=bg_events_count,
        injected_count=inj_events_count,
    )
    meta = {
        "label": label,
        "window_start": df.attrs.get("window_start"),
        "window_end": df.attrs.get("window_end"),
        "stream_names_json": json.dumps(stream_names),
        "true_subgroup_json": json.dumps([int(x) for x in subgroup]),
        "true_source_actor": int(source_actor),
        "true_receivers_json": json.dumps([int(x) for x in true_receivers]),
        "subgroup_mode": str(subgroup_mode),
        "volume_mode": str(volume_mode),
        "design_seed": int(seed),
        "active_windows_json": json.dumps([[float(s), float(e), int(z)] for s, e, z in active_windows]),

        # Original, strength-invariant background/capacity diagnostics.
        "orig_background_events": int(orig_bg_count),
        "orig_counts_by_actor_json": json.dumps([int(x) for x in orig_counts_by_actor.tolist()]),
        "orig_member_counts_json": json.dumps([int(x) for x in orig_member_counts.tolist()]),
        "orig_receiver_counts_json": json.dumps([int(x) for x in orig_receiver_counts.tolist()]),
        "orig_subgroup_bg_count": int(orig_member_counts.sum()) if orig_member_counts.size else 0,
        "orig_min_member_bg_count": int(orig_min_member_bg_count),
        "orig_mean_member_bg_count": float(orig_mean_member_bg_count),
        "orig_source_bg_count": int(orig_source_bg_count),
        "orig_min_receiver_bg_count": int(orig_min_receiver_bg_count),
        "orig_mean_receiver_bg_count": float(orig_mean_receiver_bg_count),
        "orig_bg_active_windows": int(orig_bg_active_windows),
        "orig_member_active_counts_json": json.dumps([int(x) for x in orig_member_active_counts]),

        # Realized planted exposure: theorem-facing N_eff proxy.
        "signal_counts_by_actor_json": json.dumps([int(x) for x in signal_counts_by_actor.tolist()]),
        "signal_member_counts_json": json.dumps([int(x) for x in signal_member_counts.tolist()]),
        "signal_receiver_counts_json": json.dumps([int(x) for x in signal_receiver_counts.tolist()]),
        "signal_total": int(signal_total),
        "n_eff_planted": int(signal_total),
        "signal_min_member_count": int(signal_min_member_count),
        "signal_mean_member_count": float(signal_mean_member_count),
        "signal_source_count": int(signal_source_count),
        "signal_min_receiver_count": int(signal_min_receiver_count),
        "signal_mean_receiver_count": float(signal_mean_receiver_count),
        "active_window_days": float(active_window_days),
        "pi1T_planted": float(active_window_days),
        "pi1_planted": float(active_window_days / max(float(T), 1e-9)),
        "weak_receiver_frac": float(weak_receiver_frac),
        "weak_receiver": int(weak_receiver),
        "alpha_min_proxy": int(alpha_min_proxy),
        "alpha_min_relative": float(alpha_min_relative),
        "weak_receiver_share_uniform": float(weak_receiver_share_uniform),
        "weak_receiver_share_realized": float(weak_receiver_share_realized),
        "receiver_directed_total": int(receiver_directed_total),
        "episode_length": float(episode_length),
        "n_episodes": int(n_episodes),
        "p2_fixed_active_rate": int(bool(p2_fixed_active_rate)),
        "p2_active_rate_reference_length": float(active_rate_reference_length),
        "p2_active_rate_reference_days": float(active_rate_reference_days),
        "injected_events_reference": float(base_n_inj_total),
        "injected_events_exposure_scale": float(exposure_scale),
        "expected_bg_active_windows": float(expected_bg_active_windows),
        "signal_to_expected_active_bg_ratio": float(signal_to_expected_active_bg_ratio),
        "signal_to_observed_active_bg_ratio": float(signal_to_observed_active_bg_ratio),
        "log_factor_delta_005": float(log_factor_delta_005),
        "nominal_lambda_delta_005": float(nominal_lambda_delta_005),
        "nominal_lambda_min_member_delta_005": float(nominal_lambda_min_member_delta_005),
        "nominal_lambda_min_receiver_delta_005": float(nominal_lambda_min_receiver_delta_005),

        # Volume-neutral conversion diagnostics. These are NOT exposure.
        "removed_background_events": int(removed_background_events),
        "volume_neutral_ratio": float(removed_background_events / max(len(inj_arr_true), 1)),
        "conversion_cap_hit": int(conversion_cap_hit),
        "converted_counts_by_actor_json": json.dumps([int(x) for x in converted_counts_by_actor.tolist()]),
        "converted_member_counts_json": json.dumps([int(x) for x in converted_member_counts.tolist()]),
        "converted_receiver_counts_json": json.dumps([int(x) for x in converted_receiver_counts.tolist()]),
        "converted_min_member_count": int(converted_min_member_count),
        "converted_mean_member_count": float(converted_mean_member_count),
        "converted_min_receiver_count": int(converted_min_receiver_count),
        "converted_mean_receiver_count": float(converted_mean_receiver_count),
        "leftover_counts_by_actor_json": json.dumps([int(x) for x in leftover_counts_by_actor.tolist()]),
        "leftover_member_counts_json": json.dumps([int(x) for x in leftover_member_counts.tolist()]),
        "leftover_receiver_counts_json": json.dumps([int(x) for x in leftover_receiver_counts.tolist()]),
        "leftover_subgroup_bg_count": int(leftover_member_counts.sum()) if leftover_member_counts.size else 0,
        "leftover_min_member_bg_count": int(leftover_min_member_bg_count),
        "leftover_mean_member_bg_count": float(leftover_mean_member_bg_count),
        "leftover_source_bg_count": int(leftover_source_bg_count),
        "leftover_min_receiver_bg_count": int(leftover_min_receiver_bg_count),

        # Backward-compatible aliases now point to original background counts,
        # not post-conversion leftovers.
        "subgroup_bg_count": int(orig_member_counts.sum()) if orig_member_counts.size else 0,
        "min_member_bg_count": int(orig_min_member_bg_count),
        "source_bg_count": int(orig_source_bg_count),
        "min_receiver_bg_count": int(orig_min_receiver_bg_count),
    }
    return data, meta

def active_event_truth(events: np.ndarray, z_at_event: np.ndarray, interval_edges: np.ndarray) -> np.ndarray:
    B = len(interval_edges) - 1
    truth = np.zeros(B, dtype=int)
    if len(events) == 0:
        return truth
    idx = np.searchsorted(interval_edges, events[:, 0], side="right") - 1
    idx = np.clip(idx, 0, B - 1)
    active_idx = idx[np.asarray(z_at_event, dtype=int) == 1]
    if active_idx.size:
        truth[np.unique(active_idx)] = 1
    return truth


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
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


def set_f1(pred: list[int] | np.ndarray, truth: list[int]) -> float:
    p = set(int(x) for x in pred)
    t = set(int(x) for x in truth)
    if not p and not t:
        return 1.0
    tp = len(p & t)
    prec = tp / max(len(p), 1)
    rec = tp / max(len(t), 1)
    return float(2 * prec * rec / max(prec + rec, 1e-12))


def receiver_f1_from_h(h: np.ndarray, true_receivers: list[int], hub: int) -> tuple[float, list[int]]:
    h = np.asarray(h, dtype=float)
    order = [int(x) for x in np.argsort(h)[::-1] if int(x) != int(hub)]
    pred = order[:len(true_receivers)]
    return set_f1(pred, true_receivers), pred


def _set_equal(a: list[int] | np.ndarray, b: list[int] | np.ndarray) -> int:
    return int(set(int(x) for x in a) == set(int(x) for x in b))


def exact_support_metrics(
    *,
    pred_ring: list[int] | np.ndarray,
    true_ring: list[int] | np.ndarray,
    pred_receivers: list[int] | np.ndarray,
    true_receivers: list[int] | np.ndarray,
    hub_pred: int,
    true_hub: int,
) -> dict:
    member_exact = _set_equal(pred_ring, true_ring)
    receiver_exact = _set_equal(pred_receivers, true_receivers)
    hub_exact = int(int(hub_pred) == int(true_hub))
    return {
        "member_exact": int(member_exact),
        "receiver_exact": int(receiver_exact),
        "hub_exact": int(hub_exact),
        "member_receiver_exact": int(member_exact and receiver_exact),
        "all_role_exact": int(member_exact and receiver_exact and hub_exact),
    }


def planted_interval_truth_from_windows(
    interval_edges: np.ndarray,
    active_windows: list[tuple[float, float, int]],
) -> np.ndarray:
    """Interval-level planted active path for spike-ins.

    An interval is active if it overlaps any planted active window. This is the
    hard planted-Z diagnostic used for real-background spike-ins where no true
    posterior is available.
    """
    edges = np.asarray(interval_edges, dtype=float)
    B = max(len(edges) - 1, 0)
    truth = np.zeros(B, dtype=int)
    for b in range(B):
        lo, hi = float(edges[b]), float(edges[b + 1])
        for s_win, e_win, z_win in active_windows:
            if int(z_win) == 1 and hi > float(s_win) and lo < float(e_win):
                truth[b] = 1
                break
    return truth


def interval_event_counts(events: np.ndarray, interval_edges: np.ndarray) -> np.ndarray:
    B = len(interval_edges) - 1
    counts = np.zeros(B, dtype=float)
    if len(events) == 0 or B <= 0:
        return counts
    idx = np.searchsorted(interval_edges, events[:, 0], side="right") - 1
    idx = np.clip(idx, 0, B - 1)
    np.add.at(counts, idx, 1.0)
    return counts


def posterior_accuracy_metrics(
    gamma: np.ndarray,
    *,
    events: np.ndarray,
    interval_edges: np.ndarray,
    active_windows: list[tuple[float, float, int]],
    K: int,
    n_eff: int,
    delta: float = 0.05,
) -> dict:
    """Event-weighted gap to planted interval labels, normalized by lambda_T.

    This is a spike-in diagnostic for the C6 bridge. It compares the fitted
    active posterior to the planted active-window path, not to an unavailable
    true-parameter smoothing posterior. The normalized value is therefore a
    relative posterior-quality diagnostic, not a proof of C6.
    """
    gamma = np.asarray(gamma, dtype=float)
    if gamma.ndim == 2 and gamma.shape[1] >= 2:
        active_prob = gamma[:, 1]
    else:
        active_prob = np.asarray(gamma, dtype=float).reshape(-1)
    truth = planted_interval_truth_from_windows(interval_edges, active_windows).astype(float)
    m = min(len(active_prob), len(truth))
    if m == 0:
        return {
            "gamma_gap_planted": float("nan"),
            "r_gamma_planted": float("nan"),
            "gamma_truth_corr": float("nan"),
            "lambda_T_delta_005": float("nan"),
        }
    active_prob = active_prob[:m]
    truth = truth[:m]
    counts = interval_event_counts(events, interval_edges)[:m]
    denom = float(np.sum(counts))
    if denom <= 0:
        denom = float(m)
        weights = np.ones(m, dtype=float)
    else:
        weights = counts
    gap = float(np.sum(weights * np.abs(active_prob - truth)) / denom)
    log_factor = float(math.log((float(max(int(K), 1)) ** 2) / float(delta)))
    lam = float(math.sqrt(log_factor / max(float(n_eff), 1.0)))
    if np.std(active_prob) > 1e-12 and np.std(truth) > 1e-12:
        corr = float(np.corrcoef(active_prob, truth)[0, 1])
    else:
        corr = float("nan")
    return {
        "gamma_gap_planted": gap,
        "r_gamma_planted": float(gap / max(lam, 1e-12)),
        "gamma_truth_corr": corr,
        "lambda_T_delta_005": lam,
    }




def receiver_support_diagnostics(h: np.ndarray, true_receivers: list[int], hub: int, weak_receiver: int) -> dict:
    h = np.asarray(h, dtype=float)
    order = [int(x) for x in np.argsort(h)[::-1] if int(x) != int(hub)]
    pred = order[:len(true_receivers)]
    try:
        rank = order.index(int(weak_receiver)) + 1
    except ValueError:
        rank = -1
    score = float(h[int(weak_receiver)]) if 0 <= int(weak_receiver) < len(h) else float("nan")
    return {
        "weak_receiver_recovered": int(int(weak_receiver) in set(pred)),
        "weak_receiver_rank": int(rank),
        "weak_receiver_score": float(score),
    }


def mark_escalation_lift(rho0: np.ndarray, rho1: np.ndarray) -> float:
    rho0 = np.asarray(rho0, dtype=float)
    rho1 = np.asarray(rho1, dtype=float)
    if rho0.shape[0] < 3 or rho0.shape[1] < 3:
        return float("nan")
    # Low/medium to high severity lift. Rows are parent severity, columns child severity.
    return float(0.5 * ((rho1[0, 2] - rho0[0, 2]) + (rho1[1, 2] - rho0[1, 2])))


def run_proposed(data: UCDPData, interval_width: float, max_iters: int, lr: float, lambda_g: float, lambda_h: float, lambda_e: float, lambda_0: float, n_inner_steps: int) -> dict:
    cfg = data.config
    edges = np.arange(0.0, cfg.T + interval_width + 1e-9, interval_width)
    init = create_initial_params(cfg, data.events, cfg.seed)
    kwargs = dict(
        events=data.events,
        interval_edges=edges,
        init_params=init,
        max_iters=max_iters,
        lr=lr,
        lambda_g=lambda_g,
        lambda_h=lambda_h,
        lambda_e=lambda_e,
        lambda_0=lambda_0,
        n_inner_steps=n_inner_steps,
        verbose=False,
        pi1_init=0.05,
    )
    sig = inspect.signature(run_em)
    if "stability_threshold" in sig.parameters:
        kwargs["stability_threshold"] = 0.95
    if "stability_target" in sig.parameters:
        kwargs["stability_target"] = 0.80
    result = run_em(**kwargs)
    ev = evaluate(
        gamma=result.gamma,
        interval_edges=edges,
        Z_path=data.Z_path,
        g=result.params.g,
        h=result.params.h,
        true_ring=cfg.ring_actors,
        true_hub=cfg.hub_actor,
        method="topk",
        events=data.events,
        Z_at_event=data.Z_at_event,
        auc_target="active_event",
    )
    rec_f1, pred_receivers = receiver_f1_from_h(result.params.h, data.true_receivers, cfg.hub_actor)
    weakdiag = receiver_support_diagnostics(result.params.h, data.true_receivers, cfg.hub_actor, data.weak_receiver)
    A1 = compute_A1(result.params.g, result.params.h, result.params.U, result.params.V)
    sender_scores = A1.sum(axis=1)
    hub_pred = int(np.argmax(sender_scores))
    exact = exact_support_metrics(
        pred_ring=ev.predicted_ring,
        true_ring=cfg.ring_actors,
        pred_receivers=pred_receivers,
        true_receivers=data.true_receivers,
        hub_pred=hub_pred,
        true_hub=cfg.hub_actor,
    )
    pacc = posterior_accuracy_metrics(
        result.gamma,
        events=data.events,
        interval_edges=edges,
        active_windows=data.Z_path,
        K=cfg.K,
        n_eff=data.injected_count,
    )
    return {
        "method": "Proposed",
        "member_f1": float(ev.membership_f1),
        "precision": float(ev.membership_precision),
        "recall": float(ev.membership_recall),
        "hub_correct": int(bool(ev.hub_correct)),
        "hub_pred": int(hub_pred),
        "active_auc": float(ev.field_auc),
        "receiver_f1": float(rec_f1),
        **weakdiag,
        **exact,
        **pacc,
        "pred_ring": json.dumps([int(x) for x in ev.predicted_ring]),
        "pred_receivers": json.dumps([int(x) for x in pred_receivers]),
        "mark_escalation_lift": mark_escalation_lift(result.params.rho0, result.params.rho1),
        "loglik_final": float(result.log_likelihood_trace[-1]) if result.log_likelihood_trace else float("nan"),
        "beta0_hat": float(result.params.beta0),
        "beta1_hat": float(result.params.beta1),
        "active_occ": float(np.mean(result.gamma[:, 1])),
    }


def run_b1(data: UCDPData, interval_width: float, max_iters: int) -> dict:
    if run_static_hawkes_baseline is None:
        raise ImportError(f"B1 import failed: {STATIC_IMPORT_ERROR}")
    cfg = data.config
    edges = np.arange(0.0, cfg.T + interval_width + 1e-9, interval_width)
    init = create_initial_params(cfg, data.events, cfg.seed)
    params, ll = run_static_hawkes_baseline(
        events=data.events,
        interval_edges=edges,
        init_params=init,
        max_iters=max_iters,
        lr=0.03,
        lambda_g=0.02,
        lambda_h=0.02,
        lambda_e=0.01,
        lambda_0=0.01,
        n_inner_steps=10,
        verbose=False,
    )
    B = len(edges) - 1
    gamma_uniform = np.column_stack([np.full(B, 0.5), np.full(B, 0.5)])
    ev = evaluate(
        gamma=gamma_uniform,
        interval_edges=edges,
        Z_path=data.Z_path,
        g=params.g,
        h=params.h,
        true_ring=cfg.ring_actors,
        true_hub=cfg.hub_actor,
        method="topk",
        events=data.events,
        Z_at_event=data.Z_at_event,
        auc_target="active_event",
    )
    A1 = compute_A1(params.g, params.h, params.U, params.V)
    sender_scores = A1.sum(axis=1)
    hub_pred = int(np.argmax(sender_scores))
    rec_f1, pred_receivers = receiver_f1_from_h(params.h, data.true_receivers, cfg.hub_actor)
    weakdiag = receiver_support_diagnostics(params.h, data.true_receivers, cfg.hub_actor, data.weak_receiver)
    exact = exact_support_metrics(
        pred_ring=ev.predicted_ring,
        true_ring=cfg.ring_actors,
        pred_receivers=pred_receivers,
        true_receivers=data.true_receivers,
        hub_pred=hub_pred,
        true_hub=cfg.hub_actor,
    )
    pacc = posterior_accuracy_metrics(
        gamma_uniform,
        events=data.events,
        interval_edges=edges,
        active_windows=data.Z_path,
        K=cfg.K,
        n_eff=data.injected_count,
    )
    return {
        "method": "B1_static_gates",
        "member_f1": float(ev.membership_f1),
        "precision": float(ev.membership_precision),
        "recall": float(ev.membership_recall),
        "hub_correct": int(bool(ev.hub_correct)),
        "hub_pred": int(hub_pred),
        "active_auc": float("nan"),
        "receiver_f1": float(rec_f1),
        **weakdiag,
        **exact,
        **pacc,
        "pred_ring": json.dumps([int(x) for x in ev.predicted_ring]),
        "pred_receivers": json.dumps([int(x) for x in pred_receivers]),
        "mark_escalation_lift": float("nan"),
        "loglik_final": float(ll),
        "beta0_hat": float(params.beta0),
        "beta1_hat": float(params.beta1),
        "active_occ": float("nan"),
    }


def run_b2(data: UCDPData, interval_width: float) -> dict:
    if run_modular_hmm_hawkes_baseline is None:
        raise ImportError(f"B2 import failed: {MODULAR_IMPORT_ERROR}")
    cfg = data.config
    edges = np.arange(0.0, cfg.T + interval_width + 1e-9, interval_width)
    baseline = run_modular_hmm_hawkes_baseline(
        events=data.events,
        interval_edges=edges,
        true_ring=cfg.ring_actors,
        true_hub=cfg.hub_actor,
        Z_path=None,
        hmm_threshold=0.25,
        hmm_restarts=10,
        hmm_max_iters=100,
        hawkes_min_total_events=4,
        selection="topk",
        top_k=len(cfg.ring_actors),
        score_mode="max",
        seed=cfg.seed,
    )
    truth = active_event_truth(data.events, data.Z_at_event, edges)
    auc = roc_auc_binary(truth, baseline.hmm_gamma[:, 1])
    exact = exact_support_metrics(
        pred_ring=baseline.predicted_ring,
        true_ring=cfg.ring_actors,
        pred_receivers=[],
        true_receivers=data.true_receivers,
        hub_pred=int(baseline.hub_pred),
        true_hub=cfg.hub_actor,
    )
    pacc = posterior_accuracy_metrics(
        baseline.hmm_gamma,
        events=data.events,
        interval_edges=edges,
        active_windows=data.Z_path,
        K=cfg.K,
        n_eff=data.injected_count,
    )
    return {
        "method": "B2_modular_hmm_hawkes",
        "member_f1": float(baseline.membership_f1),
        "precision": float(baseline.membership_precision),
        "recall": float(baseline.membership_recall),
        "hub_correct": int(bool(baseline.hub_correct)),
        "hub_pred": int(baseline.hub_pred),
        "active_auc": float(auc),
        "receiver_f1": float("nan"),
        "weak_receiver_recovered": 0,
        "weak_receiver_rank": -1,
        "weak_receiver_score": float("nan"),
        **exact,
        **pacc,
        "pred_ring": json.dumps([int(x) for x in baseline.predicted_ring]),
        "pred_receivers": json.dumps([]),
        "mark_escalation_lift": float("nan"),
        "loglik_final": float("nan"),
        "beta0_hat": float("nan"),
        "beta1_hat": float("nan"),
        "active_occ": float(baseline.active_fraction),
    }


def run_b3(data: UCDPData) -> dict:
    if _fit_static_hawkes_full_matrix is None or _spectral_ring_detection is None:
        raise ImportError(f"B3 import failed: {SPECTRAL_IMPORT_ERROR}")
    cfg = data.config
    _nu, A, _rho, beta = _fit_static_hawkes_full_matrix(
        events=data.events,
        K=cfg.K,
        M=cfg.M,
        T=cfg.T,
        beta_init=0.5,
        lr=0.003,
        n_iters=120,
        l1_penalty=0.01,
        verbose=False,
    )
    ring, hub = _spectral_ring_detection(A, len(cfg.ring_actors))
    f1 = set_f1(ring, cfg.ring_actors)
    exact = exact_support_metrics(
        pred_ring=ring,
        true_ring=cfg.ring_actors,
        pred_receivers=[],
        true_receivers=data.true_receivers,
        hub_pred=int(hub),
        true_hub=cfg.hub_actor,
    )
    return {
        "method": "B3_static_svd",
        "member_f1": float(f1),
        "precision": float("nan"),
        "recall": float("nan"),
        "hub_correct": int(int(hub) == int(cfg.hub_actor)),
        "hub_pred": int(hub),
        "active_auc": float("nan"),
        "receiver_f1": float("nan"),
        "weak_receiver_recovered": 0,
        "weak_receiver_rank": -1,
        "weak_receiver_score": float("nan"),
        **exact,
        "gamma_gap_planted": float("nan"),
        "r_gamma_planted": float("nan"),
        "gamma_truth_corr": float("nan"),
        "lambda_T_delta_005": float("nan"),
        "pred_ring": json.dumps([int(x) for x in ring]),
        "pred_receivers": json.dumps([]),
        "mark_escalation_lift": float("nan"),
        "loglik_final": float("nan"),
        "beta0_hat": float(beta),
        "beta1_hat": float("nan"),
        "active_occ": float("nan"),
    }


def run_task(task: dict) -> dict:
    data, meta = build_spikein_data(**task["data_kwargs"])
    method = task["method"]
    if method == "proposed":
        metrics = run_proposed(
            data,
            interval_width=task["interval_width"],
            max_iters=task["max_iters"],
            lr=task["lr"],
            lambda_g=task["lambda_g"],
            lambda_h=task["lambda_h"],
            lambda_e=task["lambda_e"],
            lambda_0=task["lambda_0"],
            n_inner_steps=task["n_inner_steps"],
        )
    elif method == "b1":
        metrics = run_b1(data, interval_width=task["interval_width"], max_iters=task["max_iters"])
    elif method == "b2":
        metrics = run_b2(data, interval_width=task["interval_width"])
    elif method == "b3":
        metrics = run_b3(data)
    else:
        raise ValueError(f"unknown method {method!r}")

    return {
        **task["row_meta"],
        **meta,
        "K": int(data.config.K),
        "M": int(data.config.M),
        "T": float(data.config.T),
        "background_events": int(data.background_count),
        "injected_events": int(data.injected_count),
        "total_events": int(len(data.events)),
        **metrics,
        "status": "ok",
        "error": "",
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    # Stable union of keys, preserving first-row order.
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    if exists:
        try:
            old = pd.read_csv(path, nrows=0)
            keys = list(old.columns)
        except Exception:
            pass
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_methods(s: str) -> list[str]:
    aliases = {"proposed": "proposed", "ours": "proposed", "b1": "b1", "static": "b1", "b2": "b2", "hmm": "b2", "b3": "b3", "svd": "b3"}
    out = []
    for tok in [x.strip().lower() for x in s.split(",") if x.strip()]:
        if tok not in aliases:
            raise ValueError(f"Unknown method {tok!r}; use proposed,b1,b2,b3")
        out.append(aliases[tok])
    return list(dict.fromkeys(out))


def main():
    p = argparse.ArgumentParser(description="MP UCDP GED real-background spike-in + temporal coarsening experiment.")
    p.add_argument("--events", required=True, help="*_events.csv from ucdp_ged_events.py")
    p.add_argument("--candidate-windows", default=None, help="CSV from screen_ucdp_ged_windows.py")
    p.add_argument("--max-windows", type=int, default=1)
    p.add_argument("--window-ranks", nargs="*", type=int, default=None)
    p.add_argument("--country", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--conflict-name", default=None)
    p.add_argument("--window-start", default=None)
    p.add_argument("--window-end", default=None)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--subgroup-size", type=int, default=4)
    p.add_argument("--source-actor", type=int, default=0)
    p.add_argument("--subgroup-mode", default="random", choices=["random", "topk"],
                   help="how planted member streams are chosen; 'random' (default) "
                        "decouples member/source identity from background volume, "
                        "'topk' reproduces the old volume-ranked behaviour for ablation")
    p.add_argument("--volume-mode", default="neutral", choices=["neutral", "additive"],
                   help="'neutral' removes matched same-actor background events before adding planted events; "
                        "'additive' preserves the older volume-increasing spike-in")
    p.add_argument("--mark-mode", default="fatality_bin", choices=["fatality_bin", "type_of_violence"])
    p.add_argument("--injection-strengths", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0, 4.0])
    p.add_argument("--base-injected-events", type=int, default=150)
    p.add_argument("--coarsen-days", nargs="+", type=float, default=[0.0, 1.0, 3.0, 7.0, 14.0])
    p.add_argument("--seeds", type=int, default=20, help="number of seeds: 0..seeds-1")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--methods", default="proposed,b1,b2,b3")
    p.add_argument("--interval-width", type=float, default=7.0)
    p.add_argument("--jitter", type=float, default=0.05, help="within-day jitter for exact dates, in days")
    p.add_argument("--n-episodes", type=int, default=4)
    p.add_argument("--episode-length", type=float, default=21.0)
    p.add_argument("--weak-receiver-fracs", nargs="+", type=float, default=[1.0],
                   help="P2 weak-link sweep: scales one designated receiver's share relative to uniform.")
    p.add_argument("--episode-lengths", nargs="+", type=float, default=None,
                   help="P2 exposure-time sweep: active episode lengths in days. Defaults to [--episode-length].")
    p.add_argument("--p2-fixed-active-rate", action="store_true",
                   help="Scale planted event count with active-window duration so episode-length sweeps change N_eff at fixed planted active rate.")
    p.add_argument("--p2-active-rate-reference-length", type=float, default=None,
                   help="Reference episode length for --p2-fixed-active-rate. Defaults to --episode-length.")
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lambda-g", type=float, default=0.02)
    p.add_argument("--lambda-h", type=float, default=0.02)
    p.add_argument("--lambda-e", type=float, default=0.01)
    p.add_argument("--lambda-0", type=float, default=0.05)
    p.add_argument("--n-inner-steps", type=int, default=10)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--outdir", default="results/ucdp_ged_spikein")
    p.add_argument("--resume", action="store_true", help="skip rows already present in checkpoint")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "ucdp_ged_spikein_raw_checkpoint.csv"

    candidates = load_candidate_windows(args.candidate_windows, args.max_windows, args.window_ranks)
    if not candidates:
        candidates = [None]

    methods = parse_methods(args.methods)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))

    done_keys = set()
    if args.resume and raw_path.exists():
        old = pd.read_csv(raw_path)
        for _, r in old.iterrows():
            if str(r.get("status", "ok")) == "ok":
                done_keys.add((
                    int(r["window_id"]), int(r["seed"]), str(r["method_key"]),
                    float(r["injection_strength"]), float(r["coarsen_days"]),
                    float(r.get("weak_receiver_frac", 1.0)) if not pd.isna(r.get("weak_receiver_frac", 1.0)) else 1.0,
                    float(r.get("episode_length", args.episode_length)) if not pd.isna(r.get("episode_length", args.episode_length)) else float(args.episode_length),
                ))

    tasks = []
    weak_fracs = list(args.weak_receiver_fracs) if args.weak_receiver_fracs else [1.0]
    episode_lengths = list(args.episode_lengths) if args.episode_lengths else [float(args.episode_length)]
    ref_len = float(args.p2_active_rate_reference_length) if args.p2_active_rate_reference_length is not None else float(args.episode_length)
    for widx, cand in enumerate(candidates):
        for seed in seeds:
            for s in args.injection_strengths:
                for cdays in args.coarsen_days:
                    for method in methods:
                        for wfrac in weak_fracs:
                            for elen in episode_lengths:
                                key = (widx, seed, method, float(s), float(cdays), float(wfrac), float(elen))
                                if key in done_keys:
                                    continue
                                data_kwargs = dict(
                                    events_csv=args.events,
                                    candidate=cand,
                                    country=args.country,
                                    region=args.region,
                                    conflict_name=args.conflict_name,
                                    window_start=args.window_start,
                                    window_end=args.window_end,
                                    top_k=args.top_k,
                                    subgroup_size=args.subgroup_size,
                                    source_actor=args.source_actor,
                                    subgroup_mode=args.subgroup_mode,
                                    volume_mode=args.volume_mode,
                                    mark_mode=args.mark_mode,
                                    injection_strength=float(s),
                                    base_injected_events=args.base_injected_events,
                                    n_episodes=args.n_episodes,
                                    episode_length=float(elen),
                                    weak_receiver_frac=float(wfrac),
                                    p2_fixed_active_rate=bool(args.p2_fixed_active_rate),
                                    p2_active_rate_reference_length=float(ref_len),
                                    coarsen_days=float(cdays),
                                    jitter=args.jitter,
                                    seed=int(seed),
                                    d=args.d,
                                )
                                row_meta = {
                                    "window_id": int(widx),
                                    "seed": int(seed),
                                    "method_key": method,
                                    "injection_strength": float(s),
                                    "coarsen_days": float(cdays),
                                    "weak_receiver_frac": float(wfrac),
                                    "episode_length": float(elen),
                                    "mark_mode": args.mark_mode,
                                }
                                tasks.append({
                                    "data_kwargs": data_kwargs,
                                    "row_meta": row_meta,
                                    "method": method,
                                    "interval_width": args.interval_width,
                                    "max_iters": args.max_iters,
                                    "lr": args.lr,
                                    "lambda_g": args.lambda_g,
                                    "lambda_h": args.lambda_h,
                                    "lambda_e": args.lambda_e,
                                    "lambda_0": args.lambda_0,
                                    "n_inner_steps": args.n_inner_steps,
                                })

    print(f"UCDP GED spike-in MP: {len(tasks)} jobs, workers={args.workers}, outdir={outdir}", flush=True)
    print("Each job = window x seed x injection strength x coarsening x method. Checkpoint rows are written as jobs finish.", flush=True)

    if not tasks:
        print("Nothing to run.")
        return

    if args.workers <= 1:
        for idx, task in enumerate(tasks, 1):
            try:
                row = run_task(task)
            except Exception as exc:
                row = {**task["row_meta"], "status": "error", "error": repr(exc)}
            write_rows(raw_path, [row])
            print(f"[{idx}/{len(tasks)}] {row.get('status')} {row.get('method_key')} seed={row.get('seed')} s={row.get('injection_strength')} coarse={row.get('coarsen_days')}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = {ex.submit(run_task, task): task for task in tasks}
            for idx, fut in enumerate(as_completed(futs), 1):
                task = futs[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {**task["row_meta"], "status": "error", "error": repr(exc)}
                write_rows(raw_path, [row])
                print(f"[{idx}/{len(tasks)}] {row.get('status')} {row.get('method_key')} seed={row.get('seed')} s={row.get('injection_strength')} coarse={row.get('coarsen_days')}", flush=True)

    print(f"Done. Raw checkpoint: {raw_path}")


if __name__ == "__main__":
    main()
