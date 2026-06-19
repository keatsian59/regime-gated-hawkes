from __future__ import annotations

"""Multiprocessing runner for powered sparse directed-edge diagnostics.

Place under ``regime_hawkes/run_sparse_directed_powered_mp.py`` and run from
repo root, for example:

    python -m regime_hawkes.run_sparse_directed_powered_mp --benchmark large --seeds 20 --workers 8 --max-iters 30

This runner is intentionally separate from the small sparse diagnostic. It uses
larger active subgroups so directed-edge AUC and F1@|E*| are meaningful and
prints the random-ranking chance baselines implied by the planted graph.

It imports sibling ``simulate_sparse_directed_patch.py`` explicitly, so you do
not need to overwrite ``regime_hawkes/simulate.py`` while testing.
"""

import argparse
import csv
import importlib.util
import inspect
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Must come before importing jax or modules that import jax.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_LOG_COMPILES", "0")
# Prevent each worker from oversubscribing CPU threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

for _name in ["jax", "jaxlib", "absl", "matplotlib", "PIL"]:
    _logger = logging.getLogger(_name)
    _logger.handlers.clear()
    _logger.setLevel(logging.CRITICAL)
    _logger.propagate = False

_root_logger = logging.getLogger()
_root_logger.handlers.clear()
_root_logger.setLevel(logging.CRITICAL)
_root_logger.propagate = False

SCRIPT_PATH = Path(__file__).resolve()
PKG_DIR = SCRIPT_PATH.parent
REPO_ROOT = PKG_DIR.parent if PKG_DIR.name == "regime_hawkes" else PKG_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from regime_hawkes.config import SimConfig
from regime_hawkes.em import run_em
from regime_hawkes.evaluate import evaluate
from regime_hawkes.mstep import MStepResult


@dataclass(frozen=True)
class RunOptions:
    large_K: int = 30
    large_k: int = 10
    large_T: float = 900.0
    large_d: int = 4
    edge_rule: str = "hub_plus_cycle"
    nu_base: float = 0.15
    alpha0_team: float = 0.01
    alpha1_max: float = 0.8
    alpha1_min: float = 0.35
    beta0: float = 1.0
    beta1: float = 3.0
    eta_on: float = 0.04
    eta_off: float = 0.3
    delta: float = 2.0
    lr: float = 0.01
    lambda_g: float = 0.02
    lambda_h: float = 0.02
    lambda_0: float = 0.05
    n_inner_steps: int = 10
    stability_threshold: float = 0.95
    stability_target: float = 0.80


def _load_sparse_sim_module():
    candidates = [
        SCRIPT_PATH.with_name("simulate_sparse_directed_patch.py"),
        PKG_DIR / "simulate_sparse_directed_patch.py",
        REPO_ROOT / "regime_hawkes" / "simulate_sparse_directed_patch.py",
        REPO_ROOT / "simulate_sparse_directed_patch.py",
        Path.cwd() / "regime_hawkes" / "simulate_sparse_directed_patch.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("simulate_sparse_directed_patch", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["simulate_sparse_directed_patch"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "Could not find simulate_sparse_directed_patch.py. Put it under "
        "regime_hawkes next to this runner."
    )


sparse_sim = _load_sparse_sim_module()
simulate_regime_hawkes = sparse_sim.simulate_regime_hawkes
summarize_simulation = sparse_sim.summarize_simulation
summarize_active_edge_rule = sparse_sim.summarize_active_edge_rule
directed_edge_truth_rows = sparse_sim.directed_edge_truth_rows
directed_edge_score_rows = sparse_sim.directed_edge_score_rows


def _empirical_mark_init(events: np.ndarray, M: int, eps: float = 1.0) -> np.ndarray:
    counts = np.full((M, M), eps, dtype=float)
    if len(events) == 0:
        return counts / counts.sum(axis=1, keepdims=True)

    order = np.lexsort((events[:, 0], events[:, 1]))
    ev = events[order]
    last_mark_by_actor: dict[int, int] = {}
    for _, actor, mark in ev:
        actor = int(actor)
        mark = int(mark)
        if actor in last_mark_by_actor:
            prev_mark = last_mark_by_actor[actor]
            counts[prev_mark, mark] += 1.0
        last_mark_by_actor[actor] = mark

    return counts / counts.sum(axis=1, keepdims=True)


def _rough_beta_init(events: np.ndarray) -> tuple[float, float]:
    if len(events) <= 1:
        return 1.0, 2.0
    t = np.sort(events[:, 0])
    dt = np.diff(t)
    dt = dt[dt > 1e-8]
    if len(dt) == 0:
        return 1.0, 2.0
    beta0 = float(np.clip(1.0 / np.median(dt), 0.1, 1.0))
    beta1 = float(np.clip(1.0 / np.percentile(dt, 10), beta0 + 0.1, 8.0))
    return beta0, beta1


def _make_cfg(benchmark: str, seed: int, opts: RunOptions) -> SimConfig:
    if benchmark == "Sparse base":
        cfg = SimConfig(
            K=10, M=2, T=500.0, ring_actors=[0, 1, 2], hub_actor=0, d=3,
            nu_base=opts.nu_base, alpha0_team=opts.alpha0_team,
            alpha1_max=opts.alpha1_max, alpha1_min=opts.alpha1_min,
            beta0=opts.beta0, beta1=opts.beta1,
            eta_on=opts.eta_on, eta_off=opts.eta_off, seed=int(seed),
        )
    elif benchmark == "Sparse scaled":
        cfg = SimConfig(
            K=20, M=2, T=600.0, ring_actors=[0, 1, 2, 3], hub_actor=0, d=3,
            nu_base=opts.nu_base, alpha0_team=opts.alpha0_team,
            alpha1_max=opts.alpha1_max, alpha1_min=opts.alpha1_min,
            beta0=opts.beta0, beta1=opts.beta1,
            eta_on=opts.eta_on, eta_off=opts.eta_off, seed=int(seed),
        )
    elif benchmark == "Sparse large":
        if opts.large_k < 3:
            raise ValueError("large_k must be at least 3")
        if opts.large_K < opts.large_k:
            raise ValueError("large_K must be >= large_k")
        cfg = SimConfig(
            K=int(opts.large_K), M=2, T=float(opts.large_T),
            ring_actors=list(range(int(opts.large_k))), hub_actor=0, d=int(opts.large_d),
            nu_base=opts.nu_base, alpha0_team=opts.alpha0_team,
            alpha1_max=opts.alpha1_max, alpha1_min=opts.alpha1_min,
            beta0=opts.beta0, beta1=opts.beta1,
            eta_on=opts.eta_on, eta_off=opts.eta_off, seed=int(seed),
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Dataclasses without slots allow this even if SimConfig has not yet been
    # extended. The patched simulator reads active_edge_rule via getattr().
    setattr(cfg, "active_edge_rule", str(opts.edge_rule))
    return cfg


def _fit_one(cfg: SimConfig, opts: RunOptions, *, max_iters: int, verbose: bool):
    data = simulate_regime_hawkes(cfg)
    summary = summarize_simulation(data)

    edges = np.arange(0.0, cfg.T + 1e-9, float(opts.delta))
    base_rate = summary["total_events"] / (cfg.K * cfg.M * cfg.T)
    fit_rng = np.random.default_rng(cfg.seed + 1000)

    rho_init = _empirical_mark_init(data.events, cfg.M)
    beta0_init, beta1_init = _rough_beta_init(data.events)

    A0_init = fit_rng.uniform(0.0, 0.02, size=(cfg.K, cfg.K))
    np.fill_diagonal(A0_init, 0.0)

    init = MStepResult(
        nu=np.full((cfg.K, cfg.M), max(base_rate, 1e-3)),
        A0=A0_init,
        g=np.full(cfg.K, 0.1),
        h=np.full(cfg.K, 0.1),
        U=fit_rng.normal(scale=0.01, size=(cfg.K, cfg.d)),
        V=fit_rng.normal(scale=0.01, size=(cfg.K, cfg.d)),
        rho0=rho_init.copy(),
        rho1=rho_init.copy(),
        beta0=beta0_init,
        beta1=beta1_init,
        eta_on=0.03,
        eta_off=0.3,
    )

    result = run_em(
        events=data.events,
        interval_edges=edges,
        init_params=init,
        max_iters=max_iters,
        lr=float(opts.lr),
        lambda_g=float(opts.lambda_g),
        lambda_h=float(opts.lambda_h),
        lambda_0=float(opts.lambda_0),
        n_inner_steps=int(opts.n_inner_steps),
        verbose=verbose,
        stability_threshold=float(opts.stability_threshold),
        stability_target=float(opts.stability_target),
    )
    return data, edges, result, summary


def _evaluate_one(result, edges, data, cfg, method: str):
    kwargs = dict(
        interval_edges=edges,
        Z_path=data.Z_path,
        true_ring=cfg.ring_actors,
        true_hub=cfg.hub_actor,
        method=method,
        events=data.events,
        Z_at_event=data.Z_at_event,
        auc_target="active_event",
        gamma=result.gamma,
        g=result.params.g,
        h=result.params.h,
    )
    active_matrix = result.a1_trace[-1] if getattr(result, "a1_trace", None) else None
    sig = inspect.signature(evaluate)
    if method == "posterior_ring" and "active_matrix" in sig.parameters:
        kwargs["active_matrix"] = active_matrix
    return evaluate(**kwargs)


def _run_one_task(task: tuple[str, int, int, bool, str, RunOptions]) -> dict[str, Any]:
    """Worker entry point. Must remain top-level for Windows spawn."""
    benchmark, seed, max_iters, verbose_em, method, opts = task
    cfg = _make_cfg(benchmark, seed, opts)
    data, edges, result, sim_summary = _fit_one(cfg, opts, max_iters=max_iters, verbose=verbose_em)

    member_f1 = float("nan")
    hub_correct: bool | None = None
    predicted_ring: list[int] | None = None
    try:
        ev = _evaluate_one(result, edges, data, cfg, method="posterior_ring")
        member_f1 = float(ev.membership_f1)
        hub_correct = bool(ev.hub_correct)
        predicted_ring = [int(x) for x in ev.predicted_ring]
    except Exception:
        pass

    truth_rows = directed_edge_truth_rows(
        data, benchmark=benchmark, seed=seed, subgroup=cfg.ring_actors
    )
    score_rows = directed_edge_score_rows(
        benchmark=benchmark,
        method=method,
        seed=seed,
        subgroup=cfg.ring_actors,
        fitted_g=result.params.g,
        fitted_h=result.params.h,
        fitted_U=result.params.U,
        fitted_V=result.params.V,
    )

    return {
        "benchmark": benchmark,
        "seed": int(seed),
        "total_events": int(sim_summary["total_events"]),
        "active_time": float(sim_summary["active_time"]),
        "pi_1": float(sim_summary["pi_1"]),
        "member_f1": member_f1,
        "hub_correct": hub_correct,
        "predicted_ring": predicted_ring,
        "score_rows": score_rows,
        "truth_rows": truth_rows,
    }


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    write_header = mode == "w" or not path.exists() or path.stat().st_size == 0
    with path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _benchmarks_from_arg(arg: str) -> list[str]:
    if arg == "base":
        return ["Sparse base"]
    if arg == "scaled":
        return ["Sparse scaled"]
    if arg == "large":
        return ["Sparse large"]
    if arg == "small":
        return ["Sparse base", "Sparse scaled"]
    return ["Sparse base", "Sparse scaled", "Sparse large"]


def _edge_chance_summary(edge_summary: dict[str, Any]) -> tuple[float, float, float]:
    n_pairs = int(edge_summary["n_pairs"])
    n_edges = int(edge_summary["n_edges"])
    density = n_edges / n_pairs if n_pairs else float("nan")
    auc_step = 1.0 / (n_edges * (n_pairs - n_edges)) if 0 < n_edges < n_pairs else float("nan")
    return density, 0.5, auc_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds per benchmark")
    ap.add_argument("--seed-start", type=int, default=0, help="first seed")
    ap.add_argument("--max-iters", type=int, default=30, help="EM iterations")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--verbose-em", action="store_true", help="print EM iteration output")
    ap.add_argument("--outdir", default="results/sparse_directed_powered")
    ap.add_argument("--append", action="store_true", help="append to existing score/truth CSVs")
    ap.add_argument(
        "--benchmark",
        choices=["all", "small", "base", "scaled", "large"],
        default="large",
        help="which benchmark to run",
    )
    ap.add_argument("--large-K", type=int, default=30)
    ap.add_argument("--large-k", type=int, default=10)
    ap.add_argument("--large-T", type=float, default=900.0)
    ap.add_argument("--large-d", type=int, default=4)
    ap.add_argument(
        "--edge-rule",
        default="hub_plus_cycle",
        choices=["cycle", "hub_to_others", "hub_plus_cycle", "dense_subgroup"],
    )
    ap.add_argument("--nu-base", type=float, default=0.15)
    ap.add_argument("--alpha0-team", type=float, default=0.01)
    ap.add_argument("--alpha1-max", type=float, default=0.8)
    ap.add_argument("--alpha1-min", type=float, default=0.35)
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--beta1", type=float, default=3.0)
    ap.add_argument("--eta-on", type=float, default=0.04)
    ap.add_argument("--eta-off", type=float, default=0.3)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lambda-g", type=float, default=0.02)
    ap.add_argument("--lambda-h", type=float, default=0.02)
    ap.add_argument("--lambda-0", type=float, default=0.05)
    ap.add_argument("--n-inner-steps", type=int, default=10)
    ap.add_argument("--stability-threshold", type=float, default=0.95)
    ap.add_argument("--stability-target", type=float, default=0.80)
    args = ap.parse_args()

    opts = RunOptions(
        large_K=args.large_K,
        large_k=args.large_k,
        large_T=args.large_T,
        large_d=args.large_d,
        edge_rule=args.edge_rule,
        nu_base=args.nu_base,
        alpha0_team=args.alpha0_team,
        alpha1_max=args.alpha1_max,
        alpha1_min=args.alpha1_min,
        beta0=args.beta0,
        beta1=args.beta1,
        eta_on=args.eta_on,
        eta_off=args.eta_off,
        delta=args.delta,
        lr=args.lr,
        lambda_g=args.lambda_g,
        lambda_h=args.lambda_h,
        lambda_0=args.lambda_0,
        n_inner_steps=args.n_inner_steps,
        stability_threshold=args.stability_threshold,
        stability_target=args.stability_target,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    score_path = outdir / "sparse_directed_scores.csv"
    truth_path = outdir / "sparse_directed_truth.csv"

    if not args.append:
        score_path.unlink(missing_ok=True)
        truth_path.unlink(missing_ok=True)

    benchmarks = _benchmarks_from_arg(args.benchmark)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    print(f"Writing score rows to: {score_path.resolve()}")
    print(f"Writing truth rows to: {truth_path.resolve()}")
    print(f"workers={args.workers}, max_iters={args.max_iters}")

    for benchmark in benchmarks:
        check_cfg = _make_cfg(benchmark, seeds[0], opts)
        edge_summary = summarize_active_edge_rule(check_cfg)
        chance_f1, chance_auc, auc_step = _edge_chance_summary(edge_summary)
        print(f"\n=== {benchmark} ===")
        print(f"active_edge_rule: {edge_summary['active_edge_rule']}")
        print(f"subgroup: {edge_summary['subgroup']}")
        print(
            f"within-subgroup pairs={edge_summary['n_pairs']}, "
            f"edges={edge_summary['n_edges']}, non-edges={edge_summary['n_non_edges']}"
        )
        print(f"edge density / random F1@|E*| baseline: {chance_f1:.4f}")
        print(f"random AUC baseline: {chance_auc:.1f}; AUC granularity approx: {auc_step:.6f}")
        print(f"edges: {edge_summary['edges']}")

    tasks = [
        (benchmark, seed, int(args.max_iters), bool(args.verbose_em), "Proposed", opts)
        for benchmark in benchmarks
        for seed in seeds
    ]

    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for task in tasks:
            res = _run_one_task(task)
            results.append(res)
            print(
                f"[{res['benchmark']}] seed={res['seed']} events={res['total_events']} "
                f"active_time={res['active_time']:.2f} pi_1={res['pi_1']:.3f} "
                f"member F1={res['member_f1']:.3f} hub={res['hub_correct']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            future_to_task = {ex.submit(_run_one_task, task): task for task in tasks}
            for fut in as_completed(future_to_task):
                benchmark, seed, *_ = future_to_task[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"[{benchmark}] seed={seed} FAILED: {exc}")
                    raise
                results.append(res)
                print(
                    f"[{res['benchmark']}] seed={res['seed']} events={res['total_events']} "
                    f"active_time={res['active_time']:.2f} pi_1={res['pi_1']:.3f} "
                    f"member F1={res['member_f1']:.3f} hub={res['hub_correct']}"
                )

    results.sort(key=lambda r: (str(r["benchmark"]), int(r["seed"])))
    score_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for res in results:
        score_rows.extend(res["score_rows"])
        truth_rows.extend(res["truth_rows"])

    _write_rows(
        score_path,
        ["benchmark", "method", "seed", "src", "dst", "score"],
        score_rows,
        append=bool(args.append),
    )
    _write_rows(
        truth_path,
        ["benchmark", "seed", "src", "dst", "y_true"],
        truth_rows,
        append=bool(args.append),
    )

    print("\nDone.")
    print(f"Scores: {score_path.resolve()}")
    print(f"Truth:  {truth_path.resolve()}")
    print("\nEvaluate from PowerShell:")
    print(
        "python -m regime_hawkes.evaluate_sparse_directed_edges_with_baselines `\n"
        f"  --scores {score_path} `\n"
        f"  --truth {truth_path} `\n"
        f"  --outdir {outdir}"
    )


if __name__ == "__main__":
    main()
