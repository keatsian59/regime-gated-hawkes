from __future__ import annotations

"""
Run sparse directed-edge synthetic diagnostics.

How to use in your repo (PowerShell):
  1. Put simulate_sparse_directed_patch.py in the repo root.
  2. Put this file at scripts/run_sparse_directed_synthetic.py.
  3. Run: python .\scripts\run_sparse_directed_synthetic.py --seeds 20
  4. Run: python .\scripts\evaluate_sparse_directed_edges.py `
           --scores .\results\sparse_directed\sparse_directed_scores.csv `
           --truth .\results\sparse_directed\sparse_directed_truth.csv `
           --outdir .\results\sparse_directed

This script deliberately imports simulation/export helpers from
simulate_sparse_directed_patch.py so you do not need to overwrite
regime_hawkes/simulate.py while testing. For the final repo, you can instead
merge the patch into regime_hawkes/simulate.py and change the imports below.
"""

import argparse
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------
# Make repo root importable when this script is run as .\scripts\...
# ---------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent if SCRIPT_PATH.parent.name == "scripts" else SCRIPT_PATH.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Must come before importing jax or modules that import jax.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_LOG_COMPILES"] = "0"

import logging

for name in ["jax", "jaxlib", "absl", "matplotlib", "PIL"]:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False

root = logging.getLogger()
root.handlers.clear()
root.setLevel(logging.CRITICAL)
root.propagate = False

from regime_hawkes.config import SimConfig
from regime_hawkes.em import run_em
from regime_hawkes.evaluate import evaluate
from regime_hawkes.mstep import MStepResult


# ---------------------------------------------------------------------
# Load sparse simulator patch explicitly.
# ---------------------------------------------------------------------
def _load_sparse_sim_module():
    candidates = [
        SCRIPT_PATH.with_name("simulate_sparse_directed_patch.py"),
        REPO_ROOT / "simulate_sparse_directed_patch.py",
        Path.cwd() / "simulate_sparse_directed_patch.py",
        Path.cwd() / "scripts" / "simulate_sparse_directed_patch.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("simulate_sparse_directed_patch", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["simulate_sparse_directed_patch"] = mod
            spec.loader.exec_module(mod)
            print(f"Using sparse simulator patch: {path}")
            return mod
    raise FileNotFoundError(
        "Could not find simulate_sparse_directed_patch.py. Put it in the repo root "
        "or next to this script."
    )


sparse_sim = _load_sparse_sim_module()
simulate_regime_hawkes = sparse_sim.simulate_regime_hawkes
summarize_simulation = sparse_sim.summarize_simulation
append_directed_edge_diagnostic_rows = sparse_sim.append_directed_edge_diagnostic_rows
summarize_active_edge_rule = sparse_sim.summarize_active_edge_rule


# ---------------------------------------------------------------------
# Helpers copied/adapted from your uploaded run_synthetic.py.
# ---------------------------------------------------------------------
def _empirical_mark_init(events: np.ndarray, M: int, eps: float = 1.0) -> np.ndarray:
    counts = np.full((M, M), eps, dtype=float)
    if len(events) == 0:
        return counts / counts.sum(axis=1, keepdims=True)

    order = np.lexsort((events[:, 0], events[:, 1]))
    ev = events[order]

    last_mark_by_actor = {}
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


def _make_cfg(benchmark: str, seed: int) -> SimConfig:
    if benchmark == "Sparse base":
        cfg = SimConfig(
            K=10,
            M=2,
            T=500.0,
            ring_actors=[0, 1, 2],
            hub_actor=0,
            d=3,
            nu_base=0.15,
            alpha0_team=0.01,
            alpha1_max=0.8,
            alpha1_min=0.35,
            beta0=1.0,
            beta1=3.0,
            eta_on=0.04,
            eta_off=0.3,
            seed=int(seed),
        )
    elif benchmark == "Sparse scaled":
        cfg = SimConfig(
            K=20,
            M=2,
            T=600.0,
            ring_actors=[0, 1, 2, 3],
            hub_actor=0,
            d=3,
            nu_base=0.15,
            alpha0_team=0.01,
            alpha1_max=0.8,
            alpha1_min=0.35,
            beta0=1.0,
            beta1=3.0,
            eta_on=0.04,
            eta_off=0.3,
            seed=int(seed),
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # This works even if SimConfig did not originally declare this field,
    # because dataclasses without slots permit dynamic attributes. The patched
    # simulator reads it via getattr(config, "active_edge_rule", ...).
    setattr(cfg, "active_edge_rule", "hub_plus_cycle")
    return cfg


def _fit_one(cfg: SimConfig, *, max_iters: int, verbose: bool):
    data = simulate_regime_hawkes(cfg)
    summary = summarize_simulation(data)

    edges = np.arange(0.0, cfg.T + 1e-9, 2.0)
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
        lr=0.01,
        lambda_g=0.02,
        lambda_h=0.02,
        lambda_0=0.05,
        n_inner_steps=10,
        verbose=verbose,
        stability_threshold=0.95,
        stability_target=0.80,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds per benchmark")
    ap.add_argument("--seed-start", type=int, default=0, help="first seed")
    ap.add_argument("--max-iters", type=int, default=50, help="EM iterations")
    ap.add_argument("--verbose-em", action="store_true", help="print EM iteration output")
    ap.add_argument("--outdir", default="results/sparse_directed")
    ap.add_argument("--append", action="store_true", help="append to existing score/truth CSVs")
    ap.add_argument(
        "--benchmark",
        choices=["all", "base", "scaled"],
        default="all",
        help="which benchmark to run",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    score_path = outdir / "sparse_directed_scores.csv"
    truth_path = outdir / "sparse_directed_truth.csv"

    if not args.append:
        score_path.unlink(missing_ok=True)
        truth_path.unlink(missing_ok=True)

    if args.benchmark == "base":
        benchmarks = ["Sparse base"]
    elif args.benchmark == "scaled":
        benchmarks = ["Sparse scaled"]
    else:
        benchmarks = ["Sparse base", "Sparse scaled"]

    print(f"Writing score rows to: {score_path.resolve()}")
    print(f"Writing truth rows to: {truth_path.resolve()}")

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    for benchmark in benchmarks:
        check_cfg = _make_cfg(benchmark, seeds[0])
        edge_summary = summarize_active_edge_rule(check_cfg)
        print(f"\n=== {benchmark} ===")
        print(f"active_edge_rule: {edge_summary['active_edge_rule']}")
        print(f"subgroup: {edge_summary['subgroup']}")
        print(
            f"within-subgroup pairs={edge_summary['n_pairs']}, "
            f"edges={edge_summary['n_edges']}, non-edges={edge_summary['n_non_edges']}"
        )
        print(f"edges: {edge_summary['edges']}")

        for seed in seeds:
            cfg = _make_cfg(benchmark, seed)
            print(f"\n[{benchmark}] seed={seed}")
            data, edges, result, sim_summary = _fit_one(
                cfg, max_iters=args.max_iters, verbose=args.verbose_em
            )
            print(
                f"events={sim_summary['total_events']}, "
                f"active_time={sim_summary['active_time']:.2f}, "
                f"pi_1={sim_summary['pi_1']:.3f}"
            )

            try:
                ev = _evaluate_one(result, edges, data, cfg, method="posterior_ring")
                print(
                    f"member F1={ev.membership_f1:.3f}, "
                    f"hub={ev.hub_correct}, predicted={ev.predicted_ring}"
                )
            except Exception as exc:
                print(f"posterior_ring evaluation skipped: {exc}")

            append_directed_edge_diagnostic_rows(
                score_path=score_path,
                truth_path=truth_path,
                data=data,
                benchmark=benchmark,
                seed=seed,
                subgroup=cfg.ring_actors,
                fitted_g=result.params.g,
                fitted_h=result.params.h,
                fitted_U=result.params.U,
                fitted_V=result.params.V,
                method="Proposed",
            )

    print("\nDone.")
    print(f"Scores: {score_path.resolve()}")
    print(f"Truth:  {truth_path.resolve()}")
    print("\nNext PowerShell command:")
    print(
        "python .\\scripts\\evaluate_sparse_directed_edges.py `\n"
        f"  --scores {score_path} `\n"
        f"  --truth {truth_path} `\n"
        f"  --outdir {outdir}"
    )


if __name__ == "__main__":
    main()
