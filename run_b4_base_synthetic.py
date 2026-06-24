#!/usr/bin/env python
"""
Run only B4_change-point Hawkes on the base K=10 synthetic benchmark.

This does not rerun Proposed/B1/B2/B3. It simulates the same fixed-seed
base benchmark streams and runs only the external B4 baseline.
"""

from pathlib import Path

from regime_hawkes.config import SimConfig
from reviewer_upgrade_experiments import run_replications, parse_method_selection


def main() -> None:
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
        eta_on=0.06,
        eta_off=0.3,
        seed=7,
    )

    seeds = list(range(7, 27))

    run_replications(
        cfg,
        seeds=seeds,
        outdir=Path("runs/synthetic_base_k10_b4_changepoint_iter30_s20"),
        method="topk",
        methods=parse_method_selection("b4"),
        include_mmhp=False,
        max_iters=30,
    )


if __name__ == "__main__":
    main()
