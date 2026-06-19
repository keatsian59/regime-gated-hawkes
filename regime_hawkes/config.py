from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimConfig:
    K: int = 10
    M: int = 2
    T: float = 120.0
    ring_actors: list[int] = field(default_factory=lambda: [0, 1, 2])
    hub_actor: int = 0
    d: int = 3
    nu_base: float = 0.25
    alpha0_team: float = 0.02
    team_size: int = 5
    alpha1_max: float = 0.22
    alpha1_min: float = 0.08
    beta0: float = 1.0
    beta1: float = 2.0
    eta_on: float = 0.03
    eta_off: float = 0.25
    seed: int = 42
    active_edge_rule: str = "dense_subgroup"


@dataclass
class ModelConfig:
    delta: float = 0.5
    eps: float = 1e-9
