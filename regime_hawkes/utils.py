"""Shared utility functions for regime_hawkes."""
from __future__ import annotations

import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def compute_A1(g: np.ndarray, h: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute K x K active coupling matrix from gates and embeddings."""
    K = len(g)
    inner = U @ V.T  # K x K
    inner = np.clip(inner, -30.0, 30.0)   # softplus saturates anyway outside ±30
    A1 = g[:, None] * h[None, :] * softplus(inner)
    np.fill_diagonal(A1, 0.0)
    return A1


def spectral_radius(matrix: np.ndarray) -> float:
    """Spectral radius (largest absolute eigenvalue)."""
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))
