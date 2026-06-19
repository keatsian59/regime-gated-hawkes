from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

try:
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    import numpy as jnp
    from scipy.special import logsumexp

from regime_hawkes.likelihood import build_interval_emissions

logger = logging.getLogger(__name__)


@dataclass
class EStepResult:
    gamma: np.ndarray
    xi: np.ndarray
    log_likelihood: float
    emission_loglik: np.ndarray


def ctmc_transition_matrix(eta_on: float, eta_off: float, delta: float) -> np.ndarray:
    s = eta_on + eta_off
    e = np.exp(-s * delta)
    p00 = (eta_off + eta_on * e) / s
    p11 = (eta_on + eta_off * e) / s
    p01 = 1.0 - p00
    p10 = 1.0 - p11
    P = np.array([[p00, p01], [p10, p11]], dtype=float)
    logger.debug(
        "E-step CTMC transition built: eta_on=%.6f eta_off=%.6f delta=%.6f P=%s",
        eta_on,
        eta_off,
        delta,
        np.array2string(P, precision=4, suppress_small=True),
    )
    return P


def build_emissions(
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
    logger.debug(
        "E-step building emissions: events=%d intervals=%d beta0=%.4f beta1=%.4f",
        len(events),
        max(len(interval_edges) - 1, 0),
        beta0,
        beta1,
    )
    emissions = build_interval_emissions(
        events=events,
        interval_edges=interval_edges,
        gamma_prev_1=gamma_prev_1,
        nu=nu,
        A0=A0,
        A1=A1,
        rho0=rho0,
        rho1=rho1,
        beta0=beta0,
        beta1=beta1,
        eps=eps,
    )
    logger.debug(
        "E-step emissions ready: shape=%s mean(z=0)=%.4f mean(z=1)=%.4f",
        emissions.shape,
        float(np.mean(emissions[:, 0])) if len(emissions) else float('nan'),
        float(np.mean(emissions[:, 1])) if len(emissions) else float('nan'),
    )
    return emissions


def forward_backward_logspace(emissions: np.ndarray, P: np.ndarray, pi: np.ndarray | None = None) -> EStepResult:
    Bn = emissions.shape[0]
    logP = np.log(np.asarray(P) + 1e-18)
    logE = np.asarray(emissions)

    if pi is None:
        pi = np.array([0.9, 0.1], dtype=float)
    log_pi = np.log(np.asarray(pi) + 1e-18)

    logger.debug(
        "Forward-backward start: intervals=%d pi=%s",
        Bn,
        np.array2string(np.asarray(pi), precision=4, suppress_small=True),
    )

    log_alpha = np.zeros((Bn, 2))
    log_alpha[0] = log_pi + logE[0]
    logger.debug(
        "Forward pass initialized: alpha[0]=%s",
        np.array2string(log_alpha[0], precision=4, suppress_small=True),
    )

    for b in range(1, Bn):
        prev = log_alpha[b - 1][:, None] + logP
        log_alpha[b] = logE[b] + logsumexp(prev, axis=0)

    logger.debug(
        "Forward pass complete: alpha[last]=%s",
        np.array2string(log_alpha[-1], precision=4, suppress_small=True),
    )

    log_beta = np.zeros((Bn, 2))
    for b in range(Bn - 2, -1, -1):
        nxt = logP + logE[b + 1][None, :] + log_beta[b + 1][None, :]
        log_beta[b] = logsumexp(nxt, axis=1)

    logger.debug(
        "Backward pass complete: beta[0]=%s beta[last]=%s",
        np.array2string(log_beta[0], precision=4, suppress_small=True),
        np.array2string(log_beta[-1], precision=4, suppress_small=True),
    )

    log_z = logsumexp(log_alpha[-1])
    if not np.isfinite(log_z):
        raise FloatingPointError("forward_backward_logspace produced non-finite log normalizer")

    log_gamma = np.asarray(log_alpha + log_beta - log_z, dtype=float)
    # Posterior probabilities should be <= 1, but numerical roundoff / extreme
    # emissions on real data can make log_gamma slightly positive.  Clip and
    # renormalize each row rather than letting exp overflow.
    log_gamma = np.clip(np.nan_to_num(log_gamma, nan=-745.0, posinf=0.0, neginf=-745.0), -745.0, 0.0)
    gamma = np.exp(log_gamma)
    gamma = gamma / np.maximum(np.sum(gamma, axis=1, keepdims=True), 1e-300)

    xi = np.zeros((Bn - 1, 2, 2), dtype=float)
    for b in range(Bn - 1):
        log_xi_b = (
            log_alpha[b][:, None]
            + logP
            + logE[b + 1][None, :]
            + log_beta[b + 1][None, :]
            - log_z
        )
        log_xi_b = np.clip(np.nan_to_num(np.asarray(log_xi_b, dtype=float), nan=-745.0, posinf=0.0, neginf=-745.0), -745.0, 0.0)
        xb = np.exp(log_xi_b)
        xi[b] = xb / max(float(np.sum(xb)), 1e-300)

    logger.debug(
        "Forward-backward complete: log_likelihood=%.6f avg_active=%.4f first_gamma=%s",
        float(log_z),
        float(np.mean(gamma[:, 1])) if len(gamma) else float('nan'),
        np.array2string(gamma[0], precision=4, suppress_small=True) if len(gamma) else '[]',
    )

    return EStepResult(gamma=gamma, xi=xi, log_likelihood=float(log_z), emission_loglik=np.array(emissions))


def run_estep(
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
    eta_on: float,
    eta_off: float,
) -> EStepResult:
    logger.debug(
        "E-step start: events=%d intervals=%d active_prior_mean=%.4f",
        len(events),
        max(len(interval_edges) - 1, 0),
        float(np.mean(gamma_prev_1)) if len(gamma_prev_1) else float('nan'),
    )
    ell = build_emissions(events, interval_edges, gamma_prev_1, nu, A0, A1, rho0, rho1, beta0, beta1)
    delta = float(np.mean(np.diff(interval_edges)))
    P = ctmc_transition_matrix(eta_on, eta_off, delta)
    s = eta_on + eta_off
    pi = np.array([eta_off / s, eta_on / s], dtype=float)
    result = forward_backward_logspace(ell, P, pi=pi)
    logger.debug(
        "E-step done: ll=%.6f avg_active=%.4f transitions=%d",
        result.log_likelihood,
        float(np.mean(result.gamma[:, 1])) if len(result.gamma) else float('nan'),
        len(result.xi),
    )
    return result
