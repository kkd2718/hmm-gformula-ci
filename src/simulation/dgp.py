"""Synthetic DGP with per-day discrete-time hazard, multinomial binned A,
and AR(1) Gaussian latent confounder Z_t.

Designed to match the real-data setup (binned MP, T=28 days, daily death
hazard with absorbing event), so that simulation results transfer to
inference on MIMIC-IV ARDS. Combines:
    Soohoo M, Arah OA (2023) IJE 52(6):1907-1913 (binary-outcome
        time-varying confounding skeleton, PMC10749778),
    augmented with an AR(1) Gaussian latent state Z_t (SSM ingredient),
    and a multinomial K-bin treatment derived by quantile-binning a
    continuous propensity score (matches MP discretization).
The gamma knob controls direct Z->A and Z->hazard strength
(Bica et al. 2020 ICML, conceptual reference only).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class DGPConfig:
    """Hyperparameters for simulate_tv_latent_confounding."""
    N: int = 2000
    T: int = 28
    K: int = 4
    gamma: float = 0.5
    psi: float = 0.9
    sigma_Z: float = 0.5
    sigma_L: float = 0.5
    beta_A: float = 0.4
    beta_L: float = 0.3
    hazard_intercept: float = -4.0
    seed: int = 0


@dataclass
class DGPSample:
    """Output of one DGP draw with per-day records.

    Z (N,T)         AR(1) latent confounder (unobserved by analysis).
    L (N,T)         Observed time-varying covariate.
    A (N,T) int     Binned treatment, values in {0, 1, ..., K-1}.
    Y (N,T) int     Daily death indicator (1 on event day, 0 elsewhere; absorbing).
    at_risk (N,T)   1 if subject still alive at start of day t, else 0.
    """
    Z: np.ndarray
    L: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    at_risk: np.ndarray


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _bin_continuous(score: np.ndarray, K: int, edges: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-bin a continuous score into K integer levels in [0, K-1]."""
    if edges is None:
        qs = np.linspace(0, 1, K + 1)[1:-1]
        edges = np.quantile(score, qs)
    binned = np.searchsorted(edges, score)
    return np.clip(binned, 0, K - 1).astype(np.int8), edges


def simulate_tv_latent_confounding(cfg: DGPConfig) -> DGPSample:
    """Simulate (Z, L, A, Y, at_risk) under the per-day hazard DGP.

    Z_0 ~ N(0, 1)
    Z_t = psi*Z_{t-1} + N(0, sigma_Z^2)
    L_t = 0.5*Z_t + 0.3*L_{t-1} + N(0, sigma_L^2)
    score_t = 0.6*L_t + gamma*Z_t + 0.4*A_{t-1}/K + N(0, 0.5^2)
    A_t = quantile_bin(score_t, K)                    in {0, ..., K-1}
    h_t = expit(hazard_intercept + beta_A * A_t/(K-1)
                + beta_L * L_t + gamma * Z_t)
    Y_t ~ Bernoulli(h_t) | survived_{t-1}; absorbing on event.
    """
    rng = np.random.default_rng(cfg.seed)
    N, T, K = cfg.N, cfg.T, cfg.K
    Z = np.zeros((N, T))
    L = np.zeros((N, T))
    A = np.zeros((N, T), dtype=np.int8)
    Y = np.zeros((N, T), dtype=np.int8)
    at_risk = np.zeros((N, T), dtype=np.float32)
    Z[:, 0] = rng.normal(0.0, 1.0, N)
    survived = np.ones(N, dtype=bool)
    edges_per_t: list[np.ndarray] = []
    A_prev_norm = np.zeros(N, dtype=np.float64)
    for t in range(T):
        if t > 0:
            Z[:, t] = cfg.psi * Z[:, t - 1] + rng.normal(0.0, cfg.sigma_Z, N)
            L[:, t] = (
                0.5 * Z[:, t] + 0.3 * L[:, t - 1]
                + rng.normal(0.0, cfg.sigma_L, N)
            )
        else:
            L[:, t] = 0.5 * Z[:, t] + rng.normal(0.0, cfg.sigma_L, N)
        score = (
            0.6 * L[:, t] + cfg.gamma * Z[:, t]
            + 0.4 * A_prev_norm + rng.normal(0.0, 0.5, N)
        )
        a_t, edges = _bin_continuous(score, K, edges=None if t == 0 else edges_per_t[0])
        edges_per_t.append(edges)
        A[:, t] = a_t
        A_prev_norm = (a_t.astype(np.float64) / max(K - 1, 1))
        at_risk[:, t] = survived.astype(np.float32)
        h_t = _expit(
            cfg.hazard_intercept
            + cfg.beta_A * A_prev_norm
            + cfg.beta_L * L[:, t]
            + cfg.gamma * Z[:, t]
        )
        new_event = (rng.uniform(size=N) < h_t) & survived
        Y[:, t] = new_event.astype(np.int8)
        survived = survived & ~new_event
    return DGPSample(Z=Z, L=L, A=A, Y=Y, at_risk=at_risk)
