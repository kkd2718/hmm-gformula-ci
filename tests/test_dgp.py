"""Smoke tests for the per-day-hazard, multinomial-A simulation DGP."""
import numpy as np

from src.simulation.dgp import simulate_tv_latent_confounding, DGPConfig


def test_dgp_shapes_and_marginals():
    cfg = DGPConfig(N=200, T=14, K=4, gamma=0.5, seed=0)
    s = simulate_tv_latent_confounding(cfg)
    assert s.Z.shape == (200, 14)
    assert s.L.shape == (200, 14)
    assert s.A.shape == (200, 14)
    assert s.Y.shape == (200, 14)
    assert s.at_risk.shape == (200, 14)
    assert set(np.unique(s.A).tolist()).issubset(set(range(4)))
    assert set(np.unique(s.Y).tolist()).issubset({0, 1})
    # at_risk is non-increasing along time (absorbing event)
    assert (s.at_risk[:, 1:] <= s.at_risk[:, :-1] + 1e-9).all()
    # At most one event per subject
    assert (s.Y.sum(axis=1) <= 1).all()


def test_dgp_gamma_controls_direct_zlink():
    """gamma=0 should give weaker partial Z->A association than gamma>0."""
    s0 = simulate_tv_latent_confounding(DGPConfig(N=2000, T=5, K=4, gamma=0.0, seed=42))
    s1 = simulate_tv_latent_confounding(DGPConfig(N=2000, T=5, K=4, gamma=1.0, seed=42))

    def _partial_zA_given_L(s):
        Z, L, A = s.Z[:, -1], s.L[:, -1], s.A[:, -1].astype(float)
        bA = np.polyfit(L, A, 1); rA = A - (bA[0] * L + bA[1])
        bZ = np.polyfit(L, Z, 1); rZ = Z - (bZ[0] * L + bZ[1])
        return float(np.corrcoef(rZ, rA)[0, 1])

    assert abs(_partial_zA_given_L(s0)) < abs(_partial_zA_given_L(s1))


def test_dgp_event_prevalence_reasonable():
    """28-day cumulative incidence should be in a sensible clinical range."""
    s = simulate_tv_latent_confounding(DGPConfig(N=1000, T=28, K=4, gamma=0.5, seed=7))
    cum_inc = float(s.Y.sum(axis=1).mean())
    assert 0.05 < cum_inc < 0.95
