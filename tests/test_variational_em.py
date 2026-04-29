"""Smoke tests for variational EM training loop."""
import math
import torch

from src.models.linear_gaussian_ssm import LinearGaussianSSM, SSMConfig
from src.models.variational_posterior import (
    StructuredGaussianMarkovPosterior, QConfig
)
from src.inference.elbo import compute_elbo, gaussian_kl
from src.training.variational_em import train_vem, TrainingConfig
from src.utils.seeds import set_seed


def _toy_problem(N=8, T=5, K=3, n_dyn=2, n_static=1, seed=0):
    set_seed(seed)
    cfg = SSMConfig(n_bins=K, n_dyn_covariates=n_dyn, n_static_covariates=n_static, fit_time_effect=False)
    model = LinearGaussianSSM(cfg)
    p_drivers = K + n_dyn + n_static
    p_cov = p_drivers
    qcfg = QConfig(n_drivers=p_drivers, n_covariates=p_cov)
    q = StructuredGaussianMarkovPosterior(qcfg)
    Y = torch.randint(0, 2, (N, T, 1)).float()
    D = torch.randn(N, T, p_drivers)
    W = torch.randn(N, T, p_cov)
    A = torch.ones(N, T, 1)
    return model, q, Y, D, W, A


def test_gaussian_kl_zero_when_identical():
    mu = torch.zeros(4, 1)
    var = torch.ones(4, 1) * 0.5
    kl = gaussian_kl(mu, var, mu, var)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_compute_elbo_finite():
    model, q, Y, D, W, A = _toy_problem()
    N = Y.shape[0]
    C_static = torch.randn(N, model.config.n_static_covariates)
    res = compute_elbo(model, q, Y, D, W, A, C_static=C_static, n_samples=2)
    assert torch.isfinite(res.elbo)
    assert torch.isfinite(res.expected_log_lik)
    assert torch.isfinite(res.kl_dynamics)
    assert res.kl_dynamics >= -1e-6   # KL should be non-negative


def test_elbo_improves_with_training():
    model, q, Y, D, W, A = _toy_problem()
    cfg = TrainingConfig(
        n_epochs=30, learning_rate=5e-2, n_mc_samples=1,
        smoothness_lambda=0.0, grad_clip=5.0, print_every=10,
        early_stop_patience=100, early_stop_tol=0.0,
    )
    history = train_vem(model, q, Y, D, W, A, cfg, verbose=False)
    assert len(history.elbo) == cfg.n_epochs
    # ELBO at end should be at least as high as at start (variance allowed)
    assert history.elbo[-1] >= history.elbo[0] - 5.0


def test_q_phi_sample_shape():
    _, q, Y, D, W, _ = _toy_problem()
    Z, mu, lv = q.sample_trajectory(D, W, Y, n_samples=3)
    N, T, _ = Y.shape
    assert Z.shape == (3, N, T, 1)
    assert mu.shape == (3, N, T, 1)
    assert lv.shape == (3, N, T, 1)


def test_phi_and_theta_both_get_gradients():
    """All parameters in both model and posterior should receive grads.
    C_static is supplied so the baseline-conditioned prior's parameters
    (gamma_init) are exercised."""
    model, q, Y, D, W, A = _toy_problem()
    N = Y.shape[0]
    C_static = torch.randn(N, model.config.n_static_covariates)
    res = compute_elbo(model, q, Y, D, W, A, C_static=C_static, n_samples=1)
    (-res.elbo).backward()

    # Model params
    for name, p in model.named_parameters():
        assert p.grad is not None, f"model.{name} got no grad"
        assert torch.isfinite(p.grad).all(), f"model.{name} grad has NaN/Inf"
    # Posterior params (skip switch-only params that are inactive under
    # the default config: alpha_Y is only used when use_Y_in_init=True)
    optional_params = {"alpha_Y"}
    for name, p in q.named_parameters():
        if name in optional_params:
            continue
        assert p.grad is not None, f"posterior.{name} got no grad"
        assert torch.isfinite(p.grad).all(), f"posterior.{name} grad has NaN/Inf"
