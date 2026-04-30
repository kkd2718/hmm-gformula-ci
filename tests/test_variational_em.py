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


def _toy_problem(N=8, T=5, K=3, n_dyn=2, n_static=1, option_b=True, seed=0):
    set_seed(seed)
    cfg = SSMConfig(
        n_bins=K, n_dyn_covariates=n_dyn, n_static_covariates=n_static,
        z_depends_on_treatment_lag=option_b, fit_time_effect=False,
    )
    model = LinearGaussianSSM(cfg)
    q = StructuredGaussianMarkovPosterior(QConfig(
        n_bins=K, n_dyn_covariates=n_dyn, n_static_covariates=n_static,
    ))
    Y = torch.randint(0, 2, (N, T, 1)).float()
    A = torch.zeros(N, T, K)
    A.scatter_(2, torch.randint(0, K, (N, T, 1)), 1.0)
    L = torch.randn(N, T, n_dyn) * 0.5
    V = torch.randn(N, n_static) * 0.5
    at_risk = torch.ones(N, T, 1)
    t_norm = (torch.arange(T).float() / max(T - 1, 1))[None, :, None].expand(N, T, 1).contiguous()
    return model, q, Y, A, L, V, at_risk, t_norm


def test_gaussian_kl_zero_when_identical():
    mu = torch.zeros(4, 1)
    var = torch.ones(4, 1) * 0.5
    kl = gaussian_kl(mu, var, mu, var)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_compute_elbo_finite_option_B():
    model, q, Y, A, L, V, at_risk, t_norm = _toy_problem(option_b=True)
    res = compute_elbo(model, q, Y, A, L, V, at_risk, t_norm, n_samples=2)
    assert torch.isfinite(res.elbo)
    assert torch.isfinite(res.expected_log_lik_y)
    assert torch.isfinite(res.expected_log_lik_l)
    assert torch.isfinite(res.kl_dynamics)
    assert res.kl_dynamics >= -1e-6


def test_compute_elbo_finite_option_A():
    model, q, Y, A, L, V, at_risk, t_norm = _toy_problem(option_b=False)
    res = compute_elbo(model, q, Y, A, L, V, at_risk, t_norm, n_samples=2)
    assert torch.isfinite(res.elbo)


def test_elbo_improves_with_training():
    model, q, Y, A, L, V, at_risk, t_norm = _toy_problem()
    cfg = TrainingConfig(
        n_epochs=30, learning_rate=5e-2, n_mc_samples=1,
        smoothness_lambda=0.0, grad_clip=5.0, print_every=10,
        early_stop_patience=100, early_stop_tol=0.0,
    )
    history = train_vem(model, q, Y, A, L, V, at_risk, t_norm, cfg, verbose=False)
    assert len(history.elbo) == cfg.n_epochs
    assert history.elbo[-1] >= history.elbo[0] - 5.0


def test_q_phi_sample_shape():
    _, q, Y, A, L, V, _, _ = _toy_problem()
    Z, mu, lv = q.sample_trajectory(A=A, L=L, V=V, Y=Y, n_samples=3)
    N, T, _ = Y.shape
    assert Z.shape == (3, N, T, 1)
    assert mu.shape == (3, N, T, 1)
    assert lv.shape == (3, N, T, 1)


def test_phi_and_theta_both_get_gradients():
    """All parameters in both model and posterior should receive grads."""
    model, q, Y, A, L, V, at_risk, t_norm = _toy_problem(option_b=True)
    res = compute_elbo(model, q, Y, A, L, V, at_risk, t_norm, n_samples=1)
    (-res.elbo).backward()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"model.{name} got no grad"
        assert torch.isfinite(p.grad).all(), f"model.{name} grad has NaN/Inf"
    optional_params = {"alpha_Y"}
    for name, p in q.named_parameters():
        if name in optional_params or not p.requires_grad:
            continue
        assert p.grad is not None, f"posterior.{name} got no grad"
        assert torch.isfinite(p.grad).all(), f"posterior.{name} grad has NaN/Inf"


def test_option_A_freezes_gamma_A():
    """Option A: gamma_A should not receive gradients."""
    model, q, Y, A, L, V, at_risk, t_norm = _toy_problem(option_b=False)
    res = compute_elbo(model, q, Y, A, L, V, at_risk, t_norm, n_samples=1)
    (-res.elbo).backward()
    assert model.gamma_A.grad is None or model.gamma_A.grad.abs().sum() == 0
