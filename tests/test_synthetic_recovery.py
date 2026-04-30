"""Synthetic-data validation: variational EM convergence check on a
known linear-Gaussian SSM with generative L emission and Bernoulli outcome."""
import math
import torch

from src.models.linear_gaussian_ssm import LinearGaussianSSM, SSMConfig
from src.models.variational_posterior import (
    StructuredGaussianMarkovPosterior, QConfig
)
from src.training.variational_em import train_vem, TrainingConfig
from src.utils.seeds import set_seed


def _simulate(N=300, T=10, K=3, p_dyn=2, p_stat=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.zeros(N, T, K)
    A.scatter_(2, torch.randint(0, K, (N, T, 1), generator=g), 1.0)
    L = torch.randn(N, T, p_dyn, generator=g) * 0.5
    V = torch.randn(N, p_stat, generator=g) * 0.5
    Y = (torch.rand(N, T, 1, generator=g) < 0.15).float()
    at_risk = torch.ones(N, T, 1)
    t_norm = (torch.arange(T).float() / max(T - 1, 1))[None, :, None].expand(N, T, 1).contiguous()
    return Y, A, L, V, at_risk, t_norm


def test_elbo_improves_on_synthetic():
    set_seed(0)
    K, p_dyn, p_stat = 3, 2, 1
    Y, A, L, V, at_risk, t_norm = _simulate(N=200, T=10, K=K, p_dyn=p_dyn, p_stat=p_stat)
    cfg = SSMConfig(
        n_bins=K, n_dyn_covariates=p_dyn, n_static_covariates=p_stat,
        fit_time_effect=False,
    )
    model = LinearGaussianSSM(cfg)
    q = StructuredGaussianMarkovPosterior(QConfig(
        n_bins=K, n_dyn_covariates=p_dyn, n_static_covariates=p_stat,
    ))
    train_cfg = TrainingConfig(
        n_epochs=80, learning_rate=2e-2, n_mc_samples=2,
        smoothness_lambda=0.0, grad_clip=5.0, print_every=200,
        early_stop_patience=300, early_stop_tol=0.0,
    )
    history = train_vem(model, q, Y, A, L, V, at_risk, t_norm, train_cfg, verbose=False)
    assert history.elbo[-1] > history.elbo[0] + 1.0
    assert math.isfinite(history.elbo[-1])
    assert history.kl[-1] >= -1e-3
    assert abs(float(model.beta_Z.detach().cpu())) < 5.0
