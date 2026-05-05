"""Laplace approximation for SplineGLMM marginal likelihood + Adam outer loop.

Per-subject random effect b_i is integrated out via Laplace approximation
following the standard hierarchical GLMM fitting procedure (Wolfinger 1993;
Pinheiro & Bates 2000; Wood 2017, ch. 5).

Algorithm
---------
Outer loop (over fixed-effect / Sigma_b parameters theta):
    For each batch of subjects:
        1. Run Newton-Raphson to find b_hat_i = argmax_b [log p(y_i | theta, b_i)
                                                         - 0.5 b_i^T Sigma_b^{-1} b_i]
        2. Compute Hessian H_i at b_hat_i.
        3. Laplace marginal log-likelihood per subject (constants dropped):
             ell_i = log p(y_i | theta, b_hat_i)
                   - 0.5 b_hat_i^T Sigma_b^{-1} b_hat_i
                   - 0.5 log|H_i|
                   - 0.5 log|Sigma_b|
        4. Gradient ascent on theta via Adam, treating b_hat_i as fixed
           (KKT/IFT-justified: first-order optimality of b_hat_i wipes out
            the implicit gradient term at the mode).

Inner Newton vectorizes across subjects: with K_re = 5 spline basis dim and
N = 17,878 subjects, each outer step costs O(N * K_re^2) per Newton iteration
plus the linear-time fixed-effect logit computation. Convergence in 5-10
inner iterations.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor

from ..models.spline_glmm import SplineGLMM


@dataclass
class LaplaceTrainingConfig:
    n_outer_epochs: int = 200
    learning_rate: float = 5e-3
    inner_max_iter: int = 30
    inner_tol: float = 1e-6
    grad_clip: float = 5.0
    print_every: int = 10
    early_stop_patience: int = 30
    early_stop_tol: float = 1e-5


@dataclass
class LaplaceTrainingHistory:
    marginal_logll: list[float]
    sigma_b_diag_norm: list[float]


def _newton_inner(
    model: SplineGLMM,
    Y: Tensor, A: Tensor, L_dyn: Tensor, V: Tensor,
    at_risk: Tensor, t_norm: Tensor,
    Sigma_inv: Tensor,
    max_iter: int = 30, tol: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Per-subject Newton mode of [log p(y_i | b_i) - 0.5 b_i^T Sigma_inv b_i].

    Vectorizes over the N subjects in the batch.

    Parameters
    ----------
    Y : (N, T) Bernoulli observations (0/1)
    A : (N, T, K_A)
    L_dyn : (N, T, p_dyn)
    V : (N, p_stat)
    at_risk : (N, T) at-risk indicator
    t_norm : (T,) or (N, T)
    Sigma_inv : (K_re, K_re)

    Returns
    -------
    b_hat : (N, K_re)        mode of the per-subject penalized log-lik
    H : (N, K_re, K_re)      observed-information Hessian at the mode
    """
    N = Y.shape[0]
    K_re = model.K_re
    device = Y.device
    dtype = Y.dtype

    # Pre-compute fixed-effect logit (theta-dependent but b-independent)
    fixed = model.fixed_logit(A, L_dyn, V, t_norm)             # (N, T)

    # B is (T, K_re); pre-compute B outer products (T, K_re, K_re)
    B = model.B
    B_outer = B.unsqueeze(-1) * B.unsqueeze(-2)                # (T, K_re, K_re)

    b = torch.zeros(N, K_re, device=device, dtype=dtype)

    Sigma_inv_bcast = Sigma_inv.unsqueeze(0).expand(N, K_re, K_re)  # (N, K, K)

    for it in range(max_iter):
        # eta = fixed + b @ B^T,  shape (N, T)
        eta = fixed + b @ B.t()
        # numerically stable sigmoid + weight
        # p = 1 / (1 + exp(-eta));  w = p * (1 - p)
        p = torch.sigmoid(eta.clamp(-30.0, 30.0))
        w = p * (1.0 - p)
        a = at_risk.float()                                    # (N, T)
        # Score: B^T (a * (y - p)) - Sigma_inv b
        residual = a * (Y - p)                                 # (N, T)
        score = residual @ B - b @ Sigma_inv.t()               # (N, K)
        # Hessian (neg log-prob curvature + prior): einsum('nt,tkl->nkl', a*w, B_outer) + Sigma_inv
        aw = a * w                                             # (N, T)
        H_data = torch.einsum("nt,tkl->nkl", aw, B_outer)      # (N, K, K)
        H = H_data + Sigma_inv_bcast
        # Newton step: solve H @ delta = score
        delta = torch.linalg.solve(H, score.unsqueeze(-1)).squeeze(-1)  # (N, K)
        b_new = b + delta

        max_step = float(delta.abs().max())
        b = b_new
        if max_step < tol:
            break

    # Recompute final Hessian at the mode (for return / log|H| in Laplace)
    eta = fixed + b @ B.t()
    p = torch.sigmoid(eta.clamp(-30.0, 30.0))
    w = p * (1.0 - p)
    aw = at_risk.float() * w
    H_data = torch.einsum("nt,tkl->nkl", aw, B_outer)
    H = H_data + Sigma_inv_bcast

    return b, H


def laplace_marginal_logll(
    model: SplineGLMM,
    Y: Tensor, A: Tensor, L_dyn: Tensor, V: Tensor,
    at_risk: Tensor, t_norm: Tensor,
    inner_max_iter: int = 30, inner_tol: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Total Laplace marginal log-likelihood (sum over subjects) and b_hat.

    Returns
    -------
    total_log_lik : scalar Tensor (gradient flows to fixed effects + Sigma_b)
    b_hat : (N, K_re) detached posterior modes
    """
    # Squeeze trailing dim if present
    if Y.ndim == 3:
        Y = Y.squeeze(-1)
    if at_risk.ndim == 3:
        at_risk = at_risk.squeeze(-1)
    Y = Y.float()

    # 1) Find modes b_hat_i WITHOUT autograd (Newton solver)
    Sigma_inv, log_det_Sigma = model.Sigma_b_inv_logdet()
    with torch.no_grad():
        b_hat, _H_at_mode = _newton_inner(
            model, Y, A, L_dyn, V, at_risk, t_norm,
            Sigma_inv.detach(),
            max_iter=inner_max_iter, tol=inner_tol,
        )
    b_hat = b_hat.detach()

    # 2) Re-evaluate Laplace marginal LL AT b_hat WITH autograd flowing to theta
    #    log p(y | b_hat) - 0.5 b_hat^T Sigma_inv b_hat - 0.5 log|H| - 0.5 log|Sigma_b|
    #    KKT/IFT: first-order term in b_hat vanishes at the mode, so we can
    #    treat b_hat as fixed in the outer gradient.

    # Conditional log-lik
    cond_loglik = model.conditional_loglik_per_subject(
        b_hat, Y, A, L_dyn, V, at_risk, t_norm,
    )                                                          # (N,)

    # Quadratic prior penalty: -0.5 b^T Sigma_inv b per subject
    quad = 0.5 * torch.einsum("nk,kl,nl->n", b_hat, Sigma_inv, b_hat)  # (N,)

    # log|H| at b_hat (recompute differentiably wrt theta)
    fixed = model.fixed_logit(A, L_dyn, V, t_norm)             # (N, T)
    eta = fixed + b_hat @ model.B.t()                          # (N, T)
    p = torch.sigmoid(eta.clamp(-30.0, 30.0))
    w = p * (1.0 - p)
    aw = at_risk.float() * w
    B_outer = model.B.unsqueeze(-1) * model.B.unsqueeze(-2)
    H_data = torch.einsum("nt,tkl->nkl", aw, B_outer)
    H = H_data + Sigma_inv.unsqueeze(0)
    # Cholesky-based logdet (S7 fix per Opus review). torch.logdet on near-PSD
    # batched matrices can silently return wrong sign / NaN; Cholesky on the
    # symmetric positive-definite H is numerically stable and explicit.
    L_H = torch.linalg.cholesky(H)
    logdet_H = 2.0 * L_H.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # (N,)

    # Per-subject Laplace approx (constants -K/2 log(2pi) drop in optimization)
    # ell_i = cond_loglik_i - quad_i - 0.5 logdet_H_i - 0.5 logdet_Sigma_b
    per_subject = cond_loglik - quad - 0.5 * logdet_H
    N = per_subject.shape[0]
    total = per_subject.sum() - 0.5 * N * log_det_Sigma

    return total, b_hat


def train_spline_glmm(
    model: SplineGLMM,
    Y: Tensor, A: Tensor, L_dyn: Tensor, V: Tensor,
    at_risk: Tensor, t_norm: Tensor,
    config: LaplaceTrainingConfig,
    verbose: bool = True,
) -> LaplaceTrainingHistory:
    """Outer Adam loop maximizing Laplace marginal log-likelihood."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = LaplaceTrainingHistory(marginal_logll=[], sigma_b_diag_norm=[])

    best_ll = -float("inf")
    patience_count = 0

    for epoch in range(config.n_outer_epochs):
        optimizer.zero_grad()
        total_ll, _ = laplace_marginal_logll(
            model, Y, A, L_dyn, V, at_risk, t_norm,
            inner_max_iter=config.inner_max_iter,
            inner_tol=config.inner_tol,
        )
        loss = -total_ll
        if not torch.isfinite(loss):
            if verbose:
                print(f"  [epoch {epoch}] non-finite loss, stopping")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        ll_val = float(total_ll.item())
        sigma_b = model.Sigma_b().detach()
        diag_norm = float(torch.diagonal(sigma_b).norm().item())
        history.marginal_logll.append(ll_val)
        history.sigma_b_diag_norm.append(diag_norm)

        if verbose and (epoch % config.print_every == 0 or epoch == config.n_outer_epochs - 1):
            print(
                f"  [epoch {epoch:3d}] marginal_logll = {ll_val:.1f}  "
                f"|diag(Σ_b)| = {diag_norm:.4f}"
            )

        # Early stopping on relative LL change
        if ll_val > best_ll + config.early_stop_tol * max(abs(best_ll), 1.0):
            best_ll = ll_val
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= config.early_stop_patience:
                if verbose:
                    print(
                        f"  early stop @ epoch {epoch}: no improvement in "
                        f"{config.early_stop_patience} epochs"
                    )
                break

    return history
