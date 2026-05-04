"""GPU-accelerated FRE-NICE dose_response (JAX) — drop-in replacement for
the slow numpy implementation in fre_nice_bayesian.py.

Loads posterior + state from existing {prefix}_state.npz files. Compiles
once via jit + scan over time, vmaps over posterior draws. On V100 GPU
expected to finish 20-bin × 200-posterior counterfactual in 5-15 min vs
1.5-6 h for the numpy version.

Xu Bayesian dose remains numpy-based (already fast: ~3h, no L sim, simpler).
"""
from __future__ import annotations
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import jax
# Enable float64 to match numpy precision (avoids ~0.3-0.5%p drift in
# cumulative incidence accumulation over T=28 timesteps).
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr


@partial(jax.jit, static_argnames=("K_A_minus_1", "p_dyn", "p_stat", "T", "n_b"))
def _one_posterior_one_bin(
    rng_key: jax.Array,
    beta: jnp.ndarray,                     # (p_outcome,)
    L_chol: jnp.ndarray,                   # (K_re, K_re)
    B_basis: jnp.ndarray,                  # (T, K_re)
    beta_L: jnp.ndarray,                   # (p_dyn, p_hist)
    sd_L: jnp.ndarray,                     # (p_dyn,)
    L_t0: jnp.ndarray,                     # (M, p_dyn)
    C_mc: jnp.ndarray,                     # (M, p_stat)
    A_onehot_dropped: jnp.ndarray,         # (M, K_A_minus_1)
    *, K_A_minus_1: int, p_dyn: int, p_stat: int, T: int, n_b: int,
) -> jnp.ndarray:
    """Mean cumulative incidence for one posterior draw, one target bin.

    Forward simulates L_t under intervention A=bin_k for t=1..T-1; uses observed L_0.
    Marginalizes over n_b draws of b ~ N(0, Σ_b).
    """
    M = L_t0.shape[0]
    K_re = B_basis.shape[1]

    # Sample b: (n_b, M, K_re), then random_logit (n_b, M, T)
    z = jr.normal(rng_key, shape=(n_b, M, K_re))
    b = z @ L_chol.T
    random_logit = b @ B_basis.T

    # Broadcast static-per-iteration tensors to (n_b, M, ...)
    A_b = jnp.broadcast_to(A_onehot_dropped, (n_b, M, K_A_minus_1))
    C_b = jnp.broadcast_to(C_mc, (n_b, M, p_stat))
    L_t0_b = jnp.broadcast_to(L_t0, (n_b, M, p_dyn))

    def step(carry, t):
        L_prev, survived, cum, key = carry
        # History features for L sim: [bias, L_prev, A_b, C_b, t/T]
        bias_col = jnp.ones((n_b, M, 1))
        t_col = jnp.full((n_b, M, 1), t / jnp.maximum(T - 1, 1))
        X_hist = jnp.concatenate([bias_col, L_prev, A_b, C_b, t_col], axis=-1)
        # mu_L: (n_b, M, p_dyn) = einsum X_hist @ beta_L^T
        mu_L = jnp.einsum("nmh,jh->nmj", X_hist, beta_L)
        key, sub = jr.split(key)
        noise = jr.normal(sub, shape=(n_b, M, p_dyn)) * sd_L
        L_new = mu_L + noise
        # At t=0, no history; use observed L_t0
        L_t = jnp.where(t == 0, L_t0_b, L_new)

        # Outcome features: [bias, A_b, L_t, C_b, t/T]
        X_out = jnp.concatenate([bias_col, A_b, L_t, C_b, t_col], axis=-1)
        eta_fix = jnp.einsum("nmh,h->nm", X_out, beta)
        logit_t = eta_fix + random_logit[:, :, t]
        p_t = jax.nn.sigmoid(jnp.clip(logit_t, -30.0, 30.0))
        cum_new = cum + survived * p_t
        survived_new = survived * (1.0 - p_t)
        return (L_t, survived_new, cum_new, key), None

    L_init = jnp.zeros((n_b, M, p_dyn))   # placeholder; t=0 branch uses L_t0_b
    survived0 = jnp.ones((n_b, M))
    cum0 = jnp.zeros((n_b, M))
    init = (L_init, survived0, cum0, rng_key)
    (_, _, cum_final, _), _ = jax.lax.scan(step, init, jnp.arange(T))
    return cum_final.mean()


@partial(jax.jit, static_argnames=("K_A_minus_1", "p_dyn", "p_stat", "T", "n_b"))
def _one_posterior_one_bin_spec2(
    rng_key: jax.Array,
    beta: jnp.ndarray,
    L_chol: jnp.ndarray,
    B_basis: jnp.ndarray,
    beta_L_aug: jnp.ndarray,                # (p_dyn, p_hist + 1) with lambda_j as last col
    sd_L: jnp.ndarray,
    L_t0: jnp.ndarray,
    C_mc: jnp.ndarray,
    A_onehot_dropped: jnp.ndarray,
    *, K_A_minus_1: int, p_dyn: int, p_stat: int, T: int, n_b: int,
) -> jnp.ndarray:
    """Spec ② counterfactual: forward L sim uses augmented beta_L
    with extra column lambda_j * b_subj^T B(t-1). L now responds to RE."""
    M = L_t0.shape[0]
    K_re = B_basis.shape[1]
    z = jr.normal(rng_key, shape=(n_b, M, K_re))
    b = z @ L_chol.T                                    # (n_b, M, K_re)
    random_logit = b @ B_basis.T                        # (n_b, M, T)

    A_b = jnp.broadcast_to(A_onehot_dropped, (n_b, M, K_A_minus_1))
    C_b = jnp.broadcast_to(C_mc, (n_b, M, p_stat))
    L_t0_b = jnp.broadcast_to(L_t0, (n_b, M, p_dyn))

    def step(carry, t):
        L_prev, survived, cum, key = carry
        bias_col = jnp.ones((n_b, M, 1))
        t_col = jnp.full((n_b, M, 1), t / jnp.maximum(T - 1, 1))
        # Spec ② extra column: b @ B(t-1), padded by B[0] at t=0 (irrelevant, L overridden)
        B_idx = jnp.maximum(t - 1, 0)
        re_col = jnp.einsum("nmk,k->nm", b, B_basis[B_idx]).reshape(n_b, M, 1)
        X_hist = jnp.concatenate(
            [bias_col, L_prev, A_b, C_b, t_col, re_col], axis=-1,
        )
        mu_L = jnp.einsum("nmh,jh->nmj", X_hist, beta_L_aug)
        key, sub = jr.split(key)
        noise = jr.normal(sub, shape=(n_b, M, p_dyn)) * sd_L
        L_new = mu_L + noise
        L_t = jnp.where(t == 0, L_t0_b, L_new)

        X_out = jnp.concatenate([bias_col, A_b, L_t, C_b, t_col], axis=-1)
        eta_fix = jnp.einsum("nmh,h->nm", X_out, beta)
        logit_t = eta_fix + random_logit[:, :, t]
        p_t = jax.nn.sigmoid(jnp.clip(logit_t, -30.0, 30.0))
        cum_new = cum + survived * p_t
        survived_new = survived * (1.0 - p_t)
        return (L_t, survived_new, cum_new, key), None

    L_init = jnp.zeros((n_b, M, p_dyn))
    survived0 = jnp.ones((n_b, M))
    cum0 = jnp.zeros((n_b, M))
    init = (L_init, survived0, cum0, rng_key)
    (_, _, cum_final, _), _ = jax.lax.scan(step, init, jnp.arange(T))
    return cum_final.mean()


def fre_nice_dose_response_jax(
    posterior: dict,
    B_basis: np.ndarray,
    beta_L_list: list[np.ndarray],
    sd_L_list: list[float],
    L_obs: np.ndarray,
    C_static: np.ndarray,
    target_bins: list[int],
    K_A: int, ref_bin: int,
    n_posterior_subset: int = 200,
    n_b_draws_per_post: int = 5,
    seed: int = 0,
    share_RE_on_L: bool = False,            # Spec ② flag
) -> np.ndarray:
    """JAX FRE-NICE dose_response. Returns (S, K_bins) risk matrix."""
    N, T, p_dyn = L_obs.shape
    p_stat = C_static.shape[1]
    K_re = B_basis.shape[1]
    K_A_minus_1 = K_A - 1
    n_b = n_b_draws_per_post

    rng_master = np.random.default_rng(seed)
    S_total = posterior["beta"].shape[0]
    S = min(n_posterior_subset, S_total)
    idx = rng_master.choice(S_total, size=S, replace=False)
    beta_post = jnp.asarray(posterior["beta"][idx], dtype=jnp.float64)
    L_chol_post = jnp.asarray(posterior["L_chol"][idx], dtype=jnp.float64)

    M = N
    idx0 = rng_master.integers(0, N, size=M)
    L_t0 = jnp.asarray(L_obs[idx0, 0, :], dtype=jnp.float64)
    C_mc = jnp.asarray(C_static[idx0], dtype=jnp.float64)
    beta_L = jnp.asarray(np.stack(beta_L_list), dtype=jnp.float64)
    sd_L = jnp.asarray(np.array(sd_L_list), dtype=jnp.float64)
    B_basis_j = jnp.asarray(B_basis, dtype=jnp.float64)

    # Choose Spec ① (no L-RE) or Spec ② (shared RE on L) implementation
    inner_fn = _one_posterior_one_bin_spec2 if share_RE_on_L else _one_posterior_one_bin
    in_axes = (0, 0, 0) + (None,) * 6
    vmapped = jax.vmap(
        partial(inner_fn,
                K_A_minus_1=K_A_minus_1, p_dyn=p_dyn, p_stat=p_stat, T=T, n_b=n_b),
        in_axes=in_axes,
    )

    rng_root = jr.PRNGKey(seed)
    risks_per_bin = []
    for ki, k in enumerate(target_bins):
        A_full = np.zeros((M, K_A), dtype=np.float32)
        A_full[:, k] = 1.0
        keep = [j for j in range(K_A) if j != ref_bin]
        A_dropped = jnp.asarray(A_full[:, keep], dtype=jnp.float64)
        rng_root, sub = jr.split(rng_root)
        keys = jr.split(sub, S)
        risks = vmapped(
            keys, beta_post, L_chol_post,
            B_basis_j, beta_L, sd_L, L_t0, C_mc, A_dropped,
        )
        risks_np = np.asarray(risks)
        risks_per_bin.append(risks_np)
        print(f"  bin {k:2d}: mean = {float(risks_np.mean()):.4f}")

    return np.stack(risks_per_bin, axis=1)            # (S, K_bins)
