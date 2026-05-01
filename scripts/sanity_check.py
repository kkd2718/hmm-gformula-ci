"""Sanity check on dose-response counterfactuals vs raw cohort mortality.

Three diagnostics:
(A) Natural-course simulation: forward-sim with each subject's OBSERVED A_t
    trajectory under the fitted VEM-SSM (no intervention). Result should
    approximate cohort 28-day mortality (~25.6%); large gap = model bias.

(B) Bin-stratified raw mortality: compute observed 28-day mortality stratified
    by day-0 MP bin. This is the confounded crude rate; counterfactual
    estimates should follow a similar shape (monotone direction at minimum).

(C) Counterfactual range comparison: contrast each method's bin 0 / bin 16
    / max-bin counterfactual vs (a) cohort overall 25.6%, (b) bin-stratified
    crude rates.
"""
from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, load_ards_cohort
from src.models.linear_gaussian_ssm import LinearGaussianSSM, SSMConfig
from src.models.variational_posterior import (
    StructuredGaussianMarkovPosterior, QConfig
)
from src.training.variational_em import train_vem, TrainingConfig


@torch.no_grad()
def natural_course_simulation(
    model: LinearGaussianSSM, cohort, n_mc_subjects: int = 5000, seed: int = 42,
) -> float:
    """Forward-simulate with each subject's OBSERVED A_t (no intervention)."""
    device = next(model.parameters()).device
    A = cohort.A_bin.to(device)               # (N, T, K)
    L0 = cohort.L_dyn[:, 0, :].to(device)
    V = cohort.C_static.to(device)
    t_norm = cohort.t_norm.to(device)
    N, T, K = A.shape
    p_dyn = L0.shape[1]
    rng_idx = np.random.default_rng(seed)
    idx = rng_idx.integers(0, N, size=min(n_mc_subjects, N))
    A_mc = A[idx]; L0_mc = L0[idx]; V_mc = V[idx]; t_mc = t_norm[idx]
    M = A_mc.shape[0]

    Z_mean, Z_var = model.initial_state(V_mc, M, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    Z = Z_mean + torch.sqrt(Z_var) * torch.randn(Z_mean.shape, generator=gen, device=device)
    L_t = L0_mc.clone()
    survived = torch.ones(M, 1, device=device)
    cumulative = torch.zeros(M, 1, device=device)

    for t in range(T):
        if t > 0:
            A_lag = A_mc[:, t - 1, :]
            L_lag = L_t
            Z_next_mean, Z_var_add = model.transition(Z, A_lag, L_lag, V_mc)
            Z_new = Z_next_mean + torch.sqrt(Z_var_add) * torch.randn(
                Z_next_mean.shape, generator=gen, device=device
            )
            Z = survived * Z_new + (1.0 - survived) * Z
            if p_dyn > 0:
                L_mean, L_var = model.l_emission(Z, A_lag, L_lag, V_mc)
                L_new = L_mean + torch.sqrt(L_var) * torch.randn(
                    L_mean.shape, generator=gen, device=device
                )
                L_t = survived * L_new + (1.0 - survived) * L_t
        logit = model.outcome_logit(Z, A_mc[:, t, :], L_t, V_mc, t_mc[:, t, :])
        p_Y = torch.sigmoid(logit)
        new_event = survived * p_Y
        cumulative = cumulative + new_event
        survived = survived * (1.0 - p_Y)

    return float(cumulative.mean().cpu())


def bin_stratified_raw_mortality(
    csv_path: Path, n_bins: int = 20, max_t: int = 28,
) -> tuple[pd.DataFrame, float]:
    """Compute 28-day mortality stratified by day-0 MP bin."""
    df = pd.read_csv(csv_path)
    df = df[df["day_num"] < max_t].copy()
    first_day = df.sort_values("day_num").groupby("stay_id").first().reset_index()
    death28 = df.groupby("stay_id")["death_event"].max().rename("mortality_28d")
    stay = first_day[["stay_id", "subject_id", "mp_j_min"]].merge(death28, on="stay_id")
    overall_mortality = float(stay["mortality_28d"].mean())

    mp = stay["mp_j_min"].dropna()
    log_mp = np.log(np.clip(mp, 1e-3, None))
    mu, sd = log_mp.mean(), log_mp.std()
    z_edges = np.linspace(-2.5, 2.5, n_bins + 1)
    edges_mp = np.exp(mu + sd * z_edges)
    stay["bin"] = np.clip(
        np.digitize(stay["mp_j_min"].fillna(0), edges_mp, right=True) - 1,
        0, n_bins - 1,
    )

    rows = []
    for k in range(n_bins):
        sub = stay[stay["bin"] == k]
        if len(sub) == 0:
            rows.append({
                "bin": k, "n_stays": 0,
                "mp_median": float("nan"), "mortality_pct": float("nan"),
            })
        else:
            rows.append({
                "bin": k, "n_stays": len(sub),
                "mp_median": float(sub["mp_j_min"].median()),
                "mortality_pct": float(sub["mortality_28d"].mean() * 100),
            })
    return pd.DataFrame(rows), overall_mortality


@torch.no_grad()
def standard_natural_course(model, cohort) -> float:
    """Standard NICE: forward sim with observed A trajectory (no intervention)."""
    K, p_dyn, p_stat, T = model._n_bins, model._n_dyn, model._n_static, model._t_max
    L_obs = cohort.L_dyn.numpy().astype(np.float64)
    A_obs = cohort.A_bin.numpy().astype(np.float64)
    C_obs = cohort.C_static.numpy().astype(np.float64)
    N = L_obs.shape[0]
    rng = np.random.default_rng(42)

    L_t = L_obs[:, 0, :].copy()
    survived = np.ones(N, dtype=np.float64)
    cum = np.zeros(N, dtype=np.float64)
    for t in range(T):
        A_t_onehot = A_obs[:, t, :]
        if t > 0:
            A_lag = A_obs[:, t - 1, :]
            X_hist = model._build_history_features(
                L_prev=L_t, A_prev_onehot=A_lag, C=C_obs, t_idx=t, T=T,
            )
            L_new = np.empty_like(L_t)
            for j in range(p_dyn):
                mu_j = X_hist @ model._beta_L[j]
                L_new[:, j] = mu_j + rng.normal(0.0, model._sd_L[j], size=N)
            L_t = L_new
        X_out = model._build_outcome_features(
            L_t=L_t, A_t_onehot=A_t_onehot, C=C_obs, t_idx=t, T=T,
        )
        from src.benchmarks.standard_gformula import _expit as _exp
        p_t = _exp(X_out @ model._beta_Y)
        cum = cum + survived * p_t
        survived = survived * (1.0 - p_t)
    return float(cum.mean())


@torch.no_grad()
def xu_natural_course(model, cohort, n_b_draws: int = 50, seed: int = 42) -> float:
    """Xu GLMM: forward sim with observed A and observed L (no L sim per Xu convention)."""
    from src.benchmarks.xu_glmm import _expit
    rng = np.random.default_rng(seed)
    L = cohort.feature_layout
    K = L["n_bins"]
    N, T = cohort.Y.shape[0], cohort.Y.shape[1]
    cov = cohort.covariates.numpy().reshape(N * T, -1).astype(np.float64)
    bias = np.ones((cov.shape[0], 1))
    X = np.concatenate([bias, cov], axis=1)
    eta_no_b = (X @ model._beta).reshape(N, T)
    risks = []
    for _ in range(n_b_draws):
        b = rng.normal(0.0, model._sigma_b, size=N)
        eta = eta_no_b + b[:, None]
        p = _expit(eta)
        survived = np.ones(N); cum = np.zeros(N)
        for t in range(T):
            cum = cum + survived * p[:, t]
            survived = survived * (1.0 - p[:, t])
        risks.append(float(cum.mean()))
    return float(np.mean(risks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--snapshot-json", type=Path, default=None,
                        help="VEM param snapshot from main_v3_optionB.")
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--max-t", type=int, default=28)
    parser.add_argument("--vem-epochs", type=int, default=50,
                        help="If snapshot not provided, fit VEM from scratch.")
    parser.add_argument("--n-mc-subjects", type=int, default=5000)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("[B/C] Loading cohort + computing bin-stratified raw mortality ...")
    bin_df, overall = bin_stratified_raw_mortality(
        args.csv, n_bins=args.n_bins, max_t=args.max_t,
    )
    print(f"  Cohort overall 28-day mortality = {overall * 100:.2f}%")
    print(f"\n  Bin-stratified raw mortality (day-0 MP bin):")
    print(bin_df.to_string(index=False))

    cohort = load_ards_cohort(ARDSConfig(
        csv_path=args.csv, n_bins=args.n_bins, max_t=args.max_t,
    ))

    print("\n[A] Fitting (or loading) VEM-SSM and running natural-course simulation ...")
    layout = cohort.feature_layout
    ssm_cfg = SSMConfig(
        n_bins=layout["n_bins"],
        n_dyn_covariates=layout["n_dyn"],
        n_static_covariates=layout["n_static"],
        z_depends_on_treatment_lag=True,
        fit_time_effect=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearGaussianSSM(ssm_cfg).to(device)
    posterior = StructuredGaussianMarkovPosterior(QConfig(
        n_bins=layout["n_bins"], n_dyn_covariates=layout["n_dyn"],
        n_static_covariates=layout["n_static"],
    )).to(device)

    if args.snapshot_json and args.snapshot_json.exists():
        snap = json.loads(args.snapshot_json.read_text())
        with torch.no_grad():
            model.psi.copy_(torch.tensor([snap["psi"]], device=device))
            model.log_sigma_Z.copy_(torch.log(torch.tensor([snap["sigma_Z"]], device=device)))
            model.beta_0_param.copy_(torch.tensor([snap["beta_0"]], device=device))
            model._beta_Z.copy_(torch.tensor([snap["beta_Z"]], device=device))
            model.beta_A.copy_(torch.tensor(snap["beta_A"], device=device))
            model.gamma_A.copy_(torch.tensor(snap["gamma_A"], device=device))
            if "delta_Z" in snap:
                model.delta_Z.copy_(torch.tensor(snap["delta_Z"], device=device))
            if "alpha_L" in snap:
                model.alpha_L.copy_(torch.tensor(snap["alpha_L"], device=device))
            if "delta_A" in snap and model.delta_A is not None:
                model.delta_A.weight.copy_(torch.tensor(snap["delta_A"], device=device))
            if "sigma_L" in snap and model.log_sigma_L is not None:
                model.log_sigma_L.copy_(torch.log(torch.tensor(snap["sigma_L"], device=device)))
        print(f"  Loaded VEM parameters from snapshot.")
    else:
        print(f"  Fitting VEM from scratch (epochs={args.vem_epochs}) ...")
        train_vem(
            model, posterior,
            Y=cohort.Y, A=cohort.A_bin, L=cohort.L_dyn, V=cohort.C_static,
            at_risk=cohort.at_risk, t_norm=cohort.t_norm,
            config=TrainingConfig(
                n_epochs=args.vem_epochs, learning_rate=1e-2, n_mc_samples=2,
                smoothness_lambda=0.02,
            ),
            verbose=True,
        )

    nat_course = natural_course_simulation(model, cohort, n_mc_subjects=args.n_mc_subjects)

    # Also fit Standard and Xu for natural-course comparison
    print("\n  Fitting Standard g-formula for natural-course comparison ...")
    from src.benchmarks import StandardGFormula, XuGLMM
    m_std = StandardGFormula(); m_std.fit(cohort)
    nc_std = standard_natural_course(m_std, cohort)
    print(f"  Standard natural-course = {nc_std * 100:.2f}%")

    print("\n  Fitting Xu GLMM for natural-course comparison ...")
    m_xu = XuGLMM(); m_xu.fit(cohort)
    nc_xu = xu_natural_course(m_xu, cohort)
    print(f"  Xu natural-course = {nc_xu * 100:.2f}%")

    print(f"\n=== NATURAL-COURSE SUMMARY ===")
    print(f"  Cohort raw                = {overall * 100:.2f}%")
    print(f"  Standard natural-course   = {nc_std * 100:.2f}%  (gap {(nc_std - overall) * 100:+.2f}%p)")
    print(f"  Xu natural-course         = {nc_xu * 100:.2f}%  (gap {(nc_xu - overall) * 100:+.2f}%p)")
    print(f"  VEM natural-course        = {nat_course * 100:.2f}%  (gap {(nat_course - overall) * 100:+.2f}%p)")

    md = [
        "# Sanity check: counterfactual estimates vs raw cohort observation",
        "",
        f"_Cohort overall 28-day mortality (raw): **{overall * 100:.2f}%**_",
        "",
        "## Diagnostic A — Natural-course simulation (no intervention) — 3 methods",
        f"Each method's forward simulation with OBSERVED A_t trajectory (no MP override).",
        f"PASS criterion: |gap vs raw| < ~5%p; large gap → model self-bias.",
        "",
        f"| Method | Natural-course (%) | Gap vs raw 25.61% (%p) | Verdict |",
        f"|---|---|---|---|",
        f"| Cohort raw | {overall * 100:.2f} | 0 | ground truth |",
        f"| Standard parametric g-formula | {nc_std * 100:.2f} | {(nc_std - overall) * 100:+.2f} | {'PASS' if abs(nc_std - overall) < 0.05 else 'FAIL'} |",
        f"| Xu 2024 GLMM | {nc_xu * 100:.2f} | {(nc_xu - overall) * 100:+.2f} | {'PASS' if abs(nc_xu - overall) < 0.05 else 'FAIL'} |",
        f"| VEM-SSM (proposed) | {nat_course * 100:.2f} | {(nat_course - overall) * 100:+.2f} | {'PASS' if abs(nat_course - overall) < 0.05 else 'FAIL'} |",
        "",
        "## Diagnostic B — Factual hazard prediction (PPC, see Appendix B)",
        "  Already validated; AUC 0.86-0.92, Brier matches baseline pooled-logistic",
        "",
        "## Diagnostic C — Bin-stratified raw mortality vs counterfactual estimates",
        "",
        "_Raw = observed 28-day mortality of patients whose day-0 MP fell in each bin._",
        "_Counterfactual = each method's estimate from main_v3_optionB._",
        "",
        "| Bin | n stays | Day-0 MP median (J/min) | Raw mortality (%) | Standard CF (%) | Xu CF (%) | VEM CF (%) |",
        "|---|---|---|---|---|---|---|",
    ]

    cf_data = np.load(
        Path(r"C:/Users/기덕/Desktop/Study/Paper/학위논문/_draft/figures/main_v3/main_v3_optionB/table2_risks.npz"),
    )
    cf_std = cf_data["standard_gformula__risk_mean"] * 100
    cf_xu = cf_data["xu_glmm__risk_mean"] * 100
    cf_vem = cf_data["vem_ssm__risk_mean"] * 100

    for k in range(args.n_bins):
        row = bin_df.iloc[k]
        n = int(row["n_stays"])
        mp_med = row["mp_median"]
        raw = row["mortality_pct"]
        md.append(
            f"| {k} | {n:,} | "
            f"{mp_med:.1f}".replace("nan", "—") + f" | "
            f"{raw:.1f}".replace("nan", "—") + f" | "
            f"{cf_std[k]:.1f} | {cf_xu[k]:.1f} | {cf_vem[k]:.1f} |"
        )

    md.extend([
        "",
        "## Interpretation",
        "",
        "- **Raw mortality is confounded** (severity → high MP → death), so monotone",
        "  raw rate increase across bins reflects confounding, not pure causal effect.",
        "- However, raw rates provide a **plausibility window** for counterfactual",
        "  estimates: estimates outside the cohort's observed range merit scrutiny.",
        "- Cohort overall is 25.6%, severity-stratified 17–39%. Counterfactual",
        "  estimates substantially outside this band imply extrapolation.",
        "",
        f"**Diagnostic A verdict**: VEM natural-course = {nat_course * 100:.1f}% vs "
        f"cohort {overall * 100:.1f}%; "
        f"{'PASS' if abs(nat_course - overall) < 0.05 else 'FAIL — model self-bias'}.",
    ])
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
