# Bayesian Functional Random-Effect NICE g-formula

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-orange.svg)](https://jax.readthedocs.io)
[![numpyro](https://img.shields.io/badge/numpyro-0.13+-red.svg)](https://num.pyro.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Causal effect of mechanical power on 28-day mortality in acute respiratory
distress syndrome — a retrospective cohort study using a Bayesian g-formula
with **functional random effects on the outcome hazard**.

---

## Overview

This repository implements a **Bayesian parametric g-formula** (NICE) with a
subject-specific time-varying functional random effect (FRE) on the outcome
hazard, parameterized via a natural cubic spline basis. The proposed method
generalizes the scalar random intercept of Xu et al. (2024) to a finite-rank
random function on the day grid, anchored in functional data analysis (Yao,
Müller & Wang, 2005) and generalized additive mixed modeling (Wood, 2017).

The repository accompanies a master's thesis on the dose-response of
mechanical power (MP, J/min) and 28-day in-hospital mortality in ARDS,
estimated on MIMIC-IV v3.1 (N = 17,878 ICU stays, 15,619 unique subjects).

---

## Headline findings

The Bayesian K=5 functional-RE NICE g-formula identifies a **strongly
nonlinear dose-response** of mechanical power on 28-day mortality in ARDS,
with marked acceleration above the Costa et al. (2021) clinical cutoff
of approximately 17 J/min.

| MP exposure (J/min) | 28-day mortality (95% CrI) |
|---|---|
| 2.7 (low) | **8.0%** (5.5–11.4) |
| 6.4 | 14.0% (10.6–17.7) |
| 12.0 | 36.4% (30.0–42.4) |
| **18.3** (Costa cutoff, ref) | **37.3%** (32.8–41.4) |
| 22.6 (high) | **46.1%** (38.9–52.4) |

Key analytical findings (full details in
[`_draft/manuscript/00_results_interpretation.md`](_draft/manuscript/00_results_interpretation.md)):

- **Framework dominance.** The three NICE-family methods (Standard
  frequentist, K=1 NICE Bayesian, K=5 FRE-NICE Bayesian) agree within
  $\pm 5$ percentage points across the dose-response. The Xu (2024)
  MSM-style observed-$L$ plug-in produces a flat $\sim 21\%$ across all
  bins. The 25-percentage-point gap at high MP is the **mediated effect
  through time-varying physiology** that the NICE forward-$L$ simulation
  captures and the MSM observed-$L$ plug-in does not (see Discussion).
- **Methodological preference.** WAIC and PSIS-LOO decisively prefer the
  K=5 functional-RE specification over the K=1 scalar-RE
  ($\Delta\mathrm{ELPD} = +641$); K=1 and Xu Bayesian are confirmed
  mathematically equivalent in fitting ($\Delta\mathrm{ELPD} = -4.9$).
- **Robustness.**
  - Knot sensitivity ($K = 4, 5, 6$): $\le 0.3$ p.p. variation.
  - Spec II ablation (shared RE on $L$): $|\lambda_j| \le 0.011$ for all
    eight $L$ equations — Y-only RE is parsimonious and sufficient.
  - Prior sensitivity (HalfCauchy vs Gamma on $\tau$): identical to
    within Monte Carlo noise.
  - Held-out 20% PPC: day-by-day calibration within $\pm 3$ p.p. across
    the 28-day course.
  - Natural-course validation: NICE-family miss $\le 1$ p.p.; Xu MSM
    overshoots by 3.7 p.p. (consistent with framework attenuation).
- **Confounder dependence.** Across 12 leave-one-covariate-out refits
  for both K=1 and K=5, the dose-response is stable except when the
  ARDS-defining $\mathrm{PaO_2}/\mathrm{FiO_2}$ ratio is excluded
  ($+43$ p.p., reported as a domain-knowledge sanity check rather than
  as a confounding signal). Lactate is the dominant proper time-varying
  confounder ($+8.6$ p.p. on K=5 exclusion).
- **Subgroups.** All nine pre-specified strata (severity, age, BMI,
  Charlson) show strong positive risk differences; severe ARDS, older
  age, and higher Charlson amplify the dose-response gradient. A BMI
  obesity-paradox-consistent signal is observed (low-BMI subjects show
  larger absolute effect).
- **Unmeasured-confounding sensitivity.** E-value at the canonical
  low-vs-reference contrast (bin 7 vs bin 16) is **8.71** at the point
  estimate and **7.23** at the credible-interval bound, indicating that
  an unmeasured confounder would need to be associated with both
  exposure and outcome by approximately 7-fold to fully explain the
  observed effect.
- **Convergence.** Across all primary NUTS fits, $\hat R \le 1.08$,
  effective sample size $\ge 71$, and **zero divergent transitions**.

---

## Methodological highlights

### Estimand and identification

Under sequential exchangeability, positivity, and consistency, the
counterfactual cumulative incidence under sustained MP exposure regime
$\bar a = (a_1, \dots, a_T)$ is

$$
\Pr\{Y^{\bar a}=1\} = \mathbb{E}_V\,\mathbb{E}_{L_1\mid V}\,\cdots
\sum_{t=1}^{T} \Pr(Y_t=1 \mid Y_{t-1}=0,\, \bar L_t,\, V;\, \bar A_t = \bar a_t)
\prod_{s=1}^{t-1}\Pr(Y_s=0\mid\cdot).
$$

### 4-method ladder

| # | Method | RE structure | $L$ handling |
|---|---|---|---|
| 1 | Standard NICE g-formula | none | forward $L$ simulation |
| 2 | Xu Bayesian GLMM (2024) | scalar intercept | observed $L$ plug-in (MSM) |
| 3 | K=1 FRE-NICE | scalar (= Xu RE in NICE framework) | forward $L$ simulation |
| 4 | **K=5 FRE-NICE** (primary) | **functional via spline basis** | forward $L$ simulation |

The outcome model for FRE-NICE is

$$
\mathrm{logit}\,\Pr(Y_t=1\mid\cdot)
= \alpha_0 + \alpha_A^{\top} A_t + \alpha_L^{\top} L_t
+ \alpha_V^{\top} V + b_i^{\top} B(t),
$$

where $B(t)$ is a QR-orthonormalized natural cubic spline basis with knots at
days {0, 3, 7, 14, 21}; $b_i \sim \mathcal{N}(0, \Sigma_b)$ with
$\Sigma_b = \mathrm{diag}(\tau)\,\Omega\,\mathrm{diag}(\tau)$,
$\tau \sim \mathrm{HalfCauchy}(0, 2.5)$, $\Omega \sim \mathrm{LKJ}(2)$.

### Inference

- Bayesian posterior via No-U-Turn Sampler (Hoffman & Gelman 2014) in
  `numpyro` (Bingham et al. 2019). 2 chains × (1,000 warm-up + 1,000 sample),
  target acceptance 0.95.
- Counterfactual dose-response computed by JAX vmap + scan (vectorized over
  posterior × subject × time), $\sim$ 144× speedup over the NumPy reference.
- Cluster-level (per-subject) log-likelihood recorded for WAIC and
  PSIS-LOO model comparison (Watanabe 2010; Vehtari et al. 2017, §4).

---

## Repository layout

```
.
├── _draft/                             # Manuscript drafts (not for code)
│   └── manuscript/                     # Section-by-section + interpretation
├── data/                               # Cohort CSV (not committed)
├── legacy/                             # Earlier exploration (HMM, VEM-SSM, etc.)
├── results/                            # Posterior states + dose-response + tables
│   ├── bayesian_main_v2/               #   Primary K=5 NUTS posterior
│   ├── bayesian_jax/                   #   Primary dose-response (JAX)
│   ├── knot_sensitivity/               #   K=4, K=6 knot sensitivity
│   ├── loco_K5/, loco_K1/              #   Leave-one-covariate-out (12 each)
│   ├── loglik_main/                    #   Refits with log_lik for WAIC/LOO
│   ├── ppc_K5/                         #   Held-out 20% PPC
│   ├── prior_sens_gamma/               #   Gamma vs HalfCauchy prior sensitivity
│   ├── bayesian_spec2/                 #   Spec II ablation (shared RE on L)
│   ├── subgroup/                       #   9 subgroup dose-response
│   ├── positivity/                     #   Per-bin sample sizes
│   ├── figures/                        #   Fig 3 + Fig 4 PNG/PDF
│   ├── standard_v2/                    #   Standard frequentist NICE
│   └── appendix_g/                     #   Standard + Xu LOCO (legacy)
├── scripts/                            # Active analysis scripts
├── src/
│   ├── benchmarks/                     # Method classes
│   ├── data/                           # Cohort assembly (ARDS)
│   └── models/                         # Spline basis, etc.
├── tests/
├── README.md
└── requirements.txt
```

### Active scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `run_bayesian_main.py` | NUTS/SVI fit + dose-response runner; phase-based execution |
| `run_jax_dose.py` | JAX dose-response from saved posterior state |
| `compute_waic_loo.py` | WAIC + PSIS-LOO from saved log_lik states |
| `extract_diagnostics.py` | R-hat / ESS / divergent counts table |
| `make_figures.py` | Fig 3 (dose-response), Fig 4 (subgroup forest) |
| `make_table1.py` | Table 1 baseline characteristics |
| `make_holdout_split.py` | Subject-stratified 80/20 holdout for PPC |
| `ppc_holdout.py` | Posterior predictive check on held-out 20% |
| `positivity_check.py` | Per-bin covariate distribution + sample sizes |
| `subgroup_jax_dose.py` | Subgroup-stratified dose-response |
| `validate_bayesian_natural_course.py` | Natural-course calibration |
| `unit_test_numpy_vs_jax.py` | Numerical equivalence test |
| `round2_chain_c.sh` | Orchestrator: 4 NUTS fits + WAIC + diagnostics + PPC |

### Source modules (`src/`)

- `src/data/ards.py` — MIMIC-IV ARDS cohort assembly, 20-bin MP discretization.
- `src/models/spline_glmm.py` — natural cubic spline basis (QR-orthonormalized).
- `src/benchmarks/standard_gformula.py` — frequentist NICE g-formula.
- `src/benchmarks/xu_glmm_bayesian.py` — Bayesian Xu MSM (scalar RE).
- `src/benchmarks/fre_nice_bayesian.py` — Bayesian K=1 / K=5 FRE-NICE g-formula.
- `src/benchmarks/dose_response_jax.py` — GPU-accelerated counterfactual.

---

## Quick start

```bash
# Environment
pip install -r requirements.txt

# Fit primary K=5 (V100 GPU recommended; ~8 min)
python scripts/run_bayesian_main.py \
    --csv data/ards_v31_v4.csv --out-dir results/bayesian_main_v2 \
    --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
    --target-accept 0.95 --n-posterior-subset 200 \
    --methods K5 --phase fit

# Dose-response (JAX, ~0.6 min on V100)
python scripts/run_jax_dose.py \
    --csv data/ards_v31_v4.csv --state-dir results/bayesian_main_v2 \
    --out-dir results/bayesian_jax --prefix fre_nice_K5

# WAIC / PSIS-LOO (after K=1, K=5, Xu fits with --record-loglik)
python scripts/compute_waic_loo.py \
    --state-files results/loglik_main/fre_nice_K1_state.npz \
                  results/loglik_main/fre_nice_K5_state.npz \
                  results/loglik_main/xu_bayesian_state.npz \
    --labels "K=1 NICE" "K=5 FRE-NICE" "Xu Bayesian" \
    --out-md results/loglik_main/waic_loo_table.md

# All sensitivity analyses (full overnight pipeline)
bash scripts/round2_chain_c.sh
```

---

## Reproducibility

- **Data**: MIMIC-IV v3.1 (PhysioNet credentialed access). Cohort assembly
  in `src/data/ards.py`; the resulting CSV is not redistributed.
- **Random seeds**: fixed in all scripts (default 0, with method-specific
  offsets).
- **Numerics**: `jax_enable_x64` is forced on for dose-response computation
  to ensure NumPy-equivalent precision.
- **Convergence**: across all primary NUTS fits, $\hat R \le 1.08$ and zero
  divergent transitions.
- **Posterior states (FRE weights)** for all primary fits and sensitivity
  analyses are saved as `results/**/*_state.npz` (32 of 35 committed; the
  three loglik-recording states exceed GitHub's 100 MB limit and are
  gitignored — regenerate with `--record-loglik`).

---

## Citing

If using this codebase, please cite the master's thesis (forthcoming) and
the methodological anchors:

- Robins, J. M. (1986). A new approach to causal inference in mortality
  studies with a sustained exposure period. *Mathematical Modelling*, 7,
  1393–1512.
- Xu, Y., et al. (2024). GLMM-based g-computation for clustered
  longitudinal data. *Biometrics*, 80(3), ujae100.
- Yao, F., Müller, H.-G., & Wang, J.-L. (2005). Functional data analysis
  for sparse longitudinal data. *JASA*, 100, 577–590.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). CRC Press.
- Costa, R., et al. (2021). Mechanical power and 28-day mortality in
  mechanically ventilated patients. *AJRCCM*, 204(3), 303–311.

---

## License

MIT — see [LICENSE](LICENSE) (if absent, MIT is intended; please confirm
with the maintainer).
