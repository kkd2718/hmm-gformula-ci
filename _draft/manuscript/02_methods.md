# Methods (manuscript draft, v1)

## Study design and cohort

We performed a retrospective cohort study using the Medical Information Mart
for Intensive Care IV (MIMIC-IV v3.1). Adult patients (≥18 y) admitted to
the intensive care unit who met the Berlin definition of acute respiratory
distress syndrome (ARDS) on the basis of arterial oxygenation, positive
end-expiratory pressure (PEEP), and mechanical ventilation status were
included. The first ICU stay per subject was retained when more than one
qualifying stay was present, yielding $N = 17{,}878$ stays from $G = 15{,}619$
unique subjects. Follow-up was capped at 28 days after ARDS onset.

The exposure of interest, mechanical power (MP), was computed daily as
$$\text{MP} = 0.098 \cdot V_T \cdot RR \cdot \left(\Delta P + \text{PEEP} - \frac{\Delta P}{2}\right),$$
where $V_T$ is tidal volume (mL), $RR$ is respiratory rate, and $\Delta P$
is the driving pressure (cmH$_2$O). MP was log-transformed and discretized
into $K_A = 20$ bins on a logarithmic grid spanning $\pm 3$ standard deviations
around the cohort mean (bin centers from approximately 0.6 J/min to 35 J/min).
The outcome was 28-day all-cause in-hospital mortality, treated as a discrete-
time competing-risks indicator censored at hospital discharge alive.

Time-varying confounders ($L_t$) included PaO$_2$/FiO$_2$ ratio, PaCO$_2$,
arterial lactate, mean arterial pressure, heart rate, Glasgow Coma Scale total,
serum creatinine, and core temperature, all aggregated to daily means.
Static confounders ($V$) were age, sex, body-mass index (with multiple
imputation for missing values), and Charlson comorbidity index. All
continuous variables were standardized to zero mean and unit variance across
the cohort. The reference bin for the dose-response contrast was chosen as
bin 16 (≈ 18.3 J/min), corresponding to the Costa et al. (2021) clinical
cutoff for risk acceleration (~17 J/min).

## Identification framework

Causal effects of sustained MP exposure regimes on 28-day mortality were
identified under sequential exchangeability, positivity, and consistency
(Robins, 1986; Hernán & Robins, 2020). Under these assumptions, the
counterfactual cumulative incidence under a deterministic regime
$\bar{a} = (a_1, \ldots, a_T)$ is
$$\Pr\{Y^{\bar{a}} = 1\} = \mathbb{E}_{V}\, \mathbb{E}_{L_1 \mid V}\, \cdots
\sum_{t=1}^{T} \Pr(Y_t = 1 \mid Y_{t-1} = 0, \bar{L}_t, V; \bar{A}_t = \bar{a}_t)
\prod_{s=1}^{t-1}\Pr(Y_s = 0 \mid \cdot).$$
We adopted the parametric NICE g-formula (Robins, 1986; Westreich et al., 2012)
because the data exhibit strong treatment-confounder feedback: $L_t$ is
affected by prior $A$ and modulates subsequent $A$. Under this dependence
structure, IPW-based marginal structural models suffer from weight instability
(Cole & Hernán, 2008) and the parametric g-formula provides a more
tractable and lower-variance estimator.

## Estimation: 4-method ladder

To benchmark the proposed approach against established methodology, we
estimated the dose-response curve using four methods that vary along two
axes: (i) framework (g-formula with forward $L$ simulation versus marginal
structural model with observed $L$ plug-in), and (ii) random-effect
structure (none; scalar; functional via spline basis).

1. **Standard parametric g-formula** (Robins 1986). Frequentist; cluster
   bootstrap (B=100) for confidence intervals; no random effect.
2. **Xu et al. (2024) GLMM g-computation** (Bayesian). Marginal structural
   model with subject-specific scalar random intercept on the outcome
   logit; counterfactual computed via observed-$L$ plug-in.
3. **K=1 FRE-NICE** (Bayesian; this study). NICE g-formula with subject-
   specific scalar random intercept (functional rank-1, equivalent to a
   constant basis). Forward $L$ simulation under counterfactual $A$.
4. **K=5 FRE-NICE** (primary; Bayesian; this study). Subject-specific
   functional random effect on the outcome logit, parameterized by a
   natural cubic spline basis with knots at days 0, 3, 7, 14, and 21. Forward
   $L$ simulation under counterfactual $A$.

For methods 3 and 4, the outcome model is
$$\text{logit}\,\Pr(Y_t = 1 \mid \cdot) = \alpha_0 + \alpha_A^\top A_t
+ \alpha_L^\top L_t + \alpha_V^\top V + b_i^\top B(t),$$
where $B(t)$ is the natural-cubic-spline basis evaluated at day $t$
(QR-orthonormalized for numerical stability), $b_i \sim \mathcal{N}(0,
\Sigma_b)$ is the subject-specific functional random effect, and $\Sigma_b
= \mathrm{diag}(\tau) \cdot \Omega \cdot \mathrm{diag}(\tau)$ with
$\tau \sim \mathrm{HalfCauchy}(0, 2.5)$ and $\Omega \sim \mathrm{LKJ}(2)$
(Lewandowski et al., 2009). For K=1 the LKJ degenerates and $\tau$ is a
scalar with the same prior. The functional formulation generalizes Xu et
al.'s scalar random intercept, which corresponds to $b_i^\top B(t) = b_i$
under a constant basis. Yao, Müller & Wang (2005) and Wood (2017) provide
the foundational treatment of functional random effects via spline bases.

Pooled $L$-equations are estimated by ridge-regularized least squares with
penalty $\lambda = 10^{-4}$. To avoid bias-vs-bins collinearity, the
reference bin column is dropped from the one-hot encoding of $A_t$ in both
the outcome and $L$ equations (analogous to the standard treatment-control
contrast coding); this removes a single rank deficiency (rank confirmed
33 of 33 in the full-cohort design) without changing the implied dose-
response structure.

Inference for methods 2-4 used Hamiltonian Monte Carlo via the No-U-Turn
sampler (NUTS; Hoffman & Gelman, 2014) implemented in numpyro
(Bingham et al., 2019), with target acceptance 0.95, 1{,}000 warm-up
iterations, 1{,}000 sampling iterations, and 2 chains. Convergence was
assessed by potential scale reduction $\hat{R}$ (Gelman & Rubin, 1992;
Vehtari et al., 2021), effective sample size, and divergent transitions.

## Counterfactual computation and dose-response

For each Bayesian method, the dose-response under sustained $A_t = a^*$
for all $t$ was computed by Monte Carlo:
$$\hat{R}(a^*) = \frac{1}{S \cdot M} \sum_{s=1}^{S} \sum_{m=1}^{M}
\frac{1}{N} \sum_{i=1}^{N} I_i^{(s,m,a^*)},$$
where $S = 200$ posterior draws, $M = 5$ random-effect draws per posterior,
and $I_i^{(s,m,a^*)}$ is the cumulative incidence under regime $a^*$
implied by posterior parameters and a draw of $b_i$ from
$\mathcal{N}(0, \Sigma_b^{(s)})$, with forward $L$ simulation under $A = a^*$.
Posterior 95% credible intervals were computed from the across-draw
quantile of $\hat{R}(a^*)$. Standard NICE used cluster bootstrap CIs over
the analogous quantity. Counterfactual computation was implemented in JAX
with double-precision arithmetic for numerical equivalence to the reference
NumPy implementation, yielding a ~144-fold speedup.

## Sensitivity analyses

We pre-specified the following sensitivity analyses to address potential
threats to validity. Each was conducted on the K=5 model unless otherwise
noted.

1. **Knot placement** (K = 4, 5, 6 spline basis dimensions; alternative knot
   positions {0,7,14,21} and {0,3,7,14,21,27}).
2. **Specification ablation (Spec II)**: refit $L$-equations with an
   additional column $\lambda_j \cdot \hat{b}_i^\top B(t)$ to test whether
   the random effect on $Y$ should be shared with $L$. Spec I (Y-only RE)
   versus Spec II (shared RE on $L$).
3. **Prior sensitivity**: $\sigma_b$ refit with $\mathrm{Gamma}(2, 0.5)$
   prior in addition to the default $\mathrm{HalfCauchy}(0, 2.5)$.
4. **Leave-one-covariate-out (LOCO)**: 12 separate refits, each excluding
   one of the 8 TV or 4 static confounders.
5. **Posterior predictive check (PPC)** on a 20% subject-stratified hold-out:
   refit on 80%, posterior-predict day-by-day mortality on 20%, compare to
   actual.
6. **Model comparison via WAIC and PSIS-LOO** (Watanabe, 2010; Vehtari et al.,
   2017) to formally rank K=1 versus K=5 outcome models.
7. **Natural-course validation**: forward simulate under observed treatment
   regime and compare cohort-level cumulative incidence to observed cohort
   28-day mortality. A miss of $\le 2$ percentage points indicates
   well-calibrated forward simulation.
8. **E-value** (VanderWeele & Ding, 2017) for unmeasured confounding
   sensitivity at key dose-response contrasts.
9. **Subgroup analysis**: dose-response stratified by Berlin severity (mild,
   moderate, severe), age (>median vs ≤median), BMI, and Charlson index,
   using the K=5 posterior of the full cohort model with subgroup-restricted
   counterfactual computation (rather than re-fitting per subgroup, to
   preserve power).
10. **Positivity check**: per-bin sample sizes and standardized covariate
    distributions to flag bins with sparse or extreme exposure.

## Software and reproducibility

All analyses were implemented in Python 3.10 using PyTorch 2 (data loading),
numpyro 0.13 (Bayesian inference), JAX 0.4 (counterfactual computation
with double precision), and NumPy/SciPy/pandas. Model code, fitting scripts,
and analysis notebooks are available at
`https://github.com/kkd2718/hmm-gformula-ci` under branch
`spline-glmm-extension`. Random seeds were fixed; convergence diagnostics,
per-method posterior traces, and all sensitivity analysis outputs are
deposited in `results/` of the same repository. Counterfactual computation
on a single NVIDIA V100 GPU completed in approximately 0.6 minutes per
method per dose-response evaluation; full posterior fits required 6-8
minutes per method.
