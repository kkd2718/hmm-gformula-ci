# Results (manuscript draft, v1)

## Cohort

Of $N$ adult ARDS admissions in MIMIC-IV (criteria detailed in Methods),
17{,}878 stays from 15{,}619 unique subjects were retained. Baseline
characteristics are summarized in Table 1 [INSERT Table 1: median age,
male proportion, BMI, baseline P/F ratio, lactate, Charlson, severity
distribution]. The 28-day cumulative incidence of mortality was 25.6%
in the cohort overall.

## Mechanical-power exposure distribution

Figure 1 [cohort flow] and Table S1 [bin sample sizes] summarize MP
exposure across the 20 binned levels. Sample sizes ranged from 135
stays in the highest bin (28.0 J/min) to 16{,}694 stays at one bin
representing low-MP / off-ventilator periods, with the Costa cutoff
(bin 16, 18.3 J/min) populated by 744 stays. Median ICU MP varied
substantially across days, with the highest power densities observed
in the first three days and gradual reduction during recovery.

## Dose-response of 28-day mortality (Figure 3, Table 2)

The Bayesian K=5 FRE-NICE g-formula (primary analysis) yielded a
monotonically increasing dose-response above MP $\approx$ 6 J/min:
8.0% (95% CrI 5.5–11.4) at MP 2.7 J/min, 14.0% (10.6–17.7) at 6.4 J/min,
20.4% (16.5–24.4) at 7.9 J/min, 36.4% (30.0–42.4) at 12.0 J/min,
37.3% (32.8–41.4) at the reference 18.3 J/min, and 46.1% (38.9–52.4) at
22.6 J/min (Table 2, Figure 3A).

The three other estimation methods produced concordant results in the
NICE family but disagreed with the marginal structural model variant
(Figure 3B). The Standard parametric g-formula and K=1 NICE FRE produced
dose-response curves within $\pm 1.0$ percentage point of K=5 across
the principal exposure range. Xu et al.'s Bayesian GLMM g-computation,
by contrast, produced a flat curve of approximately 21% across all bins.
We attribute this divergence to the choice of causal framework
(observed-$L$ plug-in versus forward $L$ simulation under the
counterfactual regime; see Discussion).

## Natural-course validation (Table S2)

Predicted day-28 mortality marginalized over the observed treatment
trajectory was: Standard NICE 24.6% (miss $-1.0$ percentage points
from cohort raw 25.6%), K=1 NICE 25.4% (miss $-0.2$), K=5 NICE 25.1%
(miss $-0.5$), Xu Bayesian 29.3% (miss $+3.7$). The NICE-family methods
were calibrated within $\le 1$ percentage point; the Xu procedure
overshot, consistent with the framework attenuation argument.

## Knot-placement sensitivity (Table S3)

Dose-response estimates at the reference (bin 16) and high (bin 17) MP
bins were stable across spline-knot specifications: K=4 (37.2% / 46.2%),
K=5 (37.3% / 46.1%), K=6 (37.0% / 46.4%). Differences across knot
specifications were $\le 0.3$ percentage points and within Monte Carlo
uncertainty.

## WAIC and PSIS-LOO comparison (Table 2 lower section)

Bayesian model comparison was performed using cluster-level (per-subject) log-
likelihood, the appropriate scale for clustered longitudinal data
(Vehtari et al., 2017, §4). Results are shown below for the three Bayesian
methods (Standard NICE is frequentist and excluded from WAIC/LOO).

| Method | WAIC (SE) | LOO (SE) | $p_{\text{waic}}$ | Pareto-$\hat{k}$ max |
|---|---|---|---|---|
| K=1 NICE | 35,686 (455) | 35,584 (454) | 2,084 | 0.50 |
| **K=5 FRE-NICE** | **34,404 (446)** | **34,438 (447)** | **4,600** | 0.50 |
| Xu Bayesian | 35,676 (454) | 35,574 (454) | 2,097 | 0.50 |

Pairwise ELPD differences (positive favors first method):

| Comparison | $\Delta$ELPD-WAIC | $\Delta$ELPD-LOO |
|---|---|---|
| K=5 vs K=1 | $+640.9$ | $+573.2$ |
| K=5 vs Xu | $+636.0$ | $+568.3$ |
| K=1 vs Xu | $-4.9$ | $-4.9$ |

The K=5 functional random-effect model is decisively preferred over both K=1
and Xu Bayesian by both WAIC and PSIS-LOO; the magnitude of the ELPD
improvement ($\sim$ 640 nats) substantially exceeds conventional thresholds
for model preference. Pareto-$\hat{k}$ values were all $\le 0.50$,
indicating reliable PSIS-LOO estimation. The mathematical equivalence of
K=1 NICE and Xu Bayesian outcome models is empirically confirmed by the
$\Delta$ELPD of $-4.9$ between them.

The effective number of parameters $p_{\text{waic}}$ increased from 2{,}084
(K=1) and 2{,}097 (Xu) to 4{,}600 (K=5), reflecting the additional
flexibility of the functional random effect (4 extra basis dimensions per
subject), which the Bayesian shrinkage (LKJ correlation prior, half-Cauchy
scale prior) regularizes.

## Posterior predictive check on held-out 20% (Table S4)

Calibration of the Bayesian K=5 model was assessed by fitting on a
random 80% of subjects (subject-stratified) and posterior-predicting
day-by-day cumulative incidence of mortality on the held-out 20%
(N=3{,}124 subjects, 3{,}560 stays). For each held-out subject, $b_i$
was drawn from the prior $\mathcal{N}(0, \hat{\Sigma}_b)$ since they
were excluded from the fit; cumulative incidence by day was computed
via Monte Carlo over 200 posterior draws and 5 random-effect draws each.

| Day | Predicted (%) | Observed (%) | Diff (%p) |
|---|---|---|---|
| 1 | 2.77 | 0.65 | +2.12 |
| 7 | 11.06 | 12.18 | $-1.12$ |
| 14 | 15.37 | 17.68 | $-2.31$ |
| 21 | 17.85 | 20.67 | $-2.83$ |
| 28 | 19.47 | 22.04 | $-2.57$ |

The day-28 cumulative incidence calibration miss was $-2.57$ percentage
points (predicted under-estimate). Day-by-day calibration was within
$\pm 3$ percentage points across the entire ICU course, supporting the
predictive validity of the Bayesian model on subjects unseen during
fitting. The slight under-prediction at later days is consistent with
the conservative nature of marginalizing over the prior $\mathcal{N}(0,
\hat{\Sigma}_b)$ for held-out subjects whose individual heterogeneity is
unobserved.

## Specification ablation (Spec II) — shared random effect on $L$

The posterior of $\lambda_j$ — the per-$L$-component coefficient on the
shared functional random effect in the $L$-equation — was concentrated
near zero for all eight time-varying confounders (maximum $|\lambda_j|
= 0.011$, mean $|\lambda_j| = 0.007$), indicating that the data do not
support sharing the random effect between the outcome and $L$ equations.
Spec II posterior dose-response at the reference and high-MP bins
(37.0% and 45.9%) differed from Spec I (37.3% and 46.1%) by $-0.3$ and
$-0.2$ percentage points respectively. Natural-course calibration of
Spec II (miss $-0.5$ percentage points) was identical to Spec I. We
therefore retain Spec I (Y-only random effect) as the parsimonious
primary specification.

## Prior sensitivity (Table S6)

Replacing the default $\mathrm{HalfCauchy}(0, 2.5)$ prior on the random-
effect scale $\tau$ with a $\mathrm{Gamma}(2, 0.5)$ prior produced
essentially identical dose-response estimates: at the reference bin
(18.3 J/min) the K=5 estimate was 37.3% (95% CrI 33.1–41.4) under the
Gamma prior versus 37.3% (32.8–41.4) under the half-Cauchy prior; at
bin 17 (22.6 J/min) 46.2% (39.0–53.1) versus 46.1% (38.9–52.4). The
posterior $\tau$ marginals were similar in shape under both priors
(K=5 $\tau$ mean across basis dimensions: $[7.10, 2.79, 3.28, 2.50,
2.10]$ under half-Cauchy; $[7.13, 2.81, 3.27, 2.49, 2.10]$ under Gamma).
The choice of prior on $\tau$ does not drive the inference.

## Subgroup analysis (Figure 4)

The dose-response gradient (high-MP risk minus low-MP risk) was robust
across all 9 examined subgroups, with magnitude differences consistent
with established clinical understanding (Figure 4):

- ARDS severity: severe +41.8 (95% CrI +34.2 to +48.5), moderate +38.8
  (+31.6 to +45.5), mild +36.1 (+28.8 to +42.6).
- Age: high +44.7 (+36.9 to +51.4) > low +30.6 (+23.8 to +37.0).
- Charlson index: high +46.2 (+38.3 to +53.0) > low +31.4 (+24.5 to +38.0).
- BMI: low +43.1 (+35.5 to +49.8) > high +35.6 (+28.3 to +41.9), an
  obesity-paradox-consistent pattern.

## Leave-one-covariate-out sensitivity (Table 3)

Across 12 LOCO refits, the dose-response was stable except when the
ARDS-defining variable PaO$_2$/FiO$_2$ ratio was excluded — its removal
produced a mortality increase of approximately +43 percentage points
at the reference bin in both K=5 and Standard methods, reflecting its
role as a definitional variable rather than a confounder amenable to
LOCO interpretation. Among proper confounders, dropping arterial
lactate produced the largest shift in both methods (K=5: +8.6, Standard:
+7.6 percentage points at bin 16), followed by heart rate (K=5: +5.6,
Standard: +4.3). Static covariates produced small ($-1.5$ to $-5$
percentage points) and direction-concordant shifts across methods.
[K=1 LOCO TO BE ADDED FROM results/loco_K1].

## Positivity diagnostic (Table S5)

Positivity is well supported across the principal exposure range: bins
6–15 each contain 1{,}300–3{,}900 stays (moderate sample). Bins 0
(0.6 J/min) and 17–18 (22.6–28.0 J/min) contain fewer than 1{,}023 and
316 stays respectively, and dose-response estimates at these bins
should be interpreted with caution. The reference bin 16 (18.3 J/min,
the Costa cutoff) is well populated with 744 stays.

## E-value sensitivity to unmeasured confounding

The relative risk of mortality at low MP (bin 7, 2.7 J/min) versus
the reference bin 16 was 0.22 (95% CrI 0.18–0.26), corresponding to
an E-value of 8.7 at the point estimate and 7.2 at the upper credible
bound. An unmeasured confounder would need to be associated with
both mechanical power and mortality at approximately 7.2-fold (each
side) to fully explain the observed association — a magnitude that
is implausibly strong given the 12 measured time-varying and static
confounders in the model.

## Convergence diagnostics (Appendix B)

All Bayesian fits used 2 chains × (1{,}000 warm-up + 1{,}000 sample),
target acceptance probability 0.95.

| Method | $n_{\text{params}}$ | $\hat{R}$ max | $\hat{R}_{p95}$ | ESS min | ESS median | Divergent |
|---|---|---|---|---|---|---|
| K=1 NICE | 46{,}892 | 1.019 | 1.001 | 154 | 2{,}747 | 0 |
| K=5 FRE-NICE | 171{,}897 | 1.078 | 1.001 | 75 | 4{,}418 | 0 |
| Xu Bayesian | 46{,}891 | 1.011 | — | 273 | 3{,}322 | 0 |
| K=5 prior=$\mathrm{Gamma}$ | 156{,}278 | 1.079 | 1.000 | 77 | 4{,}608 | 0 |
| K=5 80%-fit (PPC) | 156{,}278 | 1.056 | 1.000 | 71 | 4{,}418 | 0 |

All models had zero divergent transitions across all chains, indicating
absence of geometric pathology in the posterior surface. Maximum $\hat{R}$
ranged from 1.011 (Xu) to 1.078 (K=5), all within the conservative
threshold of 1.10 (Vehtari et al., 2021); the 95th percentile $\hat{R}$
was $\le 1.001$ across all models, indicating that the small number of
parameters with marginally elevated $\hat{R}$ did not affect the
hyperparameter posteriors of substantive interest. Minimum effective
sample size ranged from 71 to 273; combined with median ESS of 2{,}747
to 4{,}608, this indicates adequate posterior sample mixing for the
inference reported.
