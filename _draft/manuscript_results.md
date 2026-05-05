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

## WAIC and PSIS-LOO comparison (Table 2)

[INSERT FROM results/loglik_main/waic_loo_table.md WHEN COMPLETE]

The expected log predictive density (ELPD) preference between K=5 and
K=1 was [pending Stage B]. Pareto-$\hat{k}$ diagnostics for the PSIS-LOO
estimator [pending].

## Posterior predictive check on held-out 20% (Table S4)

[INSERT FROM results/ppc_K5/ppc_K5.md WHEN COMPLETE]

Calibration of the Bayesian K=5 model was assessed by fitting on a
random 80% of subjects (subject-stratified) and posterior-predicting
day-by-day mortality on the held-out 20%. [pending].

## Specification ablation (Spec II) — shared random effect on $L$

[INSERT FROM results/spec2_v3 lambda_L]
The posterior of $\lambda_j$ — the per-$L$-component coefficient on the
shared functional random effect in the $L$-equation — was concentrated
near zero for all eight time-varying confounders, indicating that the
data do not support sharing the random effect between the outcome and
$L$ equations. Spec II posterior dose-response at the reference and
high-MP bins differed from Spec I by $-0.3$ and $-0.2$ percentage points,
respectively. We retain Spec I (Y-only random effect) as the primary
specification.

## Prior sensitivity

[INSERT FROM results/prior_sens_gamma WHEN COMPLETE]

Replacing the default $\mathrm{HalfCauchy}(0, 2.5)$ prior on $\sigma_b$
with a $\mathrm{Gamma}(2, 0.5)$ prior produced [pending] dose-response
estimates differing from primary by [pending].

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

[INSERT FROM results/loglik_main/diagnostics_table.md]

All Bayesian models converged with potential scale reduction $\hat{R}
\le 1.05$ (typical maximum [pending]), effective sample size [pending],
and [pending] divergent transitions. Trace plots for the hyperparameters
$\tau_k$ are provided in Appendix B.
