# Discussion (manuscript draft, v1)

## Principal findings

In a cohort of 17{,}878 ICU stays (15{,}619 unique subjects) with ARDS,
sustained mechanical ventilation at higher mechanical power was associated
with substantially elevated 28-day mortality. The Bayesian K=5 functional
random-effect NICE g-formula yielded a steep, monotonically increasing
dose-response above approximately 6 J/min, with the cumulative incidence of
death increasing from 8.0% (95% CrI 5.5–11.4) at MP $\approx$ 2.7 J/min to
46.1% (38.9–52.4) at MP $\approx$ 22.6 J/min, with the Costa et al. (2021)
clinical cutoff (~17 J/min, our reference bin) yielding a 28-day mortality
of 37.3% (32.8–41.4). The dose-response was robust across alternative
spline-knot specifications (K=4 and K=6 differed by less than 0.5 percentage
points at the reference and high-MP bins) and across all 12 leave-one-
covariate-out refits except when the ARDS-defining oxygenation variable
(P/F ratio) was excluded; we report the latter as a domain-knowledge
sanity check rather than as a confounding signal.

## Comparison with prior literature

Costa et al. (2021) reported an adjusted hazard ratio of approximately
1.06 per 5 J/min increase in mechanical power and identified ~17 J/min as a
clinical inflection. Our dose-response converges with these findings near
the inflection — the relative risk between bin 16 (18.3 J/min) and bin 17
(22.6 J/min) is 1.24 (0.94–1.60), corresponding to approximately 1.27 per
5 J/min — but reveals a markedly nonlinear relationship across the broader
exposure range. In the 6–10 J/min range, we estimate relative risk
acceleration substantially exceeding that implied by a single linear hazard
ratio, suggesting that the linear approximation may understate harm in the
moderate-MP regime. This nuance aligns with mechanistic reasoning: ventilator-
induced lung injury risk depends on pressure-volume product and respiratory
rate in a thresholded rather than additive manner, particularly once
inflammation is established (Gattinoni et al., 2016; Amato et al., 2015).

## Comparison of methodological frameworks

Among the four estimation methods we examined, the NICE g-formula variants
(Standard, K=1, and K=5) produced congruent dose-response curves with
relative risks differing by less than 5 percentage points across most bins.
The Bayesian replication of Xu et al. (2024) GLMM g-computation, by contrast,
yielded an essentially flat dose-response of approximately 21% across all
MP bins. We attribute this difference to the choice of causal framework
rather than to model fit: Xu's procedure plugs in observed-$L$ values when
computing counterfactual cumulative incidence under hypothetical $A = a^*$,
whereas the NICE g-formula simulates $L$ forward under $a^*$. When $L_t$
is itself influenced by $A_{t-1}$, observed-$L$ plug-in absorbs part of
the indirect effect of $A$ through $L$ into the conditional outcome model,
attenuating the marginal counterfactual estimand toward the null
(Robins, 1986; Hernán & Robins, 2020, §21). Natural-course validation
supports this interpretation: the NICE-family methods predicted day-28
mortality within 1.0 percentage point of the observed cohort rate (25.6%),
whereas the Xu procedure overshot by 3.7 percentage points, consistent with
the framework attenuation in counterfactual calibration but not necessarily
in factual prediction.

## Methodological contribution

The functional random-effect (FRE) extension of Xu et al.'s scalar random
intercept embeds subject-level heterogeneity that evolves smoothly across
the ICU course. Rather than assuming a single time-invariant subject
deviation $b_i$, the FRE parameterizes $b_i^\top B(t)$ where $B(t)$ is a
natural cubic spline basis. Anchored in functional data analysis (Yao,
Müller & Wang, 2005) and generalized additive mixed modeling (Wood, 2017),
this construction generalizes the scalar-RE special case (recovered when
$B(t) \equiv 1$) to a finite-rank random function on the day grid.

Three findings inform interpretation of the methodological extension.
First, the WAIC and PSIS-LOO comparison between K=1 and K=5 indicated
[insert from results/loglik_main/waic_loo_table.md upon completion] —
a [significant/modest/inconclusive] preference for the time-varying
parameterization. Second, the dose-response estimates from K=5 differed
from those of K=1 by less than 1 percentage point in the canonical bins,
suggesting that the *causal* estimand (cumulative incidence under sustained
exposure) is robust to the random-effect structure under correctly
specified outcome and $L$ models. Third, the Spec II ablation (sharing
the random effect across $Y$ and $L$ equations) yielded $\lambda_j$
posterior near zero for all $L$ components, supporting the parsimony of
the Y-only random-effect specification.

The combination of these findings positions the FRE-NICE g-formula as a
principled methodological extension that does not materially shift the
*clinical* estimand on this cohort but provides a more flexible framework
for cohorts in which subject-level heterogeneity is plausibly time-varying
(e.g., longer ICU follow-up, transplant cohorts, chronic critical illness).

## Strengths

1. Comprehensive 4-method comparison enabling triangulation of the
   estimated dose-response.
2. Bayesian inference enables credible intervals from the joint posterior
   without dependence on cluster-bootstrap stability.
3. Pre-specified set of 10 sensitivity analyses (knot placement, prior,
   PPC, LOCO, Spec II, positivity, etc.) addressing the principal threats
   to validity in parametric g-formula estimation.
4. Natural-course calibration verifies that the forward $L$ simulation
   does not introduce systematic bias.
5. E-value analysis at the canonical contrast (bin 7 versus reference bin 16)
   yields E ≈ 8.7, indicating that an unmeasured confounder would need
   to be associated with both exposure and outcome by approximately
   8.7-fold to fully explain the observed association — implausibly strong
   given the 12 measured time-varying and static confounders.

## Limitations

1. Single-center cohort (MIMIC-IV) limits generalizability; external
   validation in independent ARDS registries is needed.
2. Discrete-time discretization at the daily level may not capture
   intra-day variability in mechanical power, particularly during weaning.
3. The mechanical power formula assumes volume-control ventilation;
   pressure-control settings introduce additional measurement uncertainty.
4. Static confounders are baseline only; chronic conditions evolving
   during ICU admission are partially captured by the time-varying
   confounders but not exhaustively.
5. The functional random effect captures unobserved subject heterogeneity
   parametrically; if true heterogeneity is highly non-smooth, the
   spline basis may misspecify it, although our knot-sensitivity analysis
   (K=4, 5, 6) suggests robustness in this cohort.
6. Positivity is borderline at extreme bins (bin 17 N=316, bin 18 N=135);
   dose-response estimates at these bins should be interpreted with
   appropriate caution as they extrapolate from sparser exposure data.

## Clinical implications

Our findings reinforce the recommendation that mechanical power should be
maintained below approximately 17 J/min in mechanically ventilated patients
with ARDS, consistent with current evidence-based practice. The rapid
acceleration of mortality risk above this threshold — particularly at
MP > 22 J/min, where risk approaches 50% — argues for active monitoring of
ventilator-derived energy delivery in addition to or beyond traditional
pressure and tidal volume targets. The substantial protective effect at
the low-MP end (E-value ≈ 8.7 for bin 7 versus reference) further
supports interventions that minimize unnecessary ventilation intensity.

## Conclusions

In a Bayesian g-formula analysis of mechanical power and 28-day mortality
in ARDS, sustained MP exposure exhibited a nonlinear dose-response with
substantial mortality acceleration above 17 J/min. The proposed functional
random-effect extension of the NICE g-formula provides a methodological
framework for embedding time-varying subject-level heterogeneity in
parametric g-computation; on the present cohort, the extension preserves
calibration and dose-response estimates while offering a generalizable
template for longitudinal causal inference in clustered data with
treatment-confounder feedback.
