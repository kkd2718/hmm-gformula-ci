# Bayesian Natural Course Validation

_Cohort raw 28-day mortality = **25.61%**. Each method forward-simulates under OBSERVED treatment trajectory and marginalizes over the random effect. A well-calibrated method should produce mortality close to cohort raw._

| Method | Mean (%) | 95% CI | Miss (%p) | Calibration |
|---|---|---|---|---|

## Interpretation

- **Xu Bayesian** uses observed L plug-in (no forward L sim), so this NC is essentially "factual" prediction averaged over b ~ N(0, sigma_b^2). Should be very close to cohort raw if model is well-fit.
- **K=1 / K=5 FRE-NICE** use NICE forward L simulation. Calibration miss indicates either (a) generative L equations have residual misspecification, (b) outcome model under model-implied L distribution drifts, or (c) implementation issue in our forward sim. For Standard parametric g-formula on this cohort: NC = 24.58% (miss -1%p) → reference baseline.

If FRE-NICE NC miss is dramatically larger than Standard's miss, the issue is in our new Bayesian implementation. If similar to Standard, the simulation is consistent with NICE algorithm.