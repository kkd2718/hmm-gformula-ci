# Appendix F: Natural-course simulation with bootstrap CI

_Cluster bootstrap on subject_id, B = 100 refit replicates. Each replicate: resample patients, refit model, forward-simulate with observed A_t trajectory (no intervention). Cohort raw 28-day mortality = **25.61%**._

| Method | Mean (%) | 95% CI (%) | Includes cohort 25.61%? |
|---|---|---|---|
| Standard parametric g-formula | 24.58 | (23.67, 25.38) | NO |
| Xu 2024 GLMM g-computation | 28.92 | (28.25, 29.72) | NO |
| VEM-SSM g-formula (proposed) | 30.04 | (28.91, 31.09) | NO |

## Interpretation

Natural-course simulation under each method's fitted generative model is a calibration check (Taubman 2009 *IJE*; Keil 2014 *Epidemiology*; McGrath 2020 *Patterns*): a method whose 95% CI contains the cohort raw rate is well-calibrated to the observed data distribution. This does NOT validate counterfactual causal estimates — only confirms that the generative model can re-produce the observed factual distribution.