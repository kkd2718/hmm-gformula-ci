# Positivity diagnostic: covariate distribution by MP bin

_Cohort N = 17878 stays. Baseline (day-1) covariates are standardized (mean 0, sd 1 across cohort)._

| MP bin | Center (J/min) | N stays ever in bin | N obs in bin | PF (z) | Lactate (z) | Age (z) | BMI (z) | Charlson (z) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6 | 1023 | 1266 | -0.16 | +0.11 | +0.08 | +0.01 | +0.09 |
| 1 | 0.8 | 462 | 500 | -0.24 | +0.04 | +0.01 | -0.02 | +0.15 |
| 2 | 0.9 | 728 | 810 | -0.18 | +0.13 | +0.06 | -0.03 | +0.09 |
| 3 | 1.2 | 1108 | 1283 | -0.23 | +0.09 | +0.11 | -0.00 | +0.13 |
| 4 | 1.4 | 1523 | 1835 | -0.20 | +0.06 | +0.09 | +0.01 | +0.12 |
| 5 | 1.8 | 2219 | 2760 | -0.20 | +0.07 | +0.06 | -0.00 | +0.07 |
| 6 | 2.2 | 3057 | 3884 | -0.23 | +0.08 | +0.06 | -0.01 | +0.08 |
| 7 | 2.7 | 3534 | 4657 | -0.25 | +0.14 | +0.02 | -0.01 | +0.03 |
| 8 | 3.4 | 3848 | 5112 | -0.28 | +0.15 | -0.01 | +0.04 | +0.05 |
| 9 | 4.2 | 3680 | 4985 | -0.30 | +0.19 | -0.01 | +0.06 | +0.05 |
| 10 | 5.1 | 3711 | 5176 | -0.35 | +0.22 | -0.04 | +0.14 | +0.05 |
| 11 | 6.4 | 3554 | 5333 | -0.40 | +0.26 | -0.12 | +0.17 | +0.02 |
| 12 | 7.9 | 3228 | 5263 | -0.46 | +0.30 | -0.15 | +0.21 | +0.01 |
| 13 | 9.7 | 2699 | 4862 | -0.53 | +0.34 | -0.23 | +0.24 | -0.02 |
| 14 | 12.0 | 1998 | 3935 | -0.60 | +0.36 | -0.29 | +0.33 | -0.04 |
| 15 | 14.8 | 1320 | 2644 | -0.65 | +0.47 | -0.42 | +0.40 | -0.10 |
| 16 | 18.3 | 744 | 1531 | -0.72 | +0.53 | -0.46 | +0.45 | -0.13 |
| 17 | 22.6 | 316 | 698 | -0.79 | +0.42 | -0.53 | +0.64 | -0.13 |
| 18 | 28.0 | 135 | 276 | -0.94 | +0.57 | -0.77 | +0.68 | -0.24 |
| 19 | 34.6 | 16694 | 357211 | -0.25 | +0.06 | -0.00 | +0.00 | -0.01 |

## Interpretation

Bins with N stays < 50 or covariate z-scores > |1.0| relative to cohort mean indicate positivity concerns: dose-response estimates at those bins extrapolate beyond well-supported regions of covariate space. The functional g-formula relies on parametric outcome and L models to extrapolate, so reviewers should consider extreme bins (e.g., bin 0 ≈ 0.6 J/min, bin 19 ≈ 35 J/min) with appropriate caution; the Costa 2021 cutoff (bin 16, 17 J/min) and neighbors fall in well-populated regions.