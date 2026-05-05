# E-value sensitivity analysis (VanderWeele & Ding 2017)

_Posterior credible intervals on RR = risk[bin] / risk[ref=16]; E-value = RR + sqrt(RR*(RR-1)) for RR > 1; E-value = 1/RR + sqrt((1/RR)*(1/RR - 1)) for RR < 1._

_E-value gives the minimum strength (on the RR scale) of an unmeasured confounder, both with treatment and outcome (each), needed to fully explain away the observed effect._

| Bin | MP (J/min) | RR (95% CrI) | E-value (point) | E-value (lower CI bound) |
|---|---|---|---|---|
| 0 | 0.6 | 0.24 (0.16-0.35) | 7.67 | 5.20 |
| 7 | 2.7 | 0.22 (0.18-0.26) | 8.71 | 7.23 |
| 11 | 6.4 | 0.38 (0.32-0.44) | 4.77 | 4.02 |
| 12 | 7.9 | 0.55 (0.48-0.63) | 3.04 | 2.56 |
| 13 | 9.7 | 0.72 (0.62-0.82) | 2.14 | 1.74 |
| 14 | 12.0 | 0.98 (0.87-1.11) | 1.17 | nan |
| 15 | 14.8 | 0.78 (0.67-0.90) | 1.90 | 1.45 |
| 17 | 22.6 | 1.24 (1.03-1.44) | 1.79 | 1.19 |
| 18 | 28.0 | 0.93 (0.69-1.20) | 1.37 | nan |
| 19 | 34.6 | 0.75 (0.68-0.86) | 1.98 | 1.61 |

## Interpretation

Following VanderWeele & Ding (2017, *Annals of Internal Medicine*):

- The E-value at the point estimate is the strength an unmeasured confounder would need (associated with both exposure and outcome) to fully explain the observed effect.
- The E-value at the CI bound is the strength needed to bring the effect to non-significance.
- An E-value of 4 means a confounder ~4-fold associated with each side is needed — clinically implausible after adjusting for the 12 tracked confounders (8 TV physiology + 4 static).
- A low E-value (<2) signals the effect could be explained by modest residual confounding.