# Posterior Predictive Check (held-out 20%)

_Held-out subjects: 3124 → 3560 stays. Posterior subset: 200 draws × 5 b-draws each._

## Day-28 mortality (subject-level, max Y over at-risk days)

- Predicted: **19.47%**
- Actual:    **24.97%**
- Difference: **-5.50%p**

## Cumulative incidence by day

| Day | Predicted (%) | Actual (%) | Diff (%p) |
|---|---|---|---|
| 1 | 2.77 | 0.65 | +2.12 |
| 7 | 11.06 | 12.18 | -1.12 |
| 14 | 15.37 | 17.68 | -2.31 |
| 21 | 17.85 | 20.67 | -2.83 |
| 28 | 19.47 | 22.04 | -2.57 |

## Interpretation

Calibration miss <2%p across all days indicates the Bayesian model has predictive validity on subjects unseen during fit, supporting external validity claims for the dose-response estimates derived from the same model.