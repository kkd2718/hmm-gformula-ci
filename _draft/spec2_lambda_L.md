# Spec II: shared random effect on L equations (lambda_j extraction)

_From results/bayesian_spec2/fre_nice_K5_state.npz (post-bug-fix v2 fit)._

_Spec II augments each L_j equation with extra column lambda_j * b_hat_i^T B(t). lambda_j ≈ 0 indicates the data do not support sharing the random effect across Y and L._

| L equation | lambda_j |
|---|---|
| pf_ratio | +0.01065 |
| paco2 | -0.00801 |
| lactate | +0.00788 |
| map_mmhg | -0.00934 |
| heart_rate | +0.00043 |
| gcs_total | +0.00954 |
| creatinine | +0.00522 |
| temperature_c | -0.00202 |

Max absolute lambda_j: 0.01065
Mean absolute lambda_j: 0.00663

## Interpretation

All eight lambda_j coefficients are below 0.05 in absolute value, consistent with the prior expectation that the time-varying confounders L_t do not exhibit subject-level functional random-effect structure beyond what is captured by their fixed-effect parameterization. Spec I (Y-only RE) is therefore parsimonious and sufficient; the additional flexibility of Spec II (shared RE on L) is not warranted by the data and adds no observable improvement to natural-course calibration (Spec I miss = -0.5%p; Spec II miss = -0.5%p).