# Table 3. Cross-method LOCO sensitivity

_Standard NICE full: 35.5/41.2; K=1 NICE full: 36.8/45.2; K=5 FRE-NICE full: 37.3/46.1 (bin16/bin17 risks)._

| Covariate | Type | Std b16 (delta) | Std b17 (delta) | K=1 b16 (delta) | K=1 b17 (delta) | K=5 b16 (delta) | K=5 b17 (delta) |
|---|---|---|---|---|---|---|---|
| pf_ratio | tv | 79.1 (+43.6) | 82.1 (+40.9) | 81.3 (+44.5) | 85.1 (+39.9) | 79.8 (+42.5) | 84.1 (+37.9) |
| lactate | tv | 43.1 (+7.6) | 47.2 (+6.0) | 45.8 (+9.0) | 46.1 (+0.9) | 45.9 (+8.6) | 46.8 (+0.7) |
| heart_rate | tv | 39.8 (+4.3) | 40.0 (-1.2) | 42.1 (+5.2) | 58.0 (+12.8) | 42.9 (+5.6) | 58.8 (+12.6) |
| anchor_age | static | 31.0 (-4.5) | 27.4 (-13.8) | 34.0 (-2.8) | 43.3 (-1.9) | 34.8 (-2.5) | 44.3 (-1.8) |
| paco2 | tv | 36.9 (+1.4) | 37.1 (-4.1) | 39.0 (+2.1) | 46.4 (+1.2) | 39.6 (+2.3) | 47.1 (+0.9) |
| charlson_index | static | 32.3 (-3.2) | 29.2 (-12.0) | 34.8 (-2.0) | 42.0 (-3.2) | 35.2 (-2.1) | 42.5 (-3.7) |
| map_mmhg | tv | 36.7 (+1.2) | 35.4 (-5.8) | 39.0 (+2.1) | 42.6 (-2.6) | 39.3 (+2.0) | 43.0 (-3.2) |
| gender_M | static | 33.0 (-2.5) | 31.1 (-10.1) | 35.5 (-1.4) | 43.5 (-1.7) | 35.6 (-1.7) | 44.6 (-1.5) |
| creatinine | tv | 36.1 (+0.6) | 37.9 (-3.3) | 38.4 (+1.5) | 50.6 (+5.4) | 38.9 (+1.6) | 51.7 (+5.5) |
| bmi_imputed | static | 33.0 (-2.5) | 30.2 (-11.0) | 35.4 (-1.4) | 42.8 (-2.4) | 35.9 (-1.4) | 43.5 (-2.7) |
| gcs_total | tv | 40.4 (+4.9) | 39.8 (-1.4) | 38.4 (+1.5) | 46.4 (+1.2) | 36.8 (-0.5) | 45.0 (-1.2) |
| temperature_c | tv | 34.2 (-1.3) | 33.2 (-8.0) | 36.4 (-0.4) | 44.6 (-0.6) | 37.0 (-0.3) | 45.3 (-0.9) |

## Interpretation (3-method concordance)

1. **PF ratio drop** (ARDS-defining variable): all three methods show large shift (Standard +43.6, K=1 +44.5, K=5 +42.5 percentage points at bin 16). Reported as sanity / domain-knowledge anchor, not as confounding signal.
2. **Lactate drop**: Standard +7.6, K=1 +9.0, K=5 +8.6 — three-method concordance identifies lactate as the dominant proper TV confounder.
3. **Static covariates** (age, BMI, charlson, gender): all methods show small negative shifts at bin 16 (-1 to -5 percentage points), direction concordant.
4. **K=1 vs K=5 LOCO**: deltas within +/- 2 percentage points at most cells. The functional random-effect parameterization does not qualitatively change confounder dependence structure.
5. **Standard bin 17 vs Bayesian bin 17** under static drops: Standard shows larger negative shifts (-10 to -14) than K=1 (-1.7 to -3.2) and K=5 (-1.5 to -3.7). This reflects Bayesian shrinkage (LKJ + HalfCauchy priors) absorbing some static-covariate variance via the random effect, which the frequentist Standard NICE method does not have.