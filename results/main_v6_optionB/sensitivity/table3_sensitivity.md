# Table 3. Sensitivity to unmeasured confounding (E-value) and covariate-set choice (LOCO + grouped exclusion)

## Panel A. E-value per MP bin (VanderWeele & Ding 2017) — three methods

_Reference bin: bin 16 (≈ 18.3 J/min, Costa 2021 cutoff). RR = P(Y|MP=bin) / P(Y|MP=ref). E-value = minimum unmeasured-confounder RR required to nullify the observed RR._

| MP bin | Center (J/min) | Standard parametric g-formula — RR | Xu 2024 GLMM — RR | VEM-SSM (proposed) — RR | Standard parametric g-formula — E-value | Xu 2024 GLMM — E-value | VEM-SSM (proposed) — E-value |
|---|---|---|---|---|---|---|---|
| 0 | 0.6 | 0.23 | 0.83 | 0.29 | 8.06 | 1.71 | 6.36 |
| 1 | 0.8 | 0.28 | 0.88 | 0.33 | 6.68 | 1.53 | 5.47 |
| 2 | 0.9 | 0.30 | 1.03 | 0.35 | 6.02 | 1.22 | 5.12 |
| 3 | 1.2 | 0.26 | 0.99 | 0.32 | 7.02 | 1.13 | 5.79 |
| 4 | 1.4 | 0.28 | 0.95 | 0.32 | 6.72 | 1.30 | 5.78 |
| 5 | 1.8 | 0.25 | 0.91 | 0.29 | 7.56 | 1.42 | 6.37 |
| 6 | 2.2 | 0.24 | 0.93 | 0.28 | 7.82 | 1.36 | 6.62 |
| 7 | 2.7 | 0.23 | 0.91 | 0.27 | 8.07 | 1.41 | 6.80 |
| 8 | 3.4 | 0.26 | 0.92 | 0.30 | 7.23 | 1.40 | 6.09 |
| 9 | 4.2 | 0.28 | 0.92 | 0.32 | 6.55 | 1.38 | 5.65 |
| 10 | 5.1 | 0.36 | 0.97 | 0.39 | 5.05 | 1.21 | 4.53 |
| 11 | 6.4 | 0.38 | 0.93 | 0.41 | 4.67 | 1.38 | 4.28 |
| 12 | 7.9 | 0.57 | 0.99 | 0.58 | 2.92 | 1.11 | 2.83 |
| 13 | 9.7 | 0.72 | 1.02 | 0.75 | 2.12 | 1.17 | 2.01 |
| 14 | 12.0 | 0.95 | 1.00 | 0.94 | 1.29 | 1.05 | 1.33 |
| 15 | 14.8 | 0.76 | 0.93 | 0.79 | 1.97 | 1.35 | 1.85 |
| 17 | 22.6 | 1.16 | 0.95 | 1.26 | 1.60 | 1.29 | 1.83 |
| 18 | 28.0 | 0.93 | 1.09 | 1.01 | 1.37 | 1.41 | 1.09 |
| 19 | 34.6 | 0.80 | 1.50 | 0.94 | 1.80 | 2.36 | 1.34 |

## Panel B. VEM-SSM LOCO + grouped exclusion sensitivity

_VEM-SSM refit excluding (i) each of 8 TV covariates, (ii) each of 4 static covariates, (iii) 5 clinical-scenario groups. Per-bin range = [min, max] of point-estimate 28-day risk across refits vs full-covariate primary._

| MP bin | Center (J/min) | Full primary (%) | TV LOCO range (%) | Static LOCO range (%) | Grouped exclusion range (%) |
|---|---|---|---|---|---|
| 0 | 0.6 | 10.8 | [9.8, 26.6] | [10.6, 11.3] | [9.8, 30.4] |
| 1 | 0.8 | 12.4 | [10.4, 30.7] | [11.8, 12.0] | [10.2, 33.1] |
| 2 | 0.9 | 13.1 | [11.9, 33.5] | [13.1, 13.5] | [11.5, 43.4] |
| 3 | 1.2 | 11.8 | [10.6, 32.4] | [11.9, 12.5] | [11.1, 39.5] |
| 4 | 1.4 | 11.8 | [10.2, 32.7] | [11.6, 12.1] | [10.7, 36.0] |
| 5 | 1.8 | 10.8 | [9.5, 30.8] | [10.9, 11.2] | [9.5, 36.6] |
| 6 | 2.2 | 10.4 | [9.2, 29.6] | [10.4, 10.8] | [9.0, 37.7] |
| 7 | 2.7 | 10.2 | [9.4, 29.7] | [10.2, 10.8] | [9.3, 41.7] |
| 8 | 3.4 | 11.2 | [10.6, 32.4] | [11.2, 11.8] | [10.4, 45.6] |
| 9 | 4.2 | 12.0 | [11.4, 32.4] | [11.8, 12.4] | [11.2, 47.6] |
| 10 | 5.1 | 14.6 | [14.0, 37.5] | [14.3, 14.8] | [13.9, 57.7] |
| 11 | 6.4 | 15.4 | [14.9, 37.1] | [15.0, 15.4] | [14.7, 57.1] |
| 12 | 7.9 | 21.7 | [21.3, 48.1] | [21.3, 21.6] | [20.5, 67.4] |
| 13 | 9.7 | 27.8 | [27.5, 56.2] | [26.9, 27.6] | [25.0, 75.5] |
| 14 | 12.0 | 34.9 | [34.5, 63.3] | [33.3, 34.6] | [30.4, 80.5] |
| 15 | 14.8 | 29.4 | [29.4, 60.9] | [27.5, 30.1] | [24.2, 84.1] |
| 16 | 18.3 | 37.2 | [36.0, 70.9] | [33.3, 37.8] | [29.0, 91.4] |
| 17 | 22.6 | 46.8 | [44.9, 77.2] | [44.6, 48.4] | [37.9, 92.5] |
| 18 | 28.0 | 37.5 | [37.3, 73.8] | [32.1, 38.2] | [26.2, 98.0] |
| 19 | 34.6 | 34.8 | [28.0, 45.4] | [35.0, 35.3] | [34.6, 46.5] |

### Panel B detail — per-exclusion VEM-SSM risk at reference bin and high bin

_Reference bin 16, max bin 19. Reports per-refit risk to identify which exclusions, if any, drive non-trivial change._

| Exclusion | Type | Risk @ ref bin (%) | Risk @ max bin (%) |
|---|---|---|---|
| (full covariate primary) | — | 37.2 | 34.8 |
| pf_ratio | TV | 70.9 | 45.4 |
| paco2 | TV | 37.4 | 34.8 |
| lactate | TV | 44.8 | 32.3 |
| map_mmhg | TV | 39.4 | 34.2 |
| heart_rate | TV | 39.7 | 35.1 |
| gcs_total | TV | 41.3 | 28.0 |
| creatinine | TV | 37.2 | 35.2 |
| temperature_c | TV | 36.0 | 35.1 |
| anchor_age | Static | 33.3 | 35.3 |
| gender_M | Static | 37.8 | 35.0 |
| bmi_imputed | Static | 36.0 | 35.2 |
| charlson_index | Static | 34.7 | 35.0 |
| All TV covariates | Group | 91.4 | 40.5 |
| All static covariates | Group | 29.0 | 36.2 |
| Oxygenation domain (P/F + PaCO2) | Group | 73.5 | 46.5 |
| Hemodynamic domain (HR + MAP) | Group | 42.4 | 34.6 |
| Metabolic/renal domain (lactate + creatinine) | Group | 47.5 | 35.0 |