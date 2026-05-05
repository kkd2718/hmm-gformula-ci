# Table 3. Sensitivity to unmeasured confounding (E-value) and covariate-set choice (LOCO + grouped exclusion)

## Panel A. E-value per MP bin (VanderWeele & Ding 2017) — three methods

_Reference bin: bin 16 (≈ 18.3 J/min, Costa 2021 cutoff). RR = P(Y|MP=bin) / P(Y|MP=ref). E-value = minimum unmeasured-confounder RR required to nullify the observed RR._

| MP bin | Center (J/min) | Standard parametric g-formula — RR | Xu 2024 GLMM — RR | VEM-SSM (proposed) — RR | Standard parametric g-formula — E-value | Xu 2024 GLMM — E-value | VEM-SSM (proposed) — E-value |
|---|---|---|---|---|---|---|---|
| 0 | 0.6 | 0.23 | 0.83 | 0.34 | 8.06 | 1.71 | 5.36 |
| 1 | 0.8 | 0.28 | 0.88 | 0.39 | 6.68 | 1.53 | 4.63 |
| 2 | 0.9 | 0.30 | 1.03 | 0.39 | 6.02 | 1.22 | 4.59 |
| 3 | 1.2 | 0.26 | 0.99 | 0.36 | 7.02 | 1.13 | 4.98 |
| 4 | 1.4 | 0.28 | 0.95 | 0.35 | 6.72 | 1.30 | 5.08 |
| 5 | 1.8 | 0.25 | 0.91 | 0.33 | 7.56 | 1.42 | 5.49 |
| 6 | 2.2 | 0.24 | 0.93 | 0.33 | 7.82 | 1.36 | 5.58 |
| 7 | 2.7 | 0.23 | 0.91 | 0.32 | 8.07 | 1.41 | 5.74 |
| 8 | 3.4 | 0.26 | 0.92 | 0.35 | 7.23 | 1.40 | 5.10 |
| 9 | 4.2 | 0.28 | 0.92 | 0.39 | 6.55 | 1.38 | 4.59 |
| 10 | 5.1 | 0.36 | 0.97 | 0.47 | 5.05 | 1.21 | 3.68 |
| 11 | 6.4 | 0.38 | 0.93 | 0.50 | 4.67 | 1.38 | 3.39 |
| 12 | 7.9 | 0.57 | 0.99 | 0.67 | 2.92 | 1.11 | 2.34 |
| 13 | 9.7 | 0.72 | 1.02 | 0.83 | 2.12 | 1.17 | 1.71 |
| 14 | 12.0 | 0.95 | 1.00 | 1.01 | 1.29 | 1.05 | 1.10 |
| 15 | 14.8 | 0.76 | 0.93 | 0.85 | 1.97 | 1.35 | 1.65 |
| 17 | 22.6 | 1.16 | 0.95 | 1.06 | 1.60 | 1.29 | 1.33 |
| 18 | 28.0 | 0.93 | 1.09 | 1.04 | 1.37 | 1.41 | 1.26 |
| 19 | 34.6 | 0.80 | 1.50 | 1.04 | 1.80 | 2.36 | 1.24 |

## Panel B. VEM-SSM LOCO + grouped exclusion sensitivity

_VEM-SSM refit excluding (i) each of 8 TV covariates, (ii) each of 4 static covariates, (iii) 5 clinical-scenario groups. Per-bin range = [min, max] of point-estimate 28-day risk across refits vs full-covariate primary._

| MP bin | Center (J/min) | Full primary (%) | TV LOCO range (%) | Static LOCO range (%) | Grouped exclusion range (%) |
|---|---|---|---|---|---|
| 0 | 0.6 | 11.2 | [10.3, 28.3] | [11.2, 11.2] | [10.1, 30.4] |
| 1 | 0.8 | 12.8 | [11.0, 31.8] | [11.5, 12.4] | [10.3, 31.9] |
| 2 | 0.9 | 12.9 | [11.8, 32.1] | [12.3, 13.1] | [11.2, 43.4] |
| 3 | 1.2 | 12.0 | [11.5, 33.7] | [12.1, 12.4] | [11.1, 39.5] |
| 4 | 1.4 | 11.8 | [10.5, 33.5] | [11.6, 12.1] | [10.6, 36.0] |
| 5 | 1.8 | 11.0 | [9.9, 32.1] | [10.9, 11.3] | [9.6, 36.6] |
| 6 | 2.2 | 10.8 | [9.7, 31.6] | [10.7, 11.1] | [9.3, 37.7] |
| 7 | 2.7 | 10.5 | [9.9, 31.0] | [10.4, 10.7] | [9.5, 41.7] |
| 8 | 3.4 | 11.7 | [11.1, 33.1] | [11.5, 11.8] | [10.7, 45.6] |
| 9 | 4.2 | 12.9 | [12.3, 34.8] | [12.7, 13.0] | [11.9, 47.6] |
| 10 | 5.1 | 15.6 | [15.1, 39.5] | [15.3, 15.7] | [14.8, 57.7] |
| 11 | 6.4 | 16.7 | [16.2, 38.9] | [16.3, 16.6] | [15.8, 57.1] |
| 12 | 7.9 | 22.3 | [21.9, 48.0] | [21.8, 22.4] | [21.1, 67.4] |
| 13 | 9.7 | 27.4 | [27.0, 54.5] | [26.5, 26.9] | [24.5, 75.5] |
| 14 | 12.0 | 33.4 | [32.7, 60.3] | [31.8, 32.7] | [29.0, 80.5] |
| 15 | 14.8 | 28.1 | [28.5, 56.2] | [26.1, 27.6] | [22.7, 84.1] |
| 16 | 18.3 | 33.2 | [33.3, 63.8] | [30.0, 32.0] | [24.8, 91.4] |
| 17 | 22.6 | 35.3 | [34.7, 65.8] | [31.4, 33.9] | [25.9, 92.5] |
| 18 | 28.0 | 34.6 | [36.1, 68.2] | [30.9, 33.9] | [23.3, 98.0] |
| 19 | 34.6 | 34.5 | [27.6, 44.1] | [34.4, 35.2] | [34.2, 42.2] |

### Panel B detail — per-exclusion VEM-SSM risk at reference bin and high bin

_Reference bin 16, max bin 19. Reports per-refit risk to identify which exclusions, if any, drive non-trivial change._

| Exclusion | Type | Risk @ ref bin (%) | Risk @ max bin (%) |
|---|---|---|---|
| (full covariate primary) | — | 33.2 | 34.5 |
| pf_ratio | TV | 63.8 | 44.1 |
| paco2 | TV | 33.3 | 34.5 |
| lactate | TV | 51.3 | 42.0 |
| map_mmhg | TV | 34.6 | 33.7 |
| heart_rate | TV | 35.8 | 35.0 |
| gcs_total | TV | 39.0 | 27.6 |
| creatinine | TV | 34.6 | 34.6 |
| temperature_c | TV | 33.6 | 34.6 |
| anchor_age | Static | 30.1 | 35.2 |
| gender_M | Static | 32.0 | 34.4 |
| bmi_imputed | Static | 31.8 | 35.0 |
| charlson_index | Static | 30.0 | 35.0 |
| All TV covariates | Group | 91.4 | 40.5 |
| All static covariates | Group | 24.8 | 36.2 |
| Oxygenation domain (P/F + PaCO2) | Group | 72.9 | 42.2 |
| Hemodynamic domain (HR + MAP) | Group | 37.3 | 34.2 |
| Metabolic/renal domain (lactate + creatinine) | Group | 52.3 | 41.5 |