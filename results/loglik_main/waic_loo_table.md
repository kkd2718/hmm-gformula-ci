# WAIC + PSIS-LOO comparison

| Method | N_obs | WAIC | WAIC SE | LOO | LOO SE | p_waic | Pareto-k max | bad k>0.7 |
|---|---|---|---|---|---|---|---|---|
| K=1 NICE | 15619 | 35686.2 | 454.5 | 35584.3 | 454.2 | 2083.8 | 0.50 | 0 |
| K=5 FRE-NICE | 15619 | 34404.4 | 446.0 | 34437.9 | 447.3 | 4600.3 | 0.50 | 0 |
| Xu Bayesian | 15619 | 35676.4 | 454.4 | 35574.4 | 454.1 | 2097.0 | 0.50 | 0 |

## Pairwise ELPD difference (positive favors first label)
| A vs B | ΔELPD-WAIC | ΔELPD-LOO |
|---|---|---|
| K=1 NICE vs K=5 FRE-NICE | -640.9 | -573.2 |
| K=1 NICE vs Xu Bayesian | -4.9 | -4.9 |
| K=5 FRE-NICE vs Xu Bayesian | +636.0 | +568.3 |

Notes: ΔELPD > 4 (with SE not overlapping zero) is conventionally considered model preference. Pareto-k > 0.7 indicates unstable PSIS-LOO estimates for those observations.