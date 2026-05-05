# Posterior diagnostics (Appendix B)

| Method | n params | R-hat max | R-hat p95 | ESS min | ESS median | Divergent |
|---|---|---|---|---|---|---|
| K=1 NICE | 46892 | 1.019 | 1.001 | 154 | 2747 | 0 |
| K=5 FRE-NICE | 171897 | 1.078 | 1.001 | 75 | 4418 | 0 |
| Xu Bayesian | 46891 | 1.011 | nan | 273 | 3322 | 0 |
| K=5 prior=Gamma | 156278 | 1.079 | 1.000 | 77 | 4608 | 0 |
| K=5 80%-fit | 156278 | 1.056 | 1.000 | 71 | 4418 | 0 |

Conventions: R-hat < 1.05 indicates convergence; ESS bulk ≥ 400 (2 chains × 1000 sample target) indicates adequate posterior sample mixing for inference; divergent transitions should be 0 or near-zero — non-zero divergences flag potential geometric issues with the posterior surface.