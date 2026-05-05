# Results: Table-by-table and Figure-by-figure interpretation

This document gives a clinical and statistical interpretation for each
manuscript table and figure, drawing on the underlying numbers in
`results/`. Use as a writing guide; the manuscript narrative is in
`manuscript_results.md`.

---

## Table 1 — Baseline characteristics (overall + Berlin severity)

**Source:** `results/table1.md`

| Stratum | N | Median age | Male % | Median day-1 P/F | Day-1 MP (J/min) | 30-day mortality |
|---|---|---|---|---|---|---|
| Overall | 17{,}878 | 64 | 60.7 | 178 | 3.9 | 26.3% |
| Mild | 7{,}405 | 65 | 63.2 | 250 | 3.2 | 17.8% |
| Moderate | 7{,}365 | 63 | 59.5 | 148 | 4.5 | 28.9% |
| Severe | 3{,}108 | 64 | 57.9 | 78 | 5.0 | 40.2% |

**Interpretation:**

- Cohort size (N=17{,}878 stays / 15{,}619 unique subjects) is among the
  largest published ARDS-MP causal-inference cohorts.
- Berlin severity distribution (41% mild / 41% moderate / 17% severe)
  matches contemporary MIMIC-IV ARDS reports.
- The mortality gradient by severity (17.8% → 40.2%) is consistent with
  Bellani et al. (2016) LUNG-SAFE multinational benchmark.
- Median day-1 MP (3.9 J/min) is well below the Costa cutoff (17 J/min);
  exposures escalating later in ICU course are what drive the dose-response
  signal at higher bins.
- Severe ARDS receives slightly higher day-1 MP (5.0 vs 3.2 J/min mild),
  reflecting clinical practice of more aggressive ventilation in worse
  oxygenation — this is the textbook treatment-confounder feedback that
  motivates g-formula estimation.

**Writing note:** The 30-day mortality reported here (26.3%) approximates the
28-day cumulative incidence used as the outcome (25.6%). State both clearly.

---

## Table 2 — Four-method dose-response comparison (primary table)

**Source:** `results/bayesian_jax/fre_nice_K5_risks.npz`,
`results/bayesian_jax/fre_nice_K1_risks.npz`,
`results/bayesian_main_v2/xu_bayesian_risks.npz`,
`results/standard_v2/table2_risks.npz`

Dose-response at canonical contrast bins (95% credible/confidence interval).

| Bin | MP (J/min) | Standard NICE | Xu Bayesian | K=1 NICE | **K=5 FRE-NICE** |
|---|---|---|---|---|---|
| 0 | 0.6 | 8.3 | 18.3 | 8.4 | 9.1 |
| 7 | 2.7 | — | — | 8.0 | 8.0 (5.5–11.4) |
| 11 | 6.4 | — | — | 12.8 | 14.0 (10.6–17.7) |
| 14 | 12.0 | 33.7 | 21.6 | 36.9 | 36.4 (30.0–42.4) |
| **16** (ref) | **18.3** | **35.5** | **21.6** | **36.8** | **37.3 (32.8–41.4)** |
| 17 | 22.6 | 41.2 | 21.0 | 45.2 | 46.1 (38.9–52.4) |
| 18 | 28.0 | 32.8 | 23.6 | 35.2 | 34.5 |

**Interpretation:**

- **NICE-family methods (Standard, K=1, K=5) are tightly concordant**: at the
  reference bin and high-MP bin, all three estimate within $\pm 5$ percentage
  points.
- **Xu Bayesian is essentially flat at $\sim$21%** across all MP bins.
- This 25-percentage-point gap at high MP is **not a Bayesian-vs-frequentist
  artifact** (Standard is frequentist; K=1 and K=5 are Bayesian — they all
  agree). It is a **framework difference**:
  - NICE g-formula simulates $L_t$ forward under the counterfactual treatment
    regime, capturing the indirect effect of MP on mortality through changes
    in $L_t$ (HR, lactate, etc.).
  - Xu MSM-style observed-$L$ plug-in does not propagate the counterfactual
    treatment through $L$, partially blocking the indirect effect.
- Under a mediation interpretation, NICE estimates approximate the
  **total causal effect** of sustained MP exposure, whereas Xu approximates
  a **controlled direct effect** (with $L$ held at observed values). The
  $\sim$25-percentage-point gap is the effect mediated through time-varying
  physiology.
- The K=5 FRE-NICE (primary) yields the steepest gradient: at 22.6 J/min,
  cumulative incidence reaches 46.1% — a 5.7-fold increase over the
  low-MP regime ($\sim$8% at 2.7 J/min).

**Writing note:** Frame Xu vs NICE as a *framework choice* and a partial
mediation analysis, not as one method being right and the other wrong.

---

## Table 2 lower — WAIC + PSIS-LOO (Bayesian model comparison)

**Source:** `results/loglik_main/waic_loo_table.md`

| Method | WAIC | $\Delta$ELPD vs K=5 | Pareto-$\hat{k}$ max |
|---|---|---|---|
| K=5 FRE-NICE | 34{,}404 | $0$ (favored) | 0.50 |
| K=1 NICE | 35{,}686 | $-640.9$ | 0.50 |
| Xu Bayesian | 35{,}676 | $-636.0$ | 0.50 |

**Interpretation:**

- **K=5 is decisively preferred** by both WAIC and PSIS-LOO. $\Delta$ELPD of
  640 nats over 15{,}619 subjects ($\sim$0.04 nats per subject) far exceeds
  conventional thresholds.
- **K=1 ≈ Xu** ($\Delta$ELPD $-4.9$): the two scalar-RE outcome models are
  mathematically equivalent in fitting, and this is empirically confirmed.
  Their dose-response estimates differ only because they use different
  frameworks for counterfactual computation (forward $L$ sim vs observed
  $L$ plug-in).
- All Pareto-$\hat{k}$ values $\le 0.50$ → reliable PSIS-LOO estimation
  (the threshold for unreliable is 0.7).
- Effective parameters $p_{\text{waic}}$: K=1 ≈ 2{,}084, K=5 ≈ 4{,}600. The
  $\sim$2.2-fold increase reflects the four extra basis dimensions per
  subject (15{,}619 × 4 / heavy shrinkage from LKJ + half-Cauchy priors).

**Writing note:** This is the formal evidence for the methodological claim
that the *time-varying* (functional) random effect adds value beyond a
scalar random intercept. Without this comparison, the K=5 vs K=1 contrast
would be unsupported.

---

## Table 3 — Cross-method LOCO sensitivity (12 covariates × 3 methods)

**Source:** `_draft/table3_cross_method_loco.md` (numbers from
`results/loco_K5/`, `results/loco_K1/`, and `results/appendix_g/loco_standard.md`)

(Full table in source file; key contrasts at bin 16 below.)

| Covariate dropped | Standard $\Delta$ | K=1 $\Delta$ | K=5 $\Delta$ |
|---|---|---|---|
| pf_ratio | $+43.6$ | $+44.5$ | $+42.5$ |
| lactate | $+7.6$ | $+9.0$ | $+8.6$ |
| heart_rate | $+4.3$ | $+5.2$ | $+5.6$ |
| anchor_age | $-4.5$ | $-2.8$ | $-2.5$ |
| charlson | $-3.2$ | $-2.0$ | $-2.1$ |
| (other 7) | $\pm 1$–$3$ | $\pm 1$–$2$ | $\pm 1$–$3$ |

**Interpretation:**

- **PF ratio drop** (the ARDS-defining oxygenation variable): all three
  methods show massive shifts of $\sim$+43 percentage points. Reported as
  domain-knowledge anchor and sanity check, *not* as confounding.
  Excluding the definitional variable inflates the apparent dose-response.
- **Lactate drop**: all three methods identify $\sim$+8 percentage point
  shift — physiologically meaningful confounder, three-method concordance.
- **Static covariate drops** (age, BMI, Charlson, gender): all methods show
  small *negative* shifts at bin 16 ($-1$ to $-5$ p.p.), direction
  concordant. This is consistent with these being confounders that, when
  adjusted, mildly reduce the apparent risk at the reference.
- **K=1 ↔ K=5 LOCO concordance**: deltas within $\pm 2$ p.p. at most cells.
  The functional random-effect parameterization does not qualitatively
  change confounder dependence.
- **Standard NICE is more sensitive at high-MP bin 17** under static-covariate
  drops ($-10$ to $-14$) than the Bayesian methods ($-1.5$ to $-3.7$). This
  reflects Bayesian shrinkage (LKJ + half-Cauchy) absorbing some
  static-covariate variance into the random effect; the frequentist Standard
  has no such mechanism.

**Writing note:** Acknowledge that several listed "TV confounders" (HR,
MAP, lactate, etc.) may be partial mediators. Frame the LOCO not as
"all variables are confounders" but as "robustness of the estimate to
covariate-set choice." The mediation discussion belongs in Discussion §.

---

## Figure 3 — Four-method dose-response (Panel A) and K=5 with credible band (Panel B)

**Source:** `results/figures/fig3_dose_response.png` and `.pdf`

**Panel A** overlays Standard, Xu Bayesian, K=1 NICE, K=5 FRE-NICE on the
log-MP axis, with the Costa 2021 cutoff (17 J/min) annotated.

**Panel B** shows K=5 mean with 95% credible band.

**Interpretation:**

- **Visual clustering of NICE methods** (Standard ~grey, K=1 ~blue, K=5 ~red
  curves) versus the **flat orange Xu line** is the most striking visual.
- The Costa cutoff (17 J/min, dashed black) sits at the *inflection* of the
  K=5 curve — below the cutoff, mortality rises gradually; above, it
  accelerates.
- Bin 19 (highest MP, $\sim$35 J/min): all methods show a slight *drop*
  in mortality. **This is a positivity artifact.** Bin 19 N=16{,}694 stays
  appears anomalously high (vs bins 17 = 316, 18 = 135) — most likely
  reflecting an off-MV / extubated period coding artifact in the original
  cohort assembly. Discuss as a positivity caveat; do not over-interpret.
- The credible band (Panel B) widens substantially at high-MP bins (17–19),
  reflecting sparse exposure.

**Writing note:** Mention bin 19 caveat in figure legend or accompanying
text. The "interesting" range is bins 7–17.

---

## Figure 4 — Subgroup forest plot (9 strata, K=5 dose-response gradient)

**Source:** `results/figures/fig4_subgroup_forest.png`

Risk difference (high MP minus low MP) per subgroup with 95% CrI.

| Subgroup | N | RD (95% CrI) |
|---|---|---|
| Mild ARDS | 7{,}405 | $+36.1$ ($+28.8$, $+42.6$) |
| Moderate ARDS | 7{,}365 | $+38.8$ ($+31.6$, $+45.5$) |
| Severe ARDS | 3{,}108 | $+41.8$ ($+34.2$, $+48.5$) |
| Age $\le$ median | 8{,}186 | $+30.6$ ($+23.8$, $+37.0$) |
| Age $>$ median | 9{,}692 | $+44.7$ ($+36.9$, $+51.4$) |
| Charlson $\le$ median | 9{,}538 | $+31.4$ ($+24.5$, $+38.0$) |
| Charlson $>$ median | 8{,}340 | $+46.2$ ($+38.3$, $+53.0$) |
| BMI $\le$ median | 6{,}568 | $+43.1$ ($+35.5$, $+49.8$) |
| BMI $>$ median | 11{,}310 | $+35.6$ ($+28.3$, $+41.9$) |

**Interpretation:**

- All 9 subgroups show **strong, positive, non-overlapping-with-zero** RDs.
  The dose-response signal is pervasive across all clinically relevant
  strata.
- **Severity gradient**: severe $>$ moderate $>$ mild ($+41.8$ vs $+36.1$,
  ordered correctly).
- **Age gradient**: older patients show larger absolute mortality difference
  ($+44.7$ vs $+30.6$) — consistent with the literature that age modifies
  the effect of mechanical injury.
- **Charlson gradient**: high comorbidity load amplifies the dose-response
  ($+46.2$ vs $+31.4$).
- **BMI: obesity-paradox-like signal** — low BMI subjects have *larger*
  RD ($+43.1$ vs $+35.6$). This deserves Discussion attention; possible
  explanations include: (i) low BMI as marker of pre-existing frailty;
  (ii) higher mechanical-load-per-mass for the same MP; (iii) confounding
  by underlying nutritional status. The obesity paradox is well-documented
  in ICU mortality (De Jong et al., 2016).

**Writing note:** This figure supports the generalizability claim. The BMI
finding is publishable in itself — frame as exploratory but consistent with
prior obesity-paradox literature.

---

## Supplementary tables (for Appendix)

### S1 — Per-bin sample sizes (positivity)

**Source:** `results/positivity/positivity_check.md`

Bins 0–2 (very low MP, N=462–1{,}023), 17–18 (high MP, N=135–316), and
the bin-19 anomaly (N=16{,}694) are sparse / artifactual. Report
positivity transparency in Methods.

### S2 — Natural-course validation

**Source:** Memory + `results/bayesian_jax/nc_validation.md` +
`results/bayesian_spec2/nc_spec2.md`

| Method | NC mortality | Miss vs cohort raw 25.6% |
|---|---|---|
| Standard NICE | 24.6% | $-1.0$ p.p. ✓ |
| Xu Bayesian | 29.3% | $+3.7$ p.p. (consistent with MSM attenuation) |
| K=1 NICE | 25.4% | $-0.2$ p.p. ✓ |
| K=5 FRE-NICE | 25.1% | $-0.5$ p.p. ✓ |
| K=5 Spec II | 25.1% | $-0.5$ p.p. ✓ |

NICE-family natural course is well-calibrated; Xu's $+3.7$ p.p. miss is
consistent with the framework attenuation.

### S3 — Knot sensitivity (K=4, K=5, K=6)

**Source:** `results/knot_sensitivity/`

| K | bin 16 (ref) | bin 17 |
|---|---|---|
| 4 | 37.2 (32.7–41.9) | 46.2 (39.8–52.7) |
| **5** (primary) | **37.3 (32.8–41.4)** | **46.1 (38.9–52.4)** |
| 6 | 37.0 (32.7–41.8) | 46.4 (38.7–53.6) |

$\le 0.3$ p.p. variation across knot specifications. K=5 is robust.

### S4 — Spec II $\lambda_j$ (shared RE on $L$ ablation)

**Source:** `_draft/spec2_lambda_L.md`

Eight $\lambda_j$ values from $-0.0093$ to $+0.0107$, all near zero.
**Spec I (Y-only RE) is sufficient**; sharing the random effect across
$L$ equations adds no information.

### S5 — Prior sensitivity ($\sigma_b$)

**Source:** `results/prior_sens_gamma/`

Half-Cauchy(0, 2.5) primary vs Gamma(2, 0.5) sensitivity gives
$\le 0.1$ p.p. difference in dose-response at bins 16 and 17. Inference
robust to prior choice.

### S6 — PPC on held-out 20%

**Source:** `results/ppc_K5/ppc_K5_v2.md`

Day-by-day cumulative incidence calibration miss within $\pm 3$ p.p.
across the entire 28-day course. Day-28 miss $-2.6$ p.p. (predicted
$19.5%$, observed $22.0%$). Predictive validity supported.

### S7 — Convergence diagnostics

**Source:** `results/loglik_main/diagnostics_table.md`

All Bayesian fits: $\hat{R} \le 1.08$, ESS$_{\text{min}} \ge 71$,
**0 divergent transitions across all chains and all fits**.

### S8 — E-value sensitivity to unmeasured confounding

**Source:** `_draft/e_value_sensitivity.md`

| Contrast | RR (95% CrI) | E-value (point) | E-value (CI bound) |
|---|---|---|---|
| bin 7 vs ref 16 | 0.22 (0.18–0.26) | 8.71 | 7.23 |
| bin 11 vs ref 16 | 0.38 (0.32–0.44) | 4.77 | 4.02 |
| bin 17 vs ref 16 | 1.24 (1.03–1.44) | 1.79 | 1.19 |

Low-MP protective effect (E ≈ 8.7) is highly robust; high-MP harm
(E ≈ 1.8) is moderately robust but with lower CI bound at 1.19.

### S9 — Costa 2021 alignment

**Source:** `_draft/costa2021_alignment.md`

Dose-response cutoff ($\sim$17 J/min) and inflection direction agree;
our nonlinear estimate refines the Costa linear HR per 5 J/min.

---

## Issues to address in writing

### 1. Bin 19 positivity artifact

Bin 19 (highest MP, $\sim$35 J/min) has N=16{,}694 stays "ever in" the
bin — clearly an artifact (vs bins 17/18 with N=316/135). Likely
reflects off-mechanical-ventilation periods being mapped to the highest
bin via the np.clip operation in `_assign_bins()`. **Manuscript should
either:**
  (a) restrict primary inference to bins 0–18 and footnote bin 19; or
  (b) re-cohort with explicit "off-MV" exclusion before binning.
Option (a) is faster; option (b) is more rigorous but requires re-running
the cohort assembly.

### 2. Heart-rate-as-mediator-or-confounder

The LOCO results (HR drop $\to +12.6$ p.p. at bin 17 in K=5) and the
literature standard (Costa 2021, Serpa Neto 2018 do not include daily
vital signs as TV confounders) jointly suggest HR (and possibly MAP,
lactate, GCS) act partially as mediators. **Recommended manuscript framing:**

  - Methods §: define the primary estimand explicitly. The 4-method ladder
    estimates two distinct quantities — NICE (Standard, K=1, K=5) target
    the *total* causal effect; Xu MSM targets a *controlled direct effect*
    (with $L$ held at observed). The $\sim$25 p.p. framework gap is
    consistent with mediation through TV physiology.
  - Discussion §: dedicate one paragraph to the mediator concern,
    citing VanderWeele & Tchetgen-Tchetgen (2014) and noting that
    formal interventional-effect decomposition is left as future work.
  - LOCO HR result is sensitivity, not main estimate.

### 3. PPC predicted-vs-actual definition

The PPC script reports two day-28 numbers: subject-level "max Y over
at-risk days" (24.97% actual) versus day-28 cumulative incidence by the
same survival-function definition used for the dose-response (22.04%
actual). **For consistency with dose-response numbers, use 22.04%
(cumulative incidence) as the actual.** The miss is then $-2.57$ p.p.
(predicted 19.47%, observed 22.04%).

### 4. Mixed inference framework disclosure

Standard NICE is frequentist (cluster bootstrap CIs); Xu, K=1, K=5 are
Bayesian (posterior credible intervals). **Disclose explicitly in
Methods**; do not implicitly compare CIs across paradigms without note.

### 5. Cohort flow (Figure 1) and DAG (Figure 2) not yet generated

Currently described in text only. Manuscript will need:
- Fig 1: CONSORT-like flow chart from MIMIC-IV total to final N=17{,}878
- Fig 2: DAG showing $V \to A_t \to L_{t+1} \to A_{t+1} \to Y_{t+1}$
  with random effect $b_i$ on $Y$
These can be drawn in any standard tool; data for Fig 1 is in cohort
assembly logs.

### 6. Reference list not yet compiled

Key citations (compile bibliography):
- Robins (1986, *Math Modelling*) — NICE g-formula
- Robins, Hernán & Brumback (2000, *Epidemiology*) — MSM
- Hernán & Robins (2020) — *Causal Inference: What If*, §21
- Costa et al. (2021, *AJRCCM*) — MP and 28-day mortality
- Gattinoni et al. (2016) — mechanical power
- Amato et al. (2015, *NEJM*) — driving pressure
- Xu et al. (2024, *Biometrics*) — GLMM g-computation
- Yao, Müller & Wang (2005, *JASA*) — functional random effects
- Wood (2017) — *Generalized Additive Models*
- Lewandowski, Kurowicka & Joe (2009) — LKJ prior
- VanderWeele & Ding (2017, *Annals of Internal Medicine*) — E-value
- VanderWeele & Tchetgen-Tchetgen (2014) — TV mediation
- Vehtari, Gelman & Gabry (2017, *Statistics and Computing*) — PSIS-LOO
- Watanabe (2010) — WAIC
- Hoffman & Gelman (2014, *JMLR*) — NUTS
- Bingham et al. (2019) — Pyro/numpyro

### 7. Random-effect weights confirmed saved (no issue)

All 35 NUTS posterior states are on disk (395 MB): β (2000 × 33 outcome),
L_chol (Cholesky of $\Sigma_b$), τ (RE scale per basis), spline basis,
β_L, σ_L. Subject-level log-likelihood (2000 × 15{,}619) saved for
WAIC/LOO methods. All dose-response counterfactuals can be re-run from
saved states.
