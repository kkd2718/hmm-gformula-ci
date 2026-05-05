# Introduction (manuscript draft, v1)

## Background

Mechanical ventilation is life-sustaining for patients with acute respiratory
distress syndrome (ARDS), but its delivery itself imposes biophysical
injury on the lung — ventilator-induced lung injury (VILI) — that
contributes substantially to the high mortality observed in this
population (Slutsky & Ranieri, 2013). Mechanical power (MP), introduced
by Gattinoni et al. (2016), aggregates the principal physical determinants
of VILI (tidal volume, respiratory rate, driving pressure, and PEEP) into
a single energy-rate metric expressed in joules per minute. Costa et al.
(2021) reported in a multicenter cohort that 28-day mortality increased
with MP, with an inflection at approximately 17 J/min above which risk
acceleration was particularly steep, and adjusted hazard ratio of
approximately 1.06 per 5 J/min increase. Subsequent work has corroborated
the prognostic utility of MP across heterogeneous ICU populations (Serpa
Neto et al., 2018) and confirmed that strategies minimizing MP — such as
reducing tidal volume or driving pressure — are associated with improved
outcomes (Amato et al., 2015; Schmidt et al., 2020).

These observational findings, however, raise a causal-inferential question
that hazard-ratio summaries do not directly address: *what would the
mortality have been had patients been ventilated at a sustained level of
mechanical power $a^*$, marginalized over baseline characteristics and
the trajectory of physiology?* Answering this question requires a method
that handles time-varying treatment, time-varying confounders that are
themselves affected by prior treatment (e.g., the PaO$_2$/FiO$_2$ ratio
responds to MP and influences subsequent ventilator adjustment), and
clustered repeated measurements within subjects.

## The methodological gap

Two principal frameworks address longitudinal causal inference under
treatment-confounder feedback: marginal structural models (MSM) with
inverse probability weighting (Robins, Hernán & Brumback, 2000) and the
parametric g-formula (Robins, 1986; Westreich et al., 2012). Recent
methodological work has integrated subject-specific random effects into
both frameworks. Xu et al. (2024) introduced a generalized linear mixed
model (GLMM) variant of g-computation with a scalar random intercept to
account for unobserved subject-level heterogeneity in clustered survival
data. To our knowledge, however, no published implementation extends this
random effect to a *time-varying* form within the NICE parametric g-formula
framework, which is the framework most natural for the ARDS-MP question
because of the strong treatment-confounder feedback in respiratory
physiology.

A scalar random effect assumes that subject-level deviation from population-
level outcome regression is constant across the ICU course — a strong
assumption when ICU physiology evolves substantially across days due to
disease progression, organ recovery, or weaning trajectories. A time-varying
generalization, in which the subject-specific deviation evolves smoothly
along the day grid, would be a natural and theoretically grounded
extension drawing on functional data analysis (Yao, Müller & Wang, 2005)
and smooth random-effects models (Wood, 2017).

## Aim and contribution

We aim to estimate the causal effect of sustained mechanical power exposure
on 28-day mortality in a large ARDS cohort and to develop a Bayesian
parametric g-formula with a functional random effect (FRE) on the outcome
hazard, parameterized by a natural cubic spline basis. To our knowledge,
this is the first implementation of NICE g-formula with a subject-specific
*time-varying* functional random effect operationalized via spline basis
expansion. The proposed FRE-NICE g-formula generalizes Xu et al.'s scalar
random intercept (recovered when the basis is constant), embeds subject-
level time-varying heterogeneity in a parametric and identifiable way,
and supports principled posterior credible intervals via Hamiltonian Monte
Carlo. We benchmark the proposed method against three established
comparators (Standard parametric g-formula, Xu et al. GLMM g-computation,
and the K=1 scalar special case) and report a comprehensive set of
sensitivity analyses to address the principal threats to validity in
parametric g-formula estimation.
