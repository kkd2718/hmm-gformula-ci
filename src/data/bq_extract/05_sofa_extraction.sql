-- First-day SOFA score extraction from MIMIC-IV v3.1 derived tables.
-- Adds total SOFA + per-organ component scores for each ICU stay.
SELECT
  stay_id,
  sofa            AS sofa_24h,
  respiration     AS sofa_respiration,
  coagulation     AS sofa_coagulation,
  liver           AS sofa_liver,
  cardiovascular  AS sofa_cardiovascular,
  cns             AS sofa_cns,
  renal           AS sofa_renal
FROM `physionet-data.mimiciv_3_1_derived.first_day_sofa`
ORDER BY stay_id;
