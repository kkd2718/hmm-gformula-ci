-- ARDS cohort v3.1 extraction, version 3 (study-design corrected)
-- ================================================================
-- Changes vs v2:
--   (A1) day_num is now defined relative to the first-observed-day
--        Berlin date (= eligibility + treatment exposure aligned),
--        per Hernán & Robins (2016) target trial emulation principle.
--        Pre-Berlin period is left-truncated (excluded from analysis).
--
--   (A2) Outcome covariates expanded with ARDS-relevant vitals and
--        labs that are NOT algebraic components of MP. The CSV now
--        contains additional clinical channels that the loader can
--        select as L_t while ensuring driving_pressure / compliance
--        / peep / pplat / ppeak / rr / tidvol_obs (MP components)
--        are NOT used as outcome covariates downstream.
--
--   New time-varying covariates (chartevents):
--     - heart_rate         (220045)
--     - spo2               (220277)
--     - map                (220052 invasive MAP, 220181 non-invasive fallback)
--     - temperature_c      (223762 Celsius; 223761 Fahrenheit converted)
--     - gcs_total          (220739 + 223900 + 223901 sum)
--   New time-varying labs:
--     - paco2              (50818)
--     - ph_arterial        (50820)
--     - hemoglobin         (51222)
--     - creatinine         (50912)
--   Existing kept: pao2, fio2 (for pf_ratio), lactate (50813)

WITH adults AS (
  SELECT
    i.stay_id, i.subject_id, i.hadm_id,
    i.intime, i.outtime, i.los,
    p.gender, p.anchor_age, p.dod
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING (subject_id)
  WHERE p.anchor_age >= 18 AND i.los >= 1
),

-- Pre-filter chartevents to relevant items (broader list now)
ce_filtered AS (
  SELECT ce.stay_id, ce.itemid, ce.charttime,
         CAST(ce.valuenum AS FLOAT64) AS valuenum,
         DATE(ce.charttime) AS d
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
  JOIN adults a USING (stay_id)
  WHERE ce.itemid IN (
    -- Ventilator parameters (for MP + Berlin filter)
    223835,                    -- FiO2
    220339, 224700,            -- PEEP, PEEP set
    224697,                    -- Peak Insp Pressure
    224696,                    -- Plateau Pressure
    224685, 224687, 224684,    -- Tidal Volume
    220210,                    -- Respiratory Rate
    -- Anthropometry (for BMI)
    226512,                    -- Admission Weight (kg)
    226707, 226730,            -- Height (Inches, cm)
    -- New: vital signs
    220045,                    -- Heart Rate
    220277,                    -- O2 saturation pulseoxymetry (SpO2)
    220052,                    -- Arterial BP mean (invasive)
    220181,                    -- Non-invasive BP mean (fallback)
    223762,                    -- Temperature Celsius
    223761,                    -- Temperature Fahrenheit
    220739,                    -- GCS - Eye Opening
    223900,                    -- GCS - Verbal Response
    223901                     -- GCS - Motor Response
  )
  AND ce.valuenum IS NOT NULL
),

-- Daily aggregates per item (mean of day, with reasonable bounds)
daily_fio2 AS (
  SELECT stay_id, d, AVG(
    CASE WHEN valuenum > 1.0 THEN valuenum / 100.0 ELSE valuenum END
  ) AS fio2
  FROM ce_filtered
  WHERE itemid = 223835 AND valuenum BETWEEN 0.21 AND 100.0
  GROUP BY stay_id, d
),
daily_peep AS (
  SELECT stay_id, d, AVG(valuenum) AS peep
  FROM ce_filtered
  WHERE itemid IN (220339, 224700) AND valuenum BETWEEN 0 AND 30
  GROUP BY stay_id, d
),
daily_ppeak AS (
  SELECT stay_id, d, AVG(valuenum) AS ppeak
  FROM ce_filtered
  WHERE itemid = 224697 AND valuenum BETWEEN 5 AND 80
  GROUP BY stay_id, d
),
daily_pplat AS (
  SELECT stay_id, d, AVG(valuenum) AS pplat
  FROM ce_filtered
  WHERE itemid = 224696 AND valuenum BETWEEN 5 AND 80
  GROUP BY stay_id, d
),
daily_tv AS (
  SELECT stay_id, d, AVG(valuenum) AS tidvol_obs
  FROM ce_filtered
  WHERE itemid IN (224685, 224687, 224684) AND valuenum BETWEEN 50 AND 2000
  GROUP BY stay_id, d
),
daily_rr AS (
  SELECT stay_id, d, AVG(valuenum) AS rr
  FROM ce_filtered
  WHERE itemid = 220210 AND valuenum BETWEEN 1 AND 70
  GROUP BY stay_id, d
),

-- New vitals
daily_hr AS (
  SELECT stay_id, d, AVG(valuenum) AS heart_rate
  FROM ce_filtered
  WHERE itemid = 220045 AND valuenum BETWEEN 20 AND 250
  GROUP BY stay_id, d
),
daily_spo2 AS (
  SELECT stay_id, d, AVG(valuenum) AS spo2
  FROM ce_filtered
  WHERE itemid = 220277 AND valuenum BETWEEN 50 AND 100
  GROUP BY stay_id, d
),
daily_map AS (
  SELECT stay_id, d,
    AVG(CASE WHEN itemid = 220052 THEN valuenum
             WHEN itemid = 220181 THEN valuenum END) AS map_mmhg
  FROM ce_filtered
  WHERE itemid IN (220052, 220181) AND valuenum BETWEEN 30 AND 200
  GROUP BY stay_id, d
),
daily_temp AS (
  SELECT stay_id, d,
    AVG(CASE
      WHEN itemid = 223762 THEN valuenum
      WHEN itemid = 223761 THEN (valuenum - 32) * 5 / 9
    END) AS temperature_c
  FROM ce_filtered
  WHERE (itemid = 223762 AND valuenum BETWEEN 30 AND 45)
     OR (itemid = 223761 AND valuenum BETWEEN 86 AND 113)
  GROUP BY stay_id, d
),
daily_gcs AS (
  SELECT stay_id, d,
    AVG(CASE WHEN itemid = 220739 THEN valuenum END) +
    AVG(CASE WHEN itemid = 223900 THEN valuenum END) +
    AVG(CASE WHEN itemid = 223901 THEN valuenum END) AS gcs_total
  FROM ce_filtered
  WHERE itemid IN (220739, 223900, 223901) AND valuenum BETWEEN 1 AND 6
  GROUP BY stay_id, d
),

-- Lab values
lab_filtered AS (
  SELECT le.subject_id, le.itemid, le.charttime,
         CAST(le.valuenum AS FLOAT64) AS valuenum,
         DATE(le.charttime) AS d
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` le
  JOIN adults a USING (subject_id)
  WHERE le.itemid IN (50821, 50813, 50818, 50820, 51222, 50912)
    AND le.valuenum IS NOT NULL
    AND le.charttime BETWEEN a.intime AND a.outtime
),
daily_pao2 AS (
  SELECT a.stay_id, l.d, AVG(l.valuenum) AS pao2
  FROM lab_filtered l JOIN adults a USING (subject_id)
  WHERE l.itemid = 50821 AND l.valuenum BETWEEN 20 AND 800
    AND l.charttime BETWEEN a.intime AND a.outtime
  GROUP BY a.stay_id, l.d
),
daily_lactate AS (
  SELECT a.stay_id, l.d, AVG(l.valuenum) AS lactate
  FROM lab_filtered l JOIN adults a USING (subject_id)
  WHERE l.itemid = 50813 AND l.valuenum BETWEEN 0 AND 30
    AND l.charttime BETWEEN a.intime AND a.outtime
  GROUP BY a.stay_id, l.d
),
daily_paco2 AS (
  SELECT a.stay_id, l.d, AVG(l.valuenum) AS paco2
  FROM lab_filtered l JOIN adults a USING (subject_id)
  WHERE l.itemid = 50818 AND l.valuenum BETWEEN 10 AND 150
    AND l.charttime BETWEEN a.intime AND a.outtime
  GROUP BY a.stay_id, l.d
),
daily_ph AS (
  SELECT a.stay_id, l.d, AVG(l.valuenum) AS ph_arterial
  FROM lab_filtered l JOIN adults a USING (subject_id)
  WHERE l.itemid = 50820 AND l.valuenum BETWEEN 6.5 AND 8.0
    AND l.charttime BETWEEN a.intime AND a.outtime
  GROUP BY a.stay_id, l.d
),
daily_hb AS (
  SELECT a.stay_id, l.d, AVG(l.valuenum) AS hemoglobin
  FROM lab_filtered l JOIN adults a USING (subject_id)
  WHERE l.itemid = 51222 AND l.valuenum BETWEEN 3 AND 25
    AND l.charttime BETWEEN a.intime AND a.outtime
  GROUP BY a.stay_id, l.d
),
daily_creat AS (
  SELECT a.stay_id, l.d, AVG(l.valuenum) AS creatinine
  FROM lab_filtered l JOIN adults a USING (subject_id)
  WHERE l.itemid = 50912 AND l.valuenum BETWEEN 0.1 AND 25
    AND l.charttime BETWEEN a.intime AND a.outtime
  GROUP BY a.stay_id, l.d
),

-- Days where Berlin is evaluable (PaO2 + FiO2 + PEEP simultaneously measured)
days_with_berlin_metrics AS (
  SELECT pa.stay_id, pa.d, pa.pao2, f.fio2, p.peep
  FROM daily_pao2 pa
  JOIN daily_fio2 f USING (stay_id, d)
  JOIN daily_peep p USING (stay_id, d)
),

-- First observed Berlin-evaluable day per stay
first_obs_day AS (
  SELECT
    stay_id,
    MIN(d) AS first_obs_date,
    ARRAY_AGG(STRUCT(d, pao2, fio2, peep) ORDER BY d LIMIT 1)[OFFSET(0)] AS first_row
  FROM days_with_berlin_metrics
  GROUP BY stay_id
),

-- Cohort: first observed Berlin day satisfies P/F<=300 AND PEEP>=5
cohort AS (
  SELECT
    a.stay_id, a.subject_id, a.hadm_id,
    a.intime, a.outtime, a.los, a.gender, a.anchor_age, a.dod,
    fod.first_obs_date,
    SAFE_DIVIDE(fod.first_row.pao2, fod.first_row.fio2) AS first_day_pf,
    fod.first_row.peep AS first_day_peep
  FROM adults a
  JOIN first_obs_day fod USING (stay_id)
  WHERE SAFE_DIVIDE(fod.first_row.pao2, fod.first_row.fio2) <= 300
    AND fod.first_row.peep >= 5
),

-- A1 FIX: 30-day window aligned to first_obs_date (NOT calendar intime).
-- Pre-Berlin period is left-truncated (not generated).
all_stay_days AS (
  SELECT c.stay_id, dd AS d
  FROM cohort c, UNNEST(GENERATE_DATE_ARRAY(
    c.first_obs_date,
    DATE_ADD(c.first_obs_date, INTERVAL 29 DAY)
  )) AS dd
),

-- Static features
weight_first AS (
  SELECT stay_id, ANY_VALUE(valuenum) AS weight_kg
  FROM ce_filtered
  WHERE itemid = 226512 AND valuenum BETWEEN 30 AND 250
  GROUP BY stay_id
),
height_cm_first AS (
  SELECT stay_id, ANY_VALUE(
    CASE WHEN itemid = 226707 THEN valuenum * 2.54
         WHEN itemid = 226730 THEN valuenum END
  ) AS height_cm
  FROM ce_filtered
  WHERE itemid IN (226707, 226730) AND valuenum BETWEEN 50 AND 300
  GROUP BY stay_id
),
bmi AS (
  SELECT
    w.stay_id,
    SAFE_DIVIDE(w.weight_kg, POW(h.height_cm / 100.0, 2)) AS bmi_imputed
  FROM weight_first w
  LEFT JOIN height_cm_first h USING (stay_id)
)

SELECT
  c.stay_id,
  c.subject_id,
  c.hadm_id,
  DATE_DIFF(asd.d, c.first_obs_date, DAY) AS day_num,        -- A1 FIX: day 0 = first Berlin day
  asd.d AS day_date,
  c.first_obs_date,                                          -- New: keep for transparency
  DATE_DIFF(c.first_obs_date, DATE(c.intime), DAY) AS pre_berlin_days,  -- New: lead-time covariate

  -- ===== MP components (DO NOT use as outcome covariates) =====
  f.fio2, p.peep, pk.ppeak, pl.pplat, r.rr, t.tidvol_obs,
  CASE
    WHEN pl.pplat IS NULL OR p.peep IS NULL THEN NULL
    WHEN (pl.pplat - p.peep) BETWEEN 1 AND 60 THEN pl.pplat - p.peep
    ELSE NULL
  END AS driving_pressure,
  CASE
    WHEN t.tidvol_obs IS NULL OR pl.pplat IS NULL OR p.peep IS NULL THEN NULL
    WHEN (pl.pplat - p.peep) <= 0 THEN NULL
    WHEN SAFE_DIVIDE(t.tidvol_obs, pl.pplat - p.peep) BETWEEN 1 AND 200
      THEN SAFE_DIVIDE(t.tidvol_obs, pl.pplat - p.peep)
    ELSE NULL
  END AS compliance,
  t.tidvol_obs / 1000.0 AS vt_liters,
  CASE
    WHEN r.rr IS NULL OR t.tidvol_obs IS NULL
      OR pk.ppeak IS NULL OR pl.pplat IS NULL OR p.peep IS NULL THEN NULL
    WHEN (0.098 * r.rr * (t.tidvol_obs / 1000.0)
          * (pk.ppeak - 0.5 * (pl.pplat - p.peep))) BETWEEN 0.1 AND 80
      THEN 0.098 * r.rr * (t.tidvol_obs / 1000.0)
           * (pk.ppeak - 0.5 * (pl.pplat - p.peep))
    ELSE NULL
  END AS mp_j_min,

  -- ===== Outcome-eligible time-varying covariates (NOT in MP) =====
  pa.pao2,
  CASE
    WHEN pa.pao2 IS NULL OR f.fio2 IS NULL OR f.fio2 <= 0 THEN NULL
    WHEN SAFE_DIVIDE(pa.pao2, f.fio2) BETWEEN 1 AND 1000 THEN SAFE_DIVIDE(pa.pao2, f.fio2)
    ELSE NULL
  END AS pf_ratio,
  lac.lactate,
  hr.heart_rate,
  spo.spo2,
  m.map_mmhg,
  tmp.temperature_c,
  gcs.gcs_total,
  paco.paco2,
  ph.ph_arterial,
  hb.hemoglobin,
  cr.creatinine,

  -- ===== Static =====
  c.anchor_age,
  CASE WHEN c.gender = 'M' THEN 1.0 ELSE 0.0 END AS gender_M,
  c.gender,
  bm.bmi_imputed,
  c.first_day_pf,
  CASE
    WHEN c.first_day_pf < 100 THEN 'severe'
    WHEN c.first_day_pf < 200 THEN 'moderate'
    ELSE 'mild'
  END AS severity,
  c.los,
  c.dod,

  -- ===== Outcome =====
  CASE WHEN c.dod IS NOT NULL AND asd.d = DATE(c.dod) THEN 1 ELSE 0 END AS death_event,
  CASE
    WHEN c.dod IS NOT NULL
     AND DATE_DIFF(DATE(c.dod), c.first_obs_date, DAY) <= 30
    THEN 1 ELSE 0
  END AS mortality_30d

FROM all_stay_days asd
JOIN cohort c ON c.stay_id = asd.stay_id
LEFT JOIN daily_fio2  f   ON f.stay_id  = asd.stay_id AND f.d  = asd.d
LEFT JOIN daily_peep  p   ON p.stay_id  = asd.stay_id AND p.d  = asd.d
LEFT JOIN daily_ppeak pk  ON pk.stay_id = asd.stay_id AND pk.d = asd.d
LEFT JOIN daily_pplat pl  ON pl.stay_id = asd.stay_id AND pl.d = asd.d
LEFT JOIN daily_tv    t   ON t.stay_id  = asd.stay_id AND t.d  = asd.d
LEFT JOIN daily_rr    r   ON r.stay_id  = asd.stay_id AND r.d  = asd.d
LEFT JOIN daily_pao2  pa  ON pa.stay_id = asd.stay_id AND pa.d = asd.d
LEFT JOIN daily_lactate lac ON lac.stay_id = asd.stay_id AND lac.d = asd.d
LEFT JOIN daily_hr    hr  ON hr.stay_id  = asd.stay_id AND hr.d  = asd.d
LEFT JOIN daily_spo2  spo ON spo.stay_id = asd.stay_id AND spo.d = asd.d
LEFT JOIN daily_map   m   ON m.stay_id   = asd.stay_id AND m.d   = asd.d
LEFT JOIN daily_temp  tmp ON tmp.stay_id = asd.stay_id AND tmp.d = asd.d
LEFT JOIN daily_gcs   gcs ON gcs.stay_id = asd.stay_id AND gcs.d = asd.d
LEFT JOIN daily_paco2 paco ON paco.stay_id = asd.stay_id AND paco.d = asd.d
LEFT JOIN daily_ph    ph  ON ph.stay_id  = asd.stay_id AND ph.d  = asd.d
LEFT JOIN daily_hb    hb  ON hb.stay_id  = asd.stay_id AND hb.d  = asd.d
LEFT JOIN daily_creat cr  ON cr.stay_id  = asd.stay_id AND cr.d  = asd.d
LEFT JOIN bmi bm ON bm.stay_id = c.stay_id
ORDER BY c.stay_id, day_num
