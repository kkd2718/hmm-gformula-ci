"""
mimic-extract.py (v4.0 — MP Formula Fix)
-----------------------------------------
Key Updates (v3 → v4):
1. FiO2 Logic: Convert % to fraction. If invalid (>1.0 or <0.21) after fix -> NaN.
2. Berlin Definition: Strictly requires PEEP >= 5 cmH2O for Severity classification.
3. Outlier Filters: Caps extreme values for Vt, RR, Ppeak to reduce noise.
4. CCI & BMI: Robust extraction logic included.
5. [NEW] MP Formula: Gattinoni Simplified (2016) as primary.
6. [NEW] Becher Simplified (2019) as sensitivity column.
7. [NEW] Driving Pressure calculated BEFORE MP (dependency order fix).
8. [NEW] MP Coverage Report for Methods section documentation.

References:
  [1] Gattinoni L, et al. Ventilator-related causes of lung injury:
      the mechanical power. Intensive Care Med. 2016;42(10):1567-1575.
  [2] Becher T, et al. Calculation of mechanical power for pressure-
      controlled ventilation. Intensive Care Med. 2019;45(9):1321-1323.
"""

import pandas as pd
import numpy as np
import os
import gc

# =============================================================================
# 1. 환경 설정
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = 'mimic_directory'
HOSP_DIR = os.path.join(ROOT_DIR, 'hosp')
ICU_DIR = os.path.join(ROOT_DIR, 'icu')
OUTPUT_DIR = os.path.join(STUDY_DIR, 'processed_data', 'final_cohort')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ITEMS = {
    'vent': {
        224685: 'tidvol_obs', # Tidal Volume (Observed)
        224695: 'ppeak',      # Peak Insp. Pressure
        224696: 'pplat',      # Plateau Pressure
        220339: 'peep',       # PEEP
        220210: 'rr',         # Respiratory Rate
        223835: 'fio2'        # FiO2
    },
    'lab': { 
        50821: 'pao2',
        50813: 'lactate' 
    }
}

# ICD-10 Mapping for CCI
CCI_MAPPING = {
    'mi':           (['410', '412'], ['I21', 'I22', 'I252']),
    'chf':          (['428'], ['I50', 'I110', 'I27', 'I42']),
    'pvd':          (['4439', '441', '7854'], ['I70', 'I71', 'I73']),
    'cva':          (['430', '431', '432', '433', '434'], ['I60', 'I61', 'I62', 'I63', 'I64']),
    'dementia':     (['290'], ['F00', 'F01', 'F02', 'F03', 'G30']),
    'copd':         (['490', '491', '492', '500', '501'], ['J40', 'J41', 'J42', 'J43', 'J44']),
    'rheum':        (['7100', '7101', '7104', '7140', '7141', '7142', '725'], ['M05', 'M06', 'M315', 'M32', 'M33', 'M34']),
    'liver_mild':   (['5712', '5714', '5715', '5716'], ['B18', 'K70', 'K71', 'K73', 'K74']),
    'diabetes':     (['2500', '2501', '2502', '2503'], ['E10', 'E11', 'E12', 'E13', 'E14']),
    'renal':        (['582', '583', '585', '586', '588'], ['N03', 'N05', 'N18', 'N19']),
    'malignancy':   (['140', '141', '142', '143', '144', '145', '146'], ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06']),
}

def map_cci(df_diag):
    """Calculate Charlson Comorbidity Index from ICD codes."""
    df_diag['icd_code'] = df_diag['icd_code'].astype(str)
    scores = pd.DataFrame(index=df_diag['hadm_id'].unique())
    
    for cond, (icd9, icd10) in CCI_MAPPING.items():
        mask9 = df_diag[df_diag['icd_version']==9]['icd_code'].str.startswith(tuple(icd9))
        mask10 = df_diag[df_diag['icd_version']==10]['icd_code'].str.startswith(tuple(icd10))
        
        ids9 = df_diag[df_diag['icd_version']==9][mask9]['hadm_id'].unique()
        ids10 = df_diag[df_diag['icd_version']==10][mask10]['hadm_id'].unique()
        ids = np.union1d(ids9, ids10)
        
        scores[f'cci_{cond}'] = 0
        scores.loc[scores.index.isin(ids), f'cci_{cond}'] = 1
        
    scores['charlson_index'] = scores.sum(axis=1)
    return scores.reset_index().rename(columns={'index': 'hadm_id'})


def run_extraction():
    print("🚀 [Start] MIMIC-IV Extraction (v4.0 — Gattinoni MP)...")
    
    # =========================================================================
    # 1. Load Cohort
    # =========================================================================
    print("1️⃣ Loading Cohort...")
    patients = pd.read_csv(
        os.path.join(HOSP_DIR, 'patients.csv.gz'),
        usecols=['subject_id', 'gender', 'anchor_age']
    )
    admissions = pd.read_csv(
        os.path.join(HOSP_DIR, 'admissions.csv.gz'),
        usecols=['subject_id', 'hadm_id', 'race', 'admission_type', 'hospital_expire_flag']
    )
    icustays = pd.read_csv(os.path.join(ICU_DIR, 'icustays.csv.gz'))
    
    cohort = icustays.merge(patients, on='subject_id').merge(admissions, on=['subject_id', 'hadm_id'])
    cohort['is_male'] = (cohort['gender'] == 'M').astype(int)
    cohort['race_group'] = cohort['race'].apply(
        lambda x: 'White' if 'WHITE' in str(x) else ('Black' if 'BLACK' in str(x) else 'Other')
    )
    cohort = cohort[(cohort['anchor_age'] >= 18) & (cohort['los'] >= 1)]
    
    # =========================================================================
    # 2. CCI Calculation
    # =========================================================================
    print("2️⃣ Calculating CCI...")
    if os.path.exists(os.path.join(HOSP_DIR, 'diagnoses_icd.csv.gz')):
        diag = pd.read_csv(os.path.join(HOSP_DIR, 'diagnoses_icd.csv.gz'))
        diag = diag[diag['hadm_id'].isin(cohort['hadm_id'])]
        cci_df = map_cci(diag)
        cohort = cohort.merge(cci_df, on='hadm_id', how='left')
        cohort['charlson_index'] = cohort['charlson_index'].fillna(0)
    else:
        cohort['charlson_index'] = 0

    # =========================================================================
    # 3. BMI (Placeholder — can be expanded to 'omr' table)
    # =========================================================================
    print("3️⃣ Extracting BMI from OMR with outlier removal...")
    omr_path = os.path.join(HOSP_DIR, 'omr.csv.gz')
    if os.path.exists(omr_path):
        omr = pd.read_csv(omr_path, usecols=['subject_id', 'result_name', 'result_value'])
        bmi_data = omr[omr['result_name'] == 'BMI (kg/m2)'].copy()        
        bmi_data['bmi'] = pd.to_numeric(bmi_data['result_value'], errors='coerce')
        bmi_data = bmi_data[(bmi_data['bmi'] >= 10.0) & (bmi_data['bmi'] <= 100.0)]
        patient_bmi = bmi_data.groupby('subject_id')['bmi'].median().reset_index()
        
        cohort = cohort.merge(patient_bmi, on='subject_id', how='left')
        
        median_bmi = cohort['bmi'].median()
        cohort['bmi_imputed'] = cohort['bmi'].fillna(median_bmi)
        
        print(f"   [BMI Stats] Valid mean: {cohort['bmi_imputed'].mean():.1f}, std: {cohort['bmi_imputed'].std():.1f}")
    else:
        print("⚠️ omr.csv.gz not found. Imputing BMI with random normal distribution.")
        cohort['bmi_imputed'] = np.random.normal(25.0, 3.0, size=len(cohort))

    # =========================================================================
    # 4. Extract Ventilation Data
    # =========================================================================
    print("3️⃣ Extracting Ventilation...")
    vent_data = []
    chunk_size = 10**6
    
    with pd.read_csv(os.path.join(ICU_DIR, 'chartevents.csv.gz'), chunksize=chunk_size, 
                     usecols=['stay_id', 'charttime', 'itemid', 'valuenum']) as reader:
        for chunk in reader:
            subset = chunk[chunk['itemid'].isin(ITEMS['vent'].keys())]
            if not subset.empty:
                vent_data.append(subset)
            
    if not vent_data:
        print("❌ No ventilation data found. Exiting.")
        return
    
    vent = pd.concat(vent_data)
    vent['item_name'] = vent['itemid'].map(ITEMS['vent'])
    vent['day_date'] = pd.to_datetime(vent['charttime']).dt.date
    
    # --- Strict Outlier Filtering before Aggregation ---
    vent = vent[vent['valuenum'] > 0]
    
    # Tidal Volume: > 2000 mL (2.0L) → likely charting error
    vent.loc[(vent['item_name'] == 'tidvol_obs') & (vent['valuenum'] > 2000), 'valuenum'] = np.nan
    # RR: > 70 → artifact
    vent.loc[(vent['item_name'] == 'rr') & (vent['valuenum'] > 70), 'valuenum'] = np.nan
    # PEEP: > 40 → extreme, likely error
    vent.loc[(vent['item_name'] == 'peep') & (vent['valuenum'] > 40), 'valuenum'] = np.nan
    # Pressures: > 100 cmH2O → impossible
    vent.loc[(vent['item_name'].isin(['ppeak', 'pplat'])) & (vent['valuenum'] > 100), 'valuenum'] = np.nan
    
    # Daily aggregation (mean of all measurements per day)
    vent_daily = vent.pivot_table(
        index=['stay_id', 'day_date'], 
        columns='item_name', values='valuenum', aggfunc='mean'
    ).reset_index()
    
    del vent, vent_data
    gc.collect()

    # =========================================================================
    # 5. Extract Lab Data
    # =========================================================================
    print("4️⃣ Extracting Lab...")
    lab_data = []
    with pd.read_csv(os.path.join(HOSP_DIR, 'labevents.csv.gz'), chunksize=chunk_size,
                     usecols=['subject_id', 'charttime', 'itemid', 'valuenum']) as reader:
        for chunk in reader:
            subset = chunk[chunk['itemid'].isin(ITEMS['lab'].keys())]
            if not subset.empty:
                lab_data.append(subset)
            
    if lab_data:
        lab = pd.concat(lab_data)
        lab['item_name'] = lab['itemid'].map(ITEMS['lab'])
        lab['day_date'] = pd.to_datetime(lab['charttime']).dt.date
        
        cohort_win = cohort[['subject_id', 'stay_id', 'intime', 'outtime']].copy()
        cohort_win['intime'] = pd.to_datetime(cohort_win['intime']).dt.date
        cohort_win['outtime'] = pd.to_datetime(cohort_win['outtime']).dt.date
        
        lab_merged = pd.merge(lab, cohort_win, on='subject_id')
        lab_merged = lab_merged[
            (lab_merged['day_date'] >= lab_merged['intime']) & 
            (lab_merged['day_date'] <= lab_merged['outtime'])
        ]
        
        lab_daily = lab_merged.pivot_table(
            index=['stay_id', 'day_date'], 
            columns='item_name', values='valuenum', aggfunc='mean'
        ).reset_index()
    else:
        lab_daily = pd.DataFrame(columns=['stay_id', 'day_date', 'pao2', 'lactate'])

    # =========================================================================
    # 6. Merge & Calculate Clinical Variables
    # =========================================================================
    print("5️⃣ Calculating Metrics (Gattinoni MP + Berlin)...")
    daily_full = pd.merge(vent_daily, lab_daily, on=['stay_id', 'day_date'], how='left')
    
    # --- FiO2 Logic Correction ---
    if 'fio2' in daily_full.columns:
        # Values > 1.0 are likely recorded as percentage → convert to fraction
        daily_full.loc[daily_full['fio2'] > 1.0, 'fio2'] = daily_full['fio2'] / 100.0
        # Validity range: 0.21 (room air) to 1.0
        daily_full.loc[
            (daily_full['fio2'] < 0.21) | (daily_full['fio2'] > 1.0), 'fio2'
        ] = np.nan
        
    # --- P/F Ratio ---
    daily_full['pf_ratio'] = daily_full['pao2'] / daily_full['fio2']
    
    # --- Unit Conversion: Tidal Volume mL → L ---
    daily_full['vt_liters'] = daily_full['tidvol_obs'] / 1000.0
    
    # --- Driving Pressure (MUST be calculated BEFORE MP) ---
    # ΔP = Pplat - PEEP
    daily_full['driving_pressure'] = daily_full['pplat'] - daily_full['peep']
    daily_full.loc[daily_full['driving_pressure'] <= 1, 'driving_pressure'] = np.nan
    
    # --- Compliance ---
    daily_full['compliance'] = daily_full['tidvol_obs'] / daily_full['driving_pressure']
    
    # --- Ppeak Imputation ---
    # When Ppeak is missing but Pplat is available:
    # Ppeak ≈ Pplat + resistive pressure drop
    # Assumed: Rrs ≈ 5 cmH2O/L/s, typical inspiratory flow ~0.5 L/s → ~2.5 cmH2O
    # Conservative estimate: +2 cmH2O (Akoumianaki et al., Ann Intensive Care, 2017)
    daily_full['ppeak_imputed'] = daily_full['ppeak'].fillna(daily_full['pplat'] + 2)

    # =====================================================================
    # PRIMARY: Gattinoni Simplified MP (Intensive Care Med, 2016)
    # MP = 0.098 × RR × Vt × (Ppeak − 0.5 × ΔP)
    #
    # Requires: RR, Vt, Ppeak (or imputed), Pplat, PEEP
    # Note: When Pplat is missing, driving_pressure is NaN → MP is NaN
    #       This is intentional (Gattinoni formula requires ΔP)
    # =====================================================================
    daily_full['mp_gattinoni'] = (
        0.098 
        * daily_full['rr'] 
        * daily_full['vt_liters'] 
        * (daily_full['ppeak_imputed'] - 0.5 * daily_full['driving_pressure'])
    )
    
    # Validity checks
    daily_full.loc[daily_full['mp_gattinoni'] <= 0, 'mp_gattinoni'] = np.nan
    daily_full.loc[daily_full['mp_gattinoni'] > 100, 'mp_gattinoni'] = np.nan

    # =====================================================================
    # SENSITIVITY: Becher Simplified MP (Intensive Care Med, 2019)
    # MP_surrogate = 0.098 × RR × Vt × Ppeak
    #
    # Advantage: Does NOT require Pplat → larger analytic sample
    # Use for sensitivity analysis to show robustness to formula choice
    # =====================================================================
    daily_full['mp_becher'] = (
        0.098 
        * daily_full['rr'] 
        * daily_full['vt_liters'] 
        * daily_full['ppeak_imputed']
    )
    
    # Validity checks
    daily_full.loc[daily_full['mp_becher'] <= 0, 'mp_becher'] = np.nan
    daily_full.loc[daily_full['mp_becher'] > 100, 'mp_becher'] = np.nan

    # =====================================================================
    # PRIMARY ANALYSIS: Use Gattinoni
    # To run sensitivity analysis with Becher, change this line to:
    #   daily_full['mp_j_min'] = daily_full['mp_becher']
    # =====================================================================
    daily_full['mp_j_min'] = daily_full['mp_gattinoni']

    # --- MP Coverage Report ---
    n_total = len(daily_full)
    n_pplat = daily_full['pplat'].notna().sum()
    n_gatt = daily_full['mp_gattinoni'].notna().sum()
    n_bech = daily_full['mp_becher'].notna().sum()
    
    print(f"\n   📊 MP Formula Coverage Report:")
    print(f"      Total daily observations:     {n_total:,}")
    print(f"      Pplat available:              {n_pplat:,} ({n_pplat/n_total*100:.1f}%)")
    print(f"      Gattinoni MP calculable:      {n_gatt:,} ({n_gatt/n_total*100:.1f}%)")
    print(f"      Becher MP calculable:         {n_bech:,} ({n_bech/n_total*100:.1f}%)")
    print(f"      Lost by choosing Gattinoni:   {n_bech - n_gatt:,} observations\n")

    # =========================================================================
    # 7. Berlin Definition Filtering
    # =========================================================================
    # Criteria: P/F ≤ 300 AND PEEP ≥ 5 cmH2O
    daily_full = daily_full.dropna(subset=['pf_ratio', 'peep'])
    ards_mask = (daily_full['pf_ratio'] <= 300) & (daily_full['peep'] >= 5)
    
    ards_stays = daily_full[ards_mask]['stay_id'].unique()
    final_daily = daily_full[daily_full['stay_id'].isin(ards_stays)].copy()
    
    # Sort & Limit to 30 days
    final_daily = final_daily.sort_values(['stay_id', 'day_date'])
    final_daily['day_num'] = final_daily.groupby('stay_id').cumcount()
    final_daily = final_daily[final_daily['day_num'] <= 30]
    
    # =========================================================================
    # 8. Merge Static Variables & Save
    # =========================================================================
    static_cols = [
        'stay_id', 'subject_id', 'anchor_age', 'is_male', 'race_group',
        'bmi_imputed', 'admission_type', 'charlson_index', 'hospital_expire_flag'
    ]
    # Add individual CCI component columns if they exist
    cci_cols = [c for c in cohort.columns if 'cci_' in c]
    static_cols.extend(cci_cols)
    static_cols = list(set(static_cols))
    
    final_cohort = pd.merge(final_daily, cohort[static_cols], on='stay_id', how='inner')
    
    # --- Final Summary ---
    n_patients = final_cohort['stay_id'].nunique()
    n_with_mp = final_cohort['mp_j_min'].notna().sum()
    n_rows = len(final_cohort)
    
    save_path = os.path.join(OUTPUT_DIR, 'ards_standard_cohort.csv')
    final_cohort.to_csv(save_path, index=False)
    
    print(f"✅ Extraction Complete.")
    print(f"   Output:          {save_path}")
    print(f"   ARDS Patients:   {n_patients:,} (Strict Berlin)")
    print(f"   Total Rows:      {n_rows:,}")
    print(f"   Rows with MP:    {n_with_mp:,} ({n_with_mp/n_rows*100:.1f}%)")
    print(f"   MP Formula:      Gattinoni Simplified (ICM, 2016)")
    print(f"   Sensitivity:     mp_becher column available for Becher (ICM, 2019)")


if __name__ == "__main__":
    run_extraction()