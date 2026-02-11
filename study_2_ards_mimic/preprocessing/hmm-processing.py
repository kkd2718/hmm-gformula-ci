"""
hmm-processing.py
-----------------
Converts Validated Cohort CSV into PyTorch Tensors.
Inputs: ./processed_data/final_cohort/ards_standard_cohort.csv
Outputs: ./processed_data/hmm_tensor/mimic_hmm_tensor.pt

Features:
- S: Log Mechanical Power Intensity (J/min)
- L: [P/F Ratio, Log Compliance, Lactate] (Dynamic)
- C: Static Covariates (Age, Sex, CCI, etc.)
- Y: Mortality Outcome
"""

import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler
import joblib

# =============================================================================
# 설정
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(STUDY_DIR, 'processed_data', 'final_cohort', 'ards_standard_cohort.csv')
OUTPUT_DIR = os.path.join(STUDY_DIR, 'processed_data', 'hmm_tensor')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SEQ_LEN = 30  # Max Sequence Length (Days)

def extract_hmm_tensor():
    print("🚀 [Start] HMM Tensor Processing...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Data file not found: {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    
    # Sort by Patient and Time
    if 'day_num' in df.columns:
        df = df.sort_values(['stay_id', 'day_num'])
    
    print(f"   Input Rows: {len(df):,}")
    print(f"   Input MP Mean: {df['mp_j_min'].mean():.2f} J/min (Validated)")

    # -------------------------------------------------------------------------
    # 2. Data Cleaning & Imputation (Tensor Ready)
    # -------------------------------------------------------------------------
    print("2️⃣ Final Cleaning & Imputation...")
    
    # Mechanical Power (Already Validated in Extraction, just safety check)
    df.loc[~df['mp_j_min'].between(0, 100), 'mp_j_min'] = np.nan
    df['mp_j_min'] = df['mp_j_min'].fillna(df['mp_j_min'].median())
    
    # Compliance
    df.loc[~df['compliance'].between(1, 200), 'compliance'] = np.nan
    df['compliance'] = df['compliance'].fillna(df['compliance'].median())
    
    # P/F Ratio
    df.loc[~df['pf_ratio'].between(10, 1000), 'pf_ratio'] = np.nan
    df['pf_ratio'] = df['pf_ratio'].fillna(df['pf_ratio'].median())

    # Lactate
    if 'lactate' in df.columns:
        df.loc[~df['lactate'].between(0, 30), 'lactate'] = np.nan
        df['lactate'] = df['lactate'].fillna(df['lactate'].median())
    else:
        # Should not happen with new extraction, but fallback
        print("⚠️ Warning: Lactate missing, using zeros.")
        df['lactate'] = 0.0

    # -------------------------------------------------------------------------
    # 3. Valid Cohort Selection
    # -------------------------------------------------------------------------
    print("3️⃣ Selecting Valid Patients...")
    # We use all patients in the validated CSV as it's already filtered.
    valid_stay_ids = df['stay_id'].unique()
    print(f"   Total Patients: {len(valid_stay_ids):,}")
    
    # -------------------------------------------------------------------------
    # 4. Feature Engineering
    # -------------------------------------------------------------------------
    # Exposure (S): Log MP Intensity
    # Clip lower to 0.1 to avoid log(0)
    df['log_intensity'] = np.log(df['mp_j_min'].clip(lower=0.1))
    
    # Covariates (L): Log Compliance, P/F, Lactate
    df['log_compliance'] = np.log(df['compliance'].clip(lower=1.0))
    
    # Define Column Groups
    exposure_col = 'log_intensity'
    cov_cols = ['pf_ratio', 'log_compliance', 'lactate'] # Dynamic L
    
    # Static Variables (C)
    # Using 'charlson_index' (Total Score)
    static_cols = ['anchor_age', 'bmi_imputed', 'charlson_index', 'is_male']
    
    outcome_col = 'hospital_expire_flag'

    # -------------------------------------------------------------------------
    # 5. Scaling
    # -------------------------------------------------------------------------
    print("5️⃣ Scaling Features...")
    scaler_C = StandardScaler()
    scaler_L = StandardScaler()
    scaler_S = StandardScaler()
    
    # Static Data (One row per patient)
    base_df = df.groupby('stay_id').first().reset_index()
    
    # Fit & Transform
    C_scaled = scaler_C.fit_transform(base_df[static_cols].fillna(0))
    L_scaled = scaler_L.fit_transform(df[cov_cols].fillna(0))
    S_scaled = scaler_S.fit_transform(df[[exposure_col]].fillna(0))
    
    # Put back to DataFrame for indexing
    df_L = pd.DataFrame(L_scaled, columns=cov_cols, index=df.index)
    df_S = pd.DataFrame(S_scaled, columns=[exposure_col], index=df.index)
    
    # -------------------------------------------------------------------------
    # 6. Build Tensors
    # -------------------------------------------------------------------------
    print("6️⃣ Building PyTorch Tensors...")
    
    n_samples = len(valid_stay_ids)
    
    # Initialize Tensors
    # S: (N, T, 1)
    S_tensor = torch.zeros(n_samples, MAX_SEQ_LEN, 1)
    # L: (N, T, 3) -> [P/F, Comp, Lactate]
    L_tensor = torch.zeros(n_samples, MAX_SEQ_LEN, len(cov_cols))
    # C: (N, 4) -> Static
    C_tensor = torch.FloatTensor(C_scaled)
    # Y: (N, T, 1) -> Outcome
    Y_tensor = torch.zeros(n_samples, MAX_SEQ_LEN, 1)
    # Mask: (N, T)
    Mask_tensor = torch.zeros(n_samples, MAX_SEQ_LEN)
    
    # Optimized Grouping
    grouped = df.groupby('stay_id')
    stay_id_list = []
    
    for i, sid in enumerate(valid_stay_ids):
        if sid not in grouped.groups: continue
        
        idx = grouped.groups[sid]
        seq_len = min(len(idx), MAX_SEQ_LEN)
        curr_idx = idx[:seq_len]
        
        # S & L
        S_tensor[i, :seq_len, :] = torch.FloatTensor(df_S.loc[curr_idx].values)
        L_tensor[i, :seq_len, :] = torch.FloatTensor(df_L.loc[curr_idx].values)
        Mask_tensor[i, :seq_len] = 1.0
        
        # Outcome (Single event at end of sequence if died)
        is_dead = base_df.loc[base_df['stay_id']==sid, outcome_col].values[0]
        if is_dead == 1:
            Y_tensor[i, seq_len-1, 0] = 1.0
            
        stay_id_list.append(sid)

    # -------------------------------------------------------------------------
    # 7. Save
    # -------------------------------------------------------------------------
    data_dict = {
        'S': S_tensor,
        'L': L_tensor,
        'C': C_tensor,
        'Y': Y_tensor,
        'mask': Mask_tensor,
        'stay_ids': stay_id_list,
        'scalers': {'C': scaler_C, 'L': scaler_L, 'S': scaler_S}
    }
    
    save_path = os.path.join(OUTPUT_DIR, 'mimic_hmm_tensor.pt')
    scaler_path = os.path.join(OUTPUT_DIR, 'scalers.pkl')
    
    torch.save(data_dict, save_path)
    joblib.dump(data_dict['scalers'], scaler_path)
    
    print(f"✅ Processing Complete.")
    print(f"   Tensor Shape S: {S_tensor.shape}")
    print(f"   Tensor Shape L: {L_tensor.shape} (Includes Lactate)")
    print(f"   Saved to: {save_path}")

if __name__ == "__main__":
    extract_hmm_tensor()