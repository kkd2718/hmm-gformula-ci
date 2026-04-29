"""
table_one.py (Complete Version)
-------------------------------
Generates Table 1 with ALL requested variables:
- Demographics: Age, Sex, BMI
- Comorbidities: CCI Total + Specific Conditions (MI, CHF, COPD, etc.)
- Respiratory: P/F, PEEP, Compliance, RR, Vt, Driving Pressure
- Mechanical Power: Intensity (J/min) AND Daily Load (J/day)
- Labs: Lactate
- Outcome: Mortality

Stratified by ARDS Severity (Mild / Moderate / Severe).
"""

import pandas as pd
import numpy as np
import os
from scipy import stats

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(CURRENT_DIR, 'processed_data', 'final_cohort', 'ards_standard_cohort.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_table1():
    print("🚀 Generating Table 1 (Comprehensive)...")
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    
    # 2. Preprocessing
    # Baseline only (Day 0)
    baseline = df[df['day_num'] == 0].copy()
    
    # [Derived Variable] Daily MP (Joules)
    baseline['mp_daily_joule'] = baseline['mp_j_min'] * 1440.0
    
    # Severity Grouping
    def get_sev(pf):
        if pd.isna(pf): return np.nan
        if pf <= 100: return 'Severe'
        elif pf <= 200: return 'Moderate'
        elif pf <= 300: return 'Mild'
        return np.nan
    
    baseline['ards_severity'] = baseline['pf_ratio'].apply(get_sev)
    baseline = baseline.dropna(subset=['ards_severity'])
    
    print(f"   Analytic Cohort N = {len(baseline):,}")
    
    # 3. Variable Definitions
    # (Label, Column Name, Is_Categorical)
    vars_config = [
        # Demographics
        ("[Demographics]", None, None),
        ("Age (years)", 'anchor_age', False),
        ("Male Sex", 'is_male', True),
        ("BMI (kg/m2)", 'bmi_imputed', False),
        
        # Clinical Scores
        ("[Comorbidities]", None, None),
        ("Charlson Index (Total)", 'charlson_index', False),
        ("Myocardial Infarction", 'cci_mi', True),
        ("Congestive Heart Failure", 'cci_chf', True),
        ("COPD", 'cci_copd', True),
        ("Diabetes", 'cci_diabetes', True),
        ("Renal Disease", 'cci_renal', True),
        ("Malignancy", 'cci_malignancy', True),
        
        # Respiratory
        ("[Respiratory Parameters]", None, None),
        ("P/F Ratio", 'pf_ratio', False),
        ("PEEP (cmH2O)", 'peep', False),
        ("Driving Pressure (cmH2O)", 'driving_pressure', False),
        ("Compliance (mL/cmH2O)", 'compliance', False),
        ("Respiratory Rate (/min)", 'rr', False),
        ("Tidal Volume (L)", 'vt_liters', False),
        
        # Mechanical Power
        ("[Mechanical Power]", None, None),
        ("Intensity (J/min)", 'mp_j_min', False),
        ("Daily Load (J/day)", 'mp_daily_joule', False),
        
        # Labs
        ("[Labs]", None, None),
        ("Lactate (mmol/L)", 'lactate', False),
        
        # Outcome
        ("[Outcome]", None, None),
        ("Hospital Mortality", 'hospital_expire_flag', True),
    ]
    
    # 4. Helper Functions
    groups = ['Mild', 'Moderate', 'Severe']
    
    def get_stats(data, col, is_cat):
        if col is None: return "" # Header row
        if is_cat:
            n = data[col].sum()
            return f"{int(n)} ({n/len(data)*100:.1f}%)"
        else:
            mean = data[col].mean()
            std = data[col].std()
            return f"{mean:.1f} ({std:.1f})"
            
    def get_p_value(data, col, is_cat):
        if col is None: return ""
        
        g_data = [data[data['ards_severity']==g][col].dropna() for g in groups]
        
        if is_cat:
            # Chi-square
            ct = pd.crosstab(data['ards_severity'], data[col])
            if ct.size == 0: return "NA"
            _, p, _, _ = stats.chi2_contingency(ct)
        else:
            # ANOVA
            if any(len(g) < 2 for g in g_data): return "NA"
            _, p = stats.f_oneway(*g_data)
            
        return f"{p:.3f}" if p >= 0.001 else "<0.001"

    # 5. Build Table
    rows = []
    headers = ["Variable", "Total", "Mild", "Moderate", "Severe", "P-value"]
    
    for label, col, is_cat in vars_config:
        row = {'Variable': label}
        
        if col is None: # Section Header
            row['Total'] = ""
            row['Mild'] = ""
            row['Moderate'] = ""
            row['Severe'] = ""
            row['P-value'] = ""
        else:
            row['Total'] = get_stats(baseline, col, is_cat)
            for g in groups:
                row[g] = get_stats(baseline[baseline['ards_severity']==g], col, is_cat)
            row['P-value'] = get_p_value(baseline, col, is_cat)
            
        rows.append(row)
        
    # 6. Save
    res_df = pd.DataFrame(rows, columns=headers)
    
    # Print formatted
    print("\n" + "="*80)
    print(res_df.to_string(index=False))
    print("="*80)
    
    save_path = os.path.join(OUTPUT_DIR, "Table1_Complete.csv")
    res_df.to_csv(save_path, index=False)
    print(f"\n✅ Table 1 Saved to: {save_path}")

if __name__ == "__main__":
    generate_table1()