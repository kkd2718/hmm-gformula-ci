"""Merge Charlson into v3 ARDS extraction."""
import sys, os
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

OUT = r"C:\Users\기덕\Desktop\Study\YMC_MPH\MR_causal\MIMIC-IV\3.1\cohort_v31"

df = pd.read_csv(os.path.join(OUT, "ards_v31_v3.csv"))
cci = pd.read_csv(os.path.join(OUT, "charlson_v31.csv"))
print(f"ARDS v3: {len(df):,} rows / {df['stay_id'].nunique():,} stays")

df_m = df.merge(cci, on=["subject_id", "hadm_id"], how="left")
weights = {
    "cci_mi": 1, "cci_chf": 1, "cci_pvd": 1, "cci_cvd": 1, "cci_dementia": 1,
    "cci_cpd": 1, "cci_rheum": 1, "cci_pud": 1, "cci_liver_mild": 1,
    "cci_liver_severe": 3, "cci_dm": 1, "cci_dm_complications": 2,
    "cci_paraplegia": 2, "cci_renal": 2, "cci_cancer": 2,
    "cci_metastatic": 6, "cci_aids": 6,
}
for c in weights:
    df_m[c] = df_m[c].fillna(0).astype(int)
df_m["charlson_index"] = sum(df_m[c] * w for c, w in weights.items())

out_path = os.path.join(OUT, "ards_v31_v3_with_cci.csv")
df_m.to_csv(out_path, index=False)
print(f"Merged: {len(df_m):,} rows, {os.path.getsize(out_path)/1e6:.1f} MB")
print(f"Charlson mean (per-stay): {df_m.drop_duplicates('stay_id')['charlson_index'].mean():.2f}")
