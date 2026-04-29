"""Merge first-day SOFA into the v3 cohort to produce the v4 analysis cohort."""
import sys
import os
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

OUT = r"C:\Users\기덕\Desktop\Study\YMC_MPH\MR_causal\MIMIC-IV\3.1\cohort_v31"
df = pd.read_csv(os.path.join(OUT, "ards_v31_v3_with_cci.csv"))
sofa = pd.read_csv(os.path.join(OUT, "sofa_v31.csv"))
print(f"v3 cohort: {len(df):,} rows / {df['stay_id'].nunique():,} stays")
print(f"SOFA table: {len(sofa):,} stays")

merged = df.merge(sofa, on="stay_id", how="left")
out_path = os.path.join(OUT, "ards_v31_v4.csv")
merged.to_csv(out_path, index=False)

per_stay = merged.drop_duplicates("stay_id")
print(f"v4 cohort: {len(merged):,} rows / {merged['stay_id'].nunique():,} stays")
print(f"SOFA mean (per-stay): {per_stay['sofa_24h'].mean():.2f}")
print(f"SOFA missing: {per_stay['sofa_24h'].isna().sum()} stays")
print(f"Wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")
