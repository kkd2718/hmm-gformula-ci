#!/bin/bash
# Overnight chain: runs sequentially after Spec II completes.
# Plan: K=4 knot + K=6 knot + Subgroup (using existing K=5 Spec I state).
# LOCO and WAIC deferred to next day (LOCO needs 12 refits, WAIC needs log_lik recording).

set +e
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
LOG=$HOME/hmm-gformula-ci/results/overnight_chain.log
cd $HOME/hmm-gformula-ci

echo "[$(ts)] === Overnight chain start ===" | tee -a $LOG

# ----- Wait for Spec II to finish (no concurrent NUTS on single GPU) -----
echo "[$(ts)] Waiting for Spec II ..." | tee -a $LOG
while pgrep -f "run_bayesian_main.py.*--share-RE-on-L" > /dev/null \
   || pgrep -f "run_jax_dose.py.*--share-RE-on-L" > /dev/null; do
    sleep 60
done
echo "[$(ts)] Spec II finished." | tee -a $LOG

mkdir -p results/knot_sensitivity results/subgroup

# ----- Knot K=4 sensitivity -----
echo "[$(ts)] [1/3] K=4 knot fit + dose ..." | tee -a $LOG
python3 -u scripts/run_bayesian_main.py \
  --csv data/ards_v31_v4.csv --out-dir results/knot_sensitivity \
  --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
  --target-accept 0.95 --n-posterior-subset 200 \
  --methods K4 --phase fit \
  >> $LOG 2>&1
python3 -u scripts/run_jax_dose.py \
  --csv data/ards_v31_v4.csv --state-dir results/knot_sensitivity \
  --out-dir results/knot_sensitivity --prefix fre_nice_K4 \
  >> $LOG 2>&1
echo "[$(ts)] [1/3] K=4 done." | tee -a $LOG

# ----- Knot K=6 sensitivity -----
echo "[$(ts)] [2/3] K=6 knot fit + dose ..." | tee -a $LOG
python3 -u scripts/run_bayesian_main.py \
  --csv data/ards_v31_v4.csv --out-dir results/knot_sensitivity \
  --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
  --target-accept 0.95 --n-posterior-subset 200 \
  --methods K6 --phase fit \
  >> $LOG 2>&1
python3 -u scripts/run_jax_dose.py \
  --csv data/ards_v31_v4.csv --state-dir results/knot_sensitivity \
  --out-dir results/knot_sensitivity --prefix fre_nice_K6 \
  >> $LOG 2>&1
echo "[$(ts)] [2/3] K=6 done." | tee -a $LOG

# ----- Subgroup analysis (uses Spec I K=5 state, existing) -----
echo "[$(ts)] [3/4] Subgroup analysis ..." | tee -a $LOG
python3 -u scripts/subgroup_jax_dose.py \
  --csv data/ards_v31_v4.csv \
  --state-path results/bayesian_main_v2/fre_nice_K5_state.npz \
  --out-md results/subgroup/subgroup_K5.md \
  --n-posterior-subset 200 --n-b-draws-per-post 5 \
  >> $LOG 2>&1
echo "[$(ts)] [3/4] Subgroup done." | tee -a $LOG

# ----- LOCO sensitivity (refit per excluded covariate) -----
mkdir -p results/loco_K5
TV_COVARIATES=("pf_ratio" "paco2" "lactate" "map_mmhg" "heart_rate" "gcs_total" "creatinine" "temperature_c")
STATIC_COVARIATES=("anchor_age" "gender_M" "bmi_imputed" "charlson_index")

run_loco_one() {
  local kind=$1   # 'tv' or 'static'
  local cov=$2
  local exclude_arg
  if [ "$kind" = "tv" ]; then exclude_arg="--exclude-tv $cov"; else exclude_arg="--exclude-static $cov"; fi
  echo "[$(ts)] LOCO ${kind}/${cov} fit + dose ..." | tee -a $LOG
  python3 -u scripts/run_bayesian_main.py \
    --csv data/ards_v31_v4.csv --out-dir results/loco_K5/${kind}_${cov} \
    --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
    --target-accept 0.95 --n-posterior-subset 200 \
    --methods K5 --phase fit $exclude_arg \
    >> $LOG 2>&1
  python3 -u scripts/run_jax_dose.py \
    --csv data/ards_v31_v4.csv --state-dir results/loco_K5/${kind}_${cov} \
    --out-dir results/loco_K5/${kind}_${cov} --prefix fre_nice_K5 \
    $exclude_arg >> $LOG 2>&1
}

echo "[$(ts)] [4/4] LOCO sensitivity (12 covariates) ..." | tee -a $LOG
for cov in "${TV_COVARIATES[@]}"; do
  run_loco_one "tv" "$cov"
done
for cov in "${STATIC_COVARIATES[@]}"; do
  run_loco_one "static" "$cov"
done
echo "[$(ts)] [4/4] LOCO done." | tee -a $LOG

echo "[$(ts)] === Overnight chain complete ===" | tee -a $LOG
