#!/bin/bash
# Round-2 reviewer-defensive chain.
# Stage 1: K=1 LOCO (12 covariates × refit + dose)
# Stage 2: WAIC/LOO refits + computation (after model log_lik patch)
# Stage 3: Prior sensitivity K=5 (HalfCauchy → Gamma)
# Stage 4: PPC held-out 20% K=5
# Stage 5: Spec II lambda_L re-extraction
# Stage 6: Convergence diagnostics extraction (4 methods)
#
# Stage 1 launches with current code (no changes). Stages 2-6 need code patch
# pushed via git. Stage 1 is independent and can run immediately.

set +e
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
LOG=$HOME/hmm-gformula-ci/results/round2_chain.log
cd $HOME/hmm-gformula-ci

echo "[$(ts)] === Round-2 chain start ===" | tee -a $LOG

# ----- Stage 1: K=1 LOCO -----
mkdir -p results/loco_K1
TV=("pf_ratio" "paco2" "lactate" "map_mmhg" "heart_rate" "gcs_total" "creatinine" "temperature_c")
ST=("anchor_age" "gender_M" "bmi_imputed" "charlson_index")

run_one_K1() {
  local kind=$1; local cov=$2; local arg
  if [ "$kind" = "tv" ]; then arg="--exclude-tv $cov"; else arg="--exclude-static $cov"; fi
  echo "[$(ts)] K=1 LOCO ${kind}/${cov} ..." | tee -a $LOG
  python3 -u scripts/run_bayesian_main.py \
    --csv data/ards_v31_v4.csv --out-dir results/loco_K1/${kind}_${cov} \
    --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
    --target-accept 0.95 --n-posterior-subset 200 \
    --methods K1 --phase fit $arg >> $LOG 2>&1
  python3 -u scripts/run_jax_dose.py \
    --csv data/ards_v31_v4.csv --state-dir results/loco_K1/${kind}_${cov} \
    --out-dir results/loco_K1/${kind}_${cov} --prefix fre_nice_K1 \
    $arg >> $LOG 2>&1
}

echo "[$(ts)] === Stage 1: K=1 LOCO (12 covariates) ===" | tee -a $LOG
for cov in "${TV[@]}"; do run_one_K1 "tv" "$cov"; done
for cov in "${ST[@]}"; do run_one_K1 "static" "$cov"; done
echo "[$(ts)] === Stage 1 done ===" | tee -a $LOG

echo "[$(ts)] === Round-2 chain Stage 1 complete (Stages 2-6 pending code patch) ===" | tee -a $LOG
