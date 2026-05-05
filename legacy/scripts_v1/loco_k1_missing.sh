#!/bin/bash
# Catch-up: re-run 3 K=1 LOCO covariates lost during git clean
# Waits for Stage B (round2_chain_b.sh) to finish before claiming GPU.
set +e
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
LOG=$HOME/hmm-gformula-ci/results/loco_k1_missing.log
cd $HOME/hmm-gformula-ci

echo "[$(ts)] === LOCO K=1 missing catch-up start ===" | tee -a $LOG
echo "[$(ts)] Waiting for round2_chain_b and any active GPU jobs ..." | tee -a $LOG
while pgrep -f "round2_chain_b.sh" > /dev/null \
   || pgrep -f "run_bayesian_main.py" > /dev/null \
   || pgrep -f "run_jax_dose.py" > /dev/null \
   || pgrep -f "ppc_holdout.py" > /dev/null; do
    sleep 60
done
echo "[$(ts)] All GPU jobs finished, starting catch-up." | tee -a $LOG

run_one() {
  local kind=$1; local cov=$2; local arg
  if [ "$kind" = "tv" ]; then arg="--exclude-tv $cov"; else arg="--exclude-static $cov"; fi
  echo "[$(ts)] catch-up K=1 LOCO ${kind}/${cov} ..." | tee -a $LOG
  mkdir -p results/loco_K1/${kind}_${cov}
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

run_one tv pf_ratio
run_one tv paco2
run_one tv lactate

echo "[$(ts)] === LOCO K=1 catch-up complete ===" | tee -a $LOG
