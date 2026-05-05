#!/bin/bash
# Round-2 chain C: re-run after log_lik OOM fix + PPC bug fix.
# Stage C1: K=1 + K=5 + Xu fits with --record-loglik (subject-level now)
# Stage C2: WAIC + PSIS-LOO computation
# Stage C3: extract_diagnostics across all available state files
# Stage C4: PPC re-run on held-out 20% (bug fix)

set +e
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
LOG=$HOME/hmm-gformula-ci/results/round2_chain_c.log
cd $HOME/hmm-gformula-ci

echo "[$(ts)] === Round-2 chain C start ===" | tee -a $LOG
# Wait for any GPU jobs
while pgrep -f "loco_k1_missing.sh" > /dev/null \
   || pgrep -f "run_bayesian_main.py" > /dev/null \
   || pgrep -f "run_jax_dose.py" > /dev/null \
   || pgrep -f "ppc_holdout.py" > /dev/null; do
    sleep 60
done
echo "[$(ts)] All GPU clear, starting." | tee -a $LOG

# Clean stale outputs from OOM-failed run
rm -rf results/loglik_main
mkdir -p results/loglik_main

# Stage C1: log_lik fits
echo "[$(ts)] === Stage C1: K=1 + K=5 + Xu with --record-loglik (subject-level) ===" | tee -a $LOG

python3 -u scripts/run_bayesian_main.py \
  --csv data/ards_v31_v4.csv --out-dir results/loglik_main \
  --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
  --target-accept 0.95 --n-posterior-subset 200 \
  --methods K1 --phase fit --record-loglik \
  >> $LOG 2>&1

python3 -u scripts/run_bayesian_main.py \
  --csv data/ards_v31_v4.csv --out-dir results/loglik_main \
  --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
  --target-accept 0.95 --n-posterior-subset 200 \
  --methods K5 --phase fit --record-loglik \
  >> $LOG 2>&1

python3 -u scripts/run_bayesian_main.py \
  --csv data/ards_v31_v4.csv --out-dir results/loglik_main \
  --inference nuts --n-warmup 1000 --n-samples 1000 --n-chains 2 \
  --target-accept 0.95 --n-posterior-subset 200 \
  --methods xu --phase fit --record-loglik \
  >> $LOG 2>&1

# Stage C2: WAIC + LOO
echo "[$(ts)] === Stage C2: WAIC + PSIS-LOO ===" | tee -a $LOG
python3 -u scripts/compute_waic_loo.py \
  --state-files results/loglik_main/fre_nice_K1_state.npz \
                results/loglik_main/fre_nice_K5_state.npz \
                results/loglik_main/xu_bayesian_state.npz \
  --labels "K=1 NICE" "K=5 FRE-NICE" "Xu Bayesian" \
  --out-md results/loglik_main/waic_loo_table.md \
  >> $LOG 2>&1

# Stage C3: diagnostics summary across all NUTS fits
echo "[$(ts)] === Stage C3: convergence diagnostics ===" | tee -a $LOG
python3 -u scripts/extract_diagnostics.py \
  --state-files results/loglik_main/fre_nice_K1_state.npz \
                results/loglik_main/fre_nice_K5_state.npz \
                results/loglik_main/xu_bayesian_state.npz \
                results/prior_sens_gamma/fre_nice_K5_state.npz \
                results/ppc_K5/fre_nice_K5_state.npz \
  --labels "K=1 NICE" "K=5 FRE-NICE" "Xu Bayesian" "K=5 prior=Gamma" "K=5 80%-fit" \
  --out-md results/loglik_main/diagnostics_table.md \
  >> $LOG 2>&1

# Stage C4: PPC re-run with bug fix (uses existing 80%-fit state from prev Stage B)
echo "[$(ts)] === Stage C4: PPC re-run (bug fix) ===" | tee -a $LOG
python3 -u scripts/ppc_holdout.py \
  --csv data/ards_v31_v4.csv \
  --state-path results/ppc_K5/fre_nice_K5_state.npz \
  --holdout-ids results/ppc_K5/holdout_subj_ids.npy \
  --out-md results/ppc_K5/ppc_K5_v2.md \
  --n-posterior-subset 200 --n-b-draws 5 \
  >> $LOG 2>&1

echo "[$(ts)] === Round-2 chain C complete ===" | tee -a $LOG
