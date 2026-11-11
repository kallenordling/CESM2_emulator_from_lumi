#!/bin/bash
# -----------------------------------------------------------------------------
# Loss-function arms (branch loss-functions). Submits one FRESH run_asinh.sh
# chain per arm. Everything except the objective is the mmlin pipeline
# (smooth sigma 4 -> min/max anchors on hist+ssp370, no PCA, no clip), so the
# baseline is the existing mmlin arm (plain cos-lat v-MSE) at matched epochs.
#
#   lossminsnr   Min-SNR-gamma 5 timestep weighting   (where capacity goes in noise level)
#   lossphuber   pseudo-Huber, c = 0.05                (robustness to residual tails)
#   lossms       + multiscale block-mean MSE, w = 2    (emphasis on coherent regional error)
#   losselbo     ELBO / maximum likelihood, log-SNR importance-sampled (calibrated spread)
#
# Each arm stops at STOP_EPOCH (default 800): the trainer saves and evaluates
# that epoch and the chain ends itself; CHAIN_REMAINING is only an upper bound
# (~90 epochs per 6 h link).
#
# See trainer/losses.py for the definitions. Each arm logs "MSE LOSS" as the
# PLAIN v-MSE, so training curves compare across arms; "Training/Loss" is the
# arm's own objective and is NOT comparable.
#
# Usage (from the checkout on LUMI that has this branch):
#   bash run_loss_arms.sh                 # all four arms
#   bash run_loss_arms.sh lossms          # one arm
#   CHAIN_REMAINING=8 bash run_loss_arms.sh
#
# The overrides travel down the chain in EXTRA_TRAIN_OVERRIDES (--export=ALL),
# and every checkpoint stores LOSS_CONFIG: a link that lost them refuses to
# resume rather than quietly training plain MSE.
#
# RESUME a broken chain (same env, FRESH=0):
#   RUN_TAG=lossms EXTRA_TRAIN_OVERRIDES="<as below>" FRESH=0 CO2FIX=1 ... bash lsubmit.sh run_asinh.sh
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export LUMI_PROJECT="${LUMI_PROJECT:-462001765}"
export LUMI_EVAL_PROJECT="${LUMI_EVAL_PROJECT:-${LUMI_PROJECT}}"
export TRAIN_DATA_CONFIG=config_data_ybias_BCprect_nopca.yaml
export STOP_EPOCH="${STOP_EPOCH:-800}"
export CHAIN_REMAINING="${CHAIN_REMAINING:-12}"
# Every watcher start purges OTHER runs' triggers older than this. Four arms
# plus the running ones share the trigger dir, and a link can sit in the queue
# for hours, so do not let these watchers eat the neighbours' triggers.
export STALE_TRIGGER_SECS="${STALE_TRIGGER_SECS:-1209600}"

HP=trainer.hyperparameters
declare -A ARMS=(
    [lossminsnr]="${HP}.snr_weighting=min_snr ${HP}.min_snr_gamma=5.0"
    [lossphuber]="${HP}.loss_type=pseudo_huber ${HP}.huber_c=0.05"
    [lossms]="${HP}.multiscale_weight=2.0"
    [losselbo]="${HP}.snr_weighting=elbo"
)

SELECTED=("$@")
[[ ${#SELECTED[@]} -eq 0 ]] && SELECTED=(lossminsnr lossphuber lossms losselbo)

for tag in "${SELECTED[@]}"; do
    [[ -n "${ARMS[$tag]:-}" ]] || { echo "unknown arm: $tag (have: ${!ARMS[*]})" >&2; exit 1; }
    save_dir="/scratch/project_${LUMI_PROJECT}/runs_${tag}"
    if compgen -G "${save_dir}/run_${tag}_co2fix_*.pt" >/dev/null; then
        echo "[$tag] checkpoints already exist in ${save_dir}; not starting it FRESH" >&2
        continue
    fi
    mkdir -p "${save_dir}"
    echo "[$tag] ${ARMS[$tag]}"
    RUN_TAG="${tag}" SAVE_DIR="${save_dir}" FRESH=1 \
        EXTRA_TRAIN_OVERRIDES="${ARMS[$tag]}" \
        bash lsubmit.sh run_asinh.sh --job-name="${tag}"
done
