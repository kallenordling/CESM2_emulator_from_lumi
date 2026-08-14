#!/bin/bash
#SBATCH --job-name=gainfix_intmatch
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Single source of truth for the LUMI project id and its paths.
# Locate the repo. Under sbatch, SLURM COPIES this script to
# /var/spool/slurmd/job<id>/, so dirname "$0" is the spool directory and a
# relative source fails with "No such file or directory" — which is exactly
# what happened on Roihu job 660362. SLURM_SUBMIT_DIR is where sbatch was
# invoked, so try that first, then the script's own directory (correct when
# run directly), then an already-exported LUMI_REPO.
_find_repo() {
    local d
    for d in "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" \
             "${LUMI_REPO:-}"; do
        [ -n "$d" ] && [ -f "$d/lumi_env.sh" ] && { echo "$d"; return 0; }
    done
    echo "ERROR: cannot locate lumi_env.sh. Submit from the repo directory, or" >&2
    echo "       export LUMI_REPO=/path/to/CESM2_emulator_from_lumi first." >&2
    return 1
}
_REPO_DIR="$(_find_repo)" || exit 1
source "${_REPO_DIR}/lumi_env.sh"
assert_account
lumi_env_banner
#
# ISOLATED A/B test (hypothesis #2 of the ssp370 warm+wet bias investigation,
# 2026-07-22): run_gainfix's persistent ssp370 warm+wet bias (GMbias
# +0.15..+0.29 K across ep0970/0990/1030) could stem from ssp370 being the
# ONLY multi-forcing (rising-CO2 + declining/changing-aerosol) scenario in
# training, while its sub-additivity target is switched OFF by default:
#   trainer/unetTrainer.py:1330  _compute_interaction_loss only pins ssp370's
#   modeled interaction field to CESM2's MEASURED sub-additive interaction
#   (I_gm vs cumCO2, trainer/unetTrainer.py:995-1049) when
#   interaction_match_ssp370=True.
#   configs/config_aero.yaml:119 sets interaction_match_ssp370: false, and
#   run2_gainfix.sh does not override it — so ssp370 currently only gets the
#   hist-only additivity→0 term (unetTrainer.py:1322-1327), never a target for
#   its own known CO2×aerosol sub-additivity. Left unconstrained here, the
#   model may default toward the (over-warm) additive sum of ghg-only +
#   aaer-only responses.
#
# This fork flips interaction_match_ssp370 on, resuming from the newest
# run_gainfix checkpoint into a SEPARATE save_name (run_gainfix_intmatch) —
# does NOT touch the production chain. No eval watcher, no self-chain.
#
# After ~30-50 epochs, eval a run_gainfix_intmatch checkpoint and compare
# ssp370 GMbias/precip-bias against the run_gainfix baseline at a matched
# epoch:
#     CHECKPOINT=.../runs/run_gainfix_intmatch_<ep>.pt sbatch run_eval_aero.sh
# Success = ssp370 GMbias/precip-bias shrinks without moving hist/ghg/aaer.
# Watch the training log for "[TRAINER] interaction target (ssp370 ...)" at
# startup — if it prints "ssp370 match disabled" (missing ghg/aaer overlap,
# see unetTrainer.py:1011-1041), this A/B silently did nothing; check the log
# before trusting a null result.
# If this DOESN'T move the bias (and the term IS active), hypothesis #2 is
# ruled out and the SGAIN-calibration hypothesis (#3, single-realization TCRE
# slope target / sparse correction) becomes the leading candidate.

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container (same as run2_gainfix.sh) ──────────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] ${SIF}"

_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
echo "[VENV] ${SINGULARITYENV_PYTHONPATH}"

# ── Networking / ROCm (same as run2_gainfix.sh) ──────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libfabric.so.1
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
export ACCELERATE_USE_FSDP=0
export HSA_ENABLE_SDMA=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE=0
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}

# ── Fork point: newest production run_gainfix checkpoint ────────────────────
RUNS_DIR="${SLURM_SUBMIT_DIR}/runs"
PROD_CKPT=$(ls -t "${RUNS_DIR}"/run_gainfix_*.pt 2>/dev/null | grep -v _best | head -1 || true)
if [[ -z "${PROD_CKPT}" ]]; then
    echo "[fork] no run_gainfix_*.pt found in ${RUNS_DIR} — aborting"; exit 1
fi
echo "[fork] forking interaction-match A/B from ${PROD_CKPT}"

# ── Launch (Hydra overrides; no watcher, no chain) ───────────────────────────
NUM_PROCESSES=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))
MAIN_PROCESS_IP=$(hostname -i)

RUN_CMD="singularity exec ${SIF} bash -c '
    accelerate launch \
        --config_file=accelerate_config.yaml \
        --num_processes=${NUM_PROCESSES} \
        --num_machines=${SLURM_NNODES} \
        --machine_rank=\${SLURM_NODEID} \
        --main_process_ip=${MAIN_PROCESS_IP} \
        main_aero.py \
        trainer.hyperparameters.save_name=run_gainfix_intmatch.pt \
        trainer.hyperparameters.load_path=${PROD_CKPT} \
        trainer.hyperparameters.sampled_gain_loss_scale=0.05 \
        trainer.hyperparameters.bc_clip_mode=populated \
        trainer.hyperparameters.interaction_match_ssp370=true
'"

srun bash -c "$RUN_CMD" || true

cp /tmp/miopen_${SLURM_JOB_ID}/*.ufdb.* "${SLURM_SUBMIT_DIR}/.miopen_cache/" 2>/dev/null || true
cp /tmp/miopen_${SLURM_JOB_ID}/*.db     "${SLURM_SUBMIT_DIR}/.miopen_cache/" 2>/dev/null || true
