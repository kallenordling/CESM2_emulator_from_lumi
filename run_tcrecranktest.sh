#!/bin/bash
#SBATCH --job-name=tcrecranktest
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
# ISOLATED A/B test: can a STRONG soft TCRE penalty force the high-forcing
# global-mean ΔT within CESM2 variability? Diagnosis (model_skill_diagnosis,
# 2026-05-24, 5-member metric): spatial pattern is perfect but global-mean
# warming is too steep with forcing — ssp370 +0.47/6.8σ, ssp126 +20σ — and the
# soft TCRE penalty at the default weight (tcre_target_fraction=0.05, ~5% of MSE)
# is too weak to counteract it (slope-match AND tcre_full_anomaly both neutral).
# This run cranks tcre_target_fraction 0.05→0.5 (10×, TCRE = 50% of MSE) AND
# scores the eval-aligned full anomaly (tcre_full_anomaly=true) — i.e. constrain
# the RIGHT quantity HARD. EVERYTHING ELSE at production ([2,2,4,1]).
# Forks the newest production run_slope-tcre checkpoint into a SEPARATE
# save_name (run_tcrecrank). No self-chain — does NOT touch production.
# After it runs, eval (5-member auto-watcher, or manual):
#     CHECKPOINT=.../runs/run_tcrecrank_<ep>.pt sbatch run_eval_aero.sh
# Success = ssp370 global-mean bias drops from ~6σ toward within ±2σ WITHOUT
# wrecking spatial pattern corr (watch patcorr stays ~0.99). If even a 10×
# penalty can't force it (or it trades away the pattern) → soft penalties are
# exhausted, build the structural global-mean head instead.

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container (same as run2_aero.sh) ─────────────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] ${SIF}"

_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
echo "[VENV] ${SINGULARITYENV_PYTHONPATH}"

# ── Networking / ROCm (same as run2_aero.sh) ─────────────────────────────────
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

# ── Fork point: newest production checkpoint ─────────────────────────────────
RUNS_DIR="${SLURM_SUBMIT_DIR}/runs"
PROD_CKPT=$(ls -t "${RUNS_DIR}"/run_slope-tcre_*.pt 2>/dev/null | grep -v _best | head -1 || true)
if [[ -z "${PROD_CKPT}" ]]; then
    echo "[fork] no run_slope-tcre_*.pt found in ${RUNS_DIR} — aborting"; exit 1
fi
echo "[fork] forking TCRE-crank test from ${PROD_CKPT}"

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
        trainer.hyperparameters.save_name=run_tcrecrank.pt \
        trainer.hyperparameters.load_path=${PROD_CKPT} \
        trainer.hyperparameters.tcre_full_anomaly=true \
        trainer.hyperparameters.tcre_target_fraction=0.5
'"

srun bash -c "$RUN_CMD" || true
