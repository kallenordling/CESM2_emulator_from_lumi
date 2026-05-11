#!/bin/bash
#SBATCH --job-name=eval_aero
#SBATCH --account=project_462001328
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container ─────────────────────────────────────────────────
# Same setup as run2_aero.sh — CSC PyTorch module no longer works after 21.1.2026.
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] Using: ${SIF}"

# Override any inherited SINGULARITYENV_PYTHONPATH from --export=ALL
_VENV_SITE=/projappl/project_462001328/venvs/diffesm_laif/lib/python3.12/site-packages
_EXTRA_PKGS=/scratch/project_462001328/python_packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}:${_EXTRA_PKGS}"
echo "[VENV] SINGULARITYENV_PYTHONPATH=${SINGULARITYENV_PYTHONPATH}"

# ── Python / Hydra ────────────────────────────────────────────────────────────
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

# ── ROCm / HIP caches ────────────────────────────────────────────────────────
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
export MIOPEN_FIND_ENFORCE=2
mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}

# Use container-internal path if available
if [ -d "/pfs/lustrep1/projappl/project_462001328/CESM2_emulator_from_lumi" ]; then
    WORK_DIR=/pfs/lustrep1/projappl/project_462001328/CESM2_emulator_from_lumi
else
    WORK_DIR=/projappl/project_462001328/CESM2_emulator_from_lumi
fi

# When submitted by the trainer, CHECKPOINT and OUTPUT_DIR are set via --export.
# Fall back to defaults for manual submission.
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_462001328/eval_output}"
# Per-channel CFG scales: < 1.0 reduces CO2 warming, > 1.0 amplifies aerosol cooling.
# Set to 1.0 (default) to disable CFG and use direct conditioning (single forward pass).
GUIDANCE_CO2="${GUIDANCE_CO2:-1.0}"
GUIDANCE_SUL="${GUIDANCE_SUL:-1.0}"
# Set RUN_XAI=1 (or SKIP_XAI=0) to run XAI figures (IG + saliency) — off by default (slow).
RUN_XAI="${RUN_XAI:-0}"
# SKIP_XAI=0 is equivalent to RUN_XAI=1 (inverted alias for convenience)
[ "${SKIP_XAI:-1}" = "0" ] && RUN_XAI="1"
# Set FORCE_CFG=1 to use 3-pass CFG decomposition even at guidance scales 1.0/1.0.
# Useful to isolate whether the additive decomposition itself introduces bias.
FORCE_CFG="${FORCE_CFG:-0}"
# Number of diffusion ensemble members to generate per experiment.
MEMBERS="${members:-${MEMBERS:-1}}"

_XAI_FLAG=""
[ "${RUN_XAI}" = "1" ] && _XAI_FLAG="--run-xai"
_CFG_FLAG=""
[ "${FORCE_CFG}" = "1" ] && _CFG_FLAG="--force-cfg"

# Multi-GPU experiment sharding: srun launches one task per GPU; each task
# runs experiments_to_run[$SLURM_PROCID::$SLURM_NTASKS] inside eval_aero.py
# (--shard-rank / --n-shards default to $SLURM_PROCID / $SLURM_NTASKS).
# ROCR_VISIBLE_DEVICES pins each task to its local GPU so they don't fight
# over device 0.
if [ -n "${CHECKPOINT}" ]; then
    CKPT_FLAG="--checkpoint ${CHECKPOINT}"
    RUNS_FLAG=""
else
    CKPT_FLAG=""
    RUNS_FLAG="--runs-dir /projappl/project_462001328/CESM2_emulator_from_lumi/runs"
fi

PY_ARGS="${CKPT_FLAG} ${RUNS_FLAG} \
    --output-dir ${OUTPUT_DIR} \
    --sample-steps 50 \
    --batch-size 16 \
    --guidance-co2 ${GUIDANCE_CO2} \
    --guidance-sul ${GUIDANCE_SUL} \
    --members ${MEMBERS} \
    ${_XAI_FLAG} ${_CFG_FLAG}"

# All variables expand here (script-launch time) except $SLURM_LOCALID which
# must be evaluated inside srun's spawned shell — hence the \$.
srun --ntasks=${SLURM_NTASKS} --ntasks-per-node=${SLURM_NTASKS} --gpus-per-task=1 \
    bash -c "
        export ROCR_VISIBLE_DEVICES=\${SLURM_LOCALID}
        singularity exec ${SIF} bash -c 'cd ${WORK_DIR} && python eval_aero.py ${PY_ARGS}'
    "
