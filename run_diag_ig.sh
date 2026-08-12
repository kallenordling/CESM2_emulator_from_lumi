#!/bin/bash
#SBATCH --job-name=diag_ig
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
assert_account
lumi_env_banner

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container ─────────────────────────────────────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] Using: ${SIF}"

_VENV_SITE=${LUMI_VENV}/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
echo "[VENV] SINGULARITYENV_PYTHONPATH=${SINGULARITYENV_PYTHONPATH}"

export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
export MIOPEN_FIND_ENFORCE=2
mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}

if [ -d "${LUMI_REPO_PFS}" ]; then
    WORK_DIR=${LUMI_REPO_PFS}
else
    WORK_DIR=${LUMI_REPO}
fi

CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${LUMI_EVAL_OUT}}"
N_IG_STEPS="${N_IG_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"

if [ -n "${CHECKPOINT}" ]; then
    singularity exec ${SIF} bash -c "
        cd ${WORK_DIR}
        python diag_integrated_gradients.py \
            --checkpoint '${CHECKPOINT}' \
            --output-dir '${OUTPUT_DIR}' \
            --n-ig-steps ${N_IG_STEPS} \
            --batch-size ${BATCH_SIZE}
    "
else
    singularity exec ${SIF} bash -c "
        cd ${WORK_DIR}
        python diag_integrated_gradients.py \
            --runs-dir   ${LUMI_REPO}/runs \
            --output-dir '${OUTPUT_DIR}' \
            --n-ig-steps ${N_IG_STEPS} \
            --batch-size ${BATCH_SIZE}
    "
fi
