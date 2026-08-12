#!/bin/bash
#SBATCH --job-name=diag_aerosol
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
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
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
N_ENSEMBLE="${N_ENSEMBLE:-3}"

if [ -n "${CHECKPOINT}" ]; then
    singularity exec ${SIF} bash -c "
        cd ${WORK_DIR}
        python diag_aerosol_sensitivity.py \
            --checkpoint   '${CHECKPOINT}' \
            --output-dir   '${OUTPUT_DIR}' \
            --sample-steps ${SAMPLE_STEPS} \
            --n-ensemble   ${N_ENSEMBLE}
    "
else
    singularity exec ${SIF} bash -c "
        cd ${WORK_DIR}
        python diag_aerosol_sensitivity.py \
            --runs-dir     ${LUMI_REPO}/runs \
            --output-dir   '${OUTPUT_DIR}' \
            --sample-steps ${SAMPLE_STEPS} \
            --n-ensemble   ${N_ENSEMBLE}
    "
fi
