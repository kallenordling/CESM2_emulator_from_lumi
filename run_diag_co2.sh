#!/bin/bash
#SBATCH --job-name=diag_co2
#SBATCH --account=project_462001328
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container ─────────────────────────────────────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] Using: ${SIF}"

_VENV_SITE=/projappl/project_462001328/venvs/diffesm_laif/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
echo "[VENV] SINGULARITYENV_PYTHONPATH=${SINGULARITYENV_PYTHONPATH}"

export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
export MIOPEN_FIND_ENFORCE=2
mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}

if [ -d "/pfs/lustrep1/projappl/project_462001328/CESM2_emulator_from_lumi" ]; then
    WORK_DIR=/pfs/lustrep1/projappl/project_462001328/CESM2_emulator_from_lumi
else
    WORK_DIR=/projappl/project_462001328/CESM2_emulator_from_lumi
fi

CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_462001328/eval_output}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
N_ENSEMBLE="${N_ENSEMBLE:-3}"

if [ -n "${CHECKPOINT}" ]; then
    singularity exec ${SIF} bash -c "
        cd ${WORK_DIR}
        python diag_co2_sensitivity.py \
            --checkpoint   '${CHECKPOINT}' \
            --output-dir   '${OUTPUT_DIR}' \
            --sample-steps ${SAMPLE_STEPS} \
            --n-ensemble   ${N_ENSEMBLE}
    "
else
    singularity exec ${SIF} bash -c "
        cd ${WORK_DIR}
        python diag_co2_sensitivity.py \
            --runs-dir     /projappl/project_462001328/CESM2_emulator_from_lumi/runs \
            --output-dir   '${OUTPUT_DIR}' \
            --sample-steps ${SAMPLE_STEPS} \
            --n-ensemble   ${N_ENSEMBLE}
    "
fi
