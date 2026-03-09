#!/bin/bash
#SBATCH --job-name=eval_aero
#SBATCH --account=project_462001112
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module --force purge
module use /appl/local/csc/modulefiles
module load LUMI
module load pytorch
source "/projappl/project_462001112/venvs/diffesm/bin/activate"

export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1

# ROCm / HIP caches
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
export MIOPEN_FIND_ENFORCE=2
mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}

cd /projappl/project_462001112/CESM2_emulator_from_lumi

# When submitted by the trainer, CHECKPOINT and OUTPUT_DIR are set via --export.
# Fall back to defaults for manual submission.
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/projappl/project_462001112/CESM2_emulator_from_lumi/eval_output}"

if [ -n "${CHECKPOINT}" ]; then
    python eval_aero.py \
        --checkpoint  "${CHECKPOINT}" \
        --output-dir  "${OUTPUT_DIR}" \
        --sample-steps 100 \
        --batch-size 16
else
    python eval_aero.py \
        --runs-dir  /projappl/project_462001112/CESM2_emulator_from_lumi/runs \
        --output-dir "${OUTPUT_DIR}" \
        --sample-steps 100 \
        --batch-size 16
fi
