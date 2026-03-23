#!/bin/bash
#SBATCH --job-name=eval_aero
#SBATCH --account=project_462001328
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module --force purge
module use /appl/local/csc/modulefiles
module load LUMI
module load pytorch
# Clear any inherited PYTHONPATH to avoid venv contamination from --export=ALL
unset PYTHONPATH
# Try container-internal path first (/pfs/lustrep1/projappl), fall back to login-node path
if [ -f "/pfs/lustrep1/projappl/project_462001328/venvs/diffesm/diffesm/bin/activate" ]; then
    source "/pfs/lustrep1/projappl/project_462001328/venvs/diffesm/diffesm/bin/activate"
else
    source "/projappl/project_462001328/venvs/diffesm/diffesm/bin/activate"
fi

export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1

# --- diagnostics (remove once venv issue is resolved) ---
echo "[DEBUG] PYTHONPATH=${PYTHONPATH:-<empty>}"
echo "[DEBUG] VIRTUAL_ENV=${VIRTUAL_ENV:-<empty>}"
echo "[DEBUG] which python=$(which python)"
python -c "import sys; print('[DEBUG] sys.path:', sys.path)"
python -c "import sklearn; print('[DEBUG] sklearn path:', sklearn.__file__)" 2>&1 || true
# ---

# ROCm / HIP caches
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
export MIOPEN_FIND_ENFORCE=2
mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}

# Use container-internal path if available
if [ -d "/pfs/lustrep1/projappl/project_462001328/CESM2_emulator_from_lumi" ]; then
    cd /pfs/lustrep1/projappl/project_462001328/CESM2_emulator_from_lumi
else
    cd /projappl/project_462001328/CESM2_emulator_from_lumi
fi

# When submitted by the trainer, CHECKPOINT and OUTPUT_DIR are set via --export.
# Fall back to defaults for manual submission.
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_462001328/eval_output}"

if [ -n "${CHECKPOINT}" ]; then
    python eval_aero.py \
        --checkpoint  "${CHECKPOINT}" \
        --output-dir  "${OUTPUT_DIR}" \
        --sample-steps 100 \
        --batch-size 16
else
    python eval_aero.py \
        --runs-dir  /projappl/project_462001328/CESM2_emulator_from_lumi/runs \
        --output-dir "${OUTPUT_DIR}" \
        --sample-steps 100 \
        --batch-size 16
fi
