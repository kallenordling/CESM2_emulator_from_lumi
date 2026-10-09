#!/bin/bash
# What emissions cause N: the 8 corners of the (CO2, SUL, BC) cube.
#
# N = ALL - GHG - AAER = CS + CB + CSB exactly, so the closure test (N from 4
# corners vs CS+CB+CSB from 8) decides whether the decomposition is valid.
#
#   CHECKPOINT=/path/run.pt SCENARIO=ssp370-126aer sbatch run_n_factorial.sh
#
# Code comes from SLURM_SUBMIT_DIR, never LUMI_REPO -- analysing an arm with
# another checkout's code is what invalidated the asinh99 evals.
#SBATCH --job-name=nfactorial
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_462001112/logs/%x_%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/lumi_env.sh"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=/path/run_xxx.pt}"
WINDOW="${WINDOW:-2041 2050}"   # ghg/aaer cond files end at 2050
MEMBERS="${MEMBERS:-5}"
# Must match the arm's training config: sigma lives there and the min/max
# anchors are fitted on the SMOOTHED field, so a mismatched sigma normalises
# the hybrid against the wrong ceiling without failing.
DATA_CONFIG="${DATA_CONFIG:-configs/config_data_ybias_BCprect_nopca.yaml}"
OUT="${OUT:-/scratch/project_462001112/analysis/n_factorial}"

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
export SINGULARITYENV_PYTHONPATH="${LUMI_VENV}/lib/python3.12/site-packages"
export HYDRA_FULL_ERROR=1 PYTHONNOUSERSITE=1
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
mkdir -p /tmp/miopen_${SLURM_JOB_ID} "${OUT}"

echo "[NFACT] code=${SLURM_SUBMIT_DIR} ckpt=${CHECKPOINT}"
echo "[NFACT] window=${WINDOW} members=${MEMBERS} (8 corners of the CO2/SUL/BC cube)"
singularity exec ${SIF} bash -c "
    cd '${SLURM_SUBMIT_DIR}' && \
    python scripts/n_factorial.py --checkpoint '${CHECKPOINT}' \
        --window ${WINDOW} \
        --members ${MEMBERS} --data-config '${DATA_CONFIG}' --out '${OUT}'
"
