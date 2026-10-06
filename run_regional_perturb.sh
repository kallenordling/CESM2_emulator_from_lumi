#!/bin/bash
# Regional aerosol perturbation: does scaling ONE region's SUL+BC move climate
# THERE? See scripts/regional_perturbation.py for what the test can and cannot
# conclude.
#
#   CHECKPOINT=/path/run.pt REGION=east_china SCALE=0.5 sbatch run_regional_perturb.sh
#
# Code is taken from the SUBMIT directory, never from LUMI_REPO: an arm trained
# from one checkout but analysed with another's code is the fault that
# invalidated every asinh99 eval up to ep0250.
#SBATCH --job-name=regperturb
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001112/logs/%x_%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/lumi_env.sh"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=/path/run_xxx.pt}"
REGION="${REGION:-east_china}"
SCALE="${SCALE:-0.5}"
YEAR="${YEAR:-2050}"
MEMBERS="${MEMBERS:-5}"
# Must match the arm's training config: sigma lives in the DATA CONFIG, and the
# normalisation anchors are fitted on the SMOOTHED field, so a mismatched sigma
# silently normalises the perturbation against the wrong ceiling.
DATA_CONFIG="${DATA_CONFIG:-configs/config_data_ybias_BCprect_nopca.yaml}"
OUT="${OUT:-/scratch/project_462001112/analysis/regional_perturbation}"

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
export SINGULARITYENV_PYTHONPATH="${LUMI_VENV}/lib/python3.12/site-packages"
export HYDRA_FULL_ERROR=1 PYTHONNOUSERSITE=1
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
mkdir -p /tmp/miopen_${SLURM_JOB_ID} "${OUT}"

echo "[REGPERTURB] code=${SLURM_SUBMIT_DIR} ckpt=${CHECKPOINT}"
echo "[REGPERTURB] region=${REGION} scale=${SCALE} year=${YEAR} members=${MEMBERS}"
echo "[REGPERTURB] data_config=${DATA_CONFIG}"
singularity exec ${SIF} bash -c "
    cd '${SLURM_SUBMIT_DIR}' && \
    python scripts/regional_perturbation.py \
        --checkpoint '${CHECKPOINT}' --region '${REGION}' --scale ${SCALE} \
        --year ${YEAR} --members ${MEMBERS} \
        --data-config '${DATA_CONFIG}' --out '${OUT}'
"
