#!/bin/bash
# Nonlinear-term maps for ONE checkpoint: the response map N_T(r) and the
# INTEGRATED aerosol source maps, plus the conditioning displacement they are
# integrated along (02_interaction_maps.py). Integrated maps decompose N; the
# old alpha=beta=1 map does not.
#
#   CHECKPOINT=/path/run.pt OUT=/scratch/.../intmaps_x.npz sbatch run_intmaps.sh
#
# Code is taken from the SUBMIT directory, never from LUMI_REPO: an arm charged
# to one project but trained from another's checkout would otherwise be
# analysed with code that does not know its conditioning transform -- the exact
# fault that invalidated every asinh99 eval up to ep0250.
#SBATCH --job-name=int_maps
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_462001112/logs/%x_%j.out

source "${SLURM_SUBMIT_DIR}/lumi_env.sh"
assert_account
lumi_env_banner
set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT}"
: "${OUT:?set OUT}"
YEAR="${YEAR:-2040}"
GRID="${GRID:-21}"

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
export SINGULARITYENV_PYTHONPATH="${LUMI_VENV}/lib/python3.12/site-packages"
export HYDRA_FULL_ERROR=1 PYTHONNOUSERSITE=1
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
mkdir -p /tmp/miopen_${SLURM_JOB_ID} "$(dirname "${OUT}")"

echo "[INTMAPS] code=${SLURM_SUBMIT_DIR} ckpt=${CHECKPOINT} year=${YEAR} grid=${GRID} out=${OUT}"
singularity exec ${SIF} bash -c "
    cd '${SLURM_SUBMIT_DIR}' && \
    python analysis/nonlinear_emission_interaction/02_interaction_maps.py \
        '${CHECKPOINT}' --year ${YEAR} --integrate-grid ${GRID} --out '${OUT}'
"
