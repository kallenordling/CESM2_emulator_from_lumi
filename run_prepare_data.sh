#!/bin/bash
# CPU-only: stage raw monthly CESM2 target data → annual-mean training chunks,
# inside the LAIF singularity container (project venv injected). Wraps
# prepare_data_lens.py (LENS2 hist+ssp370) + prepare_data_sf.py (SF AAER/GHG).
# No GPU — just xarray/numpy.
#
# Builds training_data/<VAR>/{hist,ssp370,AAER,GHG}/<member>/chunk_*.nc from
#   lens2/LENS2/<VAR>/LE2-*/   and   sf/{AAER,GHG}/<VAR>/
#
# Usage on LUMI (from the repo dir, after `git pull`):
#   bash run_prepare_data.sh [VARIABLE]      # default TREFHT; pass PRECT for precip
# e.g.
#   bash run_prepare_data.sh PRECT
#
# This is heavy I/O over many members; run it inside a batch/interactive alloc,
# NOT on the login node:
#   srun --account=project_462001328 --partition=small --time=02:00:00 \
#        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G \
#        bash run_prepare_data.sh PRECT
set -euo pipefail

VARIABLE="${1:-TREFHT}"

DATA_ROOT=/scratch/project_462001328/emulator_data
LENS_DIR="${DATA_ROOT}/lens2"
SF_DIR="${DATA_ROOT}/sf"
OUT_DIR="${DATA_ROOT}/training_data"

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# Inject the project venv's site-packages into the container (matches run_decadal_means.sh).
_VENV_SITE=$(realpath /projappl/project_462001328/venvs/diffesm_laif 2>/dev/null \
             || echo /projappl/project_462001328/venvs/diffesm_laif)/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1

echo "[prepare] VARIABLE=${VARIABLE}"
echo "[prepare] SIF=${SIF}"
echo "[prepare] LENS2 → ${OUT_DIR}/${VARIABLE}/{hist,ssp370} …"
singularity exec "${SIF}" python prepare_data_lens.py \
    --data-dir   "${LENS_DIR}" \
    --output-dir "${OUT_DIR}" \
    --variable   "${VARIABLE}"

echo "[prepare] SF (AAER,GHG) → ${OUT_DIR}/${VARIABLE}/{AAER,GHG} …"
singularity exec "${SIF}" python prepare_data_sf.py \
    --data-dir   "${SF_DIR}" \
    --output-dir "${OUT_DIR}" \
    --variable   "${VARIABLE}" \
    --ensembles  AAER GHG

echo "[prepare] done — ${OUT_DIR}/${VARIABLE}/"
