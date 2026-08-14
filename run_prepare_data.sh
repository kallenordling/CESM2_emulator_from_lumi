#!/bin/bash

# Single source of truth for the LUMI project id and its paths.
# Locate the repo. Under sbatch, SLURM COPIES this script to
# /var/spool/slurmd/job<id>/, so dirname "$0" is the spool directory and a
# relative source fails with "No such file or directory" — which is exactly
# what happened on Roihu job 660362. SLURM_SUBMIT_DIR is where sbatch was
# invoked, so try that first, then the script's own directory (correct when
# run directly), then an already-exported LUMI_REPO.
_find_repo() {
    local d
    for d in "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" \
             "${LUMI_REPO:-}"; do
        [ -n "$d" ] && [ -f "$d/lumi_env.sh" ] && { echo "$d"; return 0; }
    done
    echo "ERROR: cannot locate lumi_env.sh. Submit from the repo directory, or" >&2
    echo "       export LUMI_REPO=/path/to/CESM2_emulator_from_lumi first." >&2
    return 1
}
_REPO_DIR="$(_find_repo)" || exit 1
source "${_REPO_DIR}/lumi_env.sh"
# CPU-only: stage raw monthly CESM2 target data → annual-mean training chunks,
# inside the LAIF singularity container (project venv injected). Wraps
# prepare_data_lens.py (LENS2 hist+ssp370) + prepare_data_sf.py (SF AAER/GHG).
# No GPU — just xarray/numpy.
#
# Builds training_data/<VAR>/{hist,ssp370,AAER,GHG}/<member>/chunk_*.nc from
#   lens2/LENS2/<VAR>/LE2-*/   and   sf/{AAER,GHG}/<VAR>/
#
# Usage on LUMI (from the repo dir, after `git pull`):
#   bash run_prepare_data.sh [VARIABLE] [--overwrite]   # default TREFHT; pass PRECT for precip
# e.g.
#   bash run_prepare_data.sh PRECT --overwrite   # redo existing chunks (e.g. after re-download)
#
# This is heavy I/O over many members; run it inside a batch/interactive alloc,
# NOT on the login node:
#   srun --account=${LUMI_ACCOUNT} --partition=small --time=02:00:00 \
#        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G \
#        bash run_prepare_data.sh PRECT
set -euo pipefail

VARIABLE="${1:-TREFHT}"
OVERWRITE="${2:-}"   # pass --overwrite to redo members whose chunks already exist

DATA_ROOT=${LUMI_DATA}
LENS_DIR="${DATA_ROOT}/lens2"
SF_DIR="${DATA_ROOT}/sf"
OUT_DIR="${DATA_ROOT}/training_data"

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# Inject the project venv's site-packages into the container (matches run_decadal_means.sh).
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1

echo "[prepare] VARIABLE=${VARIABLE}"
echo "[prepare] SIF=${SIF}"
echo "[prepare] LENS2 → ${OUT_DIR}/${VARIABLE}/{hist,ssp370} …"
singularity exec "${SIF}" python prepare_data_lens.py \
    --data-dir   "${LENS_DIR}" \
    --output-dir "${OUT_DIR}" \
    --variable   "${VARIABLE}" ${OVERWRITE}

echo "[prepare] SF (AAER,GHG) → ${OUT_DIR}/${VARIABLE}/{AAER,GHG} …"
singularity exec "${SIF}" python prepare_data_sf.py \
    --data-dir   "${SF_DIR}" \
    --output-dir "${OUT_DIR}" \
    --variable   "${VARIABLE}" \
    --ensembles  AAER GHG ${OVERWRITE}

echo "[prepare] done — ${OUT_DIR}/${VARIABLE}/"
