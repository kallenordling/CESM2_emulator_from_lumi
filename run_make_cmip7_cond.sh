#!/bin/bash
# CPU-only: build the CMIP7 (h / vl) conditioning files, inside the LAIF
# singularity container (project venv injected; needs xarray + xesmf, same as
# run_make_bc_cond.sh). Wraps data/make_cmip7_cond.py.
#
# Reads the downloaded input4MIPs CMIP7 emissions from the flat inputs4mips dir
# (see run_download_input4mips_slurm.sh) and writes, to /scratch:
#     emissions_hist_cmip7_only_timefixed_bc.nc    time 1850-2023
#     emissions_h_cmip7_only_timefixed_bc.nc       time 2024-2100
#     emissions_vl_cmip7_only_timefixed_bc.nc      time 2024-2100
# each with CO2 (cumulative from 1850, incl. aircraft) + SUL + BC (annual), on
# the 192x288 CESM2 grid — i.e. the same normalized space as the CMIP6 training
# cond files, so existing checkpoints can be evaluated on them without retraining.
#
# EXISTING CMIP6 COND FILES ARE NOT TOUCHED (different filenames).
#
# Check inputs are present before doing any real work:
#     bash run_make_cmip7_cond.sh --dry-run
#
# Usage on LUMI (from the repo dir, after `git pull`):
#     bash run_make_cmip7_cond.sh
#     bash run_make_cmip7_cond.sh --hist-end 2021       # splice at scenario start
#     bash run_make_cmip7_cond.sh --scenarios h         # one scenario only
# If the login node blocks singularity, wrap it:
#     srun --account=project_462001328 --partition=small --time=01:00:00 \
#          --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
#          bash run_make_cmip7_cond.sh
set -euo pipefail

DATA_ROOT=/scratch/project_462001328/emulator_data
INPUT_DIR="${EMUL_INPUT_DIR:-${DATA_ROOT}/emission_data/inputs4mips}"
OUTPUT_DIR="${EMUL_OUTPUT_DIR:-${DATA_ROOT}}"

# Grid template: any file already on the 192x288 CESM2 f09 grid. The existing
# cond files are themselves on that grid, so they serve as the template.
TARGET="${TARGET:-${DATA_ROOT}/emissions_hist_only_timefixed_bc.nc}"

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

_VENV_SITE=$(realpath /projappl/project_462001328/venvs/diffesm_laif 2>/dev/null \
             || echo /projappl/project_462001328/venvs/diffesm_laif)/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1

echo "[cmip7-cond] INPUT_DIR  = ${INPUT_DIR}"
echo "[cmip7-cond] OUTPUT_DIR = ${OUTPUT_DIR}"
echo "[cmip7-cond] TARGET     = ${TARGET}"
echo "[cmip7-cond] SIF        = ${SIF}"
echo

if [[ ! -f "${TARGET}" ]]; then
    echo "[cmip7-cond] ERROR: grid template not found: ${TARGET}"
    echo "[cmip7-cond] Pass another file on the 192x288 CESM2 grid via TARGET=..."
    exit 1
fi

singularity exec "${SIF}" python data/make_cmip7_cond.py \
    --target     "${TARGET}" \
    --input-dir  "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

echo
echo "[cmip7-cond] done."
echo "[cmip7-cond] NOTE: h/vl have NO CESM2 reference output, so eval_aero.py can"
echo "[cmip7-cond]       only run them model-only (no truth to score against), and"
echo "[cmip7-cond]       they are OOD scenarios — eval fits a FRESH per-scenario PCA"
echo "[cmip7-cond]       basis for them (eval_aero.py:2307), as it does for ssp126."
