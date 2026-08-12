#!/bin/bash

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
# CPU-only: download CMIP7 gridded emissions (BC, SO2, CO2) from ESGF input4MIPs,
# inside the LAIF singularity container (project venv injected). Wraps
# download_input4mips_cmip7.py.
#
# The downloader SKIPS files that already exist and resumes partial .part files,
# so this is safe to re-run and only fetches what's missing. Unlike the LENS2
# scripts it DOES exit non-zero when any file fails, so this wrapper retries up
# to PASSES times and then fails loudly if anything is still missing.
#
# Default set (~12 GB, 24 files, verified on ESGF 2026-08-06):
#   historical  CEDS-CMIP-2025-04-18    1750-2023, 6 files/species
#   scenarios   IIASA-IAMC-h-1-1-0      2022-2100, 1 file/species  (high)
#               IIASA-IAMC-vl-1-1-0     2022-2100, 1 file/species  (very low)
#
# Writes to:
#   ${DATA_ROOT}/input4mips/CMIP7/<target_mip>/<source_id>/<variable_id>/*.nc
#
# Usage on LUMI (from the repo dir, after `git pull`):
#   bash run_download_input4mips.sh [PASSES]     # default 3 passes
#
# See what ESGF publishes right now (new scenarios appear over time):
#   bash run_download_input4mips.sh --list-sources
#
# Dry run:
#   bash run_download_input4mips.sh --discover-only
#
# 12 GB over HTTPS; run inside a batch/interactive alloc, NOT on the login node:
#   srun --account=${LUMI_ACCOUNT} --partition=small --time=04:00:00 \
#        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
#        bash run_download_input4mips.sh
set -euo pipefail

DATA_ROOT=${LUMI_DATA}
OUTDIR="${DATA_ROOT}/input4mips"

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# Inject the project venv's site-packages into the container (matches run_download_data.sh).
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1

# Pass-through mode: any flag argument goes straight to the python script and
# runs a single pass (used for --list-sources / --discover-only).
if [[ "${1:-}" == -* ]]; then
    exec singularity exec "${SIF}" python download_input4mips_cmip7.py \
        --outdir "${OUTDIR}" "$@"
fi

PASSES="${1:-3}"

echo "[input4mips] PASSES=${PASSES}  OUTDIR=${OUTDIR}"
echo "[input4mips] SIF=${SIF}"

rc=1
for pass in $(seq 1 "${PASSES}"); do
    echo "[input4mips] === pass ${pass}/${PASSES} ==="
    # Don't let a failed pass kill the loop; the retry IS the recovery.
    if singularity exec "${SIF}" python download_input4mips_cmip7.py \
           --outdir "${OUTDIR}"; then
        rc=0
        echo "[input4mips] pass ${pass} completed with no failures"
        break
    fi
    rc=1
    echo "[input4mips] pass ${pass} had failures; retrying …"
    sleep $((pass * 10))
done

if [[ "${rc}" -ne 0 ]]; then
    echo "[input4mips] DOWNLOAD FAILED after ${PASSES} passes — some files are still missing."
    echo "[input4mips] Re-run to resume, or use --discover-only to see what is outstanding."
    exit 1
fi

# Verify: every planned file must exist on disk with a non-trivial size. This
# re-queries ESGF, so it also catches "we never even discovered it" cases.
echo "[input4mips] verifying downloaded tree …"
singularity exec "${SIF}" python download_input4mips_cmip7.py \
    --outdir "${OUTDIR}" --discover-only \
| tee /dev/stderr | awk '
    /^  GET /  { missing++ }
    /^  HAVE / { have++ }
    END {
        printf "[input4mips] VERIFY: %d present, %d missing\n", have+0, missing+0
        if (missing+0 > 0) exit 1
    }'

echo "[input4mips] done — CMIP7 BC/SO2/CO2 emissions complete under ${OUTDIR}"
echo "[input4mips] next: regrid + build cond files (data/make_aerosol_files.py, data/make_co2_files.py)"
