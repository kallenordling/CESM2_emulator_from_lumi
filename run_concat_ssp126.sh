#!/bin/bash

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
# Rebuild the ssp126 conditioning file with the CO2 fix (drops the spurious
# concat_and_regrid.py:84 "+ hist_endpoint" ramp). Runs
# data/concat_and_regrid_ssp126.py inside the LAIF singularity container with the
# project venv injected. CPU-only; needs xesmf (regridding) in the venv/container.
#
# Reads the annual CO2/SO2 + hist files from real /scratch and writes (by default,
# with a '_co2fix' suffix so the live cond file is NOT clobbered):
#     emissions_co2_so2_regridded_ssp126_co2fix.nc
#     emissions_ssp126_only_timefixed_co2fix.nc
#
# Usage on LUMI (from the repo dir, after `git pull`):
#     bash run_concat_ssp126.sh
#     # custom grid template / overwrite live names:
#     bash run_concat_ssp126.sh --target /path/to/cesm2_grid.nc --out-suffix ""
# If the login node blocks singularity, wrap it:
#     srun --account=${LUMI_ACCOUNT} --partition=debug --time=20 \
#          --nodes=1 --ntasks=1 bash run_concat_ssp126.sh
set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# Inject the project venv's site-packages into the container (matches run_check_pca.sh).
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1

# Default grid template: the existing regridded ssp126 file is already on the
# CESM2 grid (only its lat/lon are read). Override with --target.
DEFAULT_TARGET=${LUMI_DATA}/emissions_ssp126_only_timefixed.nc

# If the caller didn't pass --target, supply the default.
case " $* " in
    *" --target "*) TARGET_ARGS=() ;;
    *)              TARGET_ARGS=(--target "${DEFAULT_TARGET}") ;;
esac

echo "[concat-ssp126] SIF=${SIF}"
echo "[concat-ssp126] venv=${_VENV_SITE}"
echo "[concat-ssp126] default target=${DEFAULT_TARGET}"
echo "[concat-ssp126] running data/concat_and_regrid_ssp126.py …"

singularity exec "${SIF}" python data/concat_and_regrid_ssp126.py "${TARGET_ARGS[@]}" "$@"
