#!/bin/bash
#SBATCH --job-name=forcing_consistency
#SBATCH --output=forcing_consistency_%j.out
#SBATCH --error=forcing_consistency_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
assert_account
lumi_env_banner
#
# Run diag_forcing_consistency.py inside the LUMI container — verify the
# single-forcing cond files match the combined hist+ssp370 forcing:
#   aaer SUL  vs  hist(≤2014)+ssp370(≥2015) SUL
#   ghg  CO2  vs  hist(≤2014)+ssp370(≥2015) CO2
#
# Usage (LUMI):
#   sbatch run_forcing_consistency.sh                  # 8 evenly-spaced columns
#   sbatch run_forcing_consistency.sh --decades        # one map column per decade
#   bash   run_forcing_consistency.sh --decades        # run on the current node
#
# Any extra args (--decades, --n-cols N, --out-prefix X, ...) pass straight
# through to the python script. Outputs land in PROJECT_DIR:
#   forcing_consistency_SUL_maps.png / _timeseries.png   (aaer vs hist+ssp370)
#   forcing_consistency_CO2_maps.png / _timeseries.png   (ghg  vs hist+ssp370)

set -euo pipefail

# ── Modules ───────────────────────────────────────────────────────────────────
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# ── Container ────────────────────────────────────────────────────────────────
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] Using: ${SIF}"

# ── Project + data paths ─────────────────────────────────────────────────────
PROJECT_DIR=${LUMI_REPO}
EMU_DIR="${EMU_DIR:-${LUMI_DATA}}"

# ── Inject host venv into container ──────────────────────────────────────────
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1
echo "[VENV] SINGULARITYENV_PYTHONPATH=${SINGULARITYENV_PYTHONPATH}"
echo "[DATA] EMU_DIR=${EMU_DIR}"
echo "[ARGS] $*"

# ── Run ──────────────────────────────────────────────────────────────────────
singularity exec \
    --bind ${LUMI_PROJAPPL} \
    --bind ${LUMI_SCRATCH} \
    "${SIF}" \
    bash -c "
        cd ${PROJECT_DIR}
        echo '[INSIDE CONTAINER]'; pwd
        python diag_forcing_consistency.py --emu-dir '${EMU_DIR}' $*
    "
