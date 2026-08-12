#!/bin/bash

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
# CPU-only: compare the emulator's CMIP7 global-mean temperature against FaIR,
# inside the LAIF singularity container (project venv injected). Wraps
# scripts/compare_gmean_emulator_vs_fair.py — no GPU, just xarray/pandas/matplotlib.
# Runs in seconds: it only touches the 1-D gmean variables in the eval NetCDFs.
#
# Reads eval_cmip7.py's <VAR>_<experiment>.nc directly, so it works as soon as
# INDIVIDUAL experiments finish — no need to wait for the combined CSV that is
# only written after all three complete.
#
# Writes into the eval output dir (so results sit beside the run they describe):
#   <EVAL_DIR>/gmean_emulator_vs_fair.png
#   <EVAL_DIR>/gmean_comparison.csv
#
# Usage on LUMI (from the repo dir, after `git pull`); extra args pass through:
#   bash run_compare_gmean.sh
#   EVAL_DIR=${LUMI_EVAL_OUT}/cmip7_smoke bash run_compare_gmean.sh
#   bash run_compare_gmean.sh --year-min 2000        # zoom the plot
#   VAR=PRECT bash run_compare_gmean.sh
# If the login node blocks singularity, wrap it:
#   srun --account=${LUMI_ACCOUNT} --partition=debug --time=10 \
#        --nodes=1 --ntasks=1 bash run_compare_gmean.sh
#
# NOTE ON INTERPRETATION: FaIR is a plausibility bracket, NOT truth. It is driven
# by the ScenarioMIP protocol paper's ILLUSTRATIVE CO2 (~25% above the final IAM
# gridded values the emulator sees) and responds to species the emulator has no
# channel for (CH4, N2O, halocarbons, volcanic). No CESM2 CMIP7 run exists, so
# neither side can be scored. Agreement is not validation.
set -euo pipefail

EVAL_DIR="${EVAL_DIR:-${LUMI_EVAL_OUT}/cmip7}"
VAR="${VAR:-TREFHT}"
FAIR_CSV="${FAIR_CSV:-reference_data/fair_cmip7_gsat.csv}"
FAIR_CO2ONLY_CSV="${FAIR_CO2ONLY_CSV:-reference_data/fair_cmip7_gsat_co2only.csv}"
OUT_PNG="${OUT_PNG:-${EVAL_DIR}/gmean_emulator_vs_fair.png}"
OUT_CSV="${OUT_CSV:-${EVAL_DIR}/gmean_comparison.csv}"

if [[ ! -d "${EVAL_DIR}" ]]; then
    echo "[gmean] ERROR: EVAL_DIR not found: ${EVAL_DIR}"
    echo "[gmean] Run the CMIP7 eval first:  sbatch run_eval_cmip7.sh"
    exit 1
fi
_n=$(find "${EVAL_DIR}" -maxdepth 1 -name "${VAR}_*.nc" 2>/dev/null | wc -l)
if [[ "${_n}" -eq 0 ]]; then
    echo "[gmean] ERROR: no ${VAR}_*.nc in ${EVAL_DIR}"
    echo "[gmean] The eval writes one per experiment as it finishes."
    exit 1
fi
for f in "${FAIR_CSV}" "${FAIR_CO2ONLY_CSV}"; do
    [[ -f "${f}" ]] || echo "[gmean] WARNING: missing FaIR reference ${f} (that curve will be skipped)"
done

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1

echo "[gmean] EVAL_DIR = ${EVAL_DIR}  (${_n} ${VAR}_*.nc found)"
echo "[gmean] FaIR ref = ${FAIR_CSV}"
echo "[gmean]          + ${FAIR_CO2ONLY_CSV}"
echo "[gmean] OUT      = ${OUT_PNG}"
echo

singularity exec "${SIF}" python scripts/compare_gmean_emulator_vs_fair.py \
    --eval-dir          "${EVAL_DIR}" \
    --var               "${VAR}" \
    --fair-csv          "${FAIR_CSV}" \
    --fair-co2only-csv  "${FAIR_CO2ONLY_CSV}" \
    --out               "${OUT_PNG}" \
    --csv               "${OUT_CSV}" \
    "$@"

echo
echo "[gmean] done — ${OUT_PNG}"
