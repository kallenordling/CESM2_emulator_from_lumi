#!/bin/bash
# -----------------------------------------------------------------------------
# Open-burning training set (branch train-openburning). One FRESH run_asinh.sh
# chain on the mmlin pipeline (smooth sigma 4 -> min/max anchors on
# hist+ssp370, no PCA, no clip) with two changes:
#
#   * cond SUL/BC = anthro + OPEN BURNING  (*_bc_co2fix_bb.nc,
#     data/add_openburning_cond.py)
#   * training scenarios hist, ssp370, aaer, ghg + CESM2-SF bmb, xaer
#     (configs/config_data_ybias_BCprect_nopca_bb.yaml)
#
# Baseline = the mmlin arm at matched epochs, ranked on SAMPLED metrics.
#
# PREREQUISITES on ${LUMI_DATA} (checked below, the job would otherwise die
# in staging): the six _bb cond files + pi1850_const/ssp126/ssp245/RAMIP _bb
# for eval, and training_data/{TREFHT,PRECT}/{BMB,xAER}.
#
# Usage (from the LUMI checkout of this branch):
#   bash run_openburning.sh
#   CHAIN_REMAINING=10 bash run_openburning.sh
# RESUME a broken chain:
#   RUN_TAG=openbb SAVE_DIR=/scratch/project_462001765/runs_openbb FRESH=0 CO2FIX=1 \
#   TRAIN_DATA_CONFIG=config_data_ybias_BCprect_nopca_bb.yaml \
#   LUMI_PROJECT=462001765 LUMI_EVAL_PROJECT=462001765 bash lsubmit.sh run_asinh.sh
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export LUMI_PROJECT="${LUMI_PROJECT:-462001765}"
export LUMI_EVAL_PROJECT="${LUMI_EVAL_PROJECT:-${LUMI_PROJECT}}"
export TRAIN_DATA_CONFIG=config_data_ybias_BCprect_nopca_bb.yaml
export CHAIN_REMAINING="${CHAIN_REMAINING:-12}"
export STALE_TRIGGER_SECS="${STALE_TRIGGER_SECS:-1209600}"
TAG="${RUN_TAG:-openbb}"
DATA="/scratch/project_${LUMI_PROJECT}/emulator_data"

missing=0
for s in hist ssp370 aaer ghg bmb xaer ssp126 ssp245; do
    f="${DATA}/emissions_${s}_only_timefixed_bc_co2fix_bb.nc"
    [[ -s "$f" ]] || { echo "MISSING $f" >&2; missing=1; }
done
for v in TREFHT PRECT; do for e in BMB xAER; do
    d="${DATA}/training_data/${v}/${e}"
    n=$(ls -d "${d}"/*/ 2>/dev/null | wc -l)
    [[ "$n" -gt 0 ]] || { echo "MISSING $d (no members)" >&2; missing=1; }
done; done
[[ "$missing" == 0 ]] || { echo "prerequisites missing; not submitting" >&2; exit 1; }

save_dir="/scratch/project_${LUMI_PROJECT}/runs_${TAG}"
if compgen -G "${save_dir}/run_${TAG}_co2fix_*.pt" >/dev/null; then
    echo "[$TAG] checkpoints already exist in ${save_dir}; not starting it FRESH" >&2
    exit 1
fi
mkdir -p "${save_dir}"
RUN_TAG="${TAG}" SAVE_DIR="${save_dir}" FRESH=1 \
    bash lsubmit.sh run_asinh.sh --job-name="${TAG}"
