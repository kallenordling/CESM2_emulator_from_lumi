#!/bin/bash
# Run diag_forcing_consistency.py — verify single-forcing cond files (aaer SUL,
# ghg CO2) match the combined hist+ssp370 forcing the model is trained on.
#
# This is a PURE xarray/numpy/matplotlib diagnostic (no torch/omegaconf), so it
# runs locally against the mounted cond files — no LUMI container needed.
#
# Usage:
#   bash run_forcing_consistency.sh                 # 8 evenly-spaced year columns
#   bash run_forcing_consistency.sh --decades       # one map column per decade
#   EMU_DIR=/mnt/lumi_sc2/emulator_data bash run_forcing_consistency.sh
#   PYTHON=~/miniconda3/envs/plotting/bin/python bash run_forcing_consistency.sh --decades
#
# Any extra args (--decades, --n-cols N, --out-prefix X, --emu-dir DIR, ...) are
# passed straight through to the python script.
#
# Outputs (in the repo dir):
#   forcing_consistency_SUL_maps.png / _timeseries.png   (aaer vs hist+ssp370)
#   forcing_consistency_CO2_maps.png / _timeseries.png   (ghg  vs hist+ssp370)

set -euo pipefail

cd "$(dirname "$0")"

# Python interpreter: override with PYTHON=... if your xarray env isn't the default.
PYTHON="${PYTHON:-python}"

# Where the emissions_*_only_timefixed.nc cond files live (local mount of LUMI scratch).
export EMU_DIR="${EMU_DIR:-/mnt/lumi_sc2/emulator_data}"

echo "[run] PYTHON=${PYTHON}"
echo "[run] EMU_DIR=${EMU_DIR}"
echo "[run] args=$*"

if [ ! -d "${EMU_DIR}" ]; then
    echo "[error] EMU_DIR not found: ${EMU_DIR}" >&2
    echo "        set EMU_DIR=... to the dir holding emissions_*_only_timefixed.nc" >&2
    exit 1
fi

"${PYTHON}" diag_forcing_consistency.py "$@"
