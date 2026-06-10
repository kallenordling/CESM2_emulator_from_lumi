#!/bin/bash
# CPU-only: plot emulator vs CESM2 global-mean anomaly from eval-generated
# NetCDFs inside the LAIF singularity container (project venv injected). Wraps
# plot_from_nc.py — no GPU, just xarray/numpy/matplotlib.
#
# Usage on LUMI (from the repo dir, after `git pull`); all args pass through:
#   bash run_plot_from_nc.sh <output_dir> <exp1> [exp2 ...] [--var TREFHT] [--out FILE] [--no-spread]
# e.g.
#   bash run_plot_from_nc.sh \
#       /scratch/project_462001328/eval_output/manual/ep0852_v2 ssp126 ssp245 ssp370
#
# Writes <output_dir>/timeseries_<exps>.png (or --out).
# If the login node blocks singularity, wrap it:
#   srun --account=project_462001328 --partition=debug --time=10 \
#        --nodes=1 --ntasks=1 bash run_plot_from_nc.sh <output_dir> <exp1> [exp2 ...]
set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# Inject the project venv's site-packages into the container (matches run_check_pca.sh).
_VENV_SITE=$(realpath /projappl/project_462001328/venvs/diffesm_laif 2>/dev/null \
             || echo /projappl/project_462001328/venvs/diffesm_laif)/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1

echo "[plot-nc] SIF=${SIF}"
echo "[plot-nc] running plot_from_nc.py $* …"

singularity exec "${SIF}" python plot_from_nc.py "$@"
