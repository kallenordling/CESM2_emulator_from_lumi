#!/bin/bash
# CPU-only: compute decadal means from an eval-generated TREFHT_<exp>.nc inside
# the LAIF singularity container (project venv injected). Wraps
# decadal_means_from_nc.py — no GPU, just xarray/numpy.
#
# Usage on LUMI (from the repo dir, after `git pull`); all args pass through:
#   bash run_decadal_means.sh <output_dir> <experiment> [--var TREFHT] [--out FILE]
# e.g.
#   bash run_decadal_means.sh \
#       /scratch/project_462001328/eval_output/manual/ep0852_v2 ssp126
#
# Writes <output_dir>/<exp>_decadal_gmean.csv and <output_dir>/<exp>_decadal_anom.nc.
# If the login node blocks singularity, wrap it:
#   srun --account=project_462001328 --partition=debug --time=10 \
#        --nodes=1 --ntasks=1 bash run_decadal_means.sh <output_dir> <experiment>
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

echo "[decadal] SIF=${SIF}"
echo "[decadal] running decadal_means_from_nc.py $* …"

singularity exec "${SIF}" python decadal_means_from_nc.py "$@"
