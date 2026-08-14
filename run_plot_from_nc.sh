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
# CPU-only: plot emulator vs CESM2 global-mean anomaly from eval-generated
# NetCDFs inside the LAIF singularity container (project venv injected). Wraps
# plot_from_nc.py — no GPU, just xarray/numpy/matplotlib.
#
# Usage on LUMI (from the repo dir, after `git pull`); all args pass through:
#   bash run_plot_from_nc.sh <output_dir> <exp1> [exp2 ...] [--var TREFHT] [--out FILE] [--no-spread]
# e.g.
#   bash run_plot_from_nc.sh \
#       ${LUMI_EVAL_OUT}/manual/ep0852_v2 ssp126 ssp245 ssp370
#
# Writes <output_dir>/timeseries_<exps>.png (or --out).
# If the login node blocks singularity, wrap it:
#   srun --account=${LUMI_ACCOUNT} --partition=debug --time=10 \
#        --nodes=1 --ntasks=1 bash run_plot_from_nc.sh <output_dir> <exp1> [exp2 ...]
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

echo "[plot-nc] SIF=${SIF}"
echo "[plot-nc] running plot_from_nc.py $* …"

singularity exec "${SIF}" python plot_from_nc.py "$@"
