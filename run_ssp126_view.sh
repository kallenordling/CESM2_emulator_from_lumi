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
# CPU-only diagnostic: show how the MODEL sees ssp126 emissions, isolating the
# aaer-PCA-basis artifact (ssp126 is never trained, so eval routes it through the
# aaer basis — see eval_aero.py:1907 and commit 851b840). Runs
# diag_ssp126_model_view.py inside the LAIF singularity container with the project
# venv injected. Reads the ssp126 + aaer cond files straight from real /scratch
# (no /tmp staging, no GPU).
#
# Usage on LUMI (from the repo dir, after `git pull`):
#     bash run_ssp126_view.sh
# If the login node blocks singularity, wrap it:
#     srun --account=${LUMI_ACCOUNT} --partition=debug --time=15 \
#          --nodes=1 --ntasks=1 bash run_ssp126_view.sh
#
# Outputs (written to CWD):
#     ssp126_model_view_timeseries.png   global-mean of each stage vs year
#     ssp126_model_view_<VAR>.png        maps @2015/2050/2100 (norm|own|aaer|diff)
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

echo "[ssp126-view] SIF=${SIF}"
echo "[ssp126-view] venv=${_VENV_SITE}"
echo "[ssp126-view] running diag_ssp126_model_view.py …"

# Pass through any extra args (e.g. --ref-scenario aaer, --out-prefix foo).
singularity exec "${SIF}" python diag_ssp126_model_view.py "$@"
