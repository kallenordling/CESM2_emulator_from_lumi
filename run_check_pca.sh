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
# Cheap CPU-only validation of the PCA-persistence fix (commit 851b840).
# Runs check_pca_persistence.py inside the LAIF singularity container with the
# project venv injected. Reads the aaer cond file straight from real /scratch
# (no /tmp staging, no GPU) and diffs the SUL channel at 2015 between the
# eval-before-fix path (no PCA) and the persisted 5-EOF basis.
#
# Usage on LUMI (from the repo dir, after `git pull`):
#     bash run_check_pca.sh
# If the login node blocks singularity, wrap it:
#     srun --account=${LUMI_ACCOUNT} --partition=debug --time=10 \
#          --nodes=1 --ntasks=1 bash run_check_pca.sh
set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# Inject the project venv's site-packages into the container (matches run_debug_aero.sh).
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1

echo "[check] SIF=${SIF}"
echo "[check] venv=${_VENV_SITE}"
echo "[check] running check_pca_persistence.py …"

singularity exec "${SIF}" python check_pca_persistence.py
