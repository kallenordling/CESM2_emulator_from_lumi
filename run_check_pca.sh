#!/bin/bash
# Cheap CPU-only validation of the PCA-persistence fix (commit 851b840).
# Runs check_pca_persistence.py inside the LAIF singularity container with the
# project venv injected. Reads the aaer cond file straight from real /scratch
# (no /tmp staging, no GPU) and diffs the SUL channel at 2015 between the
# eval-before-fix path (no PCA) and the persisted 5-EOF basis.
#
# Usage on LUMI (from the repo dir, after `git pull`):
#     bash run_check_pca.sh
# If the login node blocks singularity, wrap it:
#     srun --account=project_462001328 --partition=debug --time=10 \
#          --nodes=1 --ntasks=1 bash run_check_pca.sh
set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# Inject the project venv's site-packages into the container (matches run_debug_aero.sh).
_VENV_SITE=$(realpath /projappl/project_462001328/venvs/diffesm_laif 2>/dev/null \
             || echo /projappl/project_462001328/venvs/diffesm_laif)/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1

echo "[check] SIF=${SIF}"
echo "[check] venv=${_VENV_SITE}"
echo "[check] running check_pca_persistence.py …"

singularity exec "${SIF}" python check_pca_persistence.py
