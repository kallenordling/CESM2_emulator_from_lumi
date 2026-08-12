#!/bin/bash

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
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
