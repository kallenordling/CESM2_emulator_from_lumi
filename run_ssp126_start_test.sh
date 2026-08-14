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
# Container wrapper for diag_ssp126_start_test.py — decides why the emulator
# starts ssp126 ~1.5-2 C colder than ssp370 at 2015 by diffing the EXACT tensor
# the model consumes (Test A) and, on a GPU, the model output on those frames
# (Test B). Reuses the real eval_aero pipeline (no reimplementation).
#
# Test A is GPU-free (pure cond diff); Test B needs one GPU for the forward pass.
#
# ── Run Test A ONLY (CPU; login or debug node is fine) ───────────────────────
#     bash run_ssp126_start_test.sh --test-a-only
#   or, if the login node blocks singularity:
#     srun --account=${LUMI_ACCOUNT} --partition=debug --time=10 \
#          --nodes=1 --ntasks=1 bash run_ssp126_start_test.sh --test-a-only
#
# ── Run FULL A+B (needs a GPU; gpu-small node) ───────────────────────────────
#     srun --account=${LUMI_ACCOUNT} --partition=small-g --gpus-per-node=1 \
#          --time=20 --nodes=1 --ntasks=1 \
#          bash run_ssp126_start_test.sh \
#          --checkpoint ${LUMI_REPO}/runs/run_mseyb_852.pt
#
# Any args after the script name are forwarded to the python script
# (--checkpoint, --seed, --sample-steps, --test-a-only, --members).
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

echo "[ssp126-test] SIF=${SIF}"
echo "[ssp126-test] venv=${_VENV_SITE}"
echo "[ssp126-test] args=$*"
echo "[ssp126-test] running diag_ssp126_start_test.py …"

singularity exec "${SIF}" python diag_ssp126_start_test.py "$@"
