#!/bin/bash
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
#     srun --account=project_462001328 --partition=debug --time=10 \
#          --nodes=1 --ntasks=1 bash run_ssp126_start_test.sh --test-a-only
#
# ── Run FULL A+B (needs a GPU; gpu-small node) ───────────────────────────────
#     srun --account=project_462001328 --partition=small-g --gpus-per-node=1 \
#          --time=20 --nodes=1 --ntasks=1 \
#          bash run_ssp126_start_test.sh \
#          --checkpoint /projappl/project_462001328/CESM2_emulator_from_lumi/runs/run_mseyb_701.pt
#
# Any args after the script name are forwarded to the python script
# (--checkpoint, --seed, --sample-steps, --test-a-only, --members).
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

echo "[ssp126-test] SIF=${SIF}"
echo "[ssp126-test] venv=${_VENV_SITE}"
echo "[ssp126-test] args=$*"
echo "[ssp126-test] running diag_ssp126_start_test.py …"

singularity exec "${SIF}" python diag_ssp126_start_test.py "$@"
