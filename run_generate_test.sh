#!/bin/bash
#SBATCH --job-name=generate_test
# small-g partial-node (1 GCD). generate_test.py now sweeps EVERY year of hist
# (1850-2014) and ssp370 (2015-2100) — 251 years — sampling N_SAMPLES
# realizations per year, batched into one forward pass per diffusion step.
#
# Cost: years x SAMPLE_STEPS batched forwards. At the ~1.5 s/forward measured in
# the CMIP7 eval that is
#     251 x 100 steps ~ 25,100 forwards ~ 8-10 h
#     251 x  50 steps ~ 12,550 forwards ~ 4-5  h   (SAMPLE_STEPS=50 is what
#                                                   eval_cmip7.py uses)
# 12 h is set to cover the 100-step case with margin. Drop SAMPLE_STEPS to 50 in
# trainer/generate_test.py and this halves.
#
# NOTE: the script writes its NetCDF only at the END — a timeout loses the whole
# run. Prefer finishing inside the limit over relying on a resubmit.
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
assert_account
lumi_env_banner
#
# Run trainer/generate_test.py — sample an ensemble of stochastic realizations
# from a trained checkpoint for one scenario / one conditioning window, and
# write them to NetCDF in physical units (TREFHT degC, PRECT mm/day).
#
# Submit from the repo dir on LUMI (after `git pull`):
#     sbatch run_generate_test.sh
#
# SETTINGS LIVE IN THE PYTHON FILE, not here — generate_test.py has no CLI.
# Edit the block at the top of trainer/generate_test.py to change:
#     CHECKPOINT, SCENARIO, CONDITIONING_INDEX, N_SAMPLES, SAMPLE_STEPS, OUTPUT
# SCENARIO must be one of the data config's experiment_configs entries
# (hist / ssp370 / aaer / ghg) — the CMIP7 h and vl scenarios are NOT in there,
# so use run_eval_cmip7.sh for those.
#
# OUTPUT is a relative path, so by default it lands in the repo on /projappl.
# Point it at ${LUMI_SCRATCH}/... to keep generated data off the
# project filesystem.
set -euo pipefail
mkdir -p logs

# SLURM starts in the submit directory; the script needs the REPO ROOT as cwd
# because its paths are relative ("configs", "runs/...").
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

if [[ ! -f trainer/generate_test.py ]]; then
    echo "[gen] ERROR: trainer/generate_test.py not found in $(pwd)"
    echo "[gen] Submit from the repo root."
    exit 1
fi

# Fail before burning a GPU slot if the hardcoded checkpoint is missing.
_ckpt=$(grep -m1 '^CHECKPOINT' trainer/generate_test.py | cut -d'"' -f2)
if [[ -n "${_ckpt}" && ! -f "${_ckpt}" ]]; then
    echo "[gen] ERROR: checkpoint from generate_test.py not found: ${_ckpt}"
    echo "[gen] Newest available:"
    ls -t runs/*.pt 2>/dev/null | head -5 | sed 's/^/    /'
    exit 1
fi

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# The repo root is appended to PYTHONPATH so `from data.multi_experiment_dataset
# import ...` resolves: running `python trainer/generate_test.py` puts trainer/
# on sys.path, NOT the repo root.
_VENV_SITE=${LUMI_VENV}/lib/python3.12/site-packages
_EXTRA_PKGS=${LUMI_PKGS}
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}:${_EXTRA_PKGS}:$(pwd)"
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1

export MIOPEN_USER_DB_PATH=/tmp/miopen_gen_${SLURM_JOB_ID:-$$}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_gen_${SLURM_JOB_ID:-$$}
export HIP_CACHE_PATH=/tmp/hip_gen_${SLURM_JOB_ID:-$$}
export MIOPEN_FIND_ENFORCE=2
mkdir -p "${MIOPEN_USER_DB_PATH}" "${HIP_CACHE_PATH}"

echo "[gen] cwd        = $(pwd)"
echo "[gen] checkpoint = ${_ckpt}"
grep -E '^(SCENARIO|CONDITIONING_INDEX|N_SAMPLES|SAMPLE_STEPS|OUTPUT) ' \
     trainer/generate_test.py | sed 's/^/[gen] /'
echo "[gen] SIF        = ${SIF}"
echo

srun --unbuffered singularity exec "${SIF}" python trainer/generate_test.py

_out=$(grep -m1 '^OUTPUT' trainer/generate_test.py | cut -d'"' -f2)
echo
echo "[gen] done"
[[ -f "${_out}" ]] && ls -la "${_out}"
