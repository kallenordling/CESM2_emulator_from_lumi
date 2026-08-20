#!/bin/bash
# Single source of truth for the LUMI project id and every path derived from it.
#
# Set the project ONCE, here or in your environment:
#
#     export LUMI_PROJECT=462001112        # in ~/.bashrc on LUMI, or
#     LUMI_PROJECT=462001112 sbatch ...    # per submission
#
# and every launcher picks it up. Sourced by the run_*.sh scripts; also exports
# the variables so the python side (lumi_paths.py) and the YAML configs
# (${oc.env:LUMI_SCRATCH,...}) resolve to the same place.
#
# THE ACCOUNT IS SPECIAL. SLURM does NOT expand variables inside #SBATCH lines,
# so `#SBATCH --account=${LUMI_PROJECT}` silently does the wrong thing. The
# account therefore comes from, in order of precedence:
#   1. `sbatch --account=...` on the command line  (what lsubmit.sh does)
#   2. the SBATCH_ACCOUNT environment variable      (exported below)
# and assert_account() re-checks it INSIDE the job, so a mismatch fails in the
# first seconds instead of after the data paths turn out to be unwritable.

# shellcheck disable=SC2155

LUMI_PROJECT="${LUMI_PROJECT:-462001328}"

export LUMI_PROJECT
export LUMI_ACCOUNT="project_${LUMI_PROJECT}"
export LUMI_SCRATCH="/scratch/project_${LUMI_PROJECT}"
export LUMI_PROJAPPL="/projappl/project_${LUMI_PROJECT}"
export LUMI_REPO="${LUMI_PROJAPPL}/CESM2_emulator_from_lumi"
export LUMI_VENV="${LUMI_PROJAPPL}/venvs/diffesm_laif"
export LUMI_PKGS="${LUMI_SCRATCH}/python_packages"
export LUMI_DATA="${LUMI_SCRATCH}/emulator_data"
# Eval output does NOT have to live on LUMI_PROJECT's scratch. It is deliberately
# a separate knob: the training data, cond files and venv are on LUMI_PROJECT
# (462001328), while eval results are collected on LUMI_EVAL_PROJECT so they land
# in one place across runs regardless of which allocation the job billed to.
# Set LUMI_EVAL_PROJECT to LUMI_PROJECT to put them back on the same scratch.
LUMI_EVAL_PROJECT="${LUMI_EVAL_PROJECT:-462001112}"
export LUMI_EVAL_PROJECT
export LUMI_EVAL_OUT="${LUMI_EVAL_OUT:-/scratch/project_${LUMI_EVAL_PROJECT}/eval_output}"

# Container-internal view of projappl. Some launchers need this exact prefix
# because the bind mount inside the singularity image resolves differently.
export LUMI_REPO_PFS="/pfs/lustrep1/projappl/project_${LUMI_PROJECT}/CESM2_emulator_from_lumi"

# Make `sbatch` default to the right account even when called bare.
export SBATCH_ACCOUNT="${LUMI_ACCOUNT}"
export SALLOC_ACCOUNT="${LUMI_ACCOUNT}"

# Prefer the container-internal repo path when it exists (see run_eval_aero.sh).
if [ -d "${LUMI_REPO_PFS}" ]; then
    export LUMI_WORK_DIR="${LUMI_REPO_PFS}"
else
    export LUMI_WORK_DIR="${LUMI_REPO}"
fi

# Abort a running job whose account does not match LUMI_PROJECT. Catches the
# case where the launcher was submitted with a stale SBATCH_ACCOUNT or an
# explicit --account, which would bill (and write) to the wrong project.
assert_account() {
    if [ -n "${SLURM_JOB_ACCOUNT:-}" ] && [ "${SLURM_JOB_ACCOUNT}" != "${LUMI_ACCOUNT}" ]; then
        echo "[lumi_env] ERROR: job account '${SLURM_JOB_ACCOUNT}' != LUMI_ACCOUNT" \
             "'${LUMI_ACCOUNT}' (LUMI_PROJECT=${LUMI_PROJECT})." >&2
        echo "[lumi_env]        Submit with: bash lsubmit.sh <script.sh>" >&2
        echo "[lumi_env]        or export LUMI_PROJECT to match the account." >&2
        exit 1
    fi
    return 0
}

# One-line banner so every job log opens with the resolved configuration; a
# misfire is then visible in the first lines instead of after hours.
lumi_env_banner() {
    echo "[lumi_env] LUMI_PROJECT=${LUMI_PROJECT} account=${LUMI_ACCOUNT}" \
         "scratch=${LUMI_SCRATCH} repo=${LUMI_WORK_DIR}"
}
