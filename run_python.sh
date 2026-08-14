#!/bin/bash
#SBATCH --job-name=diag_view
#SBATCH --output=diag_view_%j.out
#SBATCH --error=diag_view_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

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
assert_account
lumi_env_banner

set -euo pipefail

# Usage:
# sbatch run_container_python.sh diag_cond_model_view.py aaer
# bash run_container_python.sh diag_cond_model_view.py aaer

if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_script> [scenario]"
    exit 1
fi

SCRIPT=$1
SCENARIO=${2:-aaer}

# ── Modules ───────────────────────────────────────────────────────────────────
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# ── Container ────────────────────────────────────────────────────────────────
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] Using: ${SIF}"

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_DIR=${LUMI_REPO}

# ── Inject host venv into container ──────────────────────────────────────────
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages

export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"

echo "[VENV] SINGULARITYENV_PYTHONPATH=${SINGULARITYENV_PYTHONPATH}"

# ── Networking ───────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn

# ── Slingshot fix ────────────────────────────────────────────────────────────
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libfabric.so.1

# ── Python / Hydra ───────────────────────────────────────────────────────────
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

# ── Diagnostics ──────────────────────────────────────────────────────────────
echo "[INFO] Script: ${SCRIPT}"
echo "[INFO] Scenario: ${SCENARIO}"

# ── Run ──────────────────────────────────────────────────────────────────────
singularity exec \
    --bind ${LUMI_PROJAPPL} \
    "${SIF}" \
    bash -c "
        cd ${PROJECT_DIR}

        echo '[INSIDE CONTAINER]'
        pwd

        python ${SCRIPT} --scenario ${SCENARIO}
    "
