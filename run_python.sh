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
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
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
