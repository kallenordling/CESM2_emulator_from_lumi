#!/bin/bash
#SBATCH --job-name=dashboard
#SBATCH --output=dashboard_%j.out
#SBATCH --error=dashboard_%j.err
#SBATCH --time=00:20:00
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

# Run plot_training_dashboard.py inside the LUMI container with LUMI-native paths
# (the script's own defaults point at the off-cluster /mnt mounts).
#
# Usage:
#   bash  run_dashboard.sh [run_name]       # quick, on a login node
#   sbatch run_dashboard.sh [run_name]
# e.g. bash run_dashboard.sh run_sensfix
#      bash run_dashboard.sh run_sensfix_b12
# Output: training_dashboard_<run>.png in the project dir.

RUN=${1:-run_sensfix}

# ── Modules / container ──────────────────────────────────────────────────────
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] Using: ${SIF}"

# ── Paths (LUMI-native) ──────────────────────────────────────────────────────
PROJECT_DIR=${LUMI_REPO}
LOG_DIR=${PROJECT_DIR}/logs
EVAL_DIR=${LUMI_EVAL_OUT}

# ── Inject host venv into container ──────────────────────────────────────────
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1

echo "[INFO] run=${RUN}"
echo "[INFO] logs=${LOG_DIR}"
echo "[INFO] evals=${EVAL_DIR}"

# ── Run ──────────────────────────────────────────────────────────────────────
singularity exec \
    --bind ${LUMI_PROJAPPL} \
    --bind ${LUMI_SCRATCH} \
    "${SIF}" \
    bash -c "
        cd ${PROJECT_DIR}
        python plot_training_dashboard.py \
            --run ${RUN} \
            --log-dir ${LOG_DIR} \
            --eval-dir ${EVAL_DIR} \
            --out ${PROJECT_DIR}/training_dashboard_${RUN}.png
    "

echo "[done] training_dashboard_${RUN}.png"
