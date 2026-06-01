#!/bin/bash
#SBATCH --job-name=dashboard
#SBATCH --output=dashboard_%j.out
#SBATCH --error=dashboard_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

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
PROJECT_DIR=/projappl/project_462001328/CESM2_emulator_from_lumi
LOG_DIR=${PROJECT_DIR}/logs
EVAL_DIR=/scratch/project_462001328/eval_output

# ── Inject host venv into container ──────────────────────────────────────────
_VENV_SITE=$(realpath /projappl/project_462001328/venvs/diffesm_laif 2>/dev/null \
             || echo /projappl/project_462001328/venvs/diffesm_laif)/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1

echo "[INFO] run=${RUN}"
echo "[INFO] logs=${LOG_DIR}"
echo "[INFO] evals=${EVAL_DIR}"

# ── Run ──────────────────────────────────────────────────────────────────────
singularity exec \
    --bind /projappl/project_462001328 \
    --bind /scratch/project_462001328 \
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
