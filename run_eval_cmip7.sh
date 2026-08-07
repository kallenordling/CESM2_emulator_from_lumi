#!/bin/bash
#SBATCH --job-name=eval_cmip7
#SBATCH --account=project_462001328
# small-g partial-node (1 GCD) — this eval is only 3 experiments (~330 sampled
# years total) and is NOT sharded across ranks like run_eval_aero.sh, so one GPU
# is enough and a 1-GCD request backfills almost immediately.
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#
# Model-only evaluation of a checkpoint on the CMIP7 h / vl scenarios.
# There is NO CESM2 reference under CMIP7 forcing, so this produces projections
# and diagnostics only — no bias/skill numbers (none are computable).
#
# Submit from the repo dir on LUMI (after `git pull`):
#     sbatch run_eval_cmip7.sh
#
# Env overrides:
#     CHECKPOINT=runs/run_gainfix_1055.pt sbatch run_eval_cmip7.sh
#     OUTPUT_DIR=/scratch/.../eval_output/cmip7_ep1055 sbatch run_eval_cmip7.sh
#     MEMBERS=1 SAMPLE_STEPS=25 sbatch run_eval_cmip7.sh        # fast smoke test
#     MODEL_CONFIG=configs/config_aero_noBCprect.yaml \
#       DATA_CONFIG=configs/config_data_noBCprect.yaml sbatch run_eval_cmip7.sh
#     SCENARIOS="h" sbatch run_eval_cmip7.sh
#
# Requires the cond files from data/make_cmip7_cond.py:
#     bash run_make_cmip7_cond.sh
set -euo pipefail
mkdir -p logs

SCRATCH=/scratch/project_462001328/emulator_data
CHECKPOINT="${CHECKPOINT:-}"                 # empty -> newest in runs/
MODEL_CONFIG="${MODEL_CONFIG:-configs/config_aero.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/config_data.yaml}"
COND_DIR="${COND_DIR:-${SCRATCH}}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_462001328/eval_output/cmip7}"
SCENARIOS="${SCENARIOS:-h vl}"
MEMBERS="${MEMBERS:-5}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TARGET_VAR="${TARGET_VAR:-TREFHT}"

# ── LUMI AI Factory container (mirrors run_eval_aero.sh) ─────────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] Using: ${SIF}"

_VENV_SITE=/projappl/project_462001328/venvs/diffesm_laif/lib/python3.12/site-packages
_EXTRA_PKGS=/scratch/project_462001328/python_packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}:${_EXTRA_PKGS}"

export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

export MIOPEN_USER_DB_PATH=/tmp/miopen_cmip7_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_cmip7_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_cmip7_${SLURM_JOB_ID}
export MIOPEN_FIND_ENFORCE=2
mkdir -p /tmp/miopen_cmip7_${SLURM_JOB_ID} /tmp/hip_cmip7_${SLURM_JOB_ID}

mkdir -p "${OUTPUT_DIR}"

echo "[cmip7-eval] CHECKPOINT   = ${CHECKPOINT:-<newest in runs/>}"
echo "[cmip7-eval] MODEL_CONFIG = ${MODEL_CONFIG}"
echo "[cmip7-eval] DATA_CONFIG  = ${DATA_CONFIG}"
echo "[cmip7-eval] COND_DIR     = ${COND_DIR}"
echo "[cmip7-eval] OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "[cmip7-eval] SCENARIOS    = ${SCENARIOS}"
echo "[cmip7-eval] MEMBERS=${MEMBERS} SAMPLE_STEPS=${SAMPLE_STEPS} BATCH=${BATCH_SIZE}"
echo

# Fail fast and clearly if the cond files were never built.
for f in emissions_hist_cmip7_only_timefixed_bc.nc \
         $(for s in ${SCENARIOS}; do echo "emissions_${s}_cmip7_only_timefixed_bc.nc"; done); do
    if [[ ! -f "${COND_DIR}/${f}" ]]; then
        echo "[cmip7-eval] ERROR: missing cond file ${COND_DIR}/${f}"
        echo "[cmip7-eval] Build them first:  bash run_make_cmip7_cond.sh"
        exit 1
    fi
done

_ckpt_arg=()
[[ -n "${CHECKPOINT}" ]] && _ckpt_arg=(--checkpoint "${CHECKPOINT}")

srun --unbuffered singularity exec "${SIF}" python eval_cmip7.py \
    "${_ckpt_arg[@]}" \
    --model-config "${MODEL_CONFIG}" \
    --data-config  "${DATA_CONFIG}" \
    --cond-dir     "${COND_DIR}" \
    --output-dir   "${OUTPUT_DIR}" \
    --scenarios    ${SCENARIOS} \
    --members      "${MEMBERS}" \
    --sample-steps "${SAMPLE_STEPS}" \
    --batch-size   "${BATCH_SIZE}" \
    --target-var   "${TARGET_VAR}"

echo
echo "[cmip7-eval] done — outputs in ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}" || true
