#!/bin/bash
#SBATCH --job-name=decomp_ssp126
#SBATCH --account=project_462001328
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#
# Single-GPU ssp126 single-forcing decomposition on dev-g (fast turnaround).
# Submit:
#     sbatch run_decompose_ssp126.sh
#     CHECKPOINT=/path/to/best.pt SAMPLE_STEPS=100 FP32=1 sbatch run_decompose_ssp126.sh
# Output: ssp126_decomp.png + the [DECOMP] table in logs/decomp_ssp126_<jobid>.out

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container (same as run_eval_aero.sh) ─────────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] ${SIF}"

_VENV_SITE=/projappl/project_462001328/venvs/diffesm_laif/lib/python3.12/site-packages
_EXTRA_PKGS=/scratch/project_462001328/python_packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}:${_EXTRA_PKGS}"

export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

# ── ROCm / HIP caches (per-job /tmp) ─────────────────────────────────────────
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
export MIOPEN_FIND_ENFORCE=2
mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}

# Container-internal repo path (matches run_eval_aero.sh).
if [ -d "/pfs/lustrep1/projappl/project_462001328/CESM2_emulator_from_lumi" ]; then
    WORK_DIR=/pfs/lustrep1/projappl/project_462001328/CESM2_emulator_from_lumi
else
    WORK_DIR=/projappl/project_462001328/CESM2_emulator_from_lumi
fi

# ── Options (override via env at submit time) ────────────────────────────────
CHECKPOINT="${CHECKPOINT:-}"          # default: newest in runs/
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SCENARIO="${SCENARIO:-ssp126}"        # ssp126/ssp370/hist/ghg/aaer
CKPT_FLAG=""; [ -n "${CHECKPOINT}" ] && CKPT_FLAG="--checkpoint ${CHECKPOINT}"
FP32_FLAG="";  [ "${FP32:-0}" = "1" ] && FP32_FLAG="--fp32"

PY_ARGS="${CKPT_FLAG} --sample-steps ${SAMPLE_STEPS} ${FP32_FLAG} --scenario ${SCENARIO} --out ${WORK_DIR}/${SCENARIO}_decomp.png"

# Let SLURM bind the single GCD (--gpus-per-task=1); do NOT set ROCR_VISIBLE_DEVICES
# manually — that races the binding and drops ranks to CPU (see run_eval_aero.sh).
srun --ntasks=1 --gpus-per-task=1 --unbuffered \
    bash -c "
        echo \"[BIND] ROCR_VISIBLE_DEVICES=\${ROCR_VISIBLE_DEVICES:-unset} HIP_VISIBLE_DEVICES=\${HIP_VISIBLE_DEVICES:-unset}\"
        singularity exec ${SIF} bash -c 'cd ${WORK_DIR} && python decompose_ssp126.py ${PY_ARGS}'
    "
