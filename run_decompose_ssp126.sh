#!/bin/bash
#SBATCH --job-name=decomp_ssp126
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

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

_VENV_SITE=${LUMI_VENV}/lib/python3.12/site-packages
_EXTRA_PKGS=${LUMI_PKGS}
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
if [ -d "${LUMI_REPO_PFS}" ]; then
    WORK_DIR=${LUMI_REPO_PFS}
else
    WORK_DIR=${LUMI_REPO}
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
