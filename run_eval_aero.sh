#!/bin/bash
#SBATCH --job-name=eval_aero
#SBATCH --account=project_462001328
# small-g allows a partial-node allocation (4 GCDs) instead of reserving a whole
# standard-g node, so the job backfills into much smaller idle gaps. Eval runs
# in ~15-20 min (bf16 + 50 sample steps + 4-GPU sharding), so a 1 h request lets
# the backfill scheduler slot it almost immediately rather than hunting for a
# 40 h hole on a full node.
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container ─────────────────────────────────────────────────
# Same setup as run2_aero.sh — CSC PyTorch module no longer works after 21.1.2026.
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] Using: ${SIF}"

# Override any inherited SINGULARITYENV_PYTHONPATH from --export=ALL
_VENV_SITE=/projappl/project_462001328/venvs/diffesm_laif/lib/python3.12/site-packages
_EXTRA_PKGS=/scratch/project_462001328/python_packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}:${_EXTRA_PKGS}"
echo "[VENV] SINGULARITYENV_PYTHONPATH=${SINGULARITYENV_PYTHONPATH}"

# ── Python / Hydra ────────────────────────────────────────────────────────────
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

# ── ROCm / HIP caches ────────────────────────────────────────────────────────
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
export MIOPEN_FIND_ENFORCE=2
mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}

# Use container-internal path if available
if [ -d "/pfs/lustrep1/projappl/project_462001328/CESM2_emulator_from_lumi" ]; then
    WORK_DIR=/pfs/lustrep1/projappl/project_462001328/CESM2_emulator_from_lumi
else
    WORK_DIR=/projappl/project_462001328/CESM2_emulator_from_lumi
fi

# When submitted by the trainer, CHECKPOINT and OUTPUT_DIR are set via --export.
# Fall back to defaults for manual submission.
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_462001328/eval_output}"
# Per-channel CFG scales: < 1.0 reduces CO2 warming, > 1.0 amplifies aerosol cooling.
# Set to 1.0 (default) to disable CFG and use direct conditioning (single forward pass).
GUIDANCE_CO2="${GUIDANCE_CO2:-1.0}"
GUIDANCE_SUL="${GUIDANCE_SUL:-1.0}"
# Set RUN_XAI=1 (or SKIP_XAI=0) to run XAI figures (IG + saliency) — off by default (slow).
RUN_XAI="${RUN_XAI:-0}"
# SKIP_XAI=0 is equivalent to RUN_XAI=1 (inverted alias for convenience)
[ "${SKIP_XAI:-1}" = "0" ] && RUN_XAI="1"
# Set FORCE_CFG=1 to use 3-pass CFG decomposition even at guidance scales 1.0/1.0.
# Useful to isolate whether the additive decomposition itself introduces bias.
FORCE_CFG="${FORCE_CFG:-0}"
# Restrict the eval to specific experiments (space-separated), e.g.
#   EXPERIMENTS=ssp126 sbatch run_eval_aero.sh        # focused re-eval, ~5× faster
#   EXPERIMENTS="hist ssp370 ssp126" sbatch ...        # keep the TCRE comparison
# Empty (default) = all experiments. Passed straight to eval_aero.py --experiments.
# NOTE: with fewer experiments than --ntasks, the extra ranks just idle (the
# cost-balanced sharding leaves their bins empty); the TCRE-comparison plot needs
# hist+ssp370+ssp126 together, so use the 3-name form if you want it.
EXPERIMENTS="${EXPERIMENTS:-}"
# Number of diffusion ensemble members to generate per experiment.
# Default 5: model field is an ensemble MEAN (matched to multi-member CESM2),
# so per-checkpoint skill is no longer dominated by single-realization sampling
# noise. ~5× the sampling time → walltime raised to 4h (see --time above).
# Override with `MEMBERS=1 sbatch ...` for a quick single-draw eval.
MEMBERS="${members:-${MEMBERS:-5}}"

# ── Echo the resolved config up front ────────────────────────────────────────
# Make a misfire obvious in the first lines of the log instead of after a 4h run.
# Env vars must reach the job via `sbatch --export=ALL,CHECKPOINT=...,...` — a
# bare `VAR=val sbatch ...` prefix does NOT reliably propagate on LUMI.
echo "[EVAL-CFG] CHECKPOINT=${CHECKPOINT:-<empty>}"
echo "[EVAL-CFG] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[EVAL-CFG] EXPERIMENTS=${EXPERIMENTS:-<all>}  MEMBERS=${MEMBERS}  GUIDANCE_CO2=${GUIDANCE_CO2} GUIDANCE_SUL=${GUIDANCE_SUL}"
if [ -z "${CHECKPOINT}" ]; then
    echo "[EVAL-CFG] WARNING: CHECKPOINT empty → eval_aero.py will use find_latest (newest run_*.pt)."
    echo "[EVAL-CFG]          If you meant a specific checkpoint, resubmit with:"
    echo "[EVAL-CFG]          sbatch --export=ALL,CHECKPOINT=/path/run_xxx.pt,OUTPUT_DIR=...,EXPERIMENTS=ssp126 run_eval_aero.sh"
fi

_XAI_FLAG=""
[ "${RUN_XAI}" = "1" ] && _XAI_FLAG="--run-xai"
_CFG_FLAG=""
[ "${FORCE_CFG}" = "1" ] && _CFG_FLAG="--force-cfg"
_EXP_FLAG=""
[ -n "${EXPERIMENTS}" ] && _EXP_FLAG="--experiments ${EXPERIMENTS}"

# Multi-GPU experiment sharding: srun launches one task per GPU; each task
# runs experiments_to_run[$SLURM_PROCID::$SLURM_NTASKS] inside eval_aero.py
# (--shard-rank / --n-shards default to $SLURM_PROCID / $SLURM_NTASKS).
# ROCR_VISIBLE_DEVICES pins each task to its local GPU so they don't fight
# over device 0.
if [ -n "${CHECKPOINT}" ]; then
    CKPT_FLAG="--checkpoint ${CHECKPOINT}"
    RUNS_FLAG=""
else
    CKPT_FLAG=""
    RUNS_FLAG="--runs-dir /projappl/project_462001328/CESM2_emulator_from_lumi/runs"
fi

PY_ARGS="${CKPT_FLAG} ${RUNS_FLAG} \
    --output-dir ${OUTPUT_DIR} \
    --sample-steps 50 \
    --batch-size 16 \
    --guidance-co2 ${GUIDANCE_CO2} \
    --guidance-sul ${GUIDANCE_SUL} \
    --members ${MEMBERS} \
    ${_XAI_FLAG} ${_CFG_FLAG} ${_EXP_FLAG}"

# All variables expand here (script-launch time) except $SLURM_LOCALID which
# must be evaluated inside srun's spawned shell — hence the \$.
#
# Device binding: --gpus-per-task=1 already restricts each task to a single
# GCD; the previous `export ROCR_VISIBLE_DEVICES=\${SLURM_LOCALID}` raced with
# that binding and left torch.cuda.is_available()==False on most ranks
# (job 18580141: ranks 0/1/3 fell back to CPU, only rank 2 got a GPU).
# Let SLURM set ROCR_VISIBLE_DEVICES itself; --unbuffered so per-rank stdout
# isn't dropped if a rank crashes mid-step.
srun --ntasks=${SLURM_NTASKS} --ntasks-per-node=${SLURM_NTASKS} --gpus-per-task=1 \
    --unbuffered \
    bash -c "
        echo \"[BIND] rank=\${SLURM_PROCID}/\${SLURM_NTASKS} localid=\${SLURM_LOCALID} ROCR_VISIBLE_DEVICES=\${ROCR_VISIBLE_DEVICES:-unset} HIP_VISIBLE_DEVICES=\${HIP_VISIBLE_DEVICES:-unset}\"
        singularity exec ${SIF} bash -c 'cd ${WORK_DIR} && python eval_aero.py ${PY_ARGS}'
    "

# ── Mirror lightweight summaries to the projappl repo ─────────────────────────
# eval outputs (NetCDF + plots) land in OUTPUT_DIR on /scratch, which is NOT
# mounted off-cluster. Copy just the small artefacts (tcre_summary.json + the
# anomaly/TCRE PNGs) into the repo's eval_output/<basename> so they ride along
# on the projappl mount and can be inspected locally. Best-effort; never fail
# the job over a mirror hiccup.
MIRROR_DST="${WORK_DIR}/eval_output/$(basename "${OUTPUT_DIR}")"
mkdir -p "${MIRROR_DST}" || true
cp -f "${OUTPUT_DIR}/tcre_summary.json" "${MIRROR_DST}/" 2>/dev/null || true
cp -f "${OUTPUT_DIR}"/anomaly_maps_*.png "${MIRROR_DST}/" 2>/dev/null || true
cp -f "${OUTPUT_DIR}"/tcre_*.png         "${MIRROR_DST}/" 2>/dev/null || true
echo "[MIRROR] summaries → ${MIRROR_DST}"
