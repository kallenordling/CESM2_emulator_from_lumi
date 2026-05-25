#!/bin/bash
#SBATCH --job-name=gmeantest
#SBATCH --account=project_462001328
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#
# ISOLATED A/B test: direct global-mean supervision.
# Diagnosis (model_skill_diagnosis, 2026-05): spatial pattern is perfect but the
# global-mean ΔT is too high at high forcing and OUTSIDE the CESM2 envelope
# (ssp370 ~7σ, ghg ~3σ, ssp126 ~20σ). Ordinary MSE under-weights the global mean
# (a tiny fraction of the per-gridpoint error); the soft TCRE penalty was too
# weak AND too narrow (hist+ssp370 only); a separate (CO2,SUL) head can't work
# (those scalars don't determine ΔT across scenarios).
# This run enables gmean_loss_scaling: a FIXED, strong term that pins the
# predicted field's area-weighted global mean to the TARGET field's global mean,
# per sample, for ALL scenarios (climatology cancels → no precompute). It works
# THROUGH the model (which has the spatial info a scalar head lacks).
# Forks the newest production run_slope-tcre ckpt into save_name=run_gmeantest.
# Everything else at production ([2,2,4,1]). No self-chain.
#
# WATCH (first few epochs): the logged "GMEAN LOSS" magnitude × GMEAN SCALE
# should be comparable-to-larger than "MSE LOSS" (i.e. actually biting). If it's
# negligible, raise gmean_loss_scaling (try 5–10); if training destabilises or
# "ANOM SKILL"/patcorr collapses, lower it.
# After it runs (5-member auto-eval, or manual):
#     CHECKPOINT=.../runs/run_gmeantest_<ep>.pt sbatch run_eval_aero.sh
# Success = ssp370/ghg global mean move toward within ±2σ AND spatial patcorr
# holds (~0.99). ssp126 (OOD) is the generalization test.

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container (same as run2_aero.sh) ─────────────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] ${SIF}"

_VENV_SITE=$(realpath /projappl/project_462001328/venvs/diffesm_laif 2>/dev/null \
             || echo /projappl/project_462001328/venvs/diffesm_laif)/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
echo "[VENV] ${SINGULARITYENV_PYTHONPATH}"

# ── Networking / ROCm (same as run2_aero.sh) ─────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libfabric.so.1
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
export ACCELERATE_USE_FSDP=0
export HSA_ENABLE_SDMA=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE=0
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}

# ── Fork point: newest production checkpoint ─────────────────────────────────
RUNS_DIR="${SLURM_SUBMIT_DIR}/runs"
PROD_CKPT=$(ls -t "${RUNS_DIR}"/run_slope-tcre_*.pt 2>/dev/null | grep -v _best | head -1 || true)
if [[ -z "${PROD_CKPT}" ]]; then
    echo "[fork] no run_slope-tcre_*.pt found in ${RUNS_DIR} — aborting"; exit 1
fi
echo "[fork] forking global-mean-supervision test from ${PROD_CKPT}"

# ── Launch (Hydra overrides; no watcher, no chain) ───────────────────────────
# gmean_loss_scaling=1.0 is a starting value — check the logged GMEAN LOSS in the
# first epochs and retune (5–10 if too weak, lower if it destabilises).
NUM_PROCESSES=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))
MAIN_PROCESS_IP=$(hostname -i)

RUN_CMD="singularity exec ${SIF} bash -c '
    accelerate launch \
        --config_file=accelerate_config.yaml \
        --num_processes=${NUM_PROCESSES} \
        --num_machines=${SLURM_NNODES} \
        --machine_rank=\${SLURM_NODEID} \
        --main_process_ip=${MAIN_PROCESS_IP} \
        main_aero.py \
        trainer.hyperparameters.save_name=run_gmeantest.pt \
        trainer.hyperparameters.load_path=${PROD_CKPT} \
        trainer.hyperparameters.gmean_loss_scaling=1.0
'"

srun bash -c "$RUN_CMD" || true
