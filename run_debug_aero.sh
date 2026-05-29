#!/bin/bash
#SBATCH --job-name=debug_aero
#SBATCH --account=project_462001328
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# Smoke test for the new cond normalisation (1-99 pct, 4-scenario reference)
# AND the unetTrainer refactor (Chunks 1-4). Goal: boot the trainer, run a
# handful of training steps, persist one checkpoint, then exit cleanly.
#
# NO self-chaining. NO eval watcher. Writes to a SEPARATE save_name so it
# never collides with the production run_normfix.pt chain.
#
# Submit:  sbatch run_debug_aero.sh
# Logs:    logs/debug_aero_<jobid>.out
#
# Pass / Fail criteria
#   PASS: log shows "[TRAINER] Multi-experiment mode",
#         "[EPOCH 0] duration: …" appears at least once,
#         a single run_normfix_debug_<N>.pt file lands in runs/,
#         no Python exception in the log tail.
#   FAIL: ImportError / NameError / shape mismatch on first forward pass,
#         missing-file error from the staging step,
#         OOM on the first step.

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container ─────────────────────────────────────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] Using: ${SIF}"

# Inject our venv's site-packages into the container.
_VENV_SITE=$(realpath /projappl/project_462001328/venvs/diffesm_laif 2>/dev/null \
             || echo /projappl/project_462001328/venvs/diffesm_laif)/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
echo "[VENV] SINGULARITYENV_PYTHONPATH=${SINGULARITYENV_PYTHONPATH}"

# ── Networking ────────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libfabric.so.1

# ── Python / Hydra ────────────────────────────────────────────────────────────
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

# ── ROCm / HIP ───────────────────────────────────────────────────────────────
export ACCELERATE_USE_FSDP=0
export HSA_ENABLE_SDMA=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE=0

# ── MIOpen cache (job-local; debug runs don't share with production) ─────────
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
export MIOPEN_FIND_ENFORCE=1
mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}

# ── Stage data to /tmp ───────────────────────────────────────────────────────
# Bind mount in the singularity exec below REPLACES /scratch/.../emulator_data
# with this /tmp directory inside the container, so ONLY files copied here are
# visible to training. Includes all four "_only_" cond files (the new
# EMISSIONS_PATHS reference + config_data feed) renamed correctly (aaer, not aero).
SRC_DATA_ROOT=/scratch/project_462001328/emulator_data
LOCAL_DATA_ROOT=/tmp/emulator_data_${SLURM_JOB_ID}

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c "
    set -euo pipefail
    mkdir -p ${LOCAL_DATA_ROOT}/training_data/TREFHT
    echo \"[stage] node \$(hostname): copying training data to /tmp …\"
    t0=\$(date +%s)
    cp -r ${SRC_DATA_ROOT}/training_data/TREFHT/hist    ${LOCAL_DATA_ROOT}/training_data/TREFHT/
    cp -r ${SRC_DATA_ROOT}/training_data/TREFHT/ssp370  ${LOCAL_DATA_ROOT}/training_data/TREFHT/
    cp -r ${SRC_DATA_ROOT}/training_data/TREFHT/AAER    ${LOCAL_DATA_ROOT}/training_data/TREFHT/
    cp -r ${SRC_DATA_ROOT}/training_data/TREFHT/GHG     ${LOCAL_DATA_ROOT}/training_data/TREFHT/
    # The four _only_ cond files needed by both EMISSIONS_PATHS (1-99 pct ref)
    # and config_data.yaml (per-scenario training cond inputs).
    cp ${SRC_DATA_ROOT}/emissions_hist_only_timefixed.nc    ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_ssp370_only_timefixed.nc  ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_aaer_only_timefixed.nc    ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_ghg_only_timefixed.nc     ${LOCAL_DATA_ROOT}/
    echo \"[stage] node \$(hostname): done in \$((\$(date +%s)-t0))s, size=\$(du -sh ${LOCAL_DATA_ROOT} | awk '{print \$1}')\"
"

# ── Launch ────────────────────────────────────────────────────────────────────
# Hydra overrides:
#   trainer.hyperparameters.save_name=run_normfix_debug.pt   — isolated save bucket
#   trainer.hyperparameters.save_every=5                     — checkpoint after ~5 sync steps
#   trainer.hyperparameters.max_epochs=3                     — bounded run length
NUM_PROCESSES=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))
MAIN_PROCESS_IP=$(hostname -i)

RUN_CMD="singularity exec --bind ${LOCAL_DATA_ROOT}:${SRC_DATA_ROOT} ${SIF} bash -c '
    accelerate launch \
        --config_file=accelerate_config.yaml \
        --num_processes=${NUM_PROCESSES} \
        --num_machines=${SLURM_NNODES} \
        --machine_rank=\${SLURM_NODEID} \
        --main_process_ip=${MAIN_PROCESS_IP} \
        main_aero.py \
        trainer.hyperparameters.save_name=run_normfix_debug.pt \
        trainer.hyperparameters.save_every=5 \
        trainer.hyperparameters.max_epochs=6
'"

srun bash -c "$RUN_CMD" || true

# ── Clean up staged data ─────────────────────────────────────────────────────
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "rm -rf ${LOCAL_DATA_ROOT} 2>/dev/null || true" || true

echo "[debug] done — check runs/run_normfix_debug_*.pt and the log tail for errors."
