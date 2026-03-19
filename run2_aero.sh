#!/bin/bash
#SBATCH --job-name=diffusion_aero
#SBATCH --account=project_462001328
#SBATCH --partition=standard-g
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=40:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --requeue

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container ─────────────────────────────────────────────────
# Replaces: module use /appl/local/csc/modulefiles && module load LUMI pytorch
# CSC PyTorch no longer works with Slingshot after LUMI service break 21.1.2026.
# Check available versions: ls /appl/local/laifs/containers/lumi-multitorch-full-*.sif
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] Using: ${SIF}"

# Inject our venv's site-packages into the container via SINGULARITYENV_PYTHONPATH.
# Regular PYTHONPATH is ignored/overridden by the container's own environment;
# SINGULARITYENV_* variables are guaranteed to be set inside singularity.
_VENV_SITE=$(realpath /projappl/project_462001328/venvs/diffesm_laif 2>/dev/null \
             || echo /projappl/project_462001328/venvs/diffesm_laif)/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
echo "[VENV] SINGULARITYENV_PYTHONPATH=${SINGULARITYENV_PYTHONPATH}"

# ── Networking ────────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn

# ── Multi-node Slingshot performance fix ─────────────────────────────────────
# https://github.com/lumi-ai-factory/laifs-container-recipes/issues/18
# Preloads the host libfabric so RCCL uses the Slingshot 11 network correctly.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libfabric.so.1

# ── Python / Hydra ────────────────────────────────────────────────────────────
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

# ── ROCm / HIP ───────────────────────────────────────────────────────────────
export ACCELERATE_USE_FSDP=0
export HSA_ENABLE_SDMA=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE=0  # ROCm Triton missing cluster_dims — inductor crash

# ── MIOpen / HIP kernel cache ─────────────────────────────────────────────────
# Strategy: copy the persistent DB to each node's /tmp at startup, run with
# the local copy (no Lustre concurrent-write races), then copy back from the
# head node after training so the next job starts with cached kernels.
PERSISTENT_CACHE="${SLURM_SUBMIT_DIR}/.miopen_cache"
mkdir -p "${PERSISTENT_CACHE}"

export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
export MIOPEN_FIND_ENFORCE=2

# Seed each node's local cache from the persistent store, then train locally.
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c "
    mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}
    cp ${PERSISTENT_CACHE}/* /tmp/miopen_${SLURM_JOB_ID}/ 2>/dev/null || true
"

# ── Launch eval watcher as a background SLURM job ────────────────────────────
# Submits watch_eval_triggers.sh on the small partition so it can call sbatch
# to dispatch eval jobs when the trainer writes trigger files.
# Runs outside the container (no GPU needed, just needs sbatch access).
WATCHER_JOB=$(sbatch --job-name=eval_watcher \
       --account=project_462001112 \
       --partition=small \
       --time=48:00:00 \
       --ntasks=1 --cpus-per-task=1 --mem=256M \
       --chdir="${SLURM_SUBMIT_DIR}" \
       --output="${SLURM_SUBMIT_DIR}/logs/eval_watcher_%j.out" \
       "${SLURM_SUBMIT_DIR}/watch_eval_triggers.sh" 2>/dev/null | awk '{print $NF}') || WATCHER_JOB=""
echo "[watcher] Submitted eval watcher job ${WATCHER_JOB:-FAILED}"

# ── Launch ────────────────────────────────────────────────────────────────────
NUM_PROCESSES=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))
MAIN_PROCESS_IP=$(hostname -i)

RUN_CMD="singularity exec ${SIF} bash -c '
    accelerate launch \
        --config_file=accelerate_config.yaml \
        --num_processes=${NUM_PROCESSES} \
        --num_machines=${SLURM_NNODES} \
        --machine_rank=\${SLURM_NODEID} \
        --main_process_ip=${MAIN_PROCESS_IP} \
        main_aero.py
'"

srun bash -c "$RUN_CMD" || true

# ── Save benchmarked MIOpen kernels back to persistent store ─────────────────
# Copy from head node's /tmp back to Lustre so next job skips re-benchmarking.
cp /tmp/miopen_${SLURM_JOB_ID}/*.ufdb.* "${PERSISTENT_CACHE}/" 2>/dev/null || true
cp /tmp/miopen_${SLURM_JOB_ID}/*.db     "${PERSISTENT_CACHE}/" 2>/dev/null || true

# ── Cancel watcher when training finishes (walltime, crash, or clean exit) ───
if [[ -n "${WATCHER_JOB}" ]]; then
    echo "[watcher] Training finished — cancelling eval watcher job ${WATCHER_JOB}"
    scancel "${WATCHER_JOB}" 2>/dev/null || true
fi
