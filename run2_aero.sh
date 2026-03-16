#!/bin/bash
#SBATCH --job-name=diffusion_aero
#SBATCH --account=project_462001328
#SBATCH --partition=standard-g
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=40:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --exclude=nid005270,nid007250,nid005391,nid005671,nid005573,nid006472

set -euo pipefail
mkdir -p logs

module --force purge
module use /appl/local/csc/modulefiles
module load LUMI
module load pytorch
source "/projappl/project_462001328/venvs/diffesm/diffesm/bin/activate"

# ── Networking ────────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn

# ── Python / Hydra ────────────────────────────────────────────────────────────
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

# ── ROCm / HIP ───────────────────────────────────────────────────────────────
export ACCELERATE_USE_FSDP=0
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export TORCH_COMPILE=0
# ── MIOpen / HIP kernel cache ─────────────────────────────────────────────────
# MIOPEN_USER_DB_PATH: persistent across jobs — stores benchmarked kernel
# selections so MIOpen doesn't re-benchmark every restart (very slow on new
# conv shapes). Shared across jobs for the same model; safe to read concurrently.
#
# MIOPEN_CUSTOM_CACHE_DIR / HIP_CACHE_PATH: per-job in /tmp to avoid the
# "Operation not permitted" crash where a previous job's lock files block MIOpen.
PERSISTENT_CACHE="${SLURM_SUBMIT_DIR}/.miopen_cache"
mkdir -p "${PERSISTENT_CACHE}"
export MIOPEN_USER_DB_PATH="${PERSISTENT_CACHE}"
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
# MIOPEN_FIND_ENFORCE=2: use cached kernels when present; benchmark on miss.
export MIOPEN_FIND_ENFORCE=2

# Create per-job cache dirs on every node's local /tmp before any GPU process starts.
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}"

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
       "${SLURM_SUBMIT_DIR}/watch_eval_triggers.sh" | awk '{print $NF}')
echo "[watcher] Submitted eval watcher job ${WATCHER_JOB}"

# ── Launch ────────────────────────────────────────────────────────────────────
NUM_PROCESSES=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))
MAIN_PROCESS_IP=$(hostname -i)

RUN_CMD="accelerate launch \
    --config_file=accelerate_config.yaml \
    --num_processes=$NUM_PROCESSES \
    --num_machines=$SLURM_NNODES \
    --machine_rank=\$SLURM_NODEID \
    --main_process_ip=$MAIN_PROCESS_IP \
    main_aero.py"

srun bash -c "$RUN_CMD" || true

# ── Cancel watcher when training finishes (walltime, crash, or clean exit) ───
if [[ -n "${WATCHER_JOB}" ]]; then
    echo "[watcher] Training finished — cancelling eval watcher job ${WATCHER_JOB}"
    scancel "${WATCHER_JOB}" 2>/dev/null || true
fi