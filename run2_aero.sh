#!/bin/bash
#SBATCH --job-name=diffusion_aero
#SBATCH --account=project_462001112
#SBATCH --partition=dev-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module --force purge
module use /appl/local/csc/modulefiles
module load LUMI
module load pytorch
source "/projappl/project_462001112/venvs/diffesm/bin/activate"

# ── Networking ────────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=hsn
export RCCL_ENABLE_SHARP=0

# ── Python / Hydra ────────────────────────────────────────────────────────────
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

# ── ROCm / HIP ───────────────────────────────────────────────────────────────
export ACCELERATE_USE_FSDP=0
export CUDA_LAUNCH_BLOCKING=0
export HSA_ENABLE_SDMA=0
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# ── Per-job private MIOpen/HIP kernel cache ───────────────────────────────────
# Prevents "filesystem error: cannot remove: Operation not permitted" crashes.
# A previous job leaves /tmp/gfx90a6e.HIP.*.ufdb.txt behind; MIOpen tries to
# update it and aborts when it can't. Using $SLURM_JOB_ID gives every job its
# own isolated cache directory on each node's local /tmp.
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
# MIOPEN_FIND_ENFORCE=2: use cached kernels when present; don't crash on miss.
export MIOPEN_FIND_ENFORCE=2
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCH_COMPILE=0
# Create cache dirs on every node's local /tmp before any GPU process starts.
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}"

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

srun bash -c "$RUN_CMD"