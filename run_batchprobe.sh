#!/bin/bash
#SBATCH --job-name=batchprobe
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
assert_account
lumi_env_banner
#
# Batch-size headroom probe for the run_sensfix config (low-t TCRE ON).
# Sweeps several per-GPU batch sizes; for each it runs ONE short epoch and
# reports FITS / FAILED(OOM). Mirrors the real training memory footprint:
# same 8-GPU DDP, same model, low-t TCRE forward enabled (config default).
#
# We already know batch_size=8 fits (the live run_sensfix run), so the sweep
# probes LARGER sizes. The first one that FAILED marks the ceiling.
#
# Submit:   sbatch run_batchprobe.sh
#   custom: BATCHES="12 16 20" sbatch run_batchprobe.sh
# Read:     grep -E 'FITS|FAILED' logs/batchprobe_<jobid>.out
#
# Isolated: save_name=run_batchprobe.pt (never checkpoints — save_every huge),
# temp per-size data configs are cleaned up. Does NOT touch run_sensfix.

set -uo pipefail   # NOT -e: a probe that OOMs must not abort the sweep
mkdir -p logs

# ── Container + venv (same as run_debug_aero.sh) ─────────────────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"

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
export MIOPEN_FIND_ENFORCE=1
mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}

# ── Stage data to /tmp once (reused by every probe) ──────────────────────────
SRC_DATA_ROOT=${LUMI_DATA}
LOCAL_DATA_ROOT=/tmp/emulator_data_${SLURM_JOB_ID}
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c "
    set -euo pipefail
    mkdir -p ${LOCAL_DATA_ROOT}/training_data/TREFHT
    for s in hist ssp370 AAER GHG; do
        cp -r ${SRC_DATA_ROOT}/training_data/TREFHT/\$s ${LOCAL_DATA_ROOT}/training_data/TREFHT/
    done
    for f in hist ssp370 aaer ghg; do
        cp ${SRC_DATA_ROOT}/emissions_\${f}_only_timefixed.nc ${LOCAL_DATA_ROOT}/
    done
    echo \"[stage] node \$(hostname): staged \$(du -sh ${LOCAL_DATA_ROOT} | awk '{print \$1}')\"
"

NUM_PROCESSES=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))
MAIN_PROCESS_IP=$(hostname -i)
BATCHES="${BATCHES:-12 16 20 24 32}"
PORT=29510

echo "[probe] sweeping per-GPU batch sizes: ${BATCHES}  (8 already known to fit)"
for B in ${BATCHES}; do
    echo "=================================================================="
    echo "[probe] >>> batch_size=${B}"
    echo "=================================================================="
    # Temp data config: this batch_size + short epoch (few realization switches
    # → few batches → OOM, if any, still hits on the first sync step).
    TMPCFG="config_data_probe_${SLURM_JOB_ID}_${B}.yaml"
    sed -e "s/^batch_size:.*/batch_size: ${B}/" \
        -e "s/^steps_per_realization:.*/steps_per_realization: 8/" \
        configs/config_data.yaml > "configs/${TMPCFG}"

    PORT=$((PORT + 1))
    RUN_CMD="singularity exec --bind ${LOCAL_DATA_ROOT}:${SRC_DATA_ROOT} ${SIF} bash -c '
        accelerate launch \
            --config_file=accelerate_config.yaml \
            --num_processes=${NUM_PROCESSES} \
            --num_machines=${SLURM_NNODES} \
            --machine_rank=\${SLURM_NODEID} \
            --main_process_ip=${MAIN_PROCESS_IP} \
            --main_process_port=${PORT} \
            main_aero.py \
            data_config=${TMPCFG} \
            trainer.hyperparameters.batch_size=${B} \
            trainer.hyperparameters.save_name=run_batchprobe.pt \
            trainer.hyperparameters.save_every=100000 \
            trainer.hyperparameters.max_epochs=1
    '"

    if srun bash -c "$RUN_CMD"; then
        echo "[probe] RESULT batch_size=${B}: ✅ FITS"
    else
        echo "[probe] RESULT batch_size=${B}: ❌ FAILED (likely OOM — see traceback above)"
    fi
    rm -f "configs/${TMPCFG}"
    sleep 5
done

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "rm -rf ${LOCAL_DATA_ROOT} 2>/dev/null || true" || true

echo "[probe] sweep done. Summary:"
grep -E '\[probe\] RESULT' "logs/batchprobe_${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "[probe] (largest FITS = your batch ceiling with the low-t pass)"
