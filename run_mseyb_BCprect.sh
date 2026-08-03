#!/bin/bash
#SBATCH --job-name=mseyb_BCprect
#SBATCH --account=project_462001328
#
# ── mseyb + BC/PRECT A/B (2026-08-03) — completes the 2×2 factorial ─────────
# ssp370 warm+wet bias investigation (see memory gainfix_ssp370_persistent_bias.md).
# Four cells:
#   run_mseyb              : mse_only=true, year_bias=1.0, 2 cond / 1 target — CLEAN
#   run_gainfix             : full aux losses+SGAIN+LR decay, 3 cond / 2 target — BIASED
#   run_gainfix_noBCprect   : full aux losses+SGAIN+LR decay, 2 cond / 1 target — pending
#   run_mseyb_BCprect (HERE): mse_only=true, year_bias=1.0, 3 cond / 2 target
#
# This is run_mseyb's exact training philosophy (mse_only=true — NO TCRE/SGAIN/
# interaction/EBM losses at all, cond_loss_scaling forced 0 in
# _update_cond_scaling, unetTrainer.py:832-850 — plus year_bias=1.0 sampling,
# constant LR — lr_decay stays "off", the config_aero.yaml default) with BC
# cond channel + PRECT target channel added back via
# configs/config_data_ybias_BCprect.yaml. model.{in,out,cond}_channels are
# LEFT AT THE CONFIG_AERO.YAML DEFAULT (2/2/3) — matches this data config, no
# override needed, but note the earlier run_gainfix_noBCprect launch crashed
# from a similar channel-count mismatch (missing overrides going the OTHER
# direction), so double-check the startup log confirms cond_channels=3 model
# built successfully before trusting a long unattended chain.
#
# If this run reproduces the ssp370 warm bias → BC/PRECT contributes
# regardless of loss config (channels are causal). If it stays clean like
# run_mseyb → channels are not the driver under EITHER loss regime, and the
# bias is specific to the aux-loss/SGAIN/LR-decay machinery itself.
#
# FRESH RUN — save_name=run_mseyb_BCprect.pt, no fork (run_mseyb's checkpoints
# are 1/1/2-channel, incompatible conv shapes).
# Fire:  CHAIN_REMAINING=6 sbatch run_mseyb_BCprect.sh
# Isolated arm — own watcher/PROD_RUN name, doesn't touch run_mseyb or
# run_gainfix's production chains/checkpoints.
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container (identical to run2_gainfix.sh) ─────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] Using: ${SIF}"

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

# ── MIOpen / HIP kernel cache ─────────────────────────────────────────────────
# Separate cache dir — 3/2-channel conv shapes match production run_gainfix's,
# so this COULD share its cache, but keep isolated to avoid any cross-run
# write races between concurrent A/B arms.
PERSISTENT_CACHE="${SLURM_SUBMIT_DIR}/.miopen_cache_mseyb_BCprect"
mkdir -p "${PERSISTENT_CACHE}"

export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
export HIP_CACHE_PATH=/tmp/hip_${SLURM_JOB_ID}
export MIOPEN_FIND_ENFORCE=1

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c "
    mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}
    cp ${PERSISTENT_CACHE}/* /tmp/miopen_${SLURM_JOB_ID}/ 2>/dev/null || true
"

# ── Stage training data to /tmp on each node ─────────────────────────────────
# BOTH target-var trees (TREFHT + PRECT) + the *_bc.nc cond files (contain
# BC) — same staging shape as run2_gainfix.sh, unlike run2_mseyb.sh's original
# TREFHT-only / non-bc staging.
SRC_DATA_ROOT=/scratch/project_462001328/emulator_data
LOCAL_DATA_ROOT=/tmp/emulator_data_${SLURM_JOB_ID}

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c "
    set -euo pipefail
    echo \"[stage] node \$(hostname): copying training data to /tmp …\"
    t0=\$(date +%s)
    for var in TREFHT PRECT; do
        mkdir -p ${LOCAL_DATA_ROOT}/training_data/\${var}
        cp -r ${SRC_DATA_ROOT}/training_data/\${var}/hist    ${LOCAL_DATA_ROOT}/training_data/\${var}/
        cp -r ${SRC_DATA_ROOT}/training_data/\${var}/ssp370  ${LOCAL_DATA_ROOT}/training_data/\${var}/
        cp -r ${SRC_DATA_ROOT}/training_data/\${var}/AAER    ${LOCAL_DATA_ROOT}/training_data/\${var}/
        cp -r ${SRC_DATA_ROOT}/training_data/\${var}/GHG     ${LOCAL_DATA_ROOT}/training_data/\${var}/
    done
    cp ${SRC_DATA_ROOT}/emissions_hist_only_timefixed_bc.nc    ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_ssp370_only_timefixed_bc.nc  ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_aaer_only_timefixed_bc.nc    ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_ghg_only_timefixed_bc.nc     ${LOCAL_DATA_ROOT}/
    echo \"[stage] node \$(hostname): done in \$((\$(date +%s)-t0))s, size=\$(du -sh ${LOCAL_DATA_ROOT} | awk '{print \$1}')\"
"

# ── Launch eval watcher as a background SLURM job ────────────────────────────
WATCHER_TIME=$(squeue -h -j "${SLURM_JOB_ID}" -o '%l' 2>/dev/null | tr -d '[:space:]' || true)
[[ -z "${WATCHER_TIME}" || "${WATCHER_TIME}" == "UNLIMITED" ]] && WATCHER_TIME="06:00:00"
EXISTING_WATCHER=$(squeue -u "$(whoami)" --name=eval_watcher_mseyb_BCprect -t PENDING,RUNNING \
                   --noheader -o '%i' 2>/dev/null | head -1 || true)
if [[ -n "${EXISTING_WATCHER}" ]]; then
    WATCHER_JOB=""
    echo "[watcher] eval watcher ${EXISTING_WATCHER} already active — not resubmitting"
else
    WATCHER_JOB=$(sbatch --job-name=eval_watcher_mseyb_BCprect \
           --account=project_462001112 \
           --partition=small \
           --time="${WATCHER_TIME}" \
           --ntasks=1 --cpus-per-task=1 --mem=256M \
           --export="ALL,PROD_RUN=run_mseyb_BCprect" \
           --chdir="${SLURM_SUBMIT_DIR}" \
           --output="${SLURM_SUBMIT_DIR}/logs/eval_watcher_mseyb_BCprect_%j.out" \
           "${SLURM_SUBMIT_DIR}/watch_eval_triggers.sh" 2>/dev/null | awk '{print $NF}') || WATCHER_JOB=""
    echo "[watcher] Submitted eval watcher job ${WATCHER_JOB:-FAILED} (time=${WATCHER_TIME})"
fi

# ── Self-chaining ──────────────────────────────────────────────────────────────
CHAIN_REMAINING="${CHAIN_REMAINING:-6}"
if [[ "${CHAIN_REMAINING}" -gt 1 ]]; then
    NEXT_JOB=$(sbatch --parsable \
           --dependency="afterany:${SLURM_JOB_ID}" \
           --export="ALL,CHAIN_REMAINING=$(( CHAIN_REMAINING - 1 ))" \
           --chdir="${SLURM_SUBMIT_DIR}" \
           "${SLURM_SUBMIT_DIR}/run_mseyb_BCprect.sh" 2>/dev/null) || NEXT_JOB=""
    echo "[chain] queued next link ${NEXT_JOB:-FAILED} (afterany:${SLURM_JOB_ID}, CHAIN_REMAINING=$(( CHAIN_REMAINING - 1 )))"
else
    echo "[chain] CHAIN_REMAINING=${CHAIN_REMAINING} — final link, not resubmitting"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
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
        data_config=config_data_ybias_BCprect.yaml \
        model.in_channels=2 \
        model.out_channels=2 \
        model.cond_channels=3 \
        trainer.hyperparameters.save_name=run_mseyb_BCprect.pt \
        trainer.hyperparameters.mse_only=true
'"

srun bash -c "$RUN_CMD" || true

# ── Save benchmarked MIOpen kernels back to persistent store ─────────────────
cp /tmp/miopen_${SLURM_JOB_ID}/*.ufdb.* "${PERSISTENT_CACHE}/" 2>/dev/null || true
cp /tmp/miopen_${SLURM_JOB_ID}/*.db     "${PERSISTENT_CACHE}/" 2>/dev/null || true

# ── Clean up staged training data from each node's /tmp ──────────────────────
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "rm -rf ${LOCAL_DATA_ROOT} 2>/dev/null || true" || true

# ── Cancel watcher when training finishes ─────────────────────────────────────
if [[ -n "${WATCHER_JOB}" ]]; then
    echo "[watcher] Training finished — cancelling eval watcher job ${WATCHER_JOB}"
    scancel "${WATCHER_JOB}" 2>/dev/null || true
fi
