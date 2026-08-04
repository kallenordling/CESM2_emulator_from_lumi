#!/bin/bash
#SBATCH --job-name=gainfix_noBCprect
#SBATCH --account=project_462001328
#
# ── Rigorous single-variable BC/PRECT-causality A/B (2026-08-02) ─────────────
# The earlier run_sensfix-vs-run_mseyb comparison used to argue BC/PRECT don't
# cause run_gainfix's persistent ssp370 warm+wet bias was RETRACTED (see memory
# gainfix_ssp370_persistent_bias.md) — mseyb differs from gainfix in 5+
# variables at once (mse_only, year_bias data config, SGAIN presence, LR decay,
# BC/PRECT), so it could not isolate BC/PRECT's contribution. The one rigorous
# test so far is the null-BC INFERENCE A/B (ep1041/ep1172, same checkpoint, BC
# zeroed only at eval time) — bias survived unchanged — but that only rules out
# the BC *input*, not PRECT (an output channel, can't be nulled the same way),
# and not any interaction between training WITH vs WITHOUT those channels from
# epoch 0 (e.g. shared-backbone capacity/gradient competition effects that a
# post-hoc null can't see).
#
# This run is the true single-variable control: IDENTICAL to run2_gainfix.sh
# in every trainer.hyperparameters setting (sampled_gain, lr_decay, mse_only=
# false, adaptive loss scaling, TCRE/interaction config) and IDENTICAL
# scenario_weights/year_bias/bsp_depth in the data config — the ONLY change is
# configs/config_data_noBCprect.yaml (TREFHT-only target, CO2+SUL-only cond)
# plus the matching model.{in,out,cond}_channels overrides below. bc_clip_mode
# is dropped (no BC channel to clip). ALSO required (both have fail-loud
# guards in trainer/unetTrainer.py that killed every link of the first launch
# attempt, 2026-08-02, jobs 20595116-20596900, zero epochs trained): dropping
# a 2/2/3-channel default config_aero.yaml onto a 1/1/2-channel model needs
# cfg_bc_drop_prob=0.0 (default 0.3 indexes a nonexistent BC channel,
# unetTrainer.py:1495) and target_var_weights=[1.0] (default [1.0,0.5] has 2
# entries vs the model's 1 output channel, unetTrainer.py:1587).
#
# ALSO required, found 2026-08-04 after the fixed run trained 448 epochs with
# ZERO successful auto-evals: the auto-eval trigger pipeline (_spawn_eval →
# eval_triggers/*.json → watch_eval_triggers.sh → sbatch run_eval_aero.sh)
# didn't know about --model-config/--data-config and was evaluating every
# checkpoint against the DEFAULT 2/2/3-channel config → state_dict shape
# mismatch, every time. Fixed by adding eval_model_config/eval_data_config
# hyperparameters (configs/config_aero.yaml) that _spawn_eval threads through
# the trigger JSON and watch_eval_triggers.sh forwards as MODEL_CONFIG/
# DATA_CONFIG env vars. Set below — takes effect on the NEXT trained epoch
# after this fix lands (git pull), not retroactively on already-trained
# checkpoints (re-eval those manually, see run_eval_aero.sh header).
#
# FRESH RUN, cannot fork from any existing checkpoint — channel counts differ
# (1/1/2 here vs 2/2/3 in run_gainfix), so conv shapes are incompatible.
# save_name=run_gainfix_noBCprect.pt starts from scratch.
#
# Compare against run_gainfix's OWN ssp370 tail-GMbias trajectory at matched
# epochs (already measured, see gainfix_ssp370_persistent_bias.md /
# global_mean_anomaly.csv history): bias is already clearly present by
# ep0170-0200 (+0.16..+0.27 tail degC) and stays in the +0.2..+0.4 range
# through ep0900. A few hundred epochs here is enough for a comparable signal
# — do NOT need to run to ep1000+ before checking.
#
# Fire:  CHAIN_REMAINING=6 sbatch run_gainfix_noBCprect.sh
# Isolated arm — no watcher/chain conflict with production run_gainfix (own
# PROD_RUN name below), doesn't touch the production chain or its checkpoints.
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
# Separate cache dir from run2_gainfix.sh: this run has different conv shapes
# (1/1/2 channels vs 2/2/3), so a shared cache would just miss every lookup.
PERSISTENT_CACHE="${SLURM_SUBMIT_DIR}/.miopen_cache_noBCprect"
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
# TREFHT tree ONLY (no PRECT — config_data_noBCprect.yaml single-tree data_dir).
# Cond files unchanged (BC variable present in file but unused, see
# config_data_noBCprect.yaml header — ClimateDataset selects cond_vars only).
SRC_DATA_ROOT=/scratch/project_462001328/emulator_data
LOCAL_DATA_ROOT=/tmp/emulator_data_${SLURM_JOB_ID}

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c "
    set -euo pipefail
    echo \"[stage] node \$(hostname): copying training data to /tmp …\"
    t0=\$(date +%s)
    mkdir -p ${LOCAL_DATA_ROOT}/training_data/TREFHT
    cp -r ${SRC_DATA_ROOT}/training_data/TREFHT/hist    ${LOCAL_DATA_ROOT}/training_data/TREFHT/
    cp -r ${SRC_DATA_ROOT}/training_data/TREFHT/ssp370  ${LOCAL_DATA_ROOT}/training_data/TREFHT/
    cp -r ${SRC_DATA_ROOT}/training_data/TREFHT/AAER    ${LOCAL_DATA_ROOT}/training_data/TREFHT/
    cp -r ${SRC_DATA_ROOT}/training_data/TREFHT/GHG     ${LOCAL_DATA_ROOT}/training_data/TREFHT/
    cp ${SRC_DATA_ROOT}/emissions_hist_only_timefixed_bc.nc    ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_ssp370_only_timefixed_bc.nc  ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_aaer_only_timefixed_bc.nc    ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_ghg_only_timefixed_bc.nc     ${LOCAL_DATA_ROOT}/
    echo \"[stage] node \$(hostname): done in \$((\$(date +%s)-t0))s, size=\$(du -sh ${LOCAL_DATA_ROOT} | awk '{print \$1}')\"
"

# ── Launch eval watcher as a background SLURM job ────────────────────────────
WATCHER_TIME=$(squeue -h -j "${SLURM_JOB_ID}" -o '%l' 2>/dev/null | tr -d '[:space:]' || true)
[[ -z "${WATCHER_TIME}" || "${WATCHER_TIME}" == "UNLIMITED" ]] && WATCHER_TIME="06:00:00"
EXISTING_WATCHER=$(squeue -u "$(whoami)" --name=eval_watcher_noBCprect -t PENDING,RUNNING \
                   --noheader -o '%i' 2>/dev/null | head -1 || true)
if [[ -n "${EXISTING_WATCHER}" ]]; then
    WATCHER_JOB=""
    echo "[watcher] eval watcher ${EXISTING_WATCHER} already active — not resubmitting"
else
    WATCHER_JOB=$(sbatch --job-name=eval_watcher_noBCprect \
           --account=project_462001112 \
           --partition=small \
           --time="${WATCHER_TIME}" \
           --ntasks=1 --cpus-per-task=1 --mem=256M \
           --export="ALL,PROD_RUN=run_gainfix_noBCprect" \
           --chdir="${SLURM_SUBMIT_DIR}" \
           --output="${SLURM_SUBMIT_DIR}/logs/eval_watcher_noBCprect_%j.out" \
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
           "${SLURM_SUBMIT_DIR}/run_gainfix_noBCprect.sh" 2>/dev/null) || NEXT_JOB=""
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
        data_config=config_data_noBCprect.yaml \
        model.in_channels=1 \
        model.out_channels=1 \
        model.cond_channels=2 \
        trainer.hyperparameters.save_name=run_gainfix_noBCprect.pt \
        trainer.hyperparameters.lr_decay=cosine \
        trainer.hyperparameters.lr_decay_horizon_steps=60000 \
        trainer.hyperparameters.sampled_gain_loss_scale=0.05 \
        trainer.hyperparameters.cfg_bc_drop_prob=0.0 \
        trainer.hyperparameters.target_var_weights=[1.0] \
        trainer.hyperparameters.eval_model_config=configs/config_aero_noBCprect.yaml \
        trainer.hyperparameters.eval_data_config=configs/config_data_noBCprect.yaml
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
