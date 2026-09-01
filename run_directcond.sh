#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# A/B ARM: conditioning by DIRECT INPUT INJECTION ONLY.
#
# Identical to run_mseyb_BCprect.sh in every respect except one flag,
# model.cond_mode=direct, so any difference in the result is attributable to
# the conditioning architecture and nothing else. Same data config, same
# mse_only training, same channel counts, same node/GPU layout.
#
# WHAT CHANGES IN THE MODEL: the SpatialCondEncoder and every per-pixel FiLM
# projection are gone. The cond map is projected once and added to x before the
# first convolution, and that is the model's only sight of the forcing.
#
# WHAT IT IS TESTING: FiLM makes the network a joint, entangled function of
# (CO2, SUL, BC) at every scale — strong where training covers the joint
# distribution, unconstrained where it does not. ssp126 and ssp245 fall in that
# gap (bias +0.34 and +0.61 degC against under 0.06 for every trained
# scenario). Additive single-point injection is a weaker hypothesis class that
# may extrapolate more predictably. Expect WORSE in-distribution skill; the
# question is whether the unseen scenarios improve enough to be worth it.
#
# READ THE RESULT ON: ssp126/ssp245 bias, and the additivity residual R, which
# under this architecture should be closer to CESM2's +0.445 K if the entangled
# FiLM was inflating it.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=directcond
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
#
# 2026-08-19: runs on project 462001328 (lumi_env.sh's default). The cond files
# come from configs/config_data_ybias_BCprect.yaml, now repointed at the
# CO2/BC-corrected `*_bc_co2fix.nc` set. See the "Fresh vs resume" block below
# before launching — the existing checkpoints predate that correction.
# Fire:  FRESH=1 CHAIN_REMAINING=6 sbatch run_directcond.sh   (clean start)
#        CHAIN_REMAINING=6 sbatch run_directcond.sh           (resume)
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

# Single source of truth for the LUMI project id and its paths.
# Under sbatch, BASH_SOURCE points at /var/spool/slurmd/job<N>/slurm_script —
# SLURM copies the script there — so the plain dirname form cannot find
# lumi_env.sh. It then failed OPEN: assert_account and lumi_env_banner were
# "command not found", every LUMI_* var stayed unset, and job 21369490 ran with
# the account guard silently absent and a PYTHONPATH pointing at the wrong
# project's venv. Same fix as commit 4121985 on monthly-temporal.
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

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container (identical to run2_gainfix.sh) ─────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] Using: ${SIF}"

_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
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
SRC_DATA_ROOT=${LUMI_DATA}
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
EXISTING_WATCHER=$(squeue -u "$(whoami)" --name=eval_watcher_directcond -t PENDING,RUNNING \
                   --noheader -o '%i' 2>/dev/null | head -1 || true)
if [[ -n "${EXISTING_WATCHER}" ]]; then
    WATCHER_JOB=""
    echo "[watcher] eval watcher ${EXISTING_WATCHER} already active — not resubmitting"
else
    # ${LUMI_ACCOUNT} rather than a literal: this is a runtime sbatch, so the
    # variable DOES expand here (unlike an #SBATCH directive, which SLURM never
    # expands — see lumi_env.sh). Hardcoding 462001112 sent the watcher to a
    # different project from the training job.
    WATCHER_JOB=$(sbatch --job-name=eval_watcher_directcond \
           --account="${LUMI_ACCOUNT}" \
           --partition=small \
           --time="${WATCHER_TIME}" \
           --ntasks=1 --cpus-per-task=1 --mem=256M \
           --export="ALL,PROD_RUN=run_directcond" \
           --chdir="${SLURM_SUBMIT_DIR}" \
           --output="${SLURM_SUBMIT_DIR}/logs/eval_watcher_directcond_%j.out" \
           "${SLURM_SUBMIT_DIR}/watch_eval_triggers.sh" 2>/dev/null | awk '{print $NF}') || WATCHER_JOB=""
    echo "[watcher] Submitted eval watcher job ${WATCHER_JOB:-FAILED} (time=${WATCHER_TIME})"
fi

# ── Fresh vs resume ───────────────────────────────────────────────────────────
# config_aero.yaml sets load_path:"newest", so a bare launch RESUMES the newest
# run_mseyb_BCprect_*.pt. As of 2026-08-19 those checkpoints (…_490 … _509) were
# trained on the PRE-FIX conditioning: ssp370/ghg cumulative CO2 doubled and
# historical BC on CEDS-2025. The cond files this script now reads are the
# corrected ones.
#
# Resuming across that change is NOT a neutral continuation. The checkpoint
# carries baked COND_NORM constants and per-scenario PCA bases fitted on the OLD
# cond distribution, and config_aero.yaml:152 re-injects them on resume — so the
# run would normalise corrected data with stale statistics and project it onto a
# basis fitted to a CO2 axis that was stretched ~1.4x. Weights also encode the
# old CO2 sensitivity.
#
#   FRESH=1 sbatch run_directcond.sh   → from scratch, own checkpoint name
#   sbatch run_directcond.sh           → resume (only sensible for a chain
#                                            ALREADY started on the fixed data)
# TWO flags, because the chain re-submits with --export=ALL and a single flag
# would propagate: FRESH=1 on every link would restart from scratch every 6h.
#   FRESH=1  → this link only: load_path=0. The chain clears it.
#   CO2FIX=1 → sticky: use the corrected-data checkpoint NAME. Set implicitly by
#              FRESH=1 and passed down the chain so later links resume the run
#              the first link started rather than the pre-fix one.
FRESH="${FRESH:-0}"
CO2FIX="${CO2FIX:-0}"
[[ "${FRESH}" == "1" ]] && CO2FIX=1
export CO2FIX

if [[ "${CO2FIX}" == "1" ]]; then
    SAVE_NAME="run_directcond_co2fix.pt"
else
    SAVE_NAME="run_directcond.pt"
fi
if [[ "${FRESH}" == "1" ]]; then
    LOAD_OVERRIDE="trainer.hyperparameters.load_path=0"
    echo "[fresh] FRESH=1 — training from scratch into ${SAVE_NAME}"
else
    LOAD_OVERRIDE=""
    echo "[fresh] resuming newest ${SAVE_NAME%.pt}_*.pt (FRESH=1 for a clean start)"
fi

# ── Self-chaining ──────────────────────────────────────────────────────────────
CHAIN_REMAINING="${CHAIN_REMAINING:-6}"
if [[ "${CHAIN_REMAINING}" -gt 1 ]]; then
    NEXT_JOB=$(sbatch --parsable \
           --dependency="afterany:${SLURM_JOB_ID}" \
           --export="ALL,CHAIN_REMAINING=$(( CHAIN_REMAINING - 1 )),FRESH=0,CO2FIX=${CO2FIX:-0}" \
           --chdir="${SLURM_SUBMIT_DIR}" \
           "${SLURM_SUBMIT_DIR}/run_directcond.sh" 2>/dev/null) || NEXT_JOB=""
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
        model.cond_mode=direct \
        trainer.hyperparameters.save_name=${SAVE_NAME} \
        ${LOAD_OVERRIDE} \
        trainer.hyperparameters.mse_only=true \
        trainer.hyperparameters.eval_data_config=configs/config_data_ybias_BCprect.yaml
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
