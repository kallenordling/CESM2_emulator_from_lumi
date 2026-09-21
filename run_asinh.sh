#!/bin/bash
# -----------------------------------------------------------------------------
# ARM: asinh CONDITIONING TRANSFORM.
#
# Identical to run_mseyb_BCprect.sh in every respect except one flag,
# trainer.hyperparameters.cond_transform=asinh, so any difference is
# attributable to the conditioning transform and nothing else. Same data
# config, same mse_only training, same channel counts, same node/GPU layout,
# same baseline architecture (branched from precip-bc, so no FiLM attention and
# no global pyramid).
#
# WHY. Measured 2026-09-08 on the raw field, at the point where the clip
# actually happens -- normalize() runs BEFORE the gaussian smoothing, which
# then hides the plateau:
#
#     BC   E China 100.0% of years pinned at +1, India 100.0%,
#          E US 96.8%, Europe 85.7%
#     SUL  E US 99.2%, Europe 82.5%, E China 74.1%
#     CO2  Europe 54.6%, E US 50.6%, E China 42.2%
#
# The industrial history of the two largest BC sources is not compressed by the
# v1 affine map, it is DESTROYED before the model sees it. No linear anchor
# fixes it: a full-record sweep showed lowering hi raises global span and global
# spatial contrast together while pushing Europe from 55% to 88% pinned, because
# the gain comes from Central Africa, which barely emits.
#
# asinh(v/s) rescaled so p99.5 of the POSITIVE cells maps to +1 is linear below
# s and logarithmic above, so the tail is compressed rather than clipped, and 0
# still maps to -1. Measured on the same record: BC worst-site pinning
# 100% -> 51% and span 13.5% -> 24%; CO2 55% -> 30% and 11.9% -> 47%.
#
# NOT log -- that was tried and made the model worse. And not the rank/quantile
# transform, which scores better on every static metric (0% pinned, span
# 33-70%) but uniformises by frequency, so shipping lanes become as prominent as
# industrial regions and the amplitude ordering the physics rests on is erased.
#
# READ THE RESULT ON: ssp245 TCRE bias and the RAMIP ssp370-126aer final-decade
# pattern correlation (baseline r = -0.049, i.e. no aerosol fingerprint at all).
# Those are the metrics with room to move if the aerosol signal was the thing
# being clipped away. Compare against run_mseyb_BCprect AT THE SAME EPOCH --
# its corrected-data evals start at ep0400.
#
# MUST BE FRESH: the transform changes the meaning of every cond channel, so a
# warm start would feed old weights a differently-scaled input. The mode is
# persisted per checkpoint as COND_TRANSFORM and re-injected at eval.
# PER-CHANNEL: cond_transform also takes a spec, e.g.
#   COND_TRANSFORM="CO2=v1,SUL=asinh,BC=asinh"
# CO2 is CUMULATIVE, so its field only grows and asinh's ceiling saturates it
# harder every decade (93.7% of E China's carbon mass pinned by 2100, up from
# 52.6% at 2014), while SUL and BC are per-year and improve through the century.
# That spec leaves CO2 exactly as the precip-bc branch had it and confines the
# new transform to the two channels it helps.
#
# Fire:  FRESH=1 CHAIN_REMAINING=6 sbatch run_asinh.sh
#        FRESH=1 RUN_TAG=asinh99 COND_TRANSFORM=asinh_aero sbatch run_asinh.sh
#
# COND_TRANSFORM=asinh_aero is the ALIAS for "CO2=v1,SUL=asinh,BC=asinh". Pass
# the alias, never the spec: Hydra's override grammar rejects a value carrying
# "=" and ",", which is what killed job 21896489 before its first step.
# RUN_TAG keeps a new arm off the previous one's checkpoints; it is sticky down
# the chain, so set it on the first submit only.
#
# SAVE_DIR: config_aero.yaml writes checkpoints to runs/ RELATIVE to the submit
# directory, i.e. onto /projappl. That filled up (55G of 54G on 462001328), and
# a full projappl is the likeliest cause of the TRUNCATED run_asinh_co2fix_9.pt
# that ended the first asinh arm -- 335 MB where epoch 4 was 784 MB. Point
# SAVE_DIR at scratch, which has terabytes free:
#   SAVE_DIR=/scratch/project_462001112/runs_asinh99
# -----------------------------------------------------------------------------
#SBATCH --job-name=asinh
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
# Fire:  FRESH=1 CHAIN_REMAINING=6 sbatch run_mseyb_BCprect.sh   (clean start)
#        CHAIN_REMAINING=6 sbatch run_mseyb_BCprect.sh           (resume)
# Isolated arm — own watcher/PROD_RUN name, doesn't touch run_mseyb or
# run_gainfix's production chains/checkpoints.
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# SLURM cannot expand variables in a directive, so this path is literal.
# It used to be logs/, relative to the submit directory on /projappl,
# which is at quota: a failed log write trips set -e and kills the job.
# That is what ended link 4 of the asinh99 chain 1:50 in, having queued
# no successor. Scratch has terabytes free.
#SBATCH --output=/scratch/project_462001112/logs/%x_%j.out

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
PERSISTENT_CACHE="${SLURM_SUBMIT_DIR}/.miopen_cache_asinh"
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
    # The _co2fix set. The plain *_bc.nc files these lines used to name were
    # DELETED 2026-09-07, and every launcher in the repo still names them: the
    # copy only survives where SRC_DATA_ROOT resolves to a scratch that still
    # holds them, which is why run_filmattn.sh (LUMI_PROJECT=462001112) does not
    # trip on it and this one (462001328, the config's own project) died in
    # staging on its first launch under set -euo pipefail.
    cp ${SRC_DATA_ROOT}/emissions_hist_only_timefixed_bc_co2fix.nc    ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_ssp370_only_timefixed_bc_co2fix.nc  ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_aaer_only_timefixed_bc_co2fix.nc    ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_ghg_only_timefixed_bc_co2fix.nc     ${LOCAL_DATA_ROOT}/
    echo \"[stage] node \$(hostname): done in \$((\$(date +%s)-t0))s, size=\$(du -sh ${LOCAL_DATA_ROOT} | awk '{print \$1}')\"
"

# RUN_TAG names the arm: checkpoints run_${RUN_TAG}_co2fix_*.pt and the eval
# watcher's PROD_RUN both derive from it, so they can never drift apart the way
# filmattn's did. Keep it out of a resume and the chain resumes the wrong arm --
# --export=ALL carries it, so only the FIRST submit needs to set it.
RUN_TAG="${RUN_TAG:-asinh}"
SAVE_DIR="${SAVE_DIR:-runs/}"
# TRAIN_DATA_CONFIG selects the data config for training AND its evals (the
# eval reads cond smoothing/PCA from it). Not DATA_CONFIG: the eval watcher
# already uses that name per trigger. Sticky down the chain like RUN_TAG.
TRAIN_DATA_CONFIG="${TRAIN_DATA_CONFIG:-config_data_ybias_BCprect.yaml}"
export RUN_TAG SAVE_DIR TRAIN_DATA_CONFIG

# ── Launch eval watcher as a background SLURM job ────────────────────────────
WATCHER_TIME=$(squeue -h -j "${SLURM_JOB_ID}" -o '%l' 2>/dev/null | tr -d '[:space:]' || true)
[[ -z "${WATCHER_TIME}" || "${WATCHER_TIME}" == "UNLIMITED" ]] && WATCHER_TIME="06:00:00"
EXISTING_WATCHER=$(squeue -u "$(whoami)" --name="eval_watcher_${RUN_TAG:-asinh}" -t PENDING,RUNNING \
                   --noheader -o '%i' 2>/dev/null | head -1 || true)
if [[ -n "${EXISTING_WATCHER}" ]]; then
    WATCHER_JOB=""
    echo "[watcher] eval watcher ${EXISTING_WATCHER} already active — not resubmitting"
else
    # ${LUMI_ACCOUNT} rather than a literal: this is a runtime sbatch, so the
    # variable DOES expand here (unlike an #SBATCH directive, which SLURM never
    # expands — see lumi_env.sh). Hardcoding 462001112 sent the watcher to a
    # different project from the training job.
    WATCHER_JOB=$(sbatch --job-name="eval_watcher_${RUN_TAG:-asinh}" \
           --account="${LUMI_ACCOUNT}" \
           --partition=small \
           --time="${WATCHER_TIME}" \
           --ntasks=1 --cpus-per-task=1 --mem=256M \
           --export="ALL,PROD_RUN=run_${RUN_TAG:-asinh}" \
           --chdir="${SLURM_SUBMIT_DIR}" \
           --output="${SLURM_SUBMIT_DIR}/logs/eval_watcher_${RUN_TAG:-asinh}_%j.out" \
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
#   FRESH=1 sbatch run_mseyb_BCprect.sh   → from scratch, own checkpoint name
#   sbatch run_mseyb_BCprect.sh           → resume (only sensible for a chain
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
    SAVE_NAME="run_${RUN_TAG}_co2fix.pt"
else
    SAVE_NAME="run_${RUN_TAG}.pt"
fi
if [[ "${FRESH}" == "1" ]]; then
    LOAD_OVERRIDE="trainer.hyperparameters.load_path=0"
    echo "[fresh] FRESH=1 — training from scratch into ${SAVE_NAME}"
else
    LOAD_OVERRIDE=""
    echo "[fresh] FRESH=0 — RESUMING newest ${SAVE_NAME%.pt}_*.pt (pass FRESH=1 to start from scratch instead)"
fi

# ── Self-chaining ──────────────────────────────────────────────────────────────
# Under sbatch, $0 is the spool copy, so take the name from the job itself and
# fall back to the literal only if that lookup fails.
SCRIPT_NAME="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
                | sed -n "s|.*Command=.*/\([^/ ]*\.sh\).*|\1|p" | head -1)"
SCRIPT_NAME="${SCRIPT_NAME:-run_asinh.sh}"
echo "[chain] this script: ${SCRIPT_NAME}"

CHAIN_REMAINING="${CHAIN_REMAINING:-6}"
if [[ "${CHAIN_REMAINING}" -gt 1 ]]; then
    # The script submitted below is THIS one, not the one it was copied from.
    # The inherited literal sent link 2 of the asinh chain into the BASELINE
    # launcher (job 21844722), which then ran run_mseyb_BCprect with 5 more
    # links queued behind it, so it is derived instead.
    #
    # These comments live ABOVE the command, never inside it. A `#` line
    # between backslash-continued lines comments out the REST of the joined
    # command — the script path went with it, sbatch read an empty stdin, and
    # job 21896489 died with "Batch script is empty!" after queueing nothing.
    NEXT_JOB=$(sbatch --parsable \
           --dependency="afterany:${SLURM_JOB_ID}" \
           --export="ALL,CHAIN_REMAINING=$(( CHAIN_REMAINING - 1 )),FRESH=0,CO2FIX=${CO2FIX:-0}" \
           --chdir="${SLURM_SUBMIT_DIR}" \
           "${SLURM_SUBMIT_DIR}/${SCRIPT_NAME}" 2>/dev/null) || NEXT_JOB=""
    echo "[chain] queued next link ${NEXT_JOB:-FAILED} (afterany:${SLURM_JOB_ID}, CHAIN_REMAINING=$(( CHAIN_REMAINING - 1 )))"
else
    echo "[chain] CHAIN_REMAINING=${CHAIN_REMAINING} — final link, not resubmitting"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
# Hydra's default run dir is ./outputs/<date>/<time>, i.e. INSIDE the checkout on
# /projappl/project_462001328 — which is at quota (55G/54G). Hydra then dies with
# "OSError: [Errno 122] Disk quota exceeded" in _run_hydra before training starts,
# which is what killed several co2smooth chain links on 2026-09-18/19 (they show
# up as COMPLETED jobs lasting 2-3 minutes). Same class of failure as the
# truncated checkpoint and the lost eval triggers: a full /projappl fails writes,
# sometimes silently. Keep this on scratch.
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-/scratch/project_${LUMI_EVAL_PROJECT:-462001112}/hydra/${SLURM_JOB_NAME:-run}_${SLURM_JOB_ID}}"
mkdir -p "${HYDRA_RUN_DIR}"
echo "[hydra] run dir = ${HYDRA_RUN_DIR}"

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
        data_config="${TRAIN_DATA_CONFIG:-config_data_ybias_BCprect.yaml}" \
        trainer.hyperparameters.cond_transform="${COND_TRANSFORM:-asinh}" \
        model.in_channels=2 \
        model.out_channels=2 \
        model.cond_channels=3 \
        trainer.hyperparameters.save_name=${SAVE_NAME} \
        trainer.hyperparameters.save_dir="${SAVE_DIR:-runs/}" \
        ${LOAD_OVERRIDE} \
        trainer.hyperparameters.mse_only=true \
        trainer.hyperparameters.eval_data_config="configs/${TRAIN_DATA_CONFIG:-config_data_ybias_BCprect.yaml}" \
        hydra.run.dir="${HYDRA_RUN_DIR}"
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
