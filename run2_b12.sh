#!/bin/bash
#SBATCH --job-name=diffusion_b12
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Single source of truth for the LUMI project id and its paths.
# Locate the repo. Under sbatch, SLURM COPIES this script to
# /var/spool/slurmd/job<id>/, so dirname "$0" is the spool directory and a
# relative source fails with "No such file or directory" — which is exactly
# what happened on Roihu job 660362. SLURM_SUBMIT_DIR is where sbatch was
# invoked, so try that first, then the script's own directory (correct when
# run directly), then an already-exported LUMI_REPO.
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
# 6 h links instead of one 48 h reservation: short jobs backfill into idle gaps
# far sooner. The run auto-resumes (load_path:"newest") and self-chains below.
#
# ── BATCH-12 EXPERIMENT (isolated fork of run2_aero.sh) ──────────────────────
# Probe (run_batchprobe.sh) found the batch ceiling = 16 with the low-t TCRE
# pass; 12 is the safe margin (expandable_segments unsupported → fragmentation
# risk at 16 over long runs). This run is a SEPARATE experiment from run_sensfix:
#   save_name = run_sensfix_b12   (own checkpoints/evals, starts from epoch 0)
#   batch_size = 12 (config_data_b12.yaml + Hydra override; the two must match)
#   lr = 7.5e-5  (×1.5 linear scaling for the larger effective batch 128→192)
# It self-chains to run2_b12.sh and tags its watcher PROD_RUN=run_sensfix_b12.
# Submit:  CHAIN_REMAINING=12 sbatch run2_b12.sh
# Best run when run_sensfix is NOT concurrently chaining (single shared watcher).

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
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
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
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
export MIOPEN_FIND_ENFORCE=1

# Seed each node's local cache from the persistent store, then train locally.
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c "
    mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}
    cp ${PERSISTENT_CACHE}/* /tmp/miopen_${SLURM_JOB_ID}/ 2>/dev/null || true
"

# ── Stage training data to /tmp on each node ─────────────────────────────────
# Lustre I/O is the dominant bottleneck (~11.5 min/epoch); local /tmp avoids
# repeated random reads across the cluster filesystem.  We stage:
#   training_data/TREFHT/{hist,ssp370,AAER,GHG}    (~3.5 GB)
#   emissions_*_timefixed.nc                       (~270 MB)
# Total ~3.8 GB per node — trivial for compute-node /tmp.  The local copy is
# bind-mounted over the original /scratch path inside the container so no
# config edits are needed.
SRC_DATA_ROOT=${LUMI_DATA}
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
    # Cond files: the four "_only_" scenario files needed by both EMISSIONS_PATHS
    # (1-99 pct reference, see data/climate_dataset.py) and the per-scenario
    # config_data.yaml inputs. ssp126 is the OOD test and is not staged here.
    cp ${SRC_DATA_ROOT}/emissions_hist_only_timefixed.nc    ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_ssp370_only_timefixed.nc  ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_aaer_only_timefixed.nc    ${LOCAL_DATA_ROOT}/
    cp ${SRC_DATA_ROOT}/emissions_ghg_only_timefixed.nc     ${LOCAL_DATA_ROOT}/
    echo \"[stage] node \$(hostname): done in \$((\$(date +%s)-t0))s, size=\$(du -sh ${LOCAL_DATA_ROOT} | awk '{print \$1}')\"
"

# ── Launch eval watcher as a background SLURM job ────────────────────────────
# Submits watch_eval_triggers.sh on the small partition so it can call sbatch
# to dispatch eval jobs when the trainer writes trigger files.
# Runs outside the container (no GPU needed, just needs sbatch access).
# Walltime: match this training job's own time limit so the watcher stops at the
# same time as the main script (auto-tracks #SBATCH --time above; falls back to
# 06:00:00 if the limit can't be read).
WATCHER_TIME=$(squeue -h -j "${SLURM_JOB_ID}" -o '%l' 2>/dev/null | tr -d '[:space:]' || true)
[[ -z "${WATCHER_TIME}" || "${WATCHER_TIME}" == "UNLIMITED" ]] && WATCHER_TIME="06:00:00"
# Chain guard: a watcher from an earlier link may still be active — don't spawn
# a duplicate (the per-job cleanup at the end only runs on a clean exit, not on
# a walltime kill, so watchers can outlive their submitting job).
EXISTING_WATCHER=$(squeue -u "$(whoami)" --name=eval_watcher -t PENDING,RUNNING \
                   --noheader -o '%i' 2>/dev/null | head -1 || true)
if [[ -n "${EXISTING_WATCHER}" ]]; then
    WATCHER_JOB=""
    echo "[watcher] eval watcher ${EXISTING_WATCHER} already active — not resubmitting"
else
    WATCHER_JOB=$(sbatch --job-name=eval_watcher \
           --account=project_462001112 \
           --partition=small \
           --time="${WATCHER_TIME}" \
           --ntasks=1 --cpus-per-task=1 --mem=256M \
           --chdir="${SLURM_SUBMIT_DIR}" \
           --export="ALL,PROD_RUN=run_sensfix,run_sensfix_b12" \
           --output="${SLURM_SUBMIT_DIR}/logs/eval_watcher_%j.out" \
           "${SLURM_SUBMIT_DIR}/watch_eval_triggers.sh" 2>/dev/null | awk '{print $NF}') || WATCHER_JOB=""
    echo "[watcher] Submitted eval watcher job ${WATCHER_JOB:-FAILED} (time=${WATCHER_TIME})"
fi

# ── Self-chaining: queue the next training link (#2 short-walltime chaining) ──
# A short 6 h job backfills far sooner than a 48 h reservation; the model
# auto-resumes from the newest checkpoint and Adam momentum carries over
# (reset_optimizer: false in configs/config_aero.yaml).
# The next link is queued BEFORE training with an `afterany` dependency, so it
# is registered even when this job is killed at the walltime limit (the normal
# end-of-link case, where the post-training cleanup below never runs).
# CHAIN_REMAINING bounds the chain length; override at first submission, e.g.
#   CHAIN_REMAINING=20 sbatch run2_b12.sh
CHAIN_REMAINING="${CHAIN_REMAINING:-12}"
if [[ "${CHAIN_REMAINING}" -gt 1 ]]; then
    NEXT_JOB=$(sbatch --parsable \
           --dependency="afterany:${SLURM_JOB_ID}" \
           --export="ALL,CHAIN_REMAINING=$(( CHAIN_REMAINING - 1 ))" \
           --chdir="${SLURM_SUBMIT_DIR}" \
           "${SLURM_SUBMIT_DIR}/run2_b12.sh" 2>/dev/null) || NEXT_JOB=""
    echo "[chain] queued next link ${NEXT_JOB:-FAILED} (afterany:${SLURM_JOB_ID}, CHAIN_REMAINING=$(( CHAIN_REMAINING - 1 )))"
else
    echo "[chain] CHAIN_REMAINING=${CHAIN_REMAINING} — final link, not resubmitting"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
NUM_PROCESSES=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))
MAIN_PROCESS_IP=$(hostname -i)

# Batch-12 Hydra overrides: separate save_name, batch-12 data config, matching
# trainer batch_size, and ×1.5 LR. data_config's batch_size must match the
# trainer override (the loader reads it from the file, not Hydra).
RUN_CMD="singularity exec --bind ${LOCAL_DATA_ROOT}:${SRC_DATA_ROOT} ${SIF} bash -c '
    accelerate launch \
        --config_file=accelerate_config.yaml \
        --num_processes=${NUM_PROCESSES} \
        --num_machines=${SLURM_NNODES} \
        --machine_rank=\${SLURM_NODEID} \
        --main_process_ip=${MAIN_PROCESS_IP} \
        main_aero.py \
        data_config=config_data_b12.yaml \
        trainer.hyperparameters.save_name=run_sensfix_b12.pt \
        trainer.hyperparameters.batch_size=12 \
        trainer.hyperparameters.lr=0.000075
'"

srun bash -c "$RUN_CMD" || true

# ── Save benchmarked MIOpen kernels back to persistent store ─────────────────
# Copy from head node's /tmp back to Lustre so next job skips re-benchmarking.
cp /tmp/miopen_${SLURM_JOB_ID}/*.ufdb.* "${PERSISTENT_CACHE}/" 2>/dev/null || true
cp /tmp/miopen_${SLURM_JOB_ID}/*.db     "${PERSISTENT_CACHE}/" 2>/dev/null || true

# ── Clean up staged training data from each node's /tmp ──────────────────────
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "rm -rf ${LOCAL_DATA_ROOT} 2>/dev/null || true" || true

# ── Cancel watcher when training finishes (walltime, crash, or clean exit) ───
if [[ -n "${WATCHER_JOB}" ]]; then
    echo "[watcher] Training finished — cancelling eval watcher job ${WATCHER_JOB}"
    scancel "${WATCHER_JOB}" 2>/dev/null || true
fi
