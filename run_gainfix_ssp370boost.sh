#!/bin/bash
#SBATCH --job-name=gainfix_ssp370boost
#SBATCH --account=project_462001328
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#
# ISOLATED A/B test (hypothesis #1 of the ssp370 warm+wet bias investigation,
# 2026-07-22): run_gainfix shows a PERSISTENT ssp370 warm+wet bias across its
# whole evaluated timeseries (GMbias +0.15..+0.29 K, precip +0.033..+0.049
# mm/day at ep0970/0990/1030). configs/config_data.yaml's own scenario_weights
# comment documents that halving ssp370's batch share (to make room for the
# aaer upweight) DOUBLED its warm bias (+0.23→+0.51) in an earlier A/B — the
# current [2,2,4,1] weights were chosen specifically to cap that at +0.23,
# which matches the magnitude re-observed here.
#
# This fork gives ssp370 one batch slot back from aaer:
#   configs/config_data_ssp370boost.yaml: scenario_weights [2,3,3,1]
#   (vs production run_gainfix's [2,2,4,1] in config_data.yaml)
# Forks the newest run_gainfix checkpoint into a SEPARATE save_name
# (run_gainfix_ssp370boost) — does NOT touch the production chain.
# No eval watcher, no self-chain — this is a short probe, not a production run.
# Reads cond/data from /scratch directly (no /tmp staging, unlike run2_gainfix.sh).
#
# After it runs ~30-50 epochs, eval a run_gainfix_ssp370boost checkpoint and
# compare ssp370 GMbias/precip-bias against the run_gainfix baseline at a
# matched epoch:
#     CHECKPOINT=.../runs/run_gainfix_ssp370boost_<ep>.pt sbatch run_eval_aero.sh
# Success = ssp370 GMbias/precip-bias shrinks toward hist/ghg's near-zero level
# while aaer patcorr doesn't relapse toward its pre-upweight 0.08-0.42 range.
# If the bias DOESN'T move, hypothesis #1 (batch-share starvation) is ruled out
# and the interaction-match / SGAIN hypotheses (run_gainfix_intmatch.sh) become
# the leading candidates.

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container (same as run2_gainfix.sh) ──────────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] ${SIF}"

_VENV_SITE=$(realpath /projappl/project_462001328/venvs/diffesm_laif 2>/dev/null \
             || echo /projappl/project_462001328/venvs/diffesm_laif)/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
echo "[VENV] ${SINGULARITYENV_PYTHONPATH}"

# ── Networking / ROCm (same as run2_gainfix.sh) ──────────────────────────────
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
mkdir -p /tmp/miopen_${SLURM_JOB_ID} /tmp/hip_${SLURM_JOB_ID}

# ── Fork point: newest production run_gainfix checkpoint ────────────────────
RUNS_DIR="${SLURM_SUBMIT_DIR}/runs"
PROD_CKPT=$(ls -t "${RUNS_DIR}"/run_gainfix_*.pt 2>/dev/null | grep -v _best | head -1 || true)
if [[ -z "${PROD_CKPT}" ]]; then
    echo "[fork] no run_gainfix_*.pt found in ${RUNS_DIR} — aborting"; exit 1
fi
echo "[fork] forking ssp370-batch-share A/B from ${PROD_CKPT}"

# ── Launch (Hydra overrides; no watcher, no chain) ───────────────────────────
NUM_PROCESSES=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))
MAIN_PROCESS_IP=$(hostname -i)

RUN_CMD="singularity exec ${SIF} bash -c '
    accelerate launch \
        --config_file=accelerate_config.yaml \
        --num_processes=${NUM_PROCESSES} \
        --num_machines=${SLURM_NNODES} \
        --machine_rank=\${SLURM_NODEID} \
        --main_process_ip=${MAIN_PROCESS_IP} \
        main_aero.py \
        trainer.hyperparameters.save_name=run_gainfix_ssp370boost.pt \
        trainer.hyperparameters.load_path=${PROD_CKPT} \
        trainer.hyperparameters.sampled_gain_loss_scale=0.05 \
        trainer.hyperparameters.bc_clip_mode=populated \
        data_config=config_data_ssp370boost.yaml
'"

srun bash -c "$RUN_CMD" || true

cp /tmp/miopen_${SLURM_JOB_ID}/*.ufdb.* "${SLURM_SUBMIT_DIR}/.miopen_cache/" 2>/dev/null || true
cp /tmp/miopen_${SLURM_JOB_ID}/*.db     "${SLURM_SUBMIT_DIR}/.miopen_cache/" 2>/dev/null || true
