#!/bin/bash
#SBATCH --job-name=aaertest
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
assert_account
lumi_env_banner
#
# ISOLATED A/B test: does upweighting aaer (aerosol-only) improve the
# under-learned aerosol response (aaer pattern corr 0.42; model_skill_diagnosis)
# and pull ssp126 down? Forks the newest production run_slope-tcre checkpoint
# into a SEPARATE save_name (run_aaertest) and uses configs/config_data_aaertest
# .yaml (scenario_weights aaer 1→4). No eval watcher, no self-chain — does NOT
# touch the production chain. Reads cond/data from /scratch (no /tmp staging).
# After it runs, eval a run_aaertest checkpoint:
#     CHECKPOINT=.../runs/run_aaertest_<ep>.pt sbatch run_eval_aero.sh
# Success = aaer pattern corr rises above 0.42 (and ideally ssp126 ratio drops)
# while hist/ssp370 hold.

set -euo pipefail
mkdir -p logs

# ── LUMI AI Factory container (same as run2_aero.sh) ─────────────────────────
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
echo "[CONTAINER] ${SIF}"

_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
echo "[VENV] ${SINGULARITYENV_PYTHONPATH}"

# ── Networking / ROCm (same as run2_aero.sh) ─────────────────────────────────
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

# ── Fork point: newest production checkpoint ─────────────────────────────────
RUNS_DIR="${SLURM_SUBMIT_DIR}/runs"
PROD_CKPT=$(ls -t "${RUNS_DIR}"/run_slope-tcre_*.pt 2>/dev/null | grep -v _best | head -1 || true)
if [[ -z "${PROD_CKPT}" ]]; then
    echo "[fork] no run_slope-tcre_*.pt found in ${RUNS_DIR} — aborting"; exit 1
fi
echo "[fork] forking aaer-upweight test from ${PROD_CKPT}"

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
        trainer.hyperparameters.save_name=run_aaertest.pt \
        trainer.hyperparameters.load_path=${PROD_CKPT} \
        data_config=config_data_aaertest.yaml
'"

srun bash -c "$RUN_CMD" || true
