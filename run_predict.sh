#!/bin/bash
#SBATCH --job-name=ssp126_get
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=2:10:00
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

set -euo pipefail
mkdir -p logs

module --force purge
module use /appl/local/csc/modulefiles
module load LUMI
module load pytorch
#module load cray-python
source "${LUMI_PROJAPPL}/venvs/diffesm/bin/activate"

# Common settings
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
export MIOPEN_FIND_ENFORCE=1
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
export MIOPEN_FIND_ENFORCE=1
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=hsn
export RCCL_ENABLE_SHARP=0
export ACCELERATE_USE_FSDP=0
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn
NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_PER_NODE)
MAIN_PROCESS_IP=$(hostname -i)

RUN_CMD="accelerate launch \
                    --config_file=accelerate_config.yaml \
                    --num_processes=$NUM_PROCESSES \
                    --num_machines=$SLURM_NNODES \
                    --machine_rank=\$SLURM_NODEID \
                    --main_process_ip=$MAIN_PROCESS_IP \
                    generate_ssp126.py"

#RUN_CMD="python -m accelerate.commands.launch \
#            --config_file=accelerate_config.yaml \
#            --num_processes=$NUM_PROCESSES \
#            --num_machines=$SLURM_NNODES \
#            --machine_rank=$SLURM_NODEID \
#            --main_process_ip=$MAIN_PROCESS_IP \
#            main.py"

srun bash -c "$RUN_CMD"
#srun bash -c "$RUN_CMD"
# Launch one 'accelerate' per node; each will spawn 8 local GPU workers
#srun --ntasks=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 bash -lc "
#  echo Node: \$(hostname), NODEID=\${SLURM_NODEID}, GPUS_ON_NODE=\${SLURM_GPUS_ON_NODE}
#  accelerate launch \
#    --num_processes \${SLURM_GPUS_ON_NODE} \
#    --num_machines \${SLURM_JOB_NUM_NODES} \
#    --machine_rank \${SLURM_NODEID} \
#    --main_process_ip ${MASTER_ADDR} \
#    --main_process_port ${MASTER_PORT} \
#    main.py \
#      hydra.job.chdir=false \
#      hydra.run.dir=. \
#      hydra.output_subdir=null
#"
