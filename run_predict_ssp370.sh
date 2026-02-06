#!/bin/bash
#SBATCH --job-name=generate_aero
#SBATCH --account=project_462001112
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=2:10:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module --force purge
module use /appl/local/csc/modulefiles
module load LUMI
module load pytorch
#module load cray-python
source "/projappl/project_462001112/venvs/diffesm/bin/activate"

# Common settings
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
export MIOPEN_FIND_ENFORCE=1
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
  export NCCL_NET_GDR_LEVEL=3
   export NCCL_CROSS_NIC=1
   export NCCL_TIMEOUT=7200
NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_PER_NODE)
MAIN_PROCESS_IP=$(hostname -i)

RUN_CMD="accelerate launch \
                    --config_file=accelerate_config.yaml \
                    --num_processes=$NUM_PROCESSES \
                    --num_machines=$SLURM_NNODES \
                    --machine_rank=\$SLURM_NODEID \
                    --main_process_ip=$MAIN_PROCESS_IP \
                    generate_ssp370.py"

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
