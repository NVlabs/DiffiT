#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --dependency=singleton
#SBATCH --job-name=diffit_sample_256

IMAGE="./images/diffit.sqsh"
MODEL="./ckpts/diffit_256.safetensors"

LOG_DIR="./log_dir/256_1"
# Set WORK_DIR to wherever sample.py lives
WORK_DIR="./diffit_code"

SAVE_DIR="./job_logs"
mkdir -p "${SAVE_DIR}"
mkdir -p "${LOG_DIR}"
DATETIME=$(date +'%Y-%m-%d_%H-%M-%S')

# -----------------
# 2) SLURM Node Info
# -----------------
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
TOTAL_NODES=$SLURM_NNODES
MASTER_NODE="${NODELIST[0]}"
MASTER_ADDR="$MASTER_NODE"
MASTER_PORT=6000
WORLD_SIZE=$((TOTAL_NODES * GPUS_PER_NODE))

echo "============================="
echo "  Total Nodes:      $TOTAL_NODES"
echo "  MASTER_ADDR:      $MASTER_ADDR"
echo "  MASTER_PORT:      $MASTER_PORT"
echo "  GPUS_PER_NODE:    $GPUS_PER_NODE"
echo "  WORLD_SIZE:       $WORLD_SIZE"
echo "============================="

# -----------------
# 3) Launch Sampling
# -----------------
srun \
  --nodes="$TOTAL_NODES" \
  --ntasks-per-node="$GPUS_PER_NODE" \
  --container-image="$IMAGE" \
  --container-env=ALL \
  --container-mounts="/lustre:/lustre,/home/${USER}:/home/${USER}" \
  --container-workdir="$WORK_DIR" \
  --output="${SAVE_DIR}/sample_%x_${DATETIME}.log" \
bash -c "
  export MASTER_ADDR=$MASTER_ADDR
  export MASTER_PORT=$MASTER_PORT
  export WORLD_SIZE=$WORLD_SIZE
  export RANK=\$SLURM_PROCID
  export LOCAL_RANK=\$SLURM_LOCALID

  echo \"[Proc] \$(hostname -s) RANK=\$RANK LOCAL_RANK=\$LOCAL_RANK\"

  python sample.py \
    --log_dir $LOG_DIR \
    --cfg_scale 4.4 \
    --model_path $MODEL \
    --image_size 256 \
    --model Diffit \
    --num_sampling_steps 250 \
    --num_samples 50000 \
    --cfg_cond True
"

echo "All tasks completed at $(date)"
