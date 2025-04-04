#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o RePaViT_Small_out.txt
#SBATCH -e RePaViT_Small_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=15556
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)


export BATCH_SIZE=$(echo "scale=0; 4096 / $WORLD_SIZE" | bc)
WANDB_MODE=online torchrun --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py \
--model=RePaViT_Small \
--batch_size=$BATCH_SIZE \
--epochs=300 \
--num_workers=20 \
--dist_eval \
--channel_idle \
--idle_ratio=0.75 \
--feature_norm=BatchNorm \
--data_path=/path/to/imagenet \
--output_dir=/path/to/output \
--lr=6.2e-3 \
--min_lr=5e-5 \
--warmup_lr=1e-6 \
--warmup_epochs=20 \
--unscale_lr \
--weight_decay=0.1444 \
--opt=lamb \
--drop_path=0.0938 \
--use_wandb \
--wandb_suffix=full_300epoch