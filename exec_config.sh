#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o RePaViT_small_patch16_224_layer12_out.txt
#SBATCH -e RePaViT_small_patch16_224_layer12_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=12555
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export BATCH_SIZE=$(echo "scale=0; 2048 / $WORLD_SIZE" | bc)

WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaViT_small_patch16_224_layer12 --dist-eval \
--model=RePaViT_small_patch16_224_layer12 \
--data-path /scratch/itee/uqxxu16/data/imagenet \
--feature_norm=BatchNorm \
--lr=3e-3 \
--min-lr=4e-5 \
--warmup-lr=1e-6 \
--warmup-epochs=20 \
--unscale-lr \
--weight-decay=0.04880 \
--opt=lamb \
--num_workers=30 \
--channel_idle \
--shortcut_gain=0.2 \
--drop-path=0.038055 \
--use_wandb \
--wandb_suffix=full_300epoch

