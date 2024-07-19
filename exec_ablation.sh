#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o RePaViT_Base_patch16_224_layer12_RatioStudy_out.txt
#SBATCH -e RePaViT_Base_patch16_224_layer12_RatioStudy_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=13479
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

export BATCH_SIZE=$(echo "scale=0; 2048 / $WORLD_SIZE" | bc)
WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaViT_Base_patch16_224_layer12 --dist-eval \
--accumulation-steps=2 \
--model=RePaViT_Base_patch16_224_layer12 \
--data-path=/scratch/itee/uqxxu16/data/imagenet \
--feature_norm=BatchNorm \
--channel_idle \
--idle_ratio=0.25 \
--lr=2e-3 \
--min-lr=2e-5 \
--warmup-lr=5e-7 \
--warmup-epochs=20 \
--unscale-lr \
--weight-decay=0.07 \
--opt=lamb \
--num_workers=30 \
--drop-path=0.1 \
--use_wandb \
--wandb_suffix=RatioStudy \
--color-jitter=0.4 \

WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaViT_Base_patch16_224_layer12 --dist-eval \
--accumulation-steps=2 \
--model=RePaViT_Base_patch16_224_layer12 \
--data-path=/scratch/itee/uqxxu16/data/imagenet \
--feature_norm=BatchNorm \
--channel_idle \
--idle_ratio=0.5 \
--lr=2e-3 \
--min-lr=2e-5 \
--warmup-lr=5e-7 \
--warmup-epochs=20 \
--unscale-lr \
--weight-decay=0.07 \
--opt=lamb \
--num_workers=30 \
--drop-path=0.1 \
--use_wandb \
--wandb_suffix=RatioStudy \
--color-jitter=0.4 \

WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaViT_Base_patch16_224_layer12 --dist-eval \
--accumulation-steps=2 \
--model=RePaViT_Base_patch16_224_layer12 \
--data-path=/scratch/itee/uqxxu16/data/imagenet \
--feature_norm=BatchNorm \
--channel_idle \
--idle_ratio=1.0 \
--lr=2e-3 \
--min-lr=2e-5 \
--warmup-lr=5e-7 \
--warmup-epochs=20 \
--unscale-lr \
--weight-decay=0.07 \
--opt=lamb \
--num_workers=30 \
--drop-path=0.1 \
--use_wandb \
--wandb_suffix=RatioStudy \
--color-jitter=0.4 \


export BATCH_SIZE=$(echo "scale=0; 1024 / $WORLD_SIZE" | bc)
WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaSwin_Base --dist-eval \
--model=RePaSwin_Base \
--data-path=/scratch/itee/uqxxu16/data/imagenet \
--feature_norm=BatchNorm \
--channel_idle \
--idle_ratio=1.0 \
--lr=4e-3 \
--min-lr=2e-5 \
--warmup-lr=1e-6 \
--warmup-epochs=20 \
--unscale-lr \
--weight-decay=0.07 \
--opt=lamb \
--num_workers=30 \
--drop-path=0.25 \
--use_wandb \
--wandb_suffix=RatioStudy \
--color-jitter=0.4 \

WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaSwin_Base --dist-eval \
--model=RePaSwin_Base \
--data-path=/scratch/itee/uqxxu16/data/imagenet \
--feature_norm=BatchNorm \
--channel_idle \
--idle_ratio=0.5 \
--lr=4e-3 \
--min-lr=2e-5 \
--warmup-lr=1e-6 \
--warmup-epochs=20 \
--unscale-lr \
--weight-decay=0.07 \
--opt=lamb \
--num_workers=30 \
--drop-path=0.25 \
--use_wandb \
--wandb_suffix=RatioStudy \
--color-jitter=0.4 \

WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaSwin_Base --dist-eval \
--model=RePaSwin_Base \
--data-path=/scratch/itee/uqxxu16/data/imagenet \
--feature_norm=BatchNorm \
--channel_idle \
--idle_ratio=0.25 \
--lr=4e-3 \
--min-lr=2e-5 \
--warmup-lr=1e-6 \
--warmup-epochs=20 \
--unscale-lr \
--weight-decay=0.07 \
--opt=lamb \
--num_workers=30 \
--drop-path=0.25 \
--use_wandb \
--wandb_suffix=RatioStudy \
--color-jitter=0.4 \

