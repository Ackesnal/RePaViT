#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o RePaViT_Tiny_rebuttal_out.txt
#SBATCH -e RePaViT_Tiny_rebuttal_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=15556
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)


export BATCH_SIZE=$(echo "scale=0; 4096 / $WORLD_SIZE" | bc)
WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaViT_Tiny_rebuttal --dist-eval \
--model=RePaViT_Tiny_patch16_224_layer12 \
--data-path=/scratch/itee/uqxxu16/data/imagenet \
--feature_norm=BatchNorm \
--channel_idle \
--lr=2.5e-3 \
--min-lr=1e-6 \
--warmup-lr=1e-6 \
--warmup-epochs=20 \
--unscale-lr \
--opt=lamb \
--num_workers=20 \
--weight-decay=0.02 \
--drop-path=0.02 \
--color-jitter=0.4 \
--use_wandb \
--wandb_suffix=full_300epoch \
--reparam 


#export BATCH_SIZE=$(echo "scale=0; 4096 / $WORLD_SIZE" | bc)
#WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaViT_Small_rebuttal --dist-eval \
#--model=RePaViT_Small_patch16_224_layer12 \
#--data-path=/scratch/itee/uqxxu16/data/imagenet \
#--feature_norm=BatchNorm \
#--channel_idle \
#--lr=3e-3 \
#--min-lr=4e-5 \
#--warmup-lr=1e-6 \
#--warmup-epochs=20 \
#--unscale-lr \
#--opt=lamb \
#--num_workers=30 \
#--weight-decay=0.05 \
#--drop-path=0.04 \
#--color-jitter=0.4 \
#--use_wandb \
#--wandb_suffix=300Epochs \
#--reparam 


#export BATCH_SIZE=$(echo "scale=0; 4096 / $WORLD_SIZE" | bc)
#WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaViT_Base_rebuttal --dist-eval \
#--model=RePaViT_Large_patch16_224_layer12 \
#--data-path=/scratch/itee/uqxxu16/data/imagenet \
#--feature_norm=BatchNorm \
#--channel_idle \
#--lr=4e-3 \
#--min-lr=4e-5 \
#--warmup-lr=1e-6 \
#--warmup-epochs=20 \
#--unscale-lr \
#--opt=lamb \
#--num_workers=30 \
#--weight-decay=0.07 \
#--drop-path=0.1 \
#--color-jitter=0.4 \
#--use_wandb \
#--wandb_suffix=full_300epoch \
#--reparam 


#export BATCH_SIZE=$(echo "scale=0; 4096 / $WORLD_SIZE" | bc)
#WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaSwin_Tiny_rebuttal --dist-eval \
#--model=RePaSwin_Tiny \
#--data-path=/scratch/itee/uqxxu16/data/imagenet \
#--feature_norm=BatchNorm \
#--channel_idle \
#--lr=5e-3 \
#--min-lr=5e-5 \
#--warmup-lr=1e-6 \
#--warmup-epochs=20 \
#--unscale-lr \
#--opt=lamb \
#--num_workers=30 \
#--weight-decay=0.2 \
#--drop-path=0.1 \
#--color-jitter=0.4 \
#--use_wandb \
#--wandb_suffix=full_300epoch \
#--reparam 


#export BATCH_SIZE=$(echo "scale=0; 4096 / $WORLD_SIZE" | bc)
#WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaSwin_Small_rebuttal --dist-eval \
#--model=RePaSwin_Small \
#--data-path=/scratch/itee/uqxxu16/data/imagenet \
#--feature_norm=BatchNorm \
#--channel_idle \
#--lr=6e-3 \
#--min-lr=5e-5 \
#--warmup-lr=1e-6 \
#--warmup-epochs=20 \
#--unscale-lr \
#--opt=lamb \
#--num_workers=30 \
#--weight-decay=0.15 \
#--drop-path=0.09 \
#--color-jitter=0.4 \
#--use_wandb \
#--wandb_suffix=full_300epoch \
#--reparam 


#export BATCH_SIZE=$(echo "scale=0; 4096 / $WORLD_SIZE" | bc)
#WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaSwin_Base_rebuttal --dist-eval \
#--model=RePaSwin_Base \
#--data-path=/scratch/itee/uqxxu16/data/imagenet \
#--feature_norm=BatchNorm \
#--channel_idle \
#--lr=4e-3 \
#--min-lr=2e-5 \
#--warmup-lr=1e-6 \
#--warmup-epochs=20 \
#--unscale-lr \
#--opt=lamb \
#--num_workers=30 \
#--weight-decay=0.1 \
#--drop-path=0.08 \
#--color-jitter=0.4 \
#--use_wandb \
#--wandb_suffix=full_300epoch \
#--reparam 