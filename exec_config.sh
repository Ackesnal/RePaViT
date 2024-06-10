#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o RePaSwin_Tiny_patch16_224_layer12_out.txt
#SBATCH -e RePaSwin_Tiny_patch16_224_layer12_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=12581
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

#export BATCH_SIZE=$(echo "scale=0; 2048 / $WORLD_SIZE" | bc)
#WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaViT_small_patch16_224_layer12 --dist-eval \
#--model=RePaViT_small_patch16_224_layer12 \
#--data-path /scratch/itee/uqxxu16/data/imagenet \
#--feature_norm=BatchNorm \
#--lr=3e-3 \
#--min-lr=1e-5 \
#--warmup-lr=1e-6 \
#--warmup-epochs=20 \
#--unscale-lr \
#--weight-decay=0.05 \
#--opt=lamb \
#--num_workers=30 \
#--channel_idle \
#--shortcut_gain=0.2 \
#--drop-path=0.05 \
#--use_wandb \
#--wandb_suffix=full_300epoch

#export BATCH_SIZE=$(echo "scale=0; 2048 / $WORLD_SIZE" | bc)
#WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaViT_Large_patch16_224_layer12 --dist-eval \
#--accumulation-steps=2 \
#--model=RePaViT_Large_patch16_224_layer12 \
#--data-path /scratch/itee/uqxxu16/data/imagenet \
#--feature_norm=BatchNorm \
#--lr=5e-3 \
#--min-lr=4e-5 \
#--warmup-lr=1e-6 \
#--warmup-epochs=20 \
#--unscale-lr \
#--weight-decay=0.07 \
#--opt=lamb \
#--num_workers=30 \
#--channel_idle \
#--shortcut_gain=1.0 \
#--drop-path=0.2 \
#--use_wandb \
#--wandb_suffix=full_300epoch


export BATCH_SIZE=$(echo "scale=0; 1024 / $WORLD_SIZE" | bc)
WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaSwin_Tiny_patch16_224_layer12 --dist-eval \
--model=RePaSwin_Tiny_patch16_224_layer12 \
--data-path /scratch/itee/uqxxu16/data/imagenet \
--lr=1e-3 \
--min-lr=1e-5 \
--clip-grad=5.0 \
--warmup-lr=1e-6 \
--warmup-epochs=20 \
--unscale-lr \
--weight-decay=0.05 \
--opt=adamw \
--num_workers=30 \
--drop-path=0.2 \
--use_wandb \
--wandb_suffix=full_300epoch \
--color-jitter=0.4 \

#export BATCH_SIZE=$(echo "scale=0; 4096 / $WORLD_SIZE" | bc)
#WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --batch-size=$BATCH_SIZE --output_dir=output/RePaViT_Tiny_patch16_224_layer12 --dist-eval \
#--model=RePaViT_Tiny_patch16_224_layer12 \
#--data-path /scratch/itee/uqxxu16/data/imagenet \
#--feature_norm=BatchNorm \
#--lr=9e-3 \
#--min-lr=1e-5 \
#--warmup-lr=1e-6 \
#--warmup-epochs=20 \
#--unscale-lr \
#--weight-decay=0.095 \
#--opt=lamb \
#--num_workers=30 \
#--channel_idle \
#--shortcut_gain=1.0 \
#--drop-path=0.05 \
#--use_wandb \
#--wandb_suffix=full_300epoch