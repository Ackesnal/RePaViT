#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o RePaViT_small_patch16_224_layer12_ChannelIdle_wandb_Sweep_out.txt
#SBATCH -e RePaViT_small_patch16_224_layer12_ChannelIdle_wandb_Sweep_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=12584
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export BATCH_SIZE=$(echo "scale=0; 2048 / $WORLD_SIZE" | bc)

WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train_wandb_sweep.py --data-path /scratch/itee/uqxxu16/data/imagenet --model RePaViT_small_patch16_224_layer12 --output_dir=output/sweep_optimization --feature_norm=BatchNorm --channel_idle --use_wandb --wandb_no_loss --wandb_suffix=sweep --wandb_sweep_count=20 --epochs=200
