#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o RePaViT_small_patch16_224_layer12_ChannelIdle+LayerScale_wandb_Sweep_out.txt
#SBATCH -e RePaViT_small_patch16_224_layer12_ChannelIdle+LayerScale_wandb_Sweep_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=12585
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train_sweep.py --data-path /scratch/itee/uqxxu16/data/imagenet --model RePaViT_small_patch16_224_layer12 --output_dir=output/sweep_optimization --feature_norm=BatchNorm --epochs=200 --channel_idle --use_wandb --wandb_no_loss --wandb_suffix=layer_scale_sweep --wandb_sweep_count=20 --layer_scale --wandb_sweep_id=vmkgxzbk

# WANDB_MODE=[online, offline, disabled] -> 用于设定是否要联网同步
# --use_wandb -> 使用wandb的功能，必须要加
# --wandb_no_loss -> 不记录训练时候的loss，如果不加的话会记录每个iteration的loss
# --wandb_suffix -> 默认是在project name后面加“_sweep” ，也可以改成其他后缀
# --wandb_sweep_count -> 设定要搜索多少组参数，默认是1
# --wandb_sweep_id -> 如果不设定，默认开一个新的sweep。如果设定id，就会继续上一个sweep已经搜索过的记录继续搜索