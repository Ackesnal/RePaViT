#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o RePaViT_small_patch16_224_layer12_ChannelIdle_Optuna_out.txt
#SBATCH -e RePaViT_small_patch16_224_layer12_ChannelIdle_Optuna_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=12584
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train_optuna.py --data-path /scratch/itee/uqxxu16/data/imagenet --model RePaViT_small_patch16_224_layer12 --output_dir=output/optuna_optimization --feature_norm=BatchNorm --epochs=300 --channel_idle --use_wandb --wandb_no_loss --wandb_suffix=optuna --optuna_ntrials=50

# WANDB_MODE=[online, offline, disabled] -> ???????????
# --use_wandb -> ??wandb???,????
# --wandb_no_loss -> ????????loss,???????????iteration?loss
# --wandb_suffix -> ????project name???“_sweep” ,?????????
# --optuna_ntrials -> ??????????,???1