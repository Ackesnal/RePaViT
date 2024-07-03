#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o RePaViT_Tiny_ChannelIdle_Optuna_out.txt
#SBATCH -e RePaViT_Tiny_ChannelIdle_Optuna_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=12777
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

# Poolformer s24
WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train_optuna.py --feature_norm=BatchNorm --channel_idle --epochs=300 --use_wandb --wandb_no_loss --wandb_suffix=optuna --optuna_ntrials=20 --num_workers=28 --dist-eval --resume_study --data-path /scratch/itee/uqxxu16/data/imagenet --output_dir=output/optuna_optimization --model=RePaPoolformer_s24 --eval-crop-ratio=0.9

# Poolformer s36
WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train_optuna.py --feature_norm=BatchNorm --channel_idle --epochs=300 --use_wandb --wandb_no_loss --wandb_suffix=optuna --optuna_ntrials=20 --num_workers=28 --dist-eval --resume_study --data-path /scratch/itee/uqxxu16/data/imagenet --output_dir=output/optuna_optimization --model=RePaPoolformer_s36 --eval-crop-ratio=0.9

# MlpMixer s32
WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train_optuna.py --feature_norm=BatchNorm --channel_idle --epochs=300 --use_wandb --wandb_no_loss --wandb_suffix=optuna --optuna_ntrials=20 --num_workers=28 --dist-eval --resume_study --data-path /scratch/itee/uqxxu16/data/imagenet --output_dir=output/optuna_optimization --model=RePaMlpMixer_s32_224

# MlpMixer s16
WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train_optuna.py --feature_norm=BatchNorm --channel_idle --epochs=300 --use_wandb --wandb_no_loss --wandb_suffix=optuna --optuna_ntrials=20 --num_workers=28 --dist-eval --resume_study --data-path /scratch/itee/uqxxu16/data/imagenet --output_dir=output/optuna_optimization --model=RePaMlpMixer_s16_224

# MlpMixer b32
WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train_optuna.py --feature_norm=BatchNorm --channel_idle --epochs=300 --use_wandb --wandb_no_loss --wandb_suffix=optuna --optuna_ntrials=20 --num_workers=28 --dist-eval --resume_study --data-path /scratch/itee/uqxxu16/data/imagenet --output_dir=output/optuna_optimization --model=RePaMlpMixer_b32_224

# MlpMixer b16
WANDB_MODE=online srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train_optuna.py --feature_norm=BatchNorm --channel_idle --epochs=300 --use_wandb --wandb_no_loss --wandb_suffix=optuna --optuna_ntrials=20 --num_workers=28 --dist-eval --resume_study --data-path /scratch/itee/uqxxu16/data/imagenet --output_dir=output/optuna_optimization --model=RePaMlpMixer_b16_224


# WANDB_MODE=[online, offline, disabled] -> 设定是否要联网同步
# --use_wandb -> 使用wandb记录数据，必须要加
# --wandb_no_loss -> 不记录每个iteration的loss。不加的话会记录loss
# --wandb_suffix -> 在project name后面加的后缀。默认是“_optuna”。
# --optuna_ntrials -> 要搜索的configuration的组合数，默认是1
# --study_name -> 给定一个study_name
# --resume_study -> 继续上次的study
# 默认使用TPESampler和MedianPruner
