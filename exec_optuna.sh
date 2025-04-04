#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH -o RePaViT_Small_Optuna_out.txt
#SBATCH -e RePaViT_Small_Optuna_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))

WANDB_MODE=online torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE train_optuna.py \
--model=RePaViT_Small \
--num_workers=20 \
--epochs=100 \
--dist_eval \
--channel_idle \
--idle_ratio=0.75 \
--feature_norm=BatchNorm \
--optuna_resume \
--optuna_ntrials=20 \
--wandb \
--wandb_entity=ackesnal-ai \
--wandb_suffix=Optuna_test \
--data_path=/path/to/imagenet \
--output_dir=/path/to/optuna_study/output

# WANDB_MODE=[online, offline, disabled] -> 设定是否要联网同步
# --use_wandb       ->   使用wandb记录数据，默认是True
# --wandb_suffix    ->   在project name后面加的后缀，默认是“_Optuna”。
# --optuna_study    ->   给定一个optuna的study name。默认使用model name
# --optuna_resume   ->   是否继续上次的study
# --optuna_ntrials  ->   要搜索的configuration的组合数，默认是10
