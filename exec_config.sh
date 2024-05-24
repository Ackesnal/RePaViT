#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o RePaViT_small_patch16_224_layer12_ChannelIdle_BS2048_lr2e-3_min-lr2e-5_warmup5epoch_gain0.1_ADAMW_weightdecay0.01_out.txt
#SBATCH -e RePaViT_small_patch16_224_layer12_ChannelIdle_BS2048_lr2e-3_min-lr2e-5_warmup5epoch_gain0.1_ADAMW_weightdecay0.01_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=12582
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export BATCH_SIZE=$(echo "scale=0; 2048 / $WORLD_SIZE" | bc)

srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --data-path /scratch/itee/uqxxu16/data/imagenet --batch-size=$BATCH_SIZE --model=RePaViT_small_patch16_224_layer12 --output_dir=output/RePaViT_small_patch16_224_layer12_ChannelIdle_BS2048_lr2e-3_min-lr2e-5_warmup5epoch_gain0.1_ADAMW_weightdecay0.01 --feature_norm=BatchNorm --shortcut_gain=0.1 --lr=1e-3 --min-lr=1e-5 --warmup-lr=1e-6 --opt=adamw --num_workers=15 --weight-decay=0.01 --warmup-epochs=5 --channel_idle