#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=train
#SBATCH --time=1-00:00:00
#SBATCH -o RePaViT_Large_out.txt
#SBATCH -e RePaViT_Large_err.txt

# Load modules if needed
# e.g., `module load miniconda3`

# Activate conda environment if needed
# e.g., `conda activate repavit`

export BATCH_SIZE=4096
export MASTER_PORT=22222
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export WORLD_SIZE=$SLURM_NTASKS
export BATCH_SIZE=$(echo "scale=0; $BATCH_SIZE / $WORLD_SIZE" | bc)
export WANDB_MODE=online

srun --export=ALL python main.py \
    --model=RePaViT_Large \
    --batch_size=$BATCH_SIZE \
    --epochs=300 \
    --num_workers=20 \
    --dist_eval \
    --channel_idle \
    --idle_ratio=0.75 \
    --feature_norm=BatchNorm \
    --data_path=/path/to/imagenet \
    --output_dir=/path/to/output \
    --opt=lamb \
    --lr=1e-3 \
    --min_lr=5e-5 \
    --warmup_lr=1e-6 \
    --warmup_epochs=20 \
    --unscale_lr \
    --weight_decay=0.05 \
    --drop_path=0.3 \
    --wandb
    #--wandb_entity=your-entity-name \
    #--wandb_suffix=your-customized-suffix