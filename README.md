
# RePaViT: Scalable Vision Transformer Acceleration via Structural Reparameterization on Feedforward Network Layers [ICML2025] [Arxiv](arxiv)

### This is the official repository for __RePaViT__.

_(For RePa-LV-ViT source code, please refer to this [repo](https://github.com/Ackesnal/RePa-LV-ViT) as LV-ViT incorporates a different training framework. For dense prediction tasks, the code based on MMDetection and MMSegmentation is under construction. The pretrained model weights will be released soon.)_

## Environment Setup

First, clone the repository locally:
```
git clone https://github.com/Ackesnal/RePaViT.git
cd RePaViT
```
Then, install environments via conda:
```
conda create -n repavit python=3.10 -y && conda activate repavit
pip install torch torchvision torchaudio timm==1.0.3 einops ptflops wandb rocksdb-py
```
After finishing the above installations, it is ready to run this repo.

We further utilize the [wandb](https://wandb.ai/site) for real-time tracking and training process visualization. The use of wandb is optional. However, you will need to register and login to wandb before using this functionality.

## Dataset Preparation
Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision `datasets.ImageFolder`, and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

## Training
### 1. Training on a single node

To train RePaViT on ImageNet on a single node with 8 gpus for 300 epochs without wandb logging, please refer to the command examples below.

__[RePaViT-Base]:__
```
torchrun --nproc_per_node=8 main.py \
  --model=RePaViT_Base \
  --batch_size=512 \
  --epochs=300 \
  --dist_eval \
  --channel_idle \
  --idle_ratio=0.75 \
  --feature_norm=BatchNorm \
  --data_path=/path/to/imagenet \
  --output_dir=/path/to/output \
  --lr=4e-3 \
  --min_lr=4e-5 \
  --warmup_lr=1e-6 \
  --warmup_epochs=20 \
  --unscale_lr \
  --weight_decay=0.05 \
  --opt=lamb \
  --drop_path=0.1
```

__[RePaViT-Large]:__
```
torchrun --nproc_per_node=8 main.py \
  --model=RePaViT_Large \
  --batch_size=512 \
  --epochs=300 \
  --dist_eval \
  --channel_idle \
  --idle_ratio=0.75 \
  --feature_norm=BatchNorm \
  --data_path=/path/to/imagenet \
  --output_dir=/path/to/output \
  --lr=1e-3 \
  --min_lr=5e-5 \
  --warmup_lr=1e-6 \
  --warmup_epochs=20 \
  --unscale_lr \
  --weight_decay=0.05 \
  --opt=lamb \
  --drop_path=0.3
```

`--channel_idle` and `--idle_ratio=0.75` are used to control channel idle mechanism in FFN layers. Please note that `--feature_norm=BatchNorm` must be added to facilitate full structural reparameterization.

If the computating resource is limited, you can add `--accumulation_steps` for training with a smaller batch size and gradient accumulation. `--accumulation_steps`$\times$`--batch_size`$\times$`--nproc_per_node` is the total batch size per batch.

### 2. Track your training with wandb

To train with wandb tracking and visualization, `--wandb` argument with environment variable `WANDB_MODE` should be set. The project name is set to the model name by default. In addition, `--wandb_suffix` can be used to nominate a customized suffix for distinguishing different projects on the same model.

__[RePaViT-Large] with wandb:__
```
WANDB_MODE=online torchrun --nproc_per_node=8 main.py \
  --model=RePaViT_Large \
  --batch_size=512 \
  --epochs=300 \
  --dist_eval \
  --channel_idle \
  --idle_ratio=0.75 \
  --feature_norm=BatchNorm \
  --data_path=/path/to/imagenet \
  --output_dir=/path/to/output \
  --lr=1e-3 \
  --min_lr=5e-5 \
  --warmup_lr=1e-6 \
  --warmup_epochs=20 \
  --unscale_lr \
  --weight_decay=0.05 \
  --opt=lamb \
  --drop_path=0.3 \
  --wandb
  #--wandb_entity=your-entity-name \
  #--wandb_suffix=your-customized-suffix
```

Please note that `WANDB_MODE` MUST be set when using `--use_wandb`. You can choose `WANDB_MODE=online` for real-time tracking on the wandb dashboard, or `WANDB_MODE=offline` for local tracking and synchronize later. 

### 3. Training on multiple nodes

Distributed multi-node multi-GPU training is available via Slurm. We provide a sample Slurm script at [exec_config.sh](https://github.com/Ackesnal/RePaViT/exec_config.sh).

A sample code snippet of _exec_config.sh_ is as shown below:

```sh
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=120gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --job-name=repavit
#SBATCH -o RePaViT_Large_out.txt
#SBATCH -e RePaViT_Large_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))
export BATCH_SIZE=$(echo "scale=0; 4096 / $WORLD_SIZE" | bc)

WANDB_MODE=online torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE main.py \
  --model=RePaViT_Large \
  --batch_size=$BATCH_SIZE \
  --epochs=300 \
  --dist_eval \
  --channel_idle \
  --idle_ratio=0.75 \
  --feature_norm=BatchNorm \
  --data_path=/path/to/imagenet \
  --output_dir=/path/to/output \
  --lr=1e-3 \
  --min_lr=5e-5 \
  --warmup_lr=1e-6 \
  --warmup_epochs=20 \
  --unscale_lr \
  --weight_decay=0.05 \
  --opt=lamb \
  --drop_path=0.3 \
  --wandb
  #--wandb_entity=your-entity-name \
  #--wandb_suffix=your-customized-suffix
```
where `--nodes` and `--gres` determines how many SLURM nodes and how many GPUs on each node you want to use. The batch size of each parallel process will be automatically calculated based on the world size.

## Supported Models

In this repo, we currently support the following backbone model(name)s:

* RePaViT-Tiny
* RePaViT-Small
* RePaViT-Base
* RePaViT-Large
* RePaViT-Huge
* RePaSwin-Tiny
* RePaSwin-Small
* RePaSwin-Base

We have also provided the implementation on MLPMixer and PoolFormer but have not tested them. The support for more backbones will be included in our future work.

## Evaluation
### 1. Accuracy evaluation

To evaluate the prediction performance, please run the following code. Please ensure `--idle_ratio` is set to the same value as the pretrained model weight.

__[RePaViT-Large] performance test:__
```
torchrun --nproc_per_node=4 main.py \
  --model=RePaViT_Large \
  --batch_size=512 \
  --eval \
  --dist_eval \
  --channel_idle \
  --idle_ratio=0.75 \
  --feature_norm=BatchNorm \
  --data_path=/path/to/imagenet \
  --output_dir=/path/to/output \
  --resume=/path/to/pretrained_weight.pth
```

### 2. Inference speed test

To test inference speed, `--test_speed` and `--only_test_speed` arguments should be utilized, and the number of processes is recommended to set to 1:

__[RePaViT-Large] speed test:__
```
torchrun --nproc_per_node=1 main.py \
  --model=RePaViT_Large \
  --batch_size=512 \
  --eval \
  --dist_eval \
  --channel_idle \
  --idle_ratio=0.75 \
  --feature_norm=BatchNorm \
  --test_speed \
  --only_test_speed
```

### 3. Evaluation with Structural Reparameterization

To enable inference-stage model compression via structural reparameterization, you can simply add the argument __`--reparam`__ as:

__[RePaViT-Large] speed test after structural reparameterization:__
```
torchrun --nproc_per_node=1 main.py \
  --model=RePaViT_Large \
  --batch_size=512 \
  --eval \
  --dist_eval \
  --channel_idle \
  --idle_ratio=0.75 \
  --feature_norm=BatchNorm \
  --test_speed \
  --only_test_speed \
  --reparam
```

`--reparam` can be combined with performance evalutation as well. The prediction accuracy before and after reparameterization should be the same.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Reference
If you use this repo or find it useful, please consider citing:
```
@inproceedings{xu2025repavit,
  title = {RePaViT: Scalable Vision Transformer Acceleration via Structural Reparameterization on Feedforward Network Layers},
  author = {Xu, Xuwei and Li, Yang and Chen, Yudong and Liu, Jiajun and Wang, Sen},
  booktitle = {The 42nd International Conference on Machine Learning (ICML)},
  year = {2025}
}
```
