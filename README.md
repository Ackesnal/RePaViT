
# RePaViT: Reparameterizable Vision Transformer

This repository contains PyTorch evaluation code, training code and pretrained models for __RePaViT__.

## Setup

First, clone the repository locally:
```
git clone https://github.com/Ackesnal/RePaViT.git
```
Then, install environments via Anaconda:
```
conda create -n repavit python=3.8.13 -y
conda activate repavit
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install timm==1.0.3 einops ptflops 
```
After the above installations, it is ready to run this repo. 

(OPTIONAL) We further utilize the [wandb](https://wandb.ai/site) for real-time track and training process visualization. However, you will need to register and login to wandb first.
```
pip install wandb
```

## Dataset preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:
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
### Ordinary training on a single node
To train RePaViT on ImageNet on a single node with 4 gpus for 300 epochs run:

RePaViT-small
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 --use_env main.py --data-path /path/to/imagenet --output_dir=output/repavit_small --model RePaViT_small_patch16_224_layer12 --feature_norm=BatchNorm --channel_idle
```

RePaViT-tiny
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 --use_env main.py --data-path /path/to/imagenet --output_dir=output/repavit_tiny --model RePaViT_tiny_patch16_224_layer12 --feature_norm=BatchNorm --channel_idle
```
Please note that `--channel_idle` argument must be used with `--feature_norm=BatchNorm`.

### Ordinary training with wandb
To train with wandb visualization:
```
WANDB_MODE=online python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 --use_env main.py --data-path /path/to/imagenet --output_dir=output/repavit_small --model RePaViT_small_patch16_224_layer12 --feature_norm=BatchNorm --channel_idle --use_wandb
```
Please note that `WANDB_MODE` MUST be set when using `--use_wandb`. You can choose `WANDB_MODE=online` for real-time tracking on the wandb dashboard, or `WANDB_MODE=offline` for local tracking and synchronize later. 


### Multinode training

Distributed training is available via Slurm. We provide Slurm script at [exec_config.sh](https://github.com/Ackesnal/RePaViT/exec_config.sh). At the moment, only multi-node single-GPU training or single-node multi-GPU training is supported. <span style="color:red">Multi-node multi-GPU training has not been supported yet.</span>

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.
