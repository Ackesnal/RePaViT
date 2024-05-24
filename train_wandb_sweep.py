# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler, create_scheduler_v2, scheduler_kwargs
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator
import torch.autograd.profiler as profiler

import models_v3

import utils
import os
import random

import wandb


# Glogal variable
args = None


def get_args_parser():
    parser = argparse.ArgumentParser('RePaViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.add_argument('--accumulation-steps', default=1, type=int)
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--use_wandb', default=False, action='store_true')
    parser.add_argument('--wandb_no_loss', default=False, action='store_true')
    parser.add_argument('--wandb_suffix', default="", type=str)
    parser.add_argument('--wandb_sweep_count', default=1, type=int,)
    
    # NFViT Ablation Augments
    parser.add_argument('--shortcut_type', default='PerLayer', type=str, choices=['PerLayer', 'PerOperation'])
    parser.add_argument('--affected_layers', default='None', type=str, choices=['None', 'Both', 'MHSA', 'FFN'])
    parser.add_argument('--feature_norm', default='LayerNorm', type=str, choices=['LayerNorm', 'BatchNorm', 'EmpiricalSTD', 'None'])
    parser.add_argument('--weight_standardization', default=False, action='store_true')
    parser.add_argument('--channel_idle', default=False, action='store_true')
    parser.add_argument('--po_shortcut', default=False, action='store_true')
    parser.add_argument('--shortcut_gain', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--finetune_gain', type=int, default=300)
    parser.add_argument('--finetune_gamma', type=int, default=300)
    parser.add_argument('--finetune_std', type=int, default=300)
    parser.add_argument('--activation', default='GELU', type=str, choices=['ReLU', 'GELU', 'Sigmoid', 'LeakyReLU', 'SiLU'])
    parser.add_argument('--reparam', default=False, action='store_true')
    return parser


def main(config=None):
    if not torch.distributed.is_initialized():
        utils.init_distributed_mode(args)
    
    if args.rank == 0:
        wandb.init(config=config, mode=os.environ['WANDB_MODE'])
        config = dict(wandb.config)
        
        args.opt = config["opt"]
        args.lr = config["lr"]
        args.min_lr = config["min_lr"]
        args.warmup_epochs = config["warmup_epochs"]
        args.weight_decay = config["weight_decay"]
        args.shortcut_gain = config["shortcut_gain"]
        args.drop_path = config["drop_path"]
        args.batch_size = config["batch_size"] // args.world_size
        args.unscale_lr = True
        torch.distributed.broadcast_object_list([config], src=0)
    else:
        config = [None]
        torch.distributed.broadcast_object_list(config, src=0)
        config = config[0]
        
        args.opt = config["opt"]
        args.lr = config["lr"]
        args.min_lr = config["min_lr"]
        args.warmup_epochs = config["warmup_epochs"]
        args.weight_decay = config["weight_decay"]
        args.shortcut_gain = config["shortcut_gain"]
        args.drop_path = config["drop_path"]
        args.batch_size = config["batch_size"] // args.world_size
        args.unscale_lr = True
    
    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    if args.activation=="ReLU":
        act_layer=torch.nn.ReLU
    elif args.activation=="GELU":
        act_layer=torch.nn.GELU
    elif args.activation=="LeakyReLU":
        act_layer=torch.nn.LeakyReLU
    elif args.activation=="Sigmoid":
        act_layer=torch.nn.Sigmoid
    elif args.activation=="SiLU":
        act_layer=torch.nn.SiLU
    
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        channel_idle=args.channel_idle,
        po_shortcut=args.po_shortcut,
        feature_norm=args.feature_norm,
        shortcut_gain=args.shortcut_gain,
        act_layer=act_layer
    )
    
    model.to(device)
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        args.lr = args.lr * args.batch_size * utils.get_world_size() / 1024.0
        args.warmup_lr = args.warmup_lr * args.batch_size * utils.get_world_size() / 1024.0
        args.min_lr = args.min_lr * args.batch_size * utils.get_world_size() / 1024.0
    args.step_on_epochs = False
    args.sched_on_updates = True
    args.updates_per_epoch = len(data_loader_train)
    # gradient accumulation also need to scale the learning rate
    if args.accumulation_steps > 1:
        args.lr = args.lr * args.accumulation_steps
        args.warmup_lr = args.warmup_lr * args.accumulation_steps
        args.min_lr = args.min_lr * args.accumulation_steps
        args.step_on_epochs = False
        args.sched_on_updates = True
        args.updates_per_epoch = len(data_loader_train)//args.accumulation_steps
        
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = utils.NativeScalerWithGradNormCount()
        
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=args.updates_per_epoch,
    )
    
    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='token',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(criterion, teacher_model, args.distillation_type, 
                                 args.distillation_alpha, args.distillation_tau)

    output_dir = Path(args.output_dir)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    
    use_amp=True
    for epoch in range(args.start_epoch, args.epochs):
        while True:
            if args.feature_norm == "EmpiricalSTD" and epoch == args.finetune_std:
                for name, param in model.module.named_parameters():
                    if "std" in name:
                        param.requires_grad_(True)
                            
                model_without_ddp = model.module
                optimizer = create_optimizer(args, model_without_ddp)
                loss_scaler = utils.NativeScalerWithGradNormCount()
                                
                lr_scheduler, num_epochs = create_scheduler_v2(
                            optimizer,
                            **scheduler_kwargs(args),
                            updates_per_epoch=args.updates_per_epoch,
                )
                lr_scheduler.step(epoch*len(data_loader_train) // args.accumulation_steps)
            if (args.po_shortcut or args.channel_idle) and epoch == args.finetune_gain:
                for name, param in model.module.named_parameters():
                    if "gain" in name:
                        param.requires_grad_(True)
                            
                model_without_ddp = model.module
                optimizer = create_optimizer(args, model_without_ddp)
                loss_scaler = utils.NativeScalerWithGradNormCount()
                                
                lr_scheduler, num_epochs = create_scheduler_v2(
                            optimizer,
                            **scheduler_kwargs(args),
                            updates_per_epoch=args.updates_per_epoch,
                )
                lr_scheduler.step(epoch*len(data_loader_train) // args.accumulation_steps)
                    
                        
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            
            train_stats, nan_loss_flag = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
                lr_scheduler = lr_scheduler,
                use_amp = use_amp,
                args = args,
            )
            
            if nan_loss_flag:
                print("Reload the last epoch's checkpoint.")
                checkpoint_path = output_dir/'checkpoint.pth'
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model_without_ddp.load_state_dict(checkpoint['model'])
                
                if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                    args.start_epoch = checkpoint['epoch']
                    model_without_ddp = model.module
                                    
                    optimizer = create_optimizer(args, model_without_ddp)
                    loss_scaler = utils.NativeScalerWithGradNormCount()
                                
                    lr_scheduler, num_epochs = create_scheduler_v2(
                            optimizer,
                            **scheduler_kwargs(args),
                            updates_per_epoch=args.updates_per_epoch,
                    )
                    lr_scheduler.step(args.start_epoch*len(data_loader_train))
                    
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    if args.model_ema:
                        utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                    if 'scaler' in checkpoint:
                        loss_scaler.load_state_dict(checkpoint['scaler'])
                lr_scheduler.step(args.start_epoch*len(data_loader_train))
                
                print("Stop torch.cuda.amp.autocast in the current epoch")
                use_amp = False
                
                continue
            
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
            
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
            print(f'Max accuracy: {max_accuracy:.2f}%')
    
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
                  
            if args.rank == 0:
                wandb.log({"accuracy": test_stats["acc1"]})
                
            break
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    # Clean up
    if args.rank == 0:
        wandb.finish()
        
                                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('RePaViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if 'SLURM_PROCID' in os.environ and int(os.environ['SLURM_NNODES']) > 1:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ['SLURM_NNODES']) * int(os.environ['SLURM_NTASKS_PER_NODE'])
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        
    # Define the search space
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "accuracy"},
        "parameters": {
            "model": {"value":args.model, "distribution": "constant"},
            "layer": {"value": "FFN" if args.channel_idle and not args.po_shortcut else "MHSA" if not args.channel_idle and args.po_shortcut else "Both" if args.channel_idle and args.po_shortcut else "None", "distribution": "constant"},
            "norm": {"value": args.feature_norm, "distribution": "constant"},
            "batch_size": {"values": [1024, 2048, 3072, 4096]},
            'opt': {'values': ["nadamw", "adamw", "lamb"]},
            'lr': {'min': 1e-4, 'max': 5e-3, 'distribution': 'uniform'},
            'min_lr': {'min': 5e-7, 'max': 1e-4, 'distribution': 'uniform'},
            'warmup_epochs': {'min': 5, 'max': 20, 'distribution': 'q_uniform', 'q': 5},
            'weight_decay': {'min': -6.90775, 'max': -2.30258, 'distribution': 'log_uniform'},
            'drop_path': {'min': 1e-2, 'max': 1e-1, 'distribution': 'uniform'},
            'shortcut_gain': {'min': -2.30258, 'max': 0.13976, 'distribution': 'q_log_uniform', 'q': 0.1}
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 100,
            "max_iter": 200,
            "eta": 3,
        }
    }
    
    if args.rank == 0:
        project_name = args.model.split("_")[0] + "_" + args.model.split("_")[1] + "_" + args.wandb_suffix
        sweep_id = wandb.sweep(sweep_configuration, project=project_name)
        wandb.agent(sweep_id, function=main, count=args.wandb_sweep_count, project=project_name, entity="ackesnal-ai")
    else:
        cnt = 0
        while cnt < args.wandb_sweep_count:
            main(config=None)
            cnt += 1