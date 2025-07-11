# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import time
import json
import utils
import wandb
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from timm.data import Mixup
from timm.models import create_model
from timm.optim import create_optimizer
from timm.utils import get_state_dict, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler_v2, scheduler_kwargs

from samplers import RASampler
from datasets import build_dataset
from augment import new_data_aug_generator
from engine import train_one_epoch, evaluate
from ptflops import get_model_complexity_info

import repavit
import repaswin
import repapoolformer
import repamlpmixer



def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce_loss', action='store_true')
    parser.add_argument('--unscale_lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model_ema', action='store_true')
    parser.add_argument('--no_model_ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.99996, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr_noise_pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr_noise_std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated_aug', action='store_true')
    parser.add_argument('--no_repeated_aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train_mode', action='store_true')
    parser.add_argument('--no_train_mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') # 3augment
    
    parser.add_argument('--src', action='store_true') # simple random crop
    
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
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat_category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--rocksdb', type=str, default=None)
    parser.add_argument('--prefetch_factor', default=2, type=int)

    # Logging parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_freq', default=10, type=int,
                        help='weight saving frequency (epochs)')
    
    # Wandb
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--wandb_suffix', default="", type=str)
    parser.add_argument('--wandb_entity', default="", type=str)
    
    # Training specs
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval_crop_ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.add_argument('--accumulation_steps', default=1, type=int)
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Speed tests and time profiling
    parser.add_argument('--test_speed', action='store_true')
    
    # RePaViT arguments
    parser.add_argument('--feature_norm', default='LayerNorm', type=str)
    parser.add_argument('--channel_idle', default=False, action='store_true')
    parser.add_argument('--idle_ratio', default=0.75, type=float)
    parser.add_argument('--heuristic', type=str, default="static")
    parser.add_argument('--reparam', default=False, action='store_true')
    
    return parser



def get_macs(model):
    macs, params = get_model_complexity_info(model, (3, 224, 224), print_per_layer_stat=False, as_strings=False)
    if next(model.parameters()).get_device()==0:
        print('{:<} {:<}{:<}'.format('Computational complexity: ', round(macs*1e-9, 2), 'GMACs'))
        print('{:<} {:<}{:<}'.format('Number of parameters: ', round(params*1e-6, 2), 'M'))
        print()



def speed_test(model, ntest=100, batchsize=128, x=None):
    if x is None:
        x = torch.rand(batchsize, 3, 224, 224).cuda()
    else:
        batchsize = x.shape[0]

    start = time.time()
    with torch.no_grad():
        for i in range(ntest):
            model(x)
    torch.cuda.synchronize()
    end = time.time()

    elapse = end - start
    speed = batchsize * ntest / elapse

    return speed



def main(args):
    
    ################################################################################################
    #----- ↓↓↓↓↓ 0. Initialize environment ↓↓↓↓↓ ------#############################################
    
    # Automatically set distributed environment under SLURM and local machine
    utils.init_distributed_mode(args)
    if args.distributed:
        cudnn.benchmark = True
    # Set torch device
    torch.device(args.device)
    
    # Set seed for reproducibility
    seed = args.seed + args.global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Present all arguments
    print(args)
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    
    #----- ↑↑↑↑↑ 0. Initialize environment ↑↑↑↑↑ ------#############################################
    ################################################################################################
    
    ################################################################################################
    #----- ↓↓↓↓↓ 1. Load dataset and initialize DataLoader (when not testing speed) ↓↓↓↓↓ ------####

    if not args.test_speed:
        # Load dataset with either raw ImageNet files or RocksDB supported
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        dataset_val, _ = build_dataset(is_train=False, args=args)

        if args.distributed:
            if args.repeated_aug:
                sampler_train = RASampler(
                    dataset_train, num_replicas=args.world_size, rank=args.global_rank, shuffle=True
                )
            else:
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=args.world_size, rank=args.global_rank, shuffle=True
                )
            if args.dist_eval:
                if len(dataset_val) % args.world_size != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible',
                          'by process number. This will slightly alter validation results as extra',
                          'duplicate entries are added to achieve equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=args.world_size, rank=args.global_rank, shuffle=False
                    )
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        # Create DataLoader for training and validation datasets
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            drop_last=True,
        )
        if args.ThreeAugment:
            data_loader_train.dataset.transform = new_data_aug_generator(args)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(args.batch_size * 1.5),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    # Mixup dataset when training
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
        
    #----- ↑↑↑↑↑ 1. Load dataset and initialize DataLoader (when not testing speed) ↑↑↑↑↑ ------####
    ################################################################################################
    
    ################################################################################################
    #----- ↓↓↓↓↓ 2. Initialize backbone model w/o reparameterization (full size) ↓↓↓↓↓ ------#######
    
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        feature_norm=args.feature_norm,
        channel_idle=args.channel_idle,
        idle_ratio=args.idle_ratio,
        heuristic=args.heuristic
    )
    
    # Send model to device
    model.to(args.device)
    
    # # Get model parameters count and computational complexity
    get_macs(model)
    model.train()
    
    # Create EMA model if specified
    model_ema = None
    if not args.eval and not args.test_speed and args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=''
        )

    # If using DDP, wrap the model in DistributedDataParallel
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device])
        model_without_ddp = model.module
        
    #----- ↑↑↑↑↑ 2. Initialize backbone model w/o reparameterization (full size) ↑↑↑↑↑ ------#######
    ################################################################################################
    
    ################################################################################################
    #----- ↓↓↓↓↓ 3. Test model inference speed only ↓↓↓↓↓ ------####################################
    
    # If only testing speed, skip training and evaluation
    if args.test_speed:
        # Get the model without DDP wrapper
        model = model_without_ddp
        # Change to eval mode
        model.eval()
        
        # If needing to reparameterize the model before speed test
        if args.reparam:
            print("Reparameterizing the backbone ...")
            model.reparam()
            model.to(args.device)
            print("Reparameterization done!")
        
        # Ensure only test speed on one process
        if args.global_rank == 0:
            x = torch.rand(128, 3, 224, 224).to(args.device)
            # test model throughput for three times to ensure accuracy
            print('Start inference speed testing...')
            inference_speed = speed_test(model, x=x)
            print('inference_speed:', inference_speed, 'images/s')
            inference_speed = speed_test(model, x=x)
            print('inference_speed:', inference_speed, 'images/s')
            inference_speed = speed_test(model, x=x)
            print('inference_speed:', inference_speed, 'images/s')
        
        # If using distributed training, destroy the process group, then exit
        if args.distributed:
            torch.distributed.destroy_process_group()
        return
    
    #----- ↑↑↑↑↑ 3. Test model inference speed only ↑↑↑↑↑ ------####################################
    ################################################################################################

    ################################################################################################
    #----- ↓↓↓↓↓ 4. Evaluation only ↓↓↓↓↓ ------####################################################
    
    if args.eval:
        # First check if model weight is provided via --resume
        # If provided, load the model weight
        if args.resume:
            # First load weights from the specified path
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            
            # Second load the model state dict
            if args.distributed:
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])
        
        # If needing to reparameterize the model before evaluation
        if args.reparam:
            print("Reparametering the backbone ...")
            if args.distributed:
                model.module.reparam()
                model.module.to(args.device)
            else:
                model.reparam()
                model.to(args.device)
            print("Reparameterization done!")
        
        test_stats = evaluate(data_loader_val, model, args.device, args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if args.distributed:
            torch.distributed.destroy_process_group()
        return
    
    #----- ↑↑↑↑↑ 4. Evaluation only ↑↑↑↑↑ ------####################################################
    ################################################################################################
    
    ################################################################################################
    #----- ↓↓↓↓↓ 5. Initialize lr scheduler, loss, loss scaler and optimizer ↓↓↓↓↓ ------###########
    
    # Set up all the learning rates based on the arguments
    if not args.unscale_lr:
        args.lr = args.lr * args.batch_size * utils.get_world_size() / 1024.0
        args.warmup_lr = args.warmup_lr * args.batch_size * utils.get_world_size() / 1024.0
        args.min_lr = args.min_lr * args.batch_size * utils.get_world_size() / 1024.0
    args.step_on_epochs = False
    args.sched_on_updates = True
    args.updates_per_epoch = len(data_loader_train)
    # Gradient accumulation also need to scale the learning rate
    if args.accumulation_steps > 1:
        args.lr = args.lr * args.accumulation_steps
        args.warmup_lr = args.warmup_lr * args.accumulation_steps
        args.min_lr = args.min_lr * args.accumulation_steps
        args.updates_per_epoch = len(data_loader_train)//args.accumulation_steps
    
    # Create optimizer
    optimizer = create_optimizer(args, model_without_ddp)
    
    # Create loss scaler for mixed precision training
    loss_scaler = utils.NativeScalerWithGradNormCount()
    
    # Create lr scheduler
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=args.updates_per_epoch,
    )
    
    # Create loss function / Criterion
    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    #----- ↑↑↑↑↑ 5. Initialize lr scheduler, loss, grad scaler and optimizer ↑↑↑↑↑ ------###########
    ################################################################################################
    
    ################################################################################################
    #----- ↓↓↓↓↓ 6. Resume from checkpoint ↓↓↓↓↓ ------#############################################
    
    if args.resume:
        # First load weights from the specified path
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        
        # Second load the model state dict
        if args.distributed:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        
        # Load optimizer, lr_scheduler, epoch, model_ema, and loss scaler if available
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch*len(data_loader_train))
        
        
    #----- ↑↑↑↑↑ 6. Resume from checkpoint ↑↑↑↑↑ ------#############################################
    ################################################################################################
    
    ################################################################################################
    #----- ↓↓↓↓↓ 7. Set up pre-training configs ↓↓↓↓↓ ------########################################
    
    if args.global_rank == 0 and args.wandb:
        project_name = f'{args.model}_{args.wandb_suffix}' if args.wandb_suffix else f'{args.model}'
        trial_name = f'{args.model}_{random.randint(0, 10000):04d}_{datetime.date.today()}'
        wandb.init(
            # set the wandb project where this run will be logged
            entity=args.wandb_entity,
            project=project_name,
            name=trial_name,
            # track hyperparameters and run metadata
            config={
                "model": args.model,
                "batch_size": args.batch_size*args.accumulation_steps*args.world_size,
                "epochs": args.epochs,
                "opt": args.opt,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "warmup_lr": args.warmup_lr,
                "warmup_epoch": args.warmup_epochs,
                "weight_decay": args.weight_decay,
                "drop_path": args.drop_path,
                "channel_idle": args.channel_idle,
                "heuristic": args.heuristic if args.channel_idle else None,
                "ffn_norm": args.feature_norm if args.channel_idle else "LayerNorm",
                "idle_ratio": args.idle_ratio if args.channel_idle else 0.0,
            }, 
            mode=os.environ['WANDB_MODE']
        )
    
    max_accuracy = 0.0
    use_amp=True
    
    #----- ↑↑↑↑↑ 7. Set up pre-training configs ↑↑↑↑↑ ------########################################
    ################################################################################################

    ################################################################################################
    #----- ↓↓↓↓↓ 8. Model training ↓↓↓↓↓ ------#####################################################
    
    print(f"Start training for {args.epochs} epochs")
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, args.device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode, 
            lr_scheduler = lr_scheduler,
            use_amp = use_amp, args = args,
        )
        
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
                    'args': args
                }, checkpoint_path)
            
            if (epoch+1) % args.save_freq == 0:
                checkpoint_paths = [output_dir / f'checkpoint_{epoch+1}epoch.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args
                    }, checkpoint_path)
        
        test_stats = evaluate(data_loader_val, model, args.device, args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args
                    }, checkpoint_path)
            
        print(f'Max accuracy: {max_accuracy:.2f}%')
        
        if args.global_rank == 0 and args.wandb:
            wandb.log({"accuracy": test_stats["acc1"], "loss": train_stats["loss"],})

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch}
            
        if args.output_dir and args.global_rank == 0:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
    #----- ↑↑↑↑↑ 8. Model training ↑↑↑↑↑ ------#####################################################
    ################################################################################################
    
    ################################################################################################
    #----- ↓↓↓↓↓ 9. Post-training processes ↓↓↓↓↓ ------############################################
    
    if args.global_rank == 0 and args.wandb:
        wandb.finish()
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if args.distributed:
        # Destroy the process group if using distributed training
        torch.distributed.destroy_process_group()
    
    #----- ↑↑↑↑↑ 9. Post-training processes ↑↑↑↑↑ ------############################################
    ################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
