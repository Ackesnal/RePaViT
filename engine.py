import utils
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss



def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, lr_scheduler = None, args = None,
                    use_amp=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(device=device, delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    idx = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        
        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(samples)
                loss = criterion(outputs, targets)
                loss = loss / args.accumulation_steps
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss = loss / args.accumulation_steps
        
        loss_value = loss.item() * args.accumulation_steps
        
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order, 
                    named_parameters=model.named_parameters(),
                    update_grad=(idx + 1) % args.accumulation_steps == 0)
            
        if (idx + 1) % args.accumulation_steps == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch*len(data_loader)+idx) // args.accumulation_steps)
            
        idx = idx + 1

        torch.cuda.synchronize()
            
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    torch.distributed.barrier()
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    

@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(device=device, delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast("cuda"):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        torch.cuda.synchronize()

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    if args.distributed:
        torch.distributed.barrier()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
