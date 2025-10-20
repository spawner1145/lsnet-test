import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    set_bn_eval=False,
                    accumulation_steps: int = 1,
                    contrastive_criterion=None,
                    contrastive_weight=0.1):
    model.train(set_training_mode)
    if set_bn_eval:
        set_bn_state(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    # Gradient accumulation variables
    accumulation_counter = 0
    accumulated_loss = 0.0
    skipped_samples = 0

    for samples, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        try:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            with torch.amp.autocast(enabled=False, device_type="cuda"):
                if contrastive_criterion is not None:
                    # For contrastive loss, get both features and outputs
                    features, outputs = model(samples, return_both=True)
                    
                    # Get base cross-entropy loss (without distillation)
                    if hasattr(criterion, 'get_base_loss'):
                        ce_loss = criterion.get_base_loss(outputs, targets)
                    else:
                        ce_loss = criterion(outputs, targets)
                    
                    contrastive_loss, vq_loss = contrastive_criterion(features, targets)
                    
                    # Combine losses: base CE + contrastive + VQ + distillation (if enabled)
                    loss = ce_loss + contrastive_weight * contrastive_loss + vq_loss
                    
                    # Add distillation loss if enabled
                    if hasattr(criterion, 'distillation_type') and criterion.distillation_type != 'none':
                        distillation_loss = criterion(samples, outputs, targets) - ce_loss
                        loss = loss + distillation_loss
                else:
                    outputs = model(samples)
                    loss = criterion(samples, outputs, targets)

            loss_value = loss.item()
            accumulated_loss += loss_value

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                        parameters=model.parameters(), create_graph=is_second_order,
                        need_update=(accumulation_counter + 1) % accumulation_steps == 0)

            accumulation_counter += 1

            # Update metrics for every step (not just accumulation steps)
            # This ensures proper logging even during accumulation
            current_loss = accumulated_loss / accumulation_counter if accumulation_counter > 0 else loss_value
            metric_logger.update(loss=current_loss)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # Perform optimizer step and reset gradients after accumulation_steps
            if accumulation_counter % accumulation_steps == 0:
                torch.cuda.synchronize()
                if model_ema is not None:
                    model_ema.update(model)

                # Reset accumulation variables
                accumulated_loss = 0.0

        except Exception as e:
            # Skip problematic samples and log warning
            skipped_samples += 1
            print(f"[Warning] Skipping problematic sample in epoch {epoch}: {str(e)}")
            continue

    # Log total skipped samples for this epoch
    if skipped_samples > 0:
        print(f"[Warning] Skipped {skipped_samples} problematic samples in epoch {epoch}")

    # Handle remaining accumulated gradients if any
    if accumulation_counter % accumulation_steps != 0:
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.update(loss=accumulated_loss / (accumulation_counter % accumulation_steps))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    skipped_samples = 0

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        try:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.amp.autocast(enabled=False, device_type="cuda"):
                output = model(images)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        except Exception as e:
            # Skip problematic samples and log warning
            skipped_samples += 1
            print(f"[Warning] Skipping problematic sample during evaluation: {str(e)}")
            continue

    # Log total skipped samples for evaluation
    if skipped_samples > 0:
        print(f"[Warning] Skipped {skipped_samples} problematic samples during evaluation")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
