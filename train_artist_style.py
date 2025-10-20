"""
训练画师风格分类模型
支持聚类和分类两种用途
"""
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import csv
import json
import os
import logging
from pathlib import Path

#python train_artist_style.py --model lsnet_t_artist --data-path artist_dataset --output-dir outputs_artist --batch-size 128 --epochs 400 --num_workers 8
#python train_artist_style.py --model lsnet_t_artist --data-path artist_dataset --output-dir outputs_artist --batch-size 128 --epochs 400 --num_workers 8 --resume outputs_artist\checkpoint.pth

# PyTorch 2.6+ 在默认 weights_only=True 的情况下，需要显式允许部分对象反序列化
try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([argparse.Namespace])
except Exception:
    pass

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from data.samplers import RASampler
from data.datasets import build_dataset
from data.threeaugment import new_data_aug_generator
from engine import train_one_epoch, evaluate
from losses import DistillationLoss, ContrastiveLoss

from model import lsnet_artist
import utils


def _patch_logging_clear_cache():
    """Work around Python logging bug where PlaceHolder lacks _cache."""
    manager_cls = logging.Manager
    if getattr(manager_cls._clear_cache, '_lsnet_patched', False):
        return

    from logging import _acquireLock, _releaseLock  # type: ignore

    def _safe_clear_cache(self):
        _acquireLock()
        try:
            for logger in self.loggerDict.values():
                cache = getattr(logger, '_cache', None)
                if cache is not None:
                    cache.clear()
            root_cache = getattr(self.root, '_cache', None)
            if root_cache is not None:
                root_cache.clear()
        finally:
            _releaseLock()

    _safe_clear_cache._lsnet_patched = True  # type: ignore[attr-defined]
    manager_cls._clear_cache = _safe_clear_cache


_patch_logging_clear_cache()


def _extract_class_to_idx(dataset):
    """Recursively extract class_to_idx mapping from dataset if available."""
    if dataset is None:
        return None
    if hasattr(dataset, 'class_to_idx') and isinstance(dataset.class_to_idx, dict):
        return dataset.class_to_idx
    if hasattr(dataset, 'dataset'):
        return _extract_class_to_idx(dataset.dataset)
    if hasattr(dataset, 'datasets'):
        for ds in getattr(dataset, 'datasets'):
            mapping = _extract_class_to_idx(ds)
            if mapping:
                return mapping
    return None


def _export_class_mapping_csv(class_to_idx, output_dir: Path) -> Path:
    """Write class mapping to CSV and return the file path."""
    if not class_to_idx:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'class_mapping.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'class_name'])
        for class_name, class_id in sorted(class_to_idx.items(), key=lambda kv: kv[1]):
            writer.writerow([class_id, class_name])
    return csv_path


def _load_finetune_weights(model, finetune_path, args):
    print(f"Finetuning from checkpoint: {finetune_path}")
    if finetune_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            finetune_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(finetune_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[new_key] = v

    # 如果分类头维度不一致，移除相关参数
    head_weight_keys = [
        'head.l.weight',
        'head.linear.weight',
        'head.weight'
    ]
    reset_head = False
    for key in head_weight_keys:
        if key in cleaned_state_dict and cleaned_state_dict[key].shape[0] != args.nb_classes:
            reset_head = True
            break
    if reset_head:
        keys_to_remove = [k for k in cleaned_state_dict.keys() if k.startswith('head.') or k.startswith('head_dist.')]
        for key in keys_to_remove:
            cleaned_state_dict.pop(key, None)
        print('Removed classification head parameters due to class mismatch; they will be randomly re-initialized.')

    msg = model.load_state_dict(cleaned_state_dict, strict=False)
    if msg.missing_keys:
        print(f'Finetune load missing keys: {msg.missing_keys}')
    if msg.unexpected_keys:
        print(f'Finetune load unexpected keys: {msg.unexpected_keys}')


def get_args_parser():
    parser = argparse.ArgumentParser('LSNet Artist Style Training', add_help=False)
    
    # 基本参数
    parser.add_argument('--batch-size', default=64, type=int,  # 为大规模训练优化
                        help='Batch size per GPU (optimized for large datasets)')
    parser.add_argument('--epochs', default=200, type=int,  # 更长的训练时间
                        help='Total training epochs (longer for large datasets)')
    parser.add_argument('--accumulation-steps', default=1, type=int,  # 可以设置梯度累积
                        help='Gradient accumulation steps (set >1 for larger effective batch size)')
    
    # 模型参数
    parser.add_argument('--model', default='lsnet_t_artist', type=str, 
                        choices=['lsnet_t_artist', 'lsnet_s_artist', 'lsnet_b_artist', 'lsnet_l_artist', 'lsnet_xl_artist'],
                        help='Model architecture')
    parser.add_argument('--input-size', default=224, type=int, 
                        help='Input image size')
    parser.add_argument('--feature-dim', default=2048, type=int,  # 更大的特征维度
                        help='Feature dimension for clustering (larger for massive classes)')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Resize validation images to fine-tune resolution')
    parser.add_argument('--use-projection', action='store_true', default=True,
                        help='Use projection layer for feature extraction')
    parser.add_argument('--no-projection', dest='use_projection', action='store_false',
                        help='Do not use projection layer')
    
    # EMA参数
    parser.add_argument('--model-ema', action='store_true', default=True)
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.add_argument('--model-ema-decay', type=float, default=0.99996)
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False)

    # 优化器参数
    parser.add_argument('--opt', default='adamw', type=str,
                        help='Optimizer')
    parser.add_argument('--opt-eps', default=1e-8, type=float,
                        help='Optimizer epsilon')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+',
                        help='Optimizer betas')
    parser.add_argument('--clip-grad', type=float, default=0.02,
                        help='Clip gradient norm')
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.1,  # 更大的weight decay
                        help='Weight decay (higher for large models)')

    # 学习率调度参数
    parser.add_argument('--sched', default='cosine', type=str,
                        help='LR scheduler')
    parser.add_argument('--lr', type=float, default=1e-3,  # 更大的学习率
                        help='Learning rate')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None,
                        help='Learning rate noise')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67)
    parser.add_argument('--lr-noise-std', type=float, default=1.0)
    parser.add_argument('--warmup-lr', type=float, default=1e-6,
                        help='Warmup learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-5,
                        help='Minimum learning rate')
    parser.add_argument('--decay-epochs', type=float, default=30)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--cooldown-epochs', type=int, default=10)
    parser.add_argument('--patience-epochs', type=int, default=10)
    parser.add_argument('--decay-rate', type=float, default=0.1)

    # 数据增强参数
    parser.add_argument('--color-jitter', type=float, default=0.4,
                        help='Color jitter factor')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1',
                        help='AutoAugment policy')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation')
    
    # Mixup/Cutmix参数
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='Mixup alpha')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='Cutmix alpha')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='Cutmix min/max ratio')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params')

    # 随机擦除参数
    parser.add_argument('--reprob', type=float, default=0.25,
                        help='Random erase prob')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # 数据集参数
    parser.add_argument('--data-path', default='./data/artist_dataset', type=str,
                        help='Dataset path (ImageFolder format)')
    parser.add_argument('--data-set', default='IMNET', type=str,
                        help='Dataset type (use IMNET for ImageFolder format)')
    parser.add_argument('--num-classes', default=None, type=int,  # 自动检测类别数
                        help='Number of artist classes')
    parser.add_argument('--inat-category', default='name', type=str,
                        help='Label category to use for INat datasets')
    parser.add_argument('--output-dir', default='./output/artist_style',
                        help='Path to save outputs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('--resume', default='', 
                        help='Resume from checkpoint')
    parser.add_argument('--finetune-from', default='',
                        help='Load pretrained weights for finetuning (model weights only)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=16, type=int,  # 更多workers
                        help='Number of data loading workers')
    parser.add_argument('--pin-mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')

    # 分布式训练参数
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--dist-eval', dest='dist_eval', action='store_true', default=False,
                        help='Enable distributed evaluation (uses DistributedSampler for validation set)')
    parser.add_argument('--no-dist-eval', dest='dist_eval', action='store_false',
                        help='Disable distributed evaluation')
    
    # 蒸馏参数
    parser.add_argument('--teacher-model', default=None, type=str,
                        help='Teacher model for distillation')
    parser.add_argument('--teacher-path', default='', type=str,
                        help='Teacher checkpoint path')
    parser.add_argument('--distillation-type', default='none', 
                        choices=['none', 'soft', 'hard'],
                        help='Distillation type')
    parser.add_argument('--distillation-alpha', default=0.5, type=float,
                        help='Distillation alpha')
    parser.add_argument('--distillation-tau', default=1.0, type=float,
                        help='Distillation temperature')
    
    # 对比损失参数
    parser.add_argument('--contrastive-loss', action='store_true', default=False,
                        help='Enable supervised contrastive loss')
    parser.add_argument('--contrastive-weight', type=float, default=0.0,
                        help='Weight for contrastive loss (0.0 = disabled, 1.0 = equal weight)')
    parser.add_argument('--contrastive-temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    parser.add_argument('--use-vq', action='store_true', default=False,
                        help='Use Vector Quantization in contrastive loss')
    parser.add_argument('--vq-num-embeddings', type=int, default=256,
                        help='Number of embeddings in VQ codebook')
    parser.add_argument('--vq-commitment-cost', type=float, default=0.25,
                        help='Commitment cost for VQ loss')
    parser.add_argument('--eval-every', default=1, type=int,
                        help='Evaluate every N epochs (default: 1, evaluate every epoch)')
    parser.add_argument('--save-every', default=None, type=int,
                        help='Save checkpoint every N epochs (default: None, only save final and best checkpoints)')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    
    print(args)
    
    device = torch.device(args.device)
    
    # 固定随机种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True
    
    # 构建数据集
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    output_dir = Path(args.output_dir) if args.output_dir else None
    class_mapping_csv = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if utils.is_main_process():
            class_to_idx = _extract_class_to_idx(dataset_train)
            class_mapping_csv = _export_class_mapping_csv(class_to_idx, output_dir)
            if class_mapping_csv:
                print(f"Saved class mapping CSV to {class_mapping_csv}")
                setattr(args, 'class_mapping_csv', str(class_mapping_csv))

    if not hasattr(args, 'class_mapping_csv'):
        setattr(args, 'class_mapping_csv', str(class_mapping_csv) if class_mapping_csv else '')
    
    if True:  # args.distributed
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
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
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    # Mixup数据增强
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    # 创建模型
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        feature_dim=args.feature_dim,
        use_projection=args.use_projection,
    )
    
    model.to(device)

    if args.finetune_from:
        if args.resume:
            raise ValueError('--finetune-from cannot be combined with --resume. Please choose one.')
        _load_finetune_weights(model, args.finetune_from, args)
        args.start_epoch = 0
    
    model_ema = None
    if args.model_ema:
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
    print(f'Number of params: {n_parameters}')
    
    # 创建优化器
    # 考虑梯度累积的有效batch size进行学习率缩放
    effective_batch_size = args.batch_size * args.accumulation_steps * utils.get_world_size()
    linear_scaled_lr = args.lr * effective_batch_size / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    
    # 创建学习率调度器
    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    # 创建损失函数
    if mixup_active:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # 创建对比损失（如果启用）
    contrastive_criterion = None
    if args.contrastive_loss:
        # 如果启用对比损失但权重为0，则使用默认权重0.1
        if args.contrastive_weight == 0.0:
            args.contrastive_weight = 0.1
        contrastive_criterion = ContrastiveLoss(
            temperature=args.contrastive_temperature,
            use_vq=args.use_vq,
            vq_num_embeddings=args.vq_num_embeddings,
            vq_embedding_dim=256,  # 使用默认特征维度
            vq_commitment_cost=args.vq_commitment_cost
        )
        contrastive_criterion.to(device)
    
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()
    
    # 包装蒸馏损失（当类型为 none 时等价于原始损失）
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    
    # output_dir 已确保在上方创建
    
    # 恢复训练
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    
    # 仅评估
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    
    # 训练循环
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode, model_ema, mixup_fn,
            set_training_mode=True,
            accumulation_steps=args.accumulation_steps,
            contrastive_criterion=contrastive_criterion,
            contrastive_weight=args.contrastive_weight
        )
        
        lr_scheduler.step(epoch)
        
        # 根据eval-every决定是否进行评估
        test_stats = None
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')
        
        # 根据save-every决定是否保存checkpoint（不影响最后epoch）
        should_save_checkpoint = (
            (args.save_every is not None and (epoch + 1) % args.save_every == 0) or  # 定期保存
            epoch + 1 == args.epochs  # 最后epoch总是保存
        )
        
        if args.output_dir and should_save_checkpoint:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema) if model_ema is not None else None,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch + 1}")
        
        # 保存最佳模型（不受save-interval影响）
        if args.output_dir and test_stats is not None and test_stats["acc1"] >= max_accuracy:
            checkpoint_paths = [output_dir / 'best_checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema) if model_ema is not None else None,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
            print(f"Saved best checkpoint at epoch {epoch + 1} with accuracy {test_stats['acc1']:.2f}%")
        
        # 记录日志
        if test_stats is not None:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LSNet Artist Style Training', parents=[get_args_parser()])
    args = parser.parse_args()
    if not hasattr(args, 'finetune'):
        setattr(args, 'finetune', False)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
