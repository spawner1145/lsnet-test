"""
Multi-label training entry point for LSNet artist style classification.
"""
import argparse
import datetime
import json
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import average_precision_score, f1_score
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, get_state_dict
from torchvision import transforms

from data.multilabel_dataset import MultiLabelImageDataset
from data.multilabel_utils import DEFAULT_LABEL_DELIMITER
from engine import train_one_epoch
from losses import DistillationLoss
import utils


# PyTorch 2.6+ compatibility for argparse.Namespace checkpoints
try:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])
except Exception:
    pass


def _load_finetune_weights(model, finetune_path, num_classes):
    print(f"Finetuning from checkpoint: {finetune_path}")
    if finetune_path.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            finetune_path, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(finetune_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[new_key] = v

    head_weight_keys = [
        "head.l.weight",
        "head.linear.weight",
        "head.weight",
    ]
    reset_head = False
    for key in head_weight_keys:
        if key in cleaned_state_dict and cleaned_state_dict[key].shape[0] != num_classes:
            reset_head = True
            break
    if reset_head:
        keys_to_remove = [k for k in cleaned_state_dict if k.startswith("head.") or k.startswith("head_dist.")]
        for key in keys_to_remove:
            cleaned_state_dict.pop(key, None)
        print("Removed classification head parameters due to class mismatch; they will be randomly re-initialized.")

    msg = model.load_state_dict(cleaned_state_dict, strict=False)
    if msg.missing_keys:
        print(f"Finetune load missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"Finetune load unexpected keys: {msg.unexpected_keys}")


def _build_transforms(args):
    train_transform = create_transform(
        input_size=args.input_size,
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.train_interpolation,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
    )
    if args.input_size <= 32:
        train_transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)

    if args.finetune:
        val_transform = transforms.Compose(
            [
                transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
    else:
        size = int((256 / 224) * args.input_size)
        val_transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
    return train_transform, val_transform


def _build_datasets(args):
    train_transform, val_transform = _build_transforms(args)

    dataset_train = MultiLabelImageDataset(
        root=args.data_path,
        ann_file=args.train_ann,
        transform=train_transform,
        label_delimiter=args.label_delimiter,
    )

    dataset_val = MultiLabelImageDataset(
        root=args.data_path,
        ann_file=args.val_ann,
        transform=val_transform,
        label_delimiter=args.label_delimiter,
        class_to_idx=dataset_train.class_to_idx,
        skip_missing=args.skip_missing,
    )

    return dataset_train, dataset_val


def _export_class_mapping(dataset: MultiLabelImageDataset, output_dir: Optional[Path]) -> Optional[Path]:
    if output_dir is None:
        return None
    return dataset.export_class_mapping(output_dir / "class_mapping.csv")


def _export_label_statistics(dataset: MultiLabelImageDataset, output_dir: Optional[Path]) -> Optional[Path]:
    if output_dir is None:
        return None
    return dataset.export_label_statistics(output_dir / "label_stats.csv")


def _average_precision(y_true: np.ndarray, y_score: np.ndarray, average: str) -> float:
    try:
        return float(average_precision_score(y_true, y_score, average=average))
    except ValueError:
        return float("nan")


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> float:
    try:
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    except ValueError:
        return float("nan")


def get_args_parser():
    parser = argparse.ArgumentParser("LSNet Multi-Label Training", add_help=False)

    # data
    parser.add_argument("--data-path", type=str, default="./data/artist_dataset", help="Dataset root directory")
    parser.add_argument("--train-ann", type=str, required=True, help="Training annotation CSV (image_path, labels)")
    parser.add_argument("--val-ann", type=str, required=True, help="Validation annotation CSV (image_path, labels)")
    parser.add_argument(
        "--label-delimiter",
        type=str,
        default=DEFAULT_LABEL_DELIMITER,
        help="Delimiter for label strings in CSV (default: comma)",
    )
    parser.add_argument("--skip-missing", action="store_true", help="Skip samples with missing image files")

    # optimization schedule
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", default=120, type=int, help="Total training epochs")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of data loading workers")
    parser.add_argument("--pin-mem", action="store_true", default=True, help="Pin CPU memory in DataLoader")
    parser.add_argument("--no-pin-mem", dest="pin_mem", action="store_false")

    parser.add_argument("--model", default="lsnet_t_artist", type=str,
                        choices=["lsnet_t_artist", "lsnet_s_artist", "lsnet_b_artist", "lsnet_l_artist", "lsnet_xl_artist", "lsnet_xl_artist_448"],
                        help="Model architecture")
    parser.add_argument("--input-size", default=224, type=int, help="Input image size")
    parser.add_argument("--feature-dim", default=None, type=int, help="Feature dimension for projection head")
    parser.add_argument("--use-projection", action="store_true", default=True)
    parser.add_argument("--no-projection", dest="use_projection", action="store_false")

    parser.add_argument("--model-ema", action="store_true", default=True)
    parser.add_argument("--no-model-ema", dest="model_ema", action="store_false")
    parser.add_argument("--model-ema-decay", type=float, default=0.9998)
    parser.add_argument("--model-ema-force-cpu", action="store_true", default=False)

    parser.add_argument("--opt", default="adamw", type=str, help="Optimizer name")
    parser.add_argument("--opt-eps", default=1e-8, type=float)
    parser.add_argument("--opt-betas", default=None, type=float, nargs="+")
    parser.add_argument("--clip-grad", type=float, default=0.02)
    parser.add_argument("--clip-mode", type=str, default="agc")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.05)

    parser.add_argument("--sched", default="cosine", type=str)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--cooldown-epochs", type=int, default=10)
    parser.add_argument("--patience-epochs", type=int, default=10)
    parser.add_argument("--decay-epochs", type=float, default=30)
    parser.add_argument("--decay-rate", type=float, default=0.1)

    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup alpha value")
    parser.add_argument("--cutmix", type=float, default=0.0, help="Cutmix alpha value")
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.0)
    parser.add_argument("--mixup-mode", type=str, default="batch")

    parser.add_argument("--color-jitter", type=float, default=0.3)
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--reprob", type=float, default=0.15)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--finetune", action="store_true", help="Resize validation images to training size")

    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold for metrics")

    parser.add_argument("--output-dir", default="./output/artist_multilabel", help="Directory to save checkpoints")
    parser.add_argument("--device", default="cuda", help="Training device")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    parser.add_argument("--finetune-from", default="", help="Load pretrained weights for finetuning")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true", help="Only run evaluation on the validation set")

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--dist-eval", dest="dist_eval", action="store_true", default=False)
    parser.add_argument("--no-dist-eval", dest="dist_eval", action="store_false")

    parser.add_argument("--teacher-model", default=None, type=str)
    parser.add_argument("--teacher-path", default="", type=str)
    parser.add_argument("--distillation-type", default="none", choices=["none", "soft", "hard"])
    parser.add_argument("--distillation-alpha", default=0.5, type=float)
    parser.add_argument("--distillation-tau", default=1.0, type=float)

    return parser


def evaluate_multilabel(data_loader, model, device, threshold: float = 0.5):
    criterion = torch.nn.BCEWithLogitsLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    model.eval()

    all_targets = []
    all_scores = []

    tp = 0.0
    fp = 0.0
    fn = 0.0
    total_labels = 0.0

    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 10, header):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            target_binary = (targets > 0).float()

            tp += (preds * target_binary).sum().item()
            fp += (preds * (1 - target_binary)).sum().item()
            fn += ((1 - preds) * target_binary).sum().item()
            total_labels += target_binary.sum().item()

            metric_logger.update(loss=loss.item())

            all_targets.append(target_binary.cpu())
            all_scores.append(probs.cpu())

    metric_logger.synchronize_between_processes()
    print("* loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_score = torch.cat(all_scores, dim=0).numpy()
    y_pred = (y_score >= threshold).astype(np.int32)

    map_macro = _average_precision(y_true, y_score, average="macro")
    map_micro = _average_precision(y_true, y_score, average="micro")
    f1_macro = _f1_score(y_true, y_pred, average="macro")
    f1_micro = _f1_score(y_true, y_pred, average="micro")

    coverage = float((y_pred.sum(axis=1) > 0).mean())
    avg_labels = float(y_true.sum(axis=1).mean())

    return {
        "loss": metric_logger.loss.global_avg,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "map_macro": map_macro,
        "map_micro": map_micro,
        "coverage": coverage,
        "avg_labels": avg_labels,
    }


def main(args):
    utils.init_distributed_mode(args)
    
    # 根据模型配置动态设置输入大小
    from model.lsnet_artist import default_cfgs_artist
    if args.model in default_cfgs_artist:
        model_cfg = default_cfgs_artist[args.model]
        configured_input_size = model_cfg.get('input_size', (3, 224, 224))[1]  # 获取高度（假设正方形）
        if args.input_size != configured_input_size:
            args.input_size = configured_input_size
            print(f"Auto-setting input_size to {configured_input_size} for model {args.model} (from config)")

    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, dataset_val = _build_datasets(args)
    args.nb_classes = dataset_train.num_classes

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    class_mapping_csv = _export_class_mapping(dataset_train, output_dir)
    label_stats_csv = _export_label_statistics(dataset_train, output_dir)
    if class_mapping_csv and utils.is_main_process():
        setattr(args, "class_mapping_csv", str(class_mapping_csv))
        print(f"Saved class mapping CSV to {class_mapping_csv}")
    if label_stats_csv and utils.is_main_process():
        setattr(args, "label_stats_csv", str(label_stats_csv))
        print(f"Saved label statistics CSV to {label_stats_csv}")

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: distributed evaluation with validation set not divisible by number of processes."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=0.0,
            num_classes=args.nb_classes,
        )

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
            raise ValueError("--finetune-from cannot be combined with --resume")
        _load_finetune_weights(model, args.finetune_from, args.nb_classes)
        args.start_epoch = 0

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    pos_weight = dataset_train.get_pos_weight().to(device)
    base_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = DistillationLoss(
        base_criterion,
        teacher_model=None,
        distillation_type=args.distillation_type,
        alpha=args.distillation_alpha,
        tau=args.distillation_tau,
    )

    teacher_model = None
    if args.distillation_type != "none":
        assert args.teacher_model, "Teacher model must be provided when using distillation"
        assert args.teacher_path, "Teacher checkpoint is required for distillation"
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool="avg",
        )
        if args.teacher_path.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.teacher_path, map_location="cpu")
        teacher_model.load_state_dict(checkpoint["model"])
        teacher_model.to(device)
        teacher_model.eval()
        criterion.teacher_model = teacher_model

    criterion.distillation_type = args.distillation_type

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.eval:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "lr_scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if "epoch" in checkpoint:
                args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema and "model_ema" in checkpoint:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])

    if args.eval:
        test_stats = evaluate_multilabel(data_loader_val, model, device, args.threshold)
        print(json.dumps(test_stats, indent=2))
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_metric = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            args.clip_mode,
            model_ema,
            mixup_fn,
            set_training_mode=True,
        )

        lr_scheduler.step(epoch)

        if output_dir:
            checkpoint_path = output_dir / "checkpoint.pth"
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "model_ema": get_state_dict(model_ema),
                    "scaler": loss_scaler.state_dict(),
                    "args": args,
                },
                checkpoint_path,
            )

        eval_stats = evaluate_multilabel(data_loader_val, model, device, args.threshold)
        print(json.dumps(eval_stats, indent=2))
        main_metric = eval_stats.get("map_micro", 0.0)
        improved = main_metric > best_metric
        if improved:
            best_metric = main_metric
        print(f"Best micro mAP so far: {best_metric:.4f}")

        if output_dir and improved:
            best_ckpt = output_dir / "best_checkpoint.pth"
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "model_ema": get_state_dict(model_ema),
                    "scaler": loss_scaler.state_dict(),
                    "args": args,
                    "metrics": eval_stats,
                },
                best_ckpt,
            )

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in eval_stats.items()}, "epoch": epoch, "n_parameters": n_parameters}

        if output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LSNet Multi-Label Training", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
