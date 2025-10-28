"""Run multi-label inference with LSNet artist models."""
import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from torchvision import transforms


def load_class_names(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Class mapping CSV not found: {csv_path}")
    names: List[str] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "class_id" not in reader.fieldnames or "class_name" not in reader.fieldnames:
            raise ValueError("Class mapping CSV must contain 'class_id' and 'class_name' columns")
        rows = sorted(reader, key=lambda row: int(row["class_id"]))
        for row in rows:
            names.append(row["class_name"])
    if not names:
        raise RuntimeError(f"No class names found in {csv_path}")
    return names


def build_transform(input_size: int, finetune: bool) -> transforms.Compose:
    if finetune:
        resize = transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = []
    else:
        resize = transforms.Resize(int((256 / 224) * input_size), interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = [transforms.CenterCrop(input_size)]
    return transforms.Compose([
        resize,
        *center_crop,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def gather_image_paths(inputs: Iterable[str], recursive: bool) -> List[Path]:
    paths: List[Path] = []
    for entry in inputs:
        p = Path(entry)
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            for candidate in p.glob(pattern):
                if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    paths.append(candidate)
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"Input path not found: {entry}")
    if not paths:
        raise RuntimeError("No images found for inference")
    return sorted(set(paths))


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        cleaned[new_key] = v
    model.load_state_dict(cleaned, strict=True)


def normalize_scores(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    total = scores.sum(dim=-1, keepdim=True)
    total = torch.clamp(total, min=eps)
    return scores / total


def main(args):
    # 根据模型配置动态设置输入大小
    from model.lsnet_artist import default_cfgs_artist
    if args.model in default_cfgs_artist:
        model_cfg = default_cfgs_artist[args.model]
        configured_input_size = model_cfg.get('input_size', (3, 224, 224))[1]  # 获取高度（假设正方形）
        if args.input_size != configured_input_size:
            args.input_size = configured_input_size
            print(f"Auto-setting input_size to {configured_input_size} for model {args.model} (from config)")
    
    device = torch.device(args.device)
    class_names = load_class_names(Path(args.class_mapping))
    num_classes = len(class_names)

    transform = build_transform(args.input_size, args.finetune_eval)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        use_projection=args.use_projection,
    )
    load_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()

    image_paths = gather_image_paths(args.inputs, args.recursive)

    results = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu()

        topk = min(args.top_k, num_classes)
        confidences, indices = torch.topk(probs, k=topk)
        confidences = confidences.tolist()
        indices = indices.tolist()

        if args.threshold > 0:
            filtered = [(idx, conf) for idx, conf in zip(indices, confidences) if conf >= args.threshold]
        else:
            filtered = list(zip(indices, confidences))

        if args.normalize_ratio:
            normalized = normalize_scores(probs.unsqueeze(0)).squeeze(0)
        else:
            normalized = probs

        labels = []
        for idx, conf in filtered:
            label_entry = {
                "label": class_names[idx],
                "confidence": conf,
            }
            if args.normalize_ratio:
                label_entry["ratio"] = float(normalized[idx].item())
            labels.append(label_entry)

        result = {
            "image": str(image_path),
            "labels": labels,
            "raw_confidence": {name: float(probs[i].item()) for i, name in enumerate(class_names)} if args.debug_full else None,
        }
        if not args.debug_full:
            result.pop("raw_confidence")
        results.append(result)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved predictions to {output_path}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LSNet Multi-Label Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--class-mapping", required=True, help="CSV file with class_id,class_name columns")
    parser.add_argument("--inputs", nargs="+", required=True, help="Image files or directories")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--model", default="lsnet_t_artist", choices=["lsnet_t_artist", "lsnet_s_artist", "lsnet_b_artist", "lsnet_l_artist", "lsnet_xl_artist", "lsnet_xl_artist_448"])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--use-projection", action="store_true", default=True)
    parser.add_argument("--no-projection", dest="use_projection", action="store_false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--finetune-eval", action="store_true", help="Use square resize instead of shortest-edge resize")
    parser.add_argument("--threshold", type=float, default=0.0, help="Drop predictions below this confidence")
    parser.add_argument("--top-k", type=int, default=5, help="Report top-K labels per image before thresholding")
    parser.add_argument("--normalize-ratio", action="store_true", help="Normalize confidences to sum to 1 per image")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directories")
    parser.add_argument("--debug-full", action="store_true", help="Include all class confidences in the output")
    args = parser.parse_args()
    main(args)
