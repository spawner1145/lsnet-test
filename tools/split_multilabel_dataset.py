"""Split a multi-label dataset into train/val CSV files with filtering."""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from data.multilabel_utils import (
    DEFAULT_LABEL_DELIMITER,
    LabelWeight,
    consolidate_duplicates,
    normalize_weights,
    parse_annotation_tokens,
    parse_label_token,
)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image_path: Path
    label_pairs: List[LabelWeight]


def _is_image(path: Path, extensions: Sequence[str] | None) -> bool:
    suffix = path.suffix.lower()
    if extensions:
        extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions]
        return suffix in extensions
    return suffix in _IMAGE_EXTENSIONS


def _load_pairs_from_txt(txt_path: Path, delimiter: str) -> List[LabelWeight]:
    content = txt_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    tokens: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens.extend(part.strip() for part in line.split(delimiter) if part.strip())
    if not tokens:
        return []
    parsed = consolidate_duplicates(parse_annotation_tokens(tokens))
    return parsed


def scan_directory(root: Path, delimiter: str, recursive: bool, extensions: Sequence[str] | None,
                   skip_missing_txt: bool) -> List[Sample]:
    samples: List[Sample] = []
    pattern = "**/*" if recursive else "*"
    for path in root.glob(pattern):
        if not path.is_file() or not _is_image(path, extensions):
            continue
        txt_path = path.with_suffix(".txt")
        if not txt_path.exists():
            if skip_missing_txt:
                continue
            raise FileNotFoundError(f"Annotation txt missing for {path}")
        label_pairs = _load_pairs_from_txt(txt_path, delimiter)
        if label_pairs:
            samples.append(Sample(image_path=path, label_pairs=label_pairs))
    return samples


def filter_by_min_frequency(samples: Sequence[Sample], min_count: int) -> Tuple[List[Sample], Dict[str, int]]:
    if min_count <= 1:
        return list(samples), {}

    label_counts: Dict[str, int] = {}
    for sample in samples:
        for pair in sample.label_pairs:
            label_counts[pair.label] = label_counts.get(pair.label, 0) + 1

    filtered: List[Sample] = []
    for sample in samples:
        kept_pairs = [pair for pair in sample.label_pairs if label_counts.get(pair.label, 0) >= min_count]
        if kept_pairs:
            filtered.append(Sample(image_path=sample.image_path, label_pairs=kept_pairs))
    removed = {label: count for label, count in label_counts.items() if count < min_count}
    return filtered, removed


def split_samples(samples: Sequence[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    val_size = int(len(indices) * val_ratio)
    val_indices = set(indices[:val_size])
    train_samples = [samples[i] for i in indices if i not in val_indices]
    val_samples = [samples[i] for i in indices if i in val_indices]
    return train_samples, val_samples


def serialize_samples(samples: Sequence[Sample], root: Path, relative_paths: bool) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for sample in samples:
        if relative_paths:
            image_ref = sample.image_path.relative_to(root)
        else:
            image_ref = sample.image_path
        weights = [max(0.0, float(pair.weight)) for pair in sample.label_pairs]
        normalized = normalize_weights(weights)
        serialised = DEFAULT_LABEL_DELIMITER.join(
            (
                f"({pair.label}:{norm_weight:g})" if abs(norm_weight - 1.0) > 1e-6 else pair.label
            )
            for pair, norm_weight in zip(sample.label_pairs, normalized)
        )
        rows.append((str(image_ref).replace("\\", "/"), serialised))
    return rows


def write_csv(path: Path, rows: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "labels"])
        writer.writerows(rows)


def run_split(args: argparse.Namespace) -> None:
    root = Path(args.input_dir).resolve()
    samples = scan_directory(
        root=root,
        delimiter=args.label_delimiter,
        recursive=args.recursive,
        extensions=args.image_ext,
        skip_missing_txt=args.skip_missing_txt,
    )
    if not samples:
        raise RuntimeError(f"No annotated images found in {root}")

    samples, removed = filter_by_min_frequency(samples, args.min_label_count)
    if not samples:
        raise RuntimeError("All samples were filtered out by min_label_count")
    if removed:
        print(f"Filtered out {len(removed)} labels below min_count={args.min_label_count}: {sorted(removed.items())[:10]}…")

    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    train_rows = serialize_samples(train_samples, root, args.relative_paths)
    val_rows = serialize_samples(val_samples, root, args.relative_paths)

    write_csv(Path(args.train_csv), train_rows)
    write_csv(Path(args.val_csv), val_rows)

    print(f"Saved {len(train_rows)} training entries to {args.train_csv}")
    print(f"Saved {len(val_rows)} validation entries to {args.val_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Split multi-label annotations into train/val CSVs")
    parser.add_argument("--input-dir", required=True, help="Root directory containing images and txt annotations")
    parser.add_argument("--train-csv", required=True, help="Output CSV path for training set")
    parser.add_argument("--val-csv", required=True, help="Output CSV path for validation set")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    parser.add_argument("--label-delimiter", default=DEFAULT_LABEL_DELIMITER, help="Delimiter inside txt annotations (default: comma)")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories")
    parser.add_argument("--relative-paths", action="store_true", help="Store image paths relative to input directory")
    parser.add_argument("--image-ext", nargs="*", default=None, help="Optional list of image extensions to include")
    parser.add_argument("--skip-missing-txt", action="store_true", help="Skip images without txt file instead of raising")
    parser.add_argument("--min-label-count", type=int, default=1, help="Discard labels appearing fewer than this number of times")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_split(args)


if __name__ == "__main__":
    main()
