"""Generate multi-label CSV annotations from image/txt pairs."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Sequence

from data.multilabel_utils import (
    DEFAULT_LABEL_DELIMITER,
    LabelWeight,
    consolidate_duplicates,
    format_label_sequence,
    parse_label_token,
)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _is_valid_image(path: Path, exts: Sequence[str] | None) -> bool:
    extensions = {ext.lower() for ext in (exts or _IMAGE_EXTENSIONS)}
    return path.suffix.lower() in extensions


def _read_annotation_tokens(txt_path: Path, fallback_delimiter: str) -> List[str]:
    content = txt_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    tokens: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # If the line already uses the expected format, keep as is
        if DEFAULT_LABEL_DELIMITER in line:
            tokens.extend(part.strip() for part in line.split(DEFAULT_LABEL_DELIMITER))
            continue
        # Split with fallback delimiter or whitespace
        raw_parts = re.split(rf"[{re.escape(fallback_delimiter)}\s]+" if fallback_delimiter else r"\s+", line)
        for part in raw_parts:
            part = part.strip()
            if part:
                tokens.append(part)
    return tokens


def _convert_token(token: str) -> LabelWeight:
    token = token.strip()
    if not token:
        raise ValueError("Encountered empty label token")
    # If token already matches the expected `(label:weight)` or `label` format
    try:
        return parse_label_token(token)
    except ValueError:
        pass

    # Attempt to split by whitespace first, treating the final segment as weight
    if " " in token:
        label, candidate = token.rsplit(" ", 1)
        try:
            weight = float(candidate)
            return LabelWeight(label=label.strip(), weight=weight)
        except ValueError:
            token = token.replace(" ", "")

    if ":" in token:
        label_part, weight_candidate = token.rsplit(":", 1)
        try:
            weight = float(weight_candidate)
            return LabelWeight(label=label_part.strip(), weight=weight)
        except ValueError:
            pass

    return LabelWeight(label=token, weight=1.0)


def collect_image_files(root: Path, recursive: bool, extensions: Sequence[str] | None) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    images = [p for p in root.glob(pattern) if p.is_file() and _is_valid_image(p, extensions)]
    return sorted(images)


def convert_directory(
    input_dir: Path,
    output_csv: Path,
    fallback_delimiter: str,
    recursive: bool,
    relative: bool,
    extensions: Sequence[str] | None,
    skip_missing_txt: bool,
) -> None:
    input_dir = input_dir.resolve()
    images = collect_image_files(input_dir, recursive, extensions)
    if not images:
        raise RuntimeError(f"No image files found under {input_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "labels"])
        for image_path in images:
            txt_path = image_path.with_suffix(".txt")
            if not txt_path.exists():
                if skip_missing_txt:
                    continue
                raise FileNotFoundError(f"Annotation file missing for {image_path.name}")

            raw_tokens = _read_annotation_tokens(txt_path, fallback_delimiter)
            if not raw_tokens:
                continue

            label_pairs = consolidate_duplicates(_convert_token(token) for token in raw_tokens)
            if not label_pairs:
                continue

            # Normalise tokens so downstream scripts can parse them consistently
            serialised = format_label_sequence(label_pairs)
            if relative:
                image_out = image_path.relative_to(input_dir)
            else:
                image_out = image_path
            writer.writerow([str(image_out).replace("\\", "/"), serialised])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Generate multi-label CSV from image/txt pairs")
    parser.add_argument("--input-dir", required=True, help="Directory containing images and annotation txt files")
    parser.add_argument("--output-csv", required=True, help="Destination CSV path")
    parser.add_argument("--fallback-delimiter", default=",", help="Delimiter to split lines when they are not already comma separated")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan sub-directories")
    parser.add_argument("--relative-paths", action="store_true", help="Store paths relative to input directory")
    parser.add_argument("--image-ext", nargs="*", default=None, help="Optional list of image extensions to include")
    parser.add_argument("--skip-missing-txt", action="store_true", help="Skip images without annotation txt instead of raising")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    convert_directory(
        input_dir=input_dir,
        output_csv=output_csv,
        fallback_delimiter=args.fallback_delimiter,
        recursive=args.recursive,
        relative=args.relative_paths,
        extensions=args.image_ext,
        skip_missing_txt=args.skip_missing_txt,
    )
    print(f"Saved multi-label annotations to {output_csv}")


if __name__ == "__main__":
    main()
