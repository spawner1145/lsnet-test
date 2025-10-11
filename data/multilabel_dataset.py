import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from .multilabel_utils import (
    DEFAULT_LABEL_DELIMITER,
    LabelWeight,
    consolidate_duplicates,
    normalize_weights,
    parse_annotation_tokens,
)


def _resolve_image_path(root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        path = root / path
    return path.expanduser().resolve()


class MultiLabelImageDataset(Dataset):
    """Image dataset that loads multi-label annotations from a CSV file.

    The annotation file must contain at least two columns: ``image_path`` and ``labels``.
    ``labels`` holds label names separated by ``label_delimiter`` (default: ``;``).
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform=None,
        *,
        class_to_idx: Optional[Dict[str, int]] = None,
        label_delimiter: str = ";",
        encoding: str = "utf-8",
        skip_missing: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.ann_file = Path(ann_file)
        self.transform = transform
        self.label_delimiter = label_delimiter or DEFAULT_LABEL_DELIMITER
        self.encoding = encoding
        self.skip_missing = skip_missing

        if not self.ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")

        self.class_to_idx: Dict[str, int] = dict(class_to_idx) if class_to_idx else {}
        self.samples: List[Path] = []
        self.label_indices: List[Tuple[int, ...]] = []
        self.label_weights: List[Tuple[float, ...]] = []
        self.raw_label_weights: List[Tuple[float, ...]] = []

        # statistics
        self.label_positive_counts: Optional[torch.Tensor] = None
        self.label_weight_sums: Optional[torch.Tensor] = None

        dynamic_positive_counts: List[float] = []
        dynamic_weight_sums: List[float] = []

        with self.ann_file.open("r", newline="", encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            if "image_path" not in reader.fieldnames or "labels" not in reader.fieldnames:
                raise ValueError(
                    f"Annotation file {self.ann_file} must contain 'image_path' and 'labels' columns."
                )
            for row in reader:
                image_rel = row["image_path"].strip()
                if not image_rel:
                    continue
                labels_raw = row["labels"].strip()
                label_tokens = [
                    token.strip()
                    for token in labels_raw.split(self.label_delimiter)
                    if token.strip()
                ]
                label_pairs = consolidate_duplicates(parse_annotation_tokens(label_tokens))
                if not label_pairs:
                    continue

                path = _resolve_image_path(self.root, image_rel)
                if not path.exists():
                    if self.skip_missing:
                        continue
                    raise FileNotFoundError(f"Image path does not exist: {path}")

                indices: List[int] = []
                raw_weights: List[float] = []
                for label_weight in label_pairs:
                    if label_weight.label not in self.class_to_idx:
                        if class_to_idx is not None:
                            raise KeyError(
                                f"Label '{label_weight.label}' in {self.ann_file} not found in provided class_to_idx mapping"
                            )
                        self.class_to_idx[label_weight.label] = len(self.class_to_idx)
                    indices.append(self.class_to_idx[label_weight.label])
                    raw_weights.append(max(0.0, float(label_weight.weight)))

                normalized_weights = normalize_weights(raw_weights)
                self.samples.append(path)
                self.label_indices.append(tuple(indices))
                self.label_weights.append(tuple(normalized_weights))
                self.raw_label_weights.append(tuple(raw_weights))

                # update statistics arrays lazily once class count known
                while len(dynamic_positive_counts) < len(self.class_to_idx):
                    dynamic_positive_counts.append(0.0)
                    dynamic_weight_sums.append(0.0)

                for idx, norm_weight in zip(indices, normalized_weights):
                    dynamic_positive_counts[idx] += 1.0
                    dynamic_weight_sums[idx] += norm_weight

        if not self.samples:
            raise RuntimeError(f"No samples were loaded from {self.ann_file}")

        self.idx_to_class: Dict[int, str] = {idx: label for label, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        self.label_positive_counts = torch.tensor(dynamic_positive_counts, dtype=torch.float32)
        self.label_weight_sums = torch.tensor(dynamic_weight_sums, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        target_indices = self.label_indices[index]
        target_weights = self.label_weights[index]

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        if target_indices:
            for idx, weight in zip(target_indices, target_weights):
                target[idx] = weight

        return image, target

    @property
    def classes(self) -> List[str]:
        return [self.idx_to_class[idx] for idx in range(self.num_classes)]

    def export_class_mapping(self, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name"])
            for idx, name in enumerate(self.classes):
                writer.writerow([idx, name])
        return output

    def export_label_statistics(self, output: Path) -> Path:
        if self.label_positive_counts is None or self.label_weight_sums is None:
            raise RuntimeError("Label statistics are not available")

        output.parent.mkdir(parents=True, exist_ok=True)
        total_positive = float(max(self.label_positive_counts.sum().item(), 1.0))
        total_weight = float(max(self.label_weight_sums.sum().item(), 1.0))

        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "class_id",
                    "class_name",
                    "sample_count",
                    "sample_ratio",
                    "weight_sum",
                    "weight_ratio",
                    "avg_weight_per_sample",
                ]
            )
            for idx, name in enumerate(self.classes):
                count = float(self.label_positive_counts[idx].item())
                weight_sum = float(self.label_weight_sums[idx].item())
                sample_ratio = count / total_positive if total_positive > 0 else 0.0
                weight_ratio = weight_sum / total_weight if total_weight > 0 else 0.0
                avg_weight = weight_sum / count if count > 0 else 0.0
                writer.writerow(
                    [
                        idx,
                        name,
                        int(round(count)),
                        f"{sample_ratio:.6f}",
                        f"{weight_sum:.6f}",
                        f"{weight_ratio:.6f}",
                        f"{avg_weight:.6f}",
                    ]
                )
        return output

    def get_pos_weight(self) -> torch.Tensor:
        """Compute positive class weights for BCEWithLogitsLoss."""
        if self.label_positive_counts is None:
            raise RuntimeError("Positive counts not initialized")
        total = torch.tensor(len(self), dtype=torch.float32)
        pos_counts = self.label_positive_counts.clone()
        neg_counts = total - pos_counts
        pos_weight = torch.ones_like(pos_counts)
        mask = pos_counts > 0
        pos_weight[mask] = neg_counts[mask] / pos_counts[mask]
        return pos_weight

    def collate_raw_targets(self) -> torch.Tensor:
        """Return stacked multi-hot targets (CPU tensor)."""
        targets = torch.zeros((len(self), self.num_classes), dtype=torch.float32)
        for i, (indices, weights) in enumerate(zip(self.label_indices, self.label_weights)):
            if indices:
                for idx, weight in zip(indices, weights):
                    targets[i, idx] = weight
        return targets
