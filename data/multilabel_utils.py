from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

DEFAULT_LABEL_DELIMITER = ","
_WEIGHT_EPS = 1e-8


def _maybe_strip_parentheses(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == "(" and token[-1] == ")":
        return token[1:-1].strip()
    return token


def _try_parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class LabelWeight:
    label: str
    weight: float

    def formatted(self) -> str:
        if abs(self.weight - 1.0) <= 1e-6:
            return self.label
        return f"({self.label}:{self.weight:g})"


def parse_label_token(token: str) -> LabelWeight:
    """Parse a single label token that may contain an explicit weight.

    Supported examples::
        "artist" -> weight 1.0
        "(artist:0.6)" -> weight 0.6
        "artist:0.3" -> weight 0.3
        "series:name:2.5" -> label "series:name", weight 2.5
    """

    token = _maybe_strip_parentheses(token)
    if not token:
        raise ValueError("Empty label token")

    # Attempt to parse using the right-most colon as separator
    label_part = token
    weight = 1.0

    if ":" in token:
        candidate_label, candidate_weight = token.rsplit(":", 1)
        parsed_weight = _try_parse_float(candidate_weight)
        if parsed_weight is not None:
            label_part = candidate_label
            weight = parsed_weight

    label_part = label_part.strip()
    if not label_part:
        raise ValueError(f"Label name missing in token: {token}")

    return LabelWeight(label=label_part, weight=weight)


def normalize_weights(weights: Sequence[float]) -> List[float]:
    if not weights:
        return []
    total = sum(max(0.0, w) for w in weights)
    if total <= _WEIGHT_EPS:
        equal = 1.0 / len(weights)
        return [equal for _ in weights]
    return [max(0.0, w) / total for w in weights]


def parse_annotation_tokens(tokens: Iterable[str]) -> List[LabelWeight]:
    parsed: List[LabelWeight] = []
    for raw in tokens:
        raw = raw.strip()
        if not raw:
            continue
        parsed.append(parse_label_token(raw))
    return parsed


def consolidate_duplicates(pairs: Iterable[LabelWeight]) -> List[LabelWeight]:
    merged = {}
    for pair in pairs:
        merged[pair.label] = merged.get(pair.label, 0.0) + pair.weight
    return [LabelWeight(label=label, weight=weight) for label, weight in merged.items()]


def format_label_sequence(pairs: Sequence[LabelWeight]) -> str:
    return DEFAULT_LABEL_DELIMITER.join(pair.formatted() for pair in pairs)
