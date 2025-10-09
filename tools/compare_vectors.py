import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

SUPPORTED_SUFFIXES = {'.npy', '.json', '.txt'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'Compare a query vector with multiple reference vectors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--query-vector', required=True, type=str,
                        help='待比较的查询向量文件 (.npy / .json / .txt)')
    parser.add_argument('--reference-vectors', required=True, nargs='+', type=str,
                        help='参考向量文件或目录列表（目录中会读取所有 .npy 文件）')
    parser.add_argument('--top-k', default=5, type=int,
                        help='返回相似度前 K 名结果')
    parser.add_argument('--normalize', action='store_true',
                        help='计算相似度前对向量执行 L2 归一化')
    parser.add_argument('--output', default=None, type=str,
                        help='可选的输出 JSON 文件路径')
    return parser.parse_args()


def load_vector_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == '.npy':
        vector = np.load(path)
    elif suffix in {'.json', '.txt'}:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        vector = np.asarray(data, dtype=np.float32)
    else:
        raise ValueError(f'不支持的向量文件格式: {path}')

    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim > 1:
        vector = vector.reshape(-1)
    if vector.size == 0:
        raise ValueError(f'向量文件为空: {path}')
    return vector


def expand_reference_paths(ref_inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for item in ref_inputs:
        p = Path(item)
        if not p.exists():
            raise FileNotFoundError(f'参考向量不存在: {item}')
        if p.is_dir():
            paths.extend(sorted(child for child in p.iterdir() if child.suffix.lower() in SUPPORTED_SUFFIXES))
        else:
            if p.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f'不支持的参考向量格式: {p}')
            paths.append(p)
    if not paths:
        raise ValueError('未找到任何参考向量文件。')
    return paths


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError('向量范数为 0，无法进行归一化。')
    return vector / norm


def compute_similarity(query_vector: np.ndarray,
                       reference_vectors: List[Tuple[Path, np.ndarray]],
                       normalize: bool) -> List[Tuple[str, float]]:
    if normalize:
        query_vector = normalize_vector(query_vector)
        reference_vectors = [(path, normalize_vector(vec)) for path, vec in reference_vectors]

    q_norm = np.linalg.norm(query_vector)
    if q_norm == 0:
        raise ValueError('查询向量范数为 0，无法计算相似度。')

    similarities: List[Tuple[str, float]] = []
    for path, vec in reference_vectors:
        denom = np.linalg.norm(vec) * q_norm
        if denom == 0:
            sim = 0.0
        else:
            sim = float(np.dot(query_vector, vec) / denom)
        similarities.append((str(path), sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities


def save_results(path: Path, query: Path, top_k: int, results: List[Tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(
            {
                'query': str(query),
                'top_k': top_k,
                'results': [
                    {'path': result_path, 'cosine_similarity': score}
                    for result_path, score in results
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f'相似度结果已保存到 {path}')


def main():
    args = parse_args()

    query_path = Path(args.query_vector)
    if not query_path.exists():
        raise FileNotFoundError(f'查询向量不存在: {query_path}')
    query_vector = load_vector_file(query_path)

    reference_paths = expand_reference_paths(args.reference_vectors)
    reference_vectors = [(path, load_vector_file(path)) for path in reference_paths]

    similarity_pairs = compute_similarity(query_vector, reference_vectors, args.normalize)
    top_k = min(args.top_k, len(similarity_pairs))
    top_results = similarity_pairs[:top_k]

    print(f'相似度 Top-{top_k} 结果:')
    for idx, (path, score) in enumerate(top_results, 1):
        print(f'  {idx}. {path}: {score:.6f}')

    if args.output:
        save_results(Path(args.output), query_path, top_k, top_results)


if __name__ == '__main__':
    main()
