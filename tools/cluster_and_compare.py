import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from inference_artist import (
    load_checkpoint_state,
    load_model,
    preprocess_image,
    resolve_num_classes,
)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(
        'Artist feature clustering and similarity utilities',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型与数据相关参数
    parser.add_argument('--images-dir', type=str, default=None,
                        help='待聚类的图像文件夹路径（将对所有支持的图像进行特征提取并聚类）')
    parser.add_argument('--model', type=str, default='lsnet_t_artist',
                        choices=['lsnet_t_artist', 'lsnet_s_artist', 'lsnet_b_artist'],
                        help='用于特征提取的模型名称')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型 checkpoint 路径（提取特征或分类需提供）')
    parser.add_argument('--feature-dim', type=int, default=None,
                        help='特征维度（若模型需要可指定）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备：cuda 或 cpu')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批量推理时的 batch size')
    parser.add_argument('--num-clusters', type=int, default=5,
                        help='KMeans 聚类簇数量')
    parser.add_argument('--cluster-output', type=str, default='./output/cluster',
                        help='聚类结果输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，用于聚类可复现')

    # 向量相似度比较相关参数
    parser.add_argument('--query-vector', type=str, default=None,
                        help='待比对的目标向量文件（.npy 或包含向量的 .json/.txt）')
    parser.add_argument('--reference-vectors', type=str, nargs='*', default=None,
                        help='参考向量文件列表（.npy 或 .json/.txt）；也可传入目录，程序会读取目录下所有 .npy 文件')
    parser.add_argument('--top-k', type=int, default=5,
                        help='返回相似度前 K 个结果（<= 参考向量数量）')
    parser.add_argument('--similarity-output', type=str, default=None,
                        help='相似度结果保存路径（JSON）')
    parser.add_argument('--normalize', action='store_true',
                        help='在计算相似度前对向量进行 L2 归一化')

    args = parser.parse_args()

    if not args.images_dir and not args.query_vector:
        parser.error('至少需要指定 --images-dir 或 --query-vector 中的一个功能。')

    if args.images_dir and not args.checkpoint:
        parser.error('--images-dir 模式需要提供 --checkpoint 以加载模型特征提取。')

    return args


def _collect_image_paths(images_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for entry in images_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(entry)
    return sorted(paths)


def _load_transform(model) -> Tuple[callable, dict]:
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform, config


def _process_batch(model, batch_tensors: List[torch.Tensor], device: torch.device) -> np.ndarray:
    batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
    features = model(batch_tensor, return_features=True)
    return features.cpu().numpy()


def _extract_directory_features(args) -> Tuple[np.ndarray, List[str]]:
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f'找不到图像文件夹: {images_dir}')

    image_paths = _collect_image_paths(images_dir)
    if not image_paths:
        raise ValueError(f'在 {images_dir} 未找到支持的图像文件，支持扩展名: {sorted(IMAGE_EXTENSIONS)}')

    print(f'共找到 {len(image_paths)} 张图像，开始提取特征...')

    state_dict = load_checkpoint_state(args.checkpoint)
    num_classes = resolve_num_classes(None, None, state_dict)
    device = torch.device(args.device)
    feature_args = argparse.Namespace(
        model=args.model,
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        device=args.device,
    )
    model = load_model(feature_args, state_dict)

    transform, _ = _load_transform(model)

    features: List[np.ndarray] = []
    names: List[str] = []

    batch_tensors: List[torch.Tensor] = []
    batch_names: List[str] = []

    for path in image_paths:
        try:
            tensor = preprocess_image(path, transform)
        except Exception as exc:
            print(f'[Warning] 无法处理 {path.name}: {exc}')
            continue
        batch_tensors.append(tensor)
        batch_names.append(path.name)

        if len(batch_tensors) == args.batch_size:
            batch_features = _process_batch(model, batch_tensors, device)
            features.append(batch_features)
            names.extend(batch_names)
            batch_tensors.clear()
            batch_names.clear()

    if batch_tensors:
        batch_features = _process_batch(model, batch_tensors, device)
        features.append(batch_features)
        names.extend(batch_names)

    if not features:
        raise RuntimeError('特征提取失败，没有有效的图像。')

    feature_matrix = np.concatenate(features, axis=0)
    print(f'特征提取完成，矩阵形状: {feature_matrix.shape}')
    return feature_matrix, names


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_clustering(args, features: np.ndarray, names: List[str]) -> dict:
    print(f'开始执行 KMeans 聚类，簇数 = {args.num_clusters} ...')
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init='auto')
    labels = kmeans.fit_predict(features)

    clusters: dict = {}
    for name, label in zip(names, labels):
        clusters.setdefault(int(label), []).append(name)

    centroids = kmeans.cluster_centers_.tolist()
    inertia = float(kmeans.inertia_)

    clustering_result = {
        'num_clusters': args.num_clusters,
        'inertia': inertia,
        'clusters': clusters,
        'cluster_sizes': {str(idx): len(items) for idx, items in clusters.items()},
        'centroids': centroids,
    }
    return clustering_result


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
    return vector


def _expand_reference_paths(ref_inputs: Optional[Iterable[str]]) -> List[Path]:
    if not ref_inputs:
        return []

    paths: List[Path] = []
    for item in ref_inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(child for child in p.iterdir() if child.suffix.lower() == '.npy'))
        elif p.exists():
            paths.append(p)
        else:
            raise FileNotFoundError(f'参考向量不存在: {item}')
    return paths


def _compute_similarity(query_vector: np.ndarray,
                        reference_vectors: List[Tuple[Path, np.ndarray]],
                        normalize: bool) -> List[Tuple[str, float]]:
    if normalize:
        query_vector = _normalize_vector(query_vector)
        ref_vectors = [(path, _normalize_vector(vec)) for path, vec in reference_vectors]
    else:
        ref_vectors = reference_vectors

    similarities: List[Tuple[str, float]] = []
    q_norm = np.linalg.norm(query_vector)
    if q_norm == 0:
        raise ValueError('查询向量范数为 0，无法计算相似度。')

    for path, vec in ref_vectors:
        denom = np.linalg.norm(vec) * q_norm
        if denom == 0:
            sim = 0.0
        else:
            sim = float(np.dot(query_vector, vec) / denom)
        similarities.append((str(path), sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def main():
    args = parse_args()

    clustering_result = None
    similarity_result = None

    if args.images_dir:
        features, names = _extract_directory_features(args)
        output_dir = Path(args.cluster_output)
        _ensure_output_dir(output_dir)

        clustering_result = _run_clustering(args, features, names)

        features_path = output_dir / 'features.npy'
        np.save(features_path, features)
        with (output_dir / 'cluster_assignments.json').open('w', encoding='utf-8') as f:
            json.dump(clustering_result, f, indent=2, ensure_ascii=False)

        print(f'聚类结果已保存到 {output_dir}')

    if args.query_vector:
        query_path = Path(args.query_vector)
        if not query_path.exists():
            raise FileNotFoundError(f'查询向量文件不存在: {query_path}')

        query_vector = load_vector_file(query_path)
        reference_paths = _expand_reference_paths(args.reference_vectors)
        if not reference_paths:
            raise ValueError('需要至少提供一个参考向量文件或目录。')

        reference_vectors = [(path, load_vector_file(path)) for path in reference_paths]
        similarity_pairs = _compute_similarity(query_vector, reference_vectors, args.normalize)

        top_k = min(args.top_k, len(similarity_pairs))
        similarity_result = similarity_pairs[:top_k]

        print('相似度 Top-{} 结果:'.format(top_k))
        for idx, (path, score) in enumerate(similarity_result, 1):
            print(f'  {idx}. {path}: {score:.6f}')

        if args.similarity_output:
            output_path = Path(args.similarity_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(
                    {
                        'query': str(query_path),
                        'top_k': top_k,
                        'results': [{'path': p, 'cosine_similarity': s} for p, s in similarity_result],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f'相似度结果已保存到 {output_path}')

    if clustering_result is None and similarity_result is None:
        raise RuntimeError('未执行任何操作，请检查输入参数。')


if __name__ == '__main__':
    main()
