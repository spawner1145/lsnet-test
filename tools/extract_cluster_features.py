import argparse
import json
from pathlib import Path
from typing import List, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'Extract LSNet artist features and cluster a folder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--images-dir', required=True, type=str,
                        help='包含图像的文件夹路径，将对其中所有支持格式的图像提取特征并聚类')
    parser.add_argument('--model', default='lsnet_t_artist', type=str,
                        choices=['lsnet_t_artist', 'lsnet_s_artist', 'lsnet_b_artist', 'lsnet_l_artist', 'lsnet_xl_artist', 'lsnet_xl_artist_448'],
                        help='用于特征提取的模型名称')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='模型 checkpoint 路径')
    parser.add_argument('--feature-dim', default=None, type=int,
                        help='特征维度（若模型需要可显式指定）')
    parser.add_argument('--device', default='cuda', type=str,
                        help='推理设备')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='批量推理时的 batch size')
    parser.add_argument('--num-clusters', default=5, type=int,
                        help='KMeans 聚类簇数量')
    parser.add_argument('--seed', default=42, type=int,
                        help='随机种子，确保聚类可复现')
    parser.add_argument('--output-dir', default='./output/cluster', type=str,
                        help='输出目录，将保存特征矩阵与聚类结果 JSON 文件')
    return parser.parse_args()


def _collect_image_paths(images_dir: Path) -> List[Path]:
    return sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def _load_transform(model) -> Tuple[torch.nn.Module, dict]:
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform, config


def _process_batch(model, tensors: List[torch.Tensor], device: torch.device) -> np.ndarray:
    batch_tensor = torch.cat(tensors, dim=0).to(device)
    features = model(batch_tensor, return_features=True)
    return features.cpu().numpy()


def extract_features(args: argparse.Namespace) -> Tuple[np.ndarray, List[str]]:
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f'找不到图像文件夹: {images_dir}')

    image_paths = _collect_image_paths(images_dir)
    if not image_paths:
        raise ValueError(
            f'在 {images_dir} 未找到支持的图像文件，支持扩展名: {sorted(IMAGE_EXTENSIONS)}'
        )

    state_dict = load_checkpoint_state(args.checkpoint)
    num_classes = resolve_num_classes(None, None, state_dict)
    feature_args = argparse.Namespace(
        model=args.model,
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        device=args.device,
    )
    model = load_model(feature_args, state_dict)
    device = torch.device(args.device)

    transform, _ = _load_transform(model)

    features: List[np.ndarray] = []
    names: List[str] = []
    batch_tensors: List[torch.Tensor] = []
    batch_names: List[str] = []

    for path in image_paths:
        try:
            tensor = preprocess_image(path, transform)
        except Exception as exc:  # pylint: disable=broad-except
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
        raise RuntimeError('特征提取失败，没有成功处理的图像。')

    feature_matrix = np.concatenate(features, axis=0)
    print(f'特征提取完成，共 {feature_matrix.shape[0]} 张图像，特征维度 {feature_matrix.shape[1]}')
    return feature_matrix, names


def run_clustering(args: argparse.Namespace, features: np.ndarray, names: List[str]) -> dict:
    print(f'开始执行 KMeans 聚类，簇数量 = {args.num_clusters}')
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init='auto')
    labels = kmeans.fit_predict(features)

    clusters = {}
    for name, label in zip(names, labels):
        clusters.setdefault(int(label), []).append(name)

    result = {
        'num_clusters': args.num_clusters,
        'inertia': float(kmeans.inertia_),
        'cluster_sizes': {str(k): len(v) for k, v in clusters.items()},
        'clusters': clusters,
        'centroids': kmeans.cluster_centers_.tolist(),
    }
    return result


def save_outputs(args: argparse.Namespace, features: np.ndarray, clustering: dict) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / 'features.npy'
    np.save(features_path, features)

    with (output_dir / 'cluster_assignments.json').open('w', encoding='utf-8') as f:
        json.dump(clustering, f, indent=2, ensure_ascii=False)

    print(f'特征矩阵与聚类结果已保存到 {output_dir}')


def main():
    args = parse_args()
    features, names = extract_features(args)
    clustering = run_clustering(args, features, names)
    save_outputs(args, features, clustering)


if __name__ == '__main__':
    main()
