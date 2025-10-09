"""
使用提取的特征进行画师风格聚类
支持多种聚类算法和可视化
"""
import argparse
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns


def get_args_parser():
    parser = argparse.ArgumentParser('Artist Style Clustering', add_help=False)
    
    parser.add_argument('--features', required=True, type=str,
                        help='Path to features.npy file')
    parser.add_argument('--image-names', required=True, type=str,
                        help='Path to image_names.txt file')
    parser.add_argument('--output', default='./output/clustering', type=str,
                        help='Output directory')
    
    # 聚类参数
    parser.add_argument('--method', default='kmeans', type=str,
                        choices=['kmeans', 'dbscan', 'hierarchical'],
                        help='Clustering method')
    parser.add_argument('--n-clusters', default=10, type=int,
                        help='Number of clusters (for kmeans and hierarchical)')
    parser.add_argument('--eps', default=0.5, type=float,
                        help='DBSCAN eps parameter')
    parser.add_argument('--min-samples', default=5, type=int,
                        help='DBSCAN min_samples parameter')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Create visualization')
    parser.add_argument('--viz-method', default='tsne', type=str,
                        choices=['tsne', 'pca'],
                        help='Dimensionality reduction method for visualization')
    parser.add_argument('--perplexity', default=30, type=int,
                        help='t-SNE perplexity parameter')
    
    return parser


def load_features(features_path, image_names_path):
    """加载特征和图像名称"""
    features = np.load(features_path)
    with open(image_names_path, 'r') as f:
        image_names = [line.strip() for line in f]
    
    print(f"Loaded features: {features.shape}")
    print(f"Number of images: {len(image_names)}")
    
    return features, image_names


def perform_clustering(features, method='kmeans', n_clusters=10, eps=0.5, min_samples=5):
    """执行聚类"""
    print(f"\nPerforming clustering with method: {method}")
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(features)
        print(f"K-Means clustering completed with {n_clusters} clusters")
        
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(features)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"DBSCAN clustering completed")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(features)
        print(f"Hierarchical clustering completed with {n_clusters} clusters")
    
    return labels


def reduce_dimensions(features, method='tsne', perplexity=30):
    """降维用于可视化"""
    print(f"\nReducing dimensions with {method}")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_2d = reducer.fit_transform(features)
        print("t-SNE reduction completed")
        
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(features)
        explained_var = reducer.explained_variance_ratio_
        print(f"PCA reduction completed")
        print(f"Explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}")
    
    return features_2d


def visualize_clusters(features_2d, labels, output_dir, method_name):
    """可视化聚类结果"""
    print("\nCreating visualization...")
    
    plt.figure(figsize=(12, 10))
    
    # 获取唯一的标签（聚类）
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # 使用不同颜色
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # 噪声点用黑色
            color = [0, 0, 0, 1]
            marker = 'x'
            label_name = 'Noise'
        else:
            marker = 'o'
            label_name = f'Cluster {label}'
        
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[color], label=label_name, marker=marker, s=50, alpha=0.6)
    
    plt.title(f'Artist Style Clustering ({method_name})\nTotal Clusters: {n_clusters}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # 保存图像
    output_path = output_dir / f'clustering_{method_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.close()


def save_clustering_results(labels, image_names, output_dir):
    """保存聚类结果"""
    # 按聚类分组
    clusters = {}
    for label, name in zip(labels, image_names):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(name)
    
    # 保存JSON格式
    json_path = output_dir / 'clustering_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    print(f"Clustering results saved to: {json_path}")
    
    # 保存文本格式（易读）
    txt_path = output_dir / 'clustering_results.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        for label in sorted(clusters.keys()):
            if label == -1:
                f.write(f"Noise ({len(clusters[label])} images):\n")
            else:
                f.write(f"Cluster {label} ({len(clusters[label])} images):\n")
            for name in clusters[label]:
                f.write(f"  - {name}\n")
            f.write("\n")
    print(f"Clustering results (text) saved to: {txt_path}")
    
    # 统计信息
    stats = {
        'total_images': len(image_names),
        'n_clusters': len([k for k in clusters.keys() if k != -1]),
        'cluster_sizes': {int(k): len(v) for k, v in clusters.items()}
    }
    
    stats_path = output_dir / 'clustering_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"Clustering statistics saved to: {stats_path}")
    
    return clusters


def print_cluster_statistics(clusters):
    """打印聚类统计信息"""
    print("\n" + "="*50)
    print("Clustering Statistics")
    print("="*50)
    
    total_images = sum(len(v) for v in clusters.values())
    n_clusters = len([k for k in clusters.keys() if k != -1])
    
    print(f"Total images: {total_images}")
    print(f"Number of clusters: {n_clusters}")
    
    if -1 in clusters:
        print(f"Noise points: {len(clusters[-1])}")
    
    print("\nCluster sizes:")
    for label in sorted(clusters.keys()):
        if label == -1:
            print(f"  Noise: {len(clusters[label])} images")
        else:
            print(f"  Cluster {label}: {len(clusters[label])} images")
    
    print("="*50)


def main(args):
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载特征
    features, image_names = load_features(args.features, args.image_names)
    
    # 执行聚类
    labels = perform_clustering(
        features, 
        method=args.method,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples
    )
    
    # 保存聚类结果
    clusters = save_clustering_results(labels, image_names, output_dir)
    
    # 打印统计信息
    print_cluster_statistics(clusters)
    
    # 可视化
    if args.visualize:
        features_2d = reduce_dimensions(features, method=args.viz_method, perplexity=args.perplexity)
        visualize_clusters(features_2d, labels, output_dir, args.method)
    
    print(f"\n✓ Clustering completed! Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Artist Style Clustering', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
