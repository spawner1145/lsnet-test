"""
画师风格模型使用示例
展示如何使用训练好的模型进行分类和聚类
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from timm.models import create_model
from torchvision import transforms


def example_load_model():
    """示例1：加载模型"""
    print("=" * 50)
    print("示例1：加载模型")
    print("=" * 50)
    
    # 创建模型
    model = create_model(
        'lsnet_t_artist',
        pretrained=False,
        num_classes=100,  # 画师类别数
        feature_dim=512,  # 特征向量维度
    )
    
    # 加载训练好的权重
    checkpoint = torch.load('path/to/checkpoint.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Number of classes: 100")
    print(f"  Feature dimension: 512")
    
    return model


def example_classify_single_image(model):
    """示例2：分类单张图像"""
    print("\n" + "=" * 50)
    print("示例2：分类单张图像")
    print("=" * 50)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 加载图像
    image = Image.open('test_image.jpg').convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        # 方法1：使用分类头
        logits = model(input_tensor, return_features=False)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1)
        confidence = probs[0, pred_class].item()
        
        print(f"✓ Classification result:")
        print(f"  Predicted class: {pred_class.item()}")
        print(f"  Confidence: {confidence:.4f}")
        
        # 方法2：使用便捷方法
        logits = model.classify(input_tensor)
        print(f"  Logits shape: {logits.shape}")


def example_extract_features(model):
    """示例3：提取特征向量（用于聚类）"""
    print("\n" + "=" * 50)
    print("示例3：提取特征向量")
    print("=" * 50)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open('test_image.jpg').convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        # 方法1：不使用分类头
        features = model(input_tensor, return_features=True)
        
        print(f"✓ Feature extraction:")
        print(f"  Feature shape: {features.shape}")
        print(f"  Feature vector (first 10 dims): {features[0][:10].numpy()}")
        
        # 方法2：使用便捷方法
        features = model.get_features(input_tensor)
        print(f"  Using get_features(): {features.shape}")


def example_batch_inference(model):
    """示例4：批量推理"""
    print("\n" + "=" * 50)
    print("示例4：批量推理")
    print("=" * 50)
    
    from torch.utils.data import Dataset, DataLoader
    
    class SimpleImageDataset(Dataset):
        def __init__(self, image_paths, transform):
            self.image_paths = image_paths
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert('RGB')
            return self.transform(image), self.image_paths[idx]
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 获取图像列表
    image_dir = Path('test_images')
    image_paths = list(image_dir.glob('*.jpg'))[:10]  # 示例：处理10张图像
    
    # 创建数据加载器
    dataset = SimpleImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    all_features = []
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch, paths in dataloader:
            # 提取特征
            features = model.get_features(batch)
            all_features.append(features.numpy())
            
            # 分类
            logits = model.classify(batch)
            preds = torch.argmax(logits, dim=-1)
            all_predictions.extend(preds.numpy())
    
    # 合并结果
    features_matrix = np.concatenate(all_features, axis=0)
    
    print(f"✓ Batch inference completed:")
    print(f"  Processed {len(image_paths)} images")
    print(f"  Features shape: {features_matrix.shape}")
    print(f"  Predictions: {all_predictions}")


def example_clustering(features_matrix):
    """示例5：特征聚类"""
    print("\n" + "=" * 50)
    print("示例5：特征聚类")
    print("=" * 50)
    
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # K-Means聚类
    print("Running K-Means clustering...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels_kmeans = kmeans.fit_predict(features_matrix)
    
    print(f"✓ K-Means clustering:")
    print(f"  Number of clusters: 5")
    print(f"  Cluster distribution: {np.bincount(labels_kmeans)}")
    
    # DBSCAN聚类
    print("\nRunning DBSCAN clustering...")
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    labels_dbscan = dbscan.fit_predict(features_matrix)
    n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise = list(labels_dbscan).count(-1)
    
    print(f"✓ DBSCAN clustering:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Noise points: {n_noise}")
    
    # t-SNE可视化
    print("\nCreating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_matrix)-1))
    features_2d = tsne.fit_transform(features_matrix)
    
    # 绘制K-Means结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels_kmeans, cmap='Spectral', s=50, alpha=0.6)
    plt.colorbar(scatter)
    plt.title('K-Means Clustering')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # 绘制DBSCAN结果
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels_dbscan, cmap='Spectral', s=50, alpha=0.6)
    plt.colorbar(scatter)
    plt.title('DBSCAN Clustering')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png', dpi=300)
    print("✓ Visualization saved to: clustering_comparison.png")


def example_compare_modes():
    """示例6：比较分类模式和聚类模式"""
    print("\n" + "=" * 50)
    print("示例6：比较分类模式和聚类模式")
    print("=" * 50)
    
    # 创建模型
    model = create_model(
        'lsnet_t_artist',
        pretrained=False,
        num_classes=100,
    )
    model.eval()
    
    # 准备输入
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        # 分类模式
        output_classify = model(dummy_input, return_features=False)
        print(f"✓ Classification mode:")
        print(f"  Output shape: {output_classify.shape}")
        print(f"  Output type: logits for {output_classify.shape[1]} classes")
        
        # 聚类模式
        output_cluster = model(dummy_input, return_features=True)
        print(f"\n✓ Clustering mode:")
        print(f"  Output shape: {output_cluster.shape}")
        print(f"  Output type: feature vector")
        
        # 比较
        print(f"\n✓ Comparison:")
        print(f"  Classification output: (batch_size, num_classes) = {output_classify.shape}")
        print(f"  Clustering output: (batch_size, feature_dim) = {output_cluster.shape}")


def main():
    """运行所有示例"""
    print("\n" + "="*50)
    print("画师风格模型使用示例")
    print("="*50)
    
    # 示例6：比较模式（不需要真实模型）
    example_compare_modes()
    
    print("\n" + "="*50)
    print("完整示例需要：")
    print("1. 训练好的模型checkpoint")
    print("2. 测试图像")
    print("3. 运行train_artist_style.py训练模型")
    print("="*50)
    
    # 如果有训练好的模型，可以运行其他示例
    # model = example_load_model()
    # example_classify_single_image(model)
    # example_extract_features(model)
    # example_batch_inference(model)
    
    # 如果有特征矩阵，可以运行聚类示例
    # features = np.random.randn(100, 512)  # 示例数据
    # example_clustering(features)


if __name__ == '__main__':
    main()
