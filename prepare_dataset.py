"""
自动分割画师数据集为训练集和验证集
支持从简单的文件夹结构自动创建ImageFolder格式的数据集
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser('Artist Dataset Split Tool', add_help=False)
    
    parser.add_argument('--source-dir', required=True, type=str,
                        help='源数据目录，包含多个画师文件夹')
    parser.add_argument('--output-dir', default='./data/artist_dataset', type=str,
                        help='输出目录')
    parser.add_argument('--val-ratio', default=0.2, type=float,
                        help='验证集比例 (0.0-1.0)')
    parser.add_argument('--seed', default=42, type=int,
                        help='随机种子')
    parser.add_argument('--min-images', default=10, type=int,
                        help='每个画师最少图像数量，少于此数量的画师将被跳过')
    parser.add_argument('--copy', action='store_true', default=True,
                        help='复制文件（默认）')
    parser.add_argument('--symlink', action='store_true',
                        help='创建符号链接而不是复制文件（节省空间）')
    parser.add_argument('--image-extensions', nargs='+', 
                        default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
                        help='支持的图像扩展名')
    
    return parser


def collect_artist_images(source_dir, image_extensions):
    """
    收集所有画师及其图像文件
    
    返回格式：
    {
        'artist_name': [image_path1, image_path2, ...]
    }
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        raise ValueError(f"源目录不存在: {source_dir}")
    
    artist_images = {}
    
    # 遍历源目录下的所有子文件夹（每个子文件夹代表一个画师）
    artist_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"发现 {len(artist_dirs)} 个画师文件夹")
    
    for artist_dir in tqdm(artist_dirs, desc="扫描画师文件夹"):
        artist_name = artist_dir.name
        
        # 收集该画师的所有图像
        images = []
        for ext in image_extensions:
            # 支持多级目录
            images.extend(list(artist_dir.rglob(f"*{ext}")))
            images.extend(list(artist_dir.rglob(f"*{ext.upper()}")))
        
        if images:
            artist_images[artist_name] = images
    
    return artist_images


def split_dataset(artist_images, val_ratio, seed, min_images):
    """
    将每个画师的图像分割为训练集和验证集
    
    返回：
    train_split: {artist_name: [train_images]}
    val_split: {artist_name: [val_images]}
    """
    random.seed(seed)
    
    train_split = {}
    val_split = {}
    skipped_artists = []
    
    for artist_name, images in artist_images.items():
        # 过滤掉图像数量太少的画师
        if len(images) < min_images:
            skipped_artists.append((artist_name, len(images)))
            continue
        
        # 随机打乱
        images_list = list(images)
        random.shuffle(images_list)
        
        # 计算分割点
        n_val = max(1, int(len(images_list) * val_ratio))  # 至少1张验证图像
        n_train = len(images_list) - n_val
        
        # 如果训练集太少，调整分割
        if n_train < 1:
            n_train = 1
            n_val = len(images_list) - 1
        
        train_split[artist_name] = images_list[:n_train]
        val_split[artist_name] = images_list[n_train:]
    
    if skipped_artists:
        print(f"\n⚠️  跳过 {len(skipped_artists)} 个图像数量不足的画师：")
        for artist, count in skipped_artists[:10]:  # 只显示前10个
            print(f"  - {artist}: {count} 张图像 (最少需要 {min_images} 张)")
        if len(skipped_artists) > 10:
            print(f"  ... 还有 {len(skipped_artists) - 10} 个")
    
    return train_split, val_split


def create_dataset_structure(train_split, val_split, output_dir, use_symlink=False):
    """
    创建ImageFolder格式的数据集结构
    """
    output_path = Path(output_dir)
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    
    # 清理旧的输出目录（可选）
    if output_path.exists():
        response = input(f"输出目录 {output_dir} 已存在，是否覆盖？(y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(output_path)
        else:
            print("取消操作")
            return False
    
    # 创建目录结构
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    total_train_images = 0
    total_val_images = 0
    
    # 处理训练集
    print("\n创建训练集...")
    for artist_name, images in tqdm(train_split.items(), desc="处理画师"):
        artist_train_dir = train_dir / artist_name
        artist_train_dir.mkdir(exist_ok=True)
        
        for img_path in images:
            dest_path = artist_train_dir / img_path.name
            
            # 处理文件名冲突
            counter = 1
            original_dest = dest_path
            while dest_path.exists():
                dest_path = original_dest.parent / f"{original_dest.stem}_{counter}{original_dest.suffix}"
                counter += 1
            
            if use_symlink:
                try:
                    dest_path.symlink_to(img_path.resolve())
                except OSError:
                    # Windows可能需要管理员权限，回退到复制
                    shutil.copy2(img_path, dest_path)
            else:
                shutil.copy2(img_path, dest_path)
            
            total_train_images += 1
    
    # 处理验证集
    print("\n创建验证集...")
    for artist_name, images in tqdm(val_split.items(), desc="处理画师"):
        artist_val_dir = val_dir / artist_name
        artist_val_dir.mkdir(exist_ok=True)
        
        for img_path in images:
            dest_path = artist_val_dir / img_path.name
            
            # 处理文件名冲突
            counter = 1
            original_dest = dest_path
            while dest_path.exists():
                dest_path = original_dest.parent / f"{original_dest.stem}_{counter}{original_dest.suffix}"
                counter += 1
            
            if use_symlink:
                try:
                    dest_path.symlink_to(img_path.resolve())
                except OSError:
                    shutil.copy2(img_path, dest_path)
            else:
                shutil.copy2(img_path, dest_path)
            
            total_val_images += 1
    
    return {
        'train_images': total_train_images,
        'val_images': total_val_images,
        'n_classes': len(train_split)
    }


def export_class_mapping_csv(class_to_idx, output_dir):
    """导出 class_id -> class_name 映射 CSV"""
    csv_path = Path(output_dir) / 'class_mapping.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['class_id', 'class_name'])
        writer.writeheader()
        for class_name, class_id in sorted(class_to_idx.items(), key=lambda x: x[1]):
            writer.writerow({'class_id': class_id, 'class_name': class_name})
    return csv_path


def save_dataset_info(train_split, val_split, output_dir, stats):
    """保存数据集信息"""
    output_path = Path(output_dir)
    
    # 类别映射
    class_names = sorted(train_split.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # 保存类别名称（用于推理时显示）
    class_names_file = output_path / 'class_names.json'
    with open(class_names_file, 'w', encoding='utf-8') as f:
        json.dump({str(idx): name for name, idx in class_to_idx.items()}, f, indent=2, ensure_ascii=False)
    
    # 保存数据集统计信息
    dataset_info = {
        'n_classes': len(class_names),
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'train_images': stats['train_images'],
        'val_images': stats['val_images'],
        'total_images': stats['train_images'] + stats['val_images'],
        'train_per_class': {name: len(images) for name, images in train_split.items()},
        'val_per_class': {name: len(images) for name, images in val_split.items()},
    }
    
    info_file = output_path / 'dataset_info.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    # 保存数据分割信息（用于复现）
    split_file = output_path / 'split_info.txt'
    with open(split_file, 'w', encoding='utf-8') as f:
        f.write("训练集:\n")
        for artist, images in sorted(train_split.items()):
            f.write(f"  {artist}: {len(images)} 张\n")
        
        f.write("\n验证集:\n")
        for artist, images in sorted(val_split.items()):
            f.write(f"  {artist}: {len(images)} 张\n")
    
    # 导出 CSV 类别映射
    class_mapping_csv = export_class_mapping_csv(class_to_idx, output_dir)

    return class_names_file, info_file, split_file, class_mapping_csv


def print_summary(train_split, val_split, stats, output_dir):
    """打印摘要信息"""
    print("\n" + "="*60)
    print("数据集创建完成！")
    print("="*60)
    
    print(f"\n📊 统计信息:")
    print(f"  画师数量: {stats['n_classes']}")
    print(f"  训练集图像: {stats['train_images']}")
    print(f"  验证集图像: {stats['val_images']}")
    print(f"  总图像数: {stats['train_images'] + stats['val_images']}")
    
    print(f"\n📁 输出目录: {output_dir}")
    print(f"  ├── train/")
    print(f"  │   ├── {list(train_split.keys())[0]}/ ({len(train_split[list(train_split.keys())[0]])} 张)")
    if len(train_split) > 1:
        print(f"  │   ├── {list(train_split.keys())[1]}/ ({len(train_split[list(train_split.keys())[1]])} 张)")
    if len(train_split) > 2:
        print(f"  │   └── ... ({len(train_split) - 2} 个其他画师)")
    print(f"  ├── val/")
    print(f"  │   ├── {list(val_split.keys())[0]}/ ({len(val_split[list(val_split.keys())[0]])} 张)")
    if len(val_split) > 1:
        print(f"  │   └── ... ({len(val_split) - 1} 个其他画师)")
    print(f"  ├── class_names.json")
    print(f"  ├── dataset_info.json")
    print(f"  └── split_info.txt")
    
    print(f"\n🎯 每个画师的图像分布（前10个）:")
    for i, (artist, images) in enumerate(sorted(train_split.items())[:10]):
        train_count = len(images)
        val_count = len(val_split[artist])
        total = train_count + val_count
        print(f"  {i+1:2d}. {artist:30s} 训练:{train_count:4d}  验证:{val_count:4d}  总计:{total:4d}")
    
    if len(train_split) > 10:
        print(f"  ... 还有 {len(train_split) - 10} 个画师")
    
    print(f"\n✅ 现在可以开始训练了！")
    print(f"\n训练命令示例:")
    print(f"python train_artist_style.py \\")
    print(f"    --model lsnet_t_artist \\")
    print(f"    --data-path {output_dir} \\")
    print(f"    --num-classes {stats['n_classes']} \\")
    print(f"    --batch-size 128 \\")
    print(f"    --epochs 300 \\")
    print(f"    --output-dir ./output/artist_model")


def main(args):
    print("="*60)
    print("画师数据集自动分割工具")
    print("="*60)
    
    print(f"\n配置:")
    print(f"  源目录: {args.source_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  验证集比例: {args.val_ratio:.1%}")
    print(f"  最少图像数: {args.min_images}")
    print(f"  文件操作: {'符号链接' if args.symlink else '复制'}")
    print(f"  随机种子: {args.seed}")
    
    # 收集画师图像
    print(f"\n步骤 1/4: 扫描源目录...")
    artist_images = collect_artist_images(args.source_dir, args.image_extensions)
    
    if not artist_images:
        print("❌ 错误：没有找到任何图像文件")
        print(f"   请检查源目录: {args.source_dir}")
        print(f"   支持的图像格式: {', '.join(args.image_extensions)}")
        return
    
    total_images = sum(len(images) for images in artist_images.values())
    print(f"✓ 找到 {len(artist_images)} 个画师，共 {total_images} 张图像")
    
    # 分割数据集
    print(f"\n步骤 2/4: 分割训练集和验证集...")
    train_split, val_split = split_dataset(
        artist_images, 
        args.val_ratio, 
        args.seed, 
        args.min_images
    )
    
    if not train_split:
        print("❌ 错误：没有足够的数据进行分割")
        print(f"   每个画师至少需要 {args.min_images} 张图像")
        return
    
    print(f"✓ 分割完成：{len(train_split)} 个画师，训练/验证比例 = {1-args.val_ratio:.0%}/{args.val_ratio:.0%}")
    
    # 创建数据集结构
    print(f"\n步骤 3/4: 创建数据集目录结构...")
    stats = create_dataset_structure(
        train_split, 
        val_split, 
        args.output_dir, 
        use_symlink=args.symlink
    )
    
    if not stats:
        return
    
    print(f"✓ 数据集结构创建完成")
    
    # 保存元信息
    print(f"\n步骤 4/4: 保存数据集信息...")
    class_names_file, info_file, split_file, class_mapping_csv = save_dataset_info(
        train_split, 
        val_split, 
        args.output_dir, 
        stats
    )
    
    print(f"✓ 信息文件已保存")
    print(f"  - {class_names_file}")
    print(f"  - {info_file}")
    print(f"  - {split_file}")
    print(f"  - {class_mapping_csv}")
    
    # 打印摘要
    print_summary(train_split, val_split, stats, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Artist Dataset Split Tool', 
        parents=[get_args_parser()],
        description='自动将画师文件夹分割为训练集和验证集'
    )
    args = parser.parse_args()
    main(args)
