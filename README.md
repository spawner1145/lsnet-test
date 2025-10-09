# LSNet 画师风格分类与聚类工作流

本仓库提供了一套面向画师风格理解的端到端流程，涵盖数据准备、模型训练、推理部署与聚类检索。以下内容帮助你确认代码是否满足需求，并在一个文档中掌握全流程使用方法。

## 功能概览

- **数据准备**：`prepare_dataset.py` 自动将原始画师文件夹划分为 ImageFolder 结构，并生成 `class_mapping.csv` 等元信息。
- **模型训练**：`train_artist_style.py` 支持多种 LSNet 画师模型，训练结束自动导出类别映射 CSV 及模型权重。
- **推理部署**：`inference_artist.py` 在分类模式下依赖训练生成的 `class_mapping.csv` 进行标签映射，可选提取特征进行聚类或二者同时执行。
- **工具脚本**：`utils.py`、`losses.py`、`robust_utils.py` 等辅助训练；`flops.py`、`speed.py`、`eval.sh` 等用于性能测算与评测。

> ✅ 代码审查结果：训练脚本自动导出 CSV，推理脚本已强制在分类场景加载该 CSV；聚类模式仅需模型权重即可。配合数据准备脚本即可完成整套流程，无额外依赖。

## 环境准备

### Python 依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 已涵盖 `torch`、`timm`、`torchvision` 等核心依赖。若使用 GPU，请确保本地 CUDA 与 PyTorch 版本匹配。

### 数据目录结构

原始数据需按照“画师 -> 多张作品”方式存放：

```
source_dir/
├── artist_A/
│   ├── img_001.jpg
│   ├── ...
├── artist_B/
│   ├── 0001.png
│   ├── ...
└── ...
```

## 步骤一：数据准备

使用 `prepare_dataset.py` 将原始数据划分为训练/验证集，并生成推理所需的 `class_mapping.csv`。

```powershell
python prepare_dataset.py ^
  --source-dir D:\datasets\raw_artists ^
  --output-dir D:\datasets\artist_dataset ^
  --val-ratio 0.2 ^
  --min-images 10
```

脚本执行后将得到：

```
artist_dataset/
├── train/
│   ├── artist_A/
│   └── ...
├── val/
│   ├── artist_A/
│   └── ...
├── class_mapping.csv       # class_id ↔ class_name
├── class_names.json
├── dataset_info.json
└── split_info.txt
```

> `class_mapping.csv` 会在训练与分类推理阶段重复使用，请妥善保留。

## 步骤二：模型训练

运行 `train_artist_style.py` 以训练 LSNet 画师模型。脚本会在输出目录中生成：

- `checkpoint.pth`：最新模型权重
- `model_best.pth`：性能最佳权重
- `class_mapping.csv`：训练时根据数据集自动导出的类别映射
- `train_log.txt` / TensorBoard 日志等

示例命令：

```powershell
python train_artist_style.py ^
  --model lsnet_t_artist ^
  --data-path D:\datasets\artist_dataset ^
  --output-dir D:\experiments\lsnet_t ^
  --batch-size 128 ^
  --epochs 300 ^
  --num-workers 8
```

常用参数说明：

- `--model`：可选 `lsnet_t_artist`、`lsnet_s_artist`、`lsnet_b_artist`
- `--amp`：启用混合精度训练
- `--resume`：断点续训
- `--pretrained`：加载预训练权重作为初始化

训练结束后，`output-dir` 下的 `class_mapping.csv` 将作为后续分类推理的唯一标签映射文件。

## 步骤三：推理与特征提取

`inference_artist.py` 支持三种模式：

- `classify`：使用分类头输出前 Top-K 画师
- `cluster`：仅提取特征向量（无需 CSV）
- `both`：同时输出分类结果与特征

### 单张图像分类

```powershell
python inference_artist.py ^
  --mode classify ^
  --model lsnet_t_artist ^
  --checkpoint D:\experiments\lsnet_t\model_best.pth ^
  --class-csv D:\experiments\lsnet_t\class_mapping.csv ^
  --input D:\samples\test.jpg ^
  --output D:\results\single
```

输出位于 `output\test_result.json`，内含 Top-K 预测类别及概率。

### 批量分类 + 特征保存

```powershell
python inference_artist.py ^
  --mode both ^
  --model lsnet_t_artist ^
  --checkpoint D:\experiments\lsnet_t\model_best.pth ^
  --class-csv D:\experiments\lsnet_t\class_mapping.csv ^
  --input D:\samples\batch ^
  --output D:\results\batch ^
  --batch-size 64
```

当输入为目录时：

- `batch_results.json`：逐图像的分类结果与特征向量
- `features.npy`：堆叠后的特征矩阵，可用于聚类或相似检索
- `image_names.txt`：特征矩阵的文件名顺序

### 仅提取特征

```powershell
python inference_artist.py ^
  --mode cluster ^
  --model lsnet_t_artist ^
  --checkpoint D:\experiments\lsnet_t\model_best.pth ^
  --input D:\samples\batch ^
  --output D:\results\features_only
```

聚类模式不需要 `--class-csv`，只需模型权重即可获取特征表示。

### 进阶：批量聚类与向量相似度

**提取并聚类一个文件夹：**

```powershell
python tools/extract_cluster_features.py ^
  --images-dir D:\samples\artist_A ^
  --model lsnet_t_artist ^
  --checkpoint D:\experiments\lsnet_t\model_best.pth ^
  --num-clusters 6 ^
  --output-dir D:\results\artist_A_cluster
```

输出：

- `features.npy`：上述目录内全部图像的特征矩阵
- `cluster_assignments.json`：KMeans 聚类结果，包含每个簇内的文件名、质心与簇大小

**比较一个向量与多个参考向量的相似度：**

```powershell
python tools/compare_vectors.py ^
  --query-vector D:\results\query.npy ^
  --reference-vectors refs\vector1.npy refs\vector2.npy ^
  --top-k 3 ^
  --normalize ^
  --output D:\results\similarity.json
```

脚本会对查询向量与参考向量执行余弦相似度排序，并在终端及可选 JSON 文件中给出前 `top-k` 结果。`--reference-vectors` 同时支持传入目录（会自动读取其中的 `.npy` 文件）。

## 常见问题排查

| 问题 | 排查建议 |
| --- | --- |
| `timm` 导入失败 | 确认已执行 `pip install -r requirements.txt`，或手动安装 `pip install timm` |
| 分类推理提示缺少 CSV | 分类或 `both` 模式必须提供 `--class-csv`，请使用训练输出目录中的同名文件 |
| 数据集划分脚本覆盖提示 | 若输出目录已存在，需要在提示后输入 `y` 允许覆盖 |
| Windows 下符号链接失败 | 默认为复制模式；若想使用 `--symlink` 需以管理员方式运行或保持复制 |

## 项目结构速览

```
lsnet/
├── train_artist_style.py      # 训练入口
├── inference_artist.py        # 推理/特征提取脚本
├── prepare_dataset.py         # 数据集划分与 CSV 生成
├── model/                     # 模型定义
├── data/                      # 数据增强与数据集实现
├── utils.py / losses.py       # 训练工具
├── requirements.txt           # 依赖列表
└── ...                        # 其他性能测试与评估脚本
```

## 后续建议

- 若需集成到 Web 服务，可将 `inference_artist.py` 封装为 API，输出 JSON 结果或特征库查询。
- 聚类模式生成的 `features.npy` 可直接接入 Faiss、Milvus 等相似度检索系统。
- 如需扩展新的画师类别，重复执行“数据准备 → 训练 → 推理”流程即可。

祝你使用顺利！如果流程中遇到新的需求或问题，欢迎继续反馈。
# [LSNet: See Large, Focus Small](https://arxiv.org/abs/2503.23135)


Official PyTorch implementation of **LSNet**. CVPR 2025.

<p align="center">
  <img src="figures/throughput.svg" width=60%> <br>
  Models are trained on ImageNet-1K and the throughput
 is tested on a Nvidia RTX3090.
</p>

[LSNet: See Large, Focus Small](https://arxiv.org/abs/2503.23135).\
Ao Wang, Hui Chen, Zijia Lin, Jungong Han, and Guiguang Ding\
[![arXiv](https://img.shields.io/badge/arXiv-2503.23135-b31b1b.svg)](https://arxiv.org/abs/2503.23135) [![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/jameslahm/lsnet/tree/main) [![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-blue)](https://huggingface.co/collections/jameslahm/lsnet-67ebec0ab4e220e7918d9565)

We introduce LSNet, a new family of lightweight vision models inspired by dynamic heteroscale capability of the human visual system, i.e., "See Large, Focus Small". LSNet achieves state-of-the-art performance and efficiency trade-offs across various vision tasks.

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Vision network designs, including Convolutional Neural Networks and Vision Transformers, have significantly advanced the field of computer vision. Yet, their complex computations pose challenges for practical deployments, particularly in real-time applications. To tackle this issue, researchers have explored various lightweight and efficient network designs. However, existing lightweight models predominantly leverage self-attention mechanisms and convolutions for token mixing. This dependence brings limitations in effectiveness and efficiency in the perception and aggregation processes of lightweight networks, hindering the balance between performance and efficiency under limited computational budgets. In this paper, we draw inspiration from the dynamic heteroscale vision ability inherent in the efficient human vision system and propose a "See Large, Focus Small" strategy for lightweight vision network design. We introduce LS (<b>L</b>arge-<b>S</b>mall) convolution, which combines large-kernel perception and small-kernel aggregation. It can efficiently capture a wide range of perceptual information and achieve precise feature aggregation for dynamic and complex visual representations, thus enabling proficient processing of visual information. Based on LS convolution, we present LSNet, a new family of lightweight models. Extensive experiments demonstrate that LSNet achieves superior performance and efficiency over existing lightweight networks in various vision tasks.
</details>

## Classification on ImageNet-1K

### Models
- \* denotes the results with distillation.
- The throughput is tested on a Nvidia RTX3090 using [speed.py](./speed.py).

| Model | Top-1 | Params | FLOPs | Throughput | Ckpt | Log |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| LSNet-T | 74.9 / 76.1* | 11.4M | 0.3G | 14708 | [T](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_t.pth) / [T*](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_t_distill.pth) | [T](logs/lsnet_t.log) / [T*](logs/lsnet_t_distill.log) |
| LSNet-S | 77.8 / 79.0* | 16.1M | 0.5G | 9023  | [S](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_s.pth) / [S*](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_s_distill.pth) | [S](logs/lsnet_s.log) / [S*](logs/lsnet_s_distill.log) |
| LSNet-B | 80.3 / 81.6* | 23.2M | 1.3G | 3996  | [B](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_b.pth) / [B*](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_b_distill.pth) | [B](logs/lsnet_b.log) / [B*](logs/lsnet_b_distill.log) |

## ImageNet  

### Prerequisites
`conda` virtual environment is recommended. 
```bash
conda create -n lsnet python=3.8
pip install -r requirements.txt
```

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/. The training and validation data are expected to be in the `train` folder and `val` folder respectively:
```
|-- /path/to/imagenet/
    |-- train
    |-- val
```

### Training
To train LSNet-T on an 8-GPU machine:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model lsnet_t --data-path ~/imagenet --dist-eval
# For training with distillation, please add `--distillation-type hard`
# For LSNet-B, please add `--weight-decay 0.05`
```

### Testing 
```bash
python main.py --eval --model lsnet_t --resume ./pretrain/lsnet_t.pth --data-path ~/imagenet
```
Models can also be automatically downloaded from 🤗 like below.
```python
import timm

model = timm.create_model(
    f'hf_hub:jameslahm/lsnet_{t/t_distill/s/s_distill/b/b_distill}',
    pretrained=True
)
```

## Downstream Tasks
[Object Detection and Instance Segmentation](detection/README.md)<br>
[Semantic Segmentation](segmentation/README.md)<br>
[Robustness Evaluation](README_robustness.md)

## Acknowledgement

Classification (ImageNet) code base is partly built with [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT), [LeViT](https://github.com/facebookresearch/LeViT), [PoolFormer](https://github.com/sail-sg/poolformer) and [EfficientFormer](https://github.com/snap-research/EfficientFormer). 

The detection and segmentation pipeline is from [MMCV](https://github.com/open-mmlab/mmcv) ([MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)). 

Thanks for the great implementations! 

## Citation

If our code or models help your work, please cite our paper:
```BibTeX
@misc{wang2025lsnetlargefocussmall,
      title={LSNet: See Large, Focus Small}, 
      author={Ao Wang and Hui Chen and Zijia Lin and Jungong Han and Guiguang Ding},
      year={2025},
      eprint={2503.23135},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.23135}, 
}
```