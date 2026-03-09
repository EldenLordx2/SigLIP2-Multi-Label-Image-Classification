# SigLIP2 Multi-Label Image Classification

一个 **工程化多标签图像分类仓库**，基于 **SigLIP2 + ML-Decoder**
构建，支持本地模型、离线环境以及多 GPU 训练。

该项目主要用于快速构建
**多标签视觉分类系统**，适用于工业视觉、内容审核、商品识别、场景理解等任务。

------------------------------------------------------------------------

# Features

-   支持 **SigLIP2 + ML-Decoder** 多标签分类架构
-   支持 **本地模型目录 / 离线训练**
-   支持 **Accelerate 多 GPU 训练**
-   提供 **工程化仓库结构**
-   支持 **自动保存训练配置**
-   支持 **灵活模型选择方式**
-   内置 **ASL Loss / EMA / RandAugment / Cutout**
-   提供 **mAP / Precision / Recall / F1** 评估指标

------------------------------------------------------------------------

# Repository Structure

    siglip2-multilabel-repo
    ├── README.md
    ├── requirements.txt
    ├── configs
    │   ├── train
    │   ├── predict
    │   └── dataset
    ├── scripts
    │   ├── train.sh
    │   └── predict.sh
    └── src
        └── siglip2_multilabel
            ├── modeling.py
            ├── data.py
            ├── augmentations.py
            ├── losses.py
            ├── metrics.py
            ├── train.py
            ├── predict.py
            └── utils.py

------------------------------------------------------------------------

# Quick Start

## 1. 克隆仓库

``` bash
git clone https://github.com/yourname/siglip2-multilabel-repo
cd siglip2-multilabel-repo
```

## 2. 安装环境

推荐 Python ≥ 3.10

``` bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
export PYTHONPATH=./src
```

------------------------------------------------------------------------

# Docker 环境（推荐）

推荐使用 ModelScope 官方环境：

``` bash
docker pull modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py311-torch2.8.0-vllm0.11.0-modelscope1.32.0-swift3.11.3
```

运行容器：

``` bash
docker run --gpus all -it --rm -v /workspace/project:/workspace/project -v /workspace/docker_share:/workspace/docker_share -w /workspace/project modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py311-torch2.8.0-vllm0.11.0-modelscope1.32.0-swift3.11.3 /bin/bash
```

------------------------------------------------------------------------

# 数据格式

训练数据格式：

    /path/img1.jpg  0,1,0,0,1
    /path/img2.jpg  1,0,1,0,0

预测输入：

    /path/img1.jpg
    /path/img2.jpg

------------------------------------------------------------------------

# Training

使用 accelerate 启动多卡训练：

``` bash
PYTHONPATH=./src accelerate launch --num_processes 4 -m siglip2_multilabel.train --model_root /workspace/docker_share --model_size base --model_patch 16 --image_size 256 --train_txt train.txt --val_txt val.txt --labels_txt labels.txt --batch_size 128 --epochs 10
```

------------------------------------------------------------------------

# Inference

``` bash
PYTHONPATH=./src python -m siglip2_multilabel.predict --ckpt outputs/best_model.pt --labels_txt labels.txt --input_txt images.txt --output_txt result.txt
```

------------------------------------------------------------------------

# Benchmark

在相同训练数据、验证集和训练参数下：

| Model | mAP |
|------|------|
| CLIP-ViT-B/16 (PaddleX) | 62.67 |
| SigLIP2-base-patch16-224 | **67.34** |

验证集（3.4万张图片）：

| Metric | CLIP | SigLIP2 |
|------|------|------|
| Accuracy | 88.02 | **93.46** |
| Recall | 68.47 | **74.80** |
| F1 | 77.02 | **83.10** |

------------------------------------------------------------------------

# Multi-GPU Training

首次使用：

``` bash
accelerate config
```

启动训练：

``` bash
accelerate launch --num_processes 4 -m siglip2_multilabel.train
```

------------------------------------------------------------------------

# FAQ

### 模型找不到

检查：

-   model_root
-   model_patch
-   image_size

### 训练与预测不一致

确保以下参数一致：

-   labels.txt
-   query_num
-   token_stride

------------------------------------------------------------------------

# References

SIGLIP\
https://github.com/google-research/big_vision

CLIP\
https://github.com/openai/CLIP

PaddleX\
https://github.com/PaddlePaddle/PaddleX
