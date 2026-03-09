# SigLIP2 Multi-Label Image Classification

An **engineering-oriented repository for multi-label image
classification** built with **SigLIP2 + ML-Decoder**.

This project is designed for **large-scale multi-label vision tasks**
and supports:

-   Local model directories
-   Offline environments
-   Multi-GPU distributed training

Typical use cases include:

-   Industrial image classification
-   Content moderation
-   Product recognition
-   Scene understanding

------------------------------------------------------------------------

# Features

-   SigLIP2 + ML-Decoder architecture
-   Local model directory support
-   Offline training support
-   Accelerate multi-GPU training
-   Modular engineering code structure
-   Automatic training configuration saving
-   Built-in ASL Loss, EMA, RandAugment, Cutout
-   mAP / Precision / Recall / F1 evaluation

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

## 1. Clone Repository

``` bash
git clone https://github.com/yourname/siglip2-multilabel-repo
cd siglip2-multilabel-repo
```

## 2. Install Dependencies

Recommended Python ≥ 3.10

``` bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
export PYTHONPATH=./src
```

------------------------------------------------------------------------

# Docker Environment (Recommended)

Use the official ModelScope environment:

``` bash
docker pull modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py311-torch2.8.0-vllm0.11.0-modelscope1.32.0-swift3.11.3
```

Run container:

``` bash
docker run --gpus all -it --rm -v /workspace/project:/workspace/project -v /workspace/docker_share:/workspace/docker_share -w /workspace/project modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py311-torch2.8.0-vllm0.11.0-modelscope1.32.0-swift3.11.3 /bin/bash
```

------------------------------------------------------------------------

# Dataset Format

Training format:

    /path/img1.jpg  0,1,0,0,1
    /path/img2.jpg  1,0,1,0,0

Inference input:

    /path/img1.jpg
    /path/img2.jpg

------------------------------------------------------------------------

# Training

Example multi-GPU training:

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

Under the same dataset and training configuration:

  Model                      mAP
  -------------------------- -----------
  SigLIP2-base-patch16-224   62.67
  CLIP-ViT-B/16 (PaddleX)    **67.34**

Validation dataset (34k images):

  Metric     SigLIP2   CLIP
  ---------- --------- -----------
  Accuracy   88.02     **93.46**
  Recall     68.47     **74.80**
  F1         77.02     **83.10**

------------------------------------------------------------------------

# Multi-GPU Training

Configure accelerate:

``` bash
accelerate config
```

Launch training:

``` bash
accelerate launch --num_processes 4 -m siglip2_multilabel.train
```

------------------------------------------------------------------------

# FAQ

### Model path not found

Check:

-   model_root
-   model_patch
-   image_size

### Training / inference mismatch

Ensure consistency of:

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
