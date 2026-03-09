#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=./src

python -m siglip2_multilabel.predict \
  --model_root /workspace/docker_share \
  --model_size base \
  --model_patch 16 \
  --image_size 256 \
  --ckpt outputs/exp_patch16_256/best_model.pt \
  --labels_txt labels.txt \
  --input_txt wps_0302.txt \
  --output_txt result/result_patch16_256.txt \
  --batch_size 64 \
  --threshold 0.5 \
  --query_num 70 \
  --local_files_only
