#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1,2,4,6
export PYTHONPATH=./src

SAVE_DIR=outputs/exp_patch16_256
mkdir -p "${SAVE_DIR}"

nohup accelerate launch --num_processes 4 -m siglip2_multilabel.train \
  --model_root /workspace/docker_share \
  --model_size base \
  --model_patch 16 \
  --image_size 256 \
  --train_txt train.txt \
  --val_txt val.txt \
  --labels_txt labels.txt \
  --batch_size 128 \
  --batch_size_eval 128 \
  --epochs 10 \
  --query_num 70 \
  --token_stride 2 \
  --lr_backbone 1e-5 \
  --lr_head 1e-4 \
  --weight_decay 1e-4 \
  --warmup_epochs 5 \
  --gamma_pos 0 \
  --gamma_neg 4 \
  --asl_clip 0.05 \
  --use_ema \
  --ema_decay 0.9997 \
  --cutout_length 224 \
  --save_every_n_steps 2000 \
  --save_dir "${SAVE_DIR}" \
  --num_workers 8 \
  --local_files_only \
  > "${SAVE_DIR}/train.log" 2>&1 &
