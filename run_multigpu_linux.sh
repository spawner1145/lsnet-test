#!/bin/bash
# Multi-GPU training script for Linux
# Usage: bash run_multigpu_linux.sh

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

# Launch training with torchrun
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  train_artist_style.py \
  --model lsnet_t_artist \
  --data-path "/path/to/your/artist_dataset" \
  --output-dir "./output_multigpu" \
  --batch-size 16 \
  --accumulation-steps 2 \
  --epochs 300 \
  --lr 0.001 \
  --weight-decay 0.1 \
  --num_workers 8