@echo off
REM Multi-GPU training script for Windows
REM Usage: run_multigpu.bat

set CUDA_VISIBLE_DEVICES=0
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=12345
set WORLD_SIZE=1
set RANK=0
set LOCAL_RANK=0

python train_artist_style.py ^
  --model lsnet_t_artist ^
  --data-path "d:\Downloads\lsnet-test\data\artist_dataset" ^
  --batch-size 8 ^
  --accumulation-steps 1 ^
  --epochs 1 ^
  --lr 0.001 ^
  --weight-decay 0.05 ^
  --feature-dim 256 ^
  --num_workers 2 ^
  --output-dir "d:\Downloads\lsnet-test\output\artist_multigpu_test" ^
  --dist-eval

echo Training completed!