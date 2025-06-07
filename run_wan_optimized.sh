#!/bin/bash
# 高速化モードでWAN推論を実行するサンプルスクリプト

# 環境変数で最適化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# 速度優先モード（最速）
echo "Running WAN inference with speed optimization..."
python -m vace.vace_wan_inference \
    --ckpt_dir models/Wan2.1-VACE-1.3B/ \
    --model_name vace-1.3B \
    --use_optimized \
    --optimization_mode speed \
    --size 480p \
    --frame_num 81 \
    --sample_steps 50 \
    --save_dir results/optimized_speed/

# メモリ優先モード（省メモリ）
echo "Running WAN inference with memory optimization..."
python -m vace.vace_wan_inference \
    --ckpt_dir models/Wan2.1-VACE-1.3B/ \
    --model_name vace-1.3B \
    --use_optimized \
    --optimization_mode memory \
    --size 480p \
    --frame_num 81 \
    --offload_model True \
    --save_dir results/optimized_memory/

# バランスモード（推奨）
echo "Running WAN inference with balanced optimization..."
python -m vace.vace_wan_inference \
    --ckpt_dir models/Wan2.1-VACE-1.3B/ \
    --model_name vace-1.3B \
    --use_optimized \
    --optimization_mode balanced \
    --size 480p \
    --frame_num 81 \
    --save_dir results/optimized_balanced/

# 通常モード（比較用）
echo "Running WAN inference without optimization (for comparison)..."
python -m vace.vace_wan_inference \
    --ckpt_dir models/Wan2.1-VACE-1.3B/ \
    --model_name vace-1.3B \
    --size 480p \
    --frame_num 81 \
    --sample_steps 50 \
    --save_dir results/standard/