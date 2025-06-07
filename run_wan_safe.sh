#!/bin/bash
# 安全な実行設定（メモリエラーを回避）

echo "Setting up safe execution environment..."

# torch.compileを無効化
export TORCH_COMPILE_DISABLE=1

# メモリ最適化設定
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

# CUDAグラフを無効化（メモリ節約）
export CUDA_LAUNCH_BLOCKING=1

# キャッシュをクリア
python -c "import torch; torch.cuda.empty_cache()"

echo "Running WAN inference with safe settings..."

# 1. Flash Attentionのみ使用（torch.compile無効）
python -m vace.vace_wan_inference \
    --ckpt_dir models/Wan2.1-VACE-1.3B/ \
    --model_name vace-1.3B \
    --use_optimized \
    --optimization_mode memory \
    --size 480p \
    --frame_num 81 \
    --sample_steps 50 \
    --offload_model True \
    --save_dir results/safe_optimized/

# 2. さらにメモリが不足する場合は、解像度を下げる
# python -m vace.vace_wan_inference \
#     --ckpt_dir models/Wan2.1-VACE-1.3B/ \
#     --model_name vace-1.3B \
#     --use_optimized \
#     --optimization_mode memory \
#     --size 360p \
#     --frame_num 41 \
#     --offload_model True \
#     --save_dir results/safe_optimized_small/