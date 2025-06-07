#!/bin/bash
# 実用的な高速化設定

# 1. 環境変数で最適化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # GPUアーキテクチャに合わせて調整

# 2. xFormersインストール（メモリ効率的なアテンション）
pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118

# 3. 最適化されたコマンド
python -m vace.vace_pipeline \
    --base ltx \
    --task text2video \
    --prompt "your prompt" \
    --num_inference_steps 25 \
    --decode_timestep 0.0 \
    --decode_noise_scale 0.0 \
    --seed 42  # 固定シードで再現性確保

# 4. プロファイリングで実際のボトルネックを特定
python -m torch.profiler \
    -m vace.vace_pipeline \
    --base ltx \
    --task text2video \
    --prompt "test" \
    --num_inference_steps 5