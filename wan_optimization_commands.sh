#!/bin/bash
# WANモデルの最適化コマンド例

# 1. 基本的な実行（既に混合精度を使用）
python -m vace.vace_pipeline \
    --base wan \
    --task text2video \
    --prompt "your prompt" \
    --num_inference_steps 30

# 2. CPUオフロードでさらにメモリ節約
python -m vace.vace_pipeline \
    --base wan \
    --offload_to_cpu \
    --task text2video \
    --prompt "your prompt" \
    --num_inference_steps 30

# 3. マルチGPU使用（利用可能な場合）
# WanVaceMPクラスが自動的に並列処理
python -m vace.vace_pipeline \
    --base wan \
    --task text2video \
    --prompt "your prompt" \
    --device_ids 0,1  # GPU 0と1を使用

# 4. 最大限の最適化
python -m vace.vace_pipeline \
    --base wan \
    --offload_to_cpu \
    --num_inference_steps 25 \
    --solver dpm++ \
    --guidance_scale 5.5 \
    --task text2video \
    --prompt "your prompt"