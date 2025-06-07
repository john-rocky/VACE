#!/bin/bash
# Colab専用の安全な実行スクリプト

echo "Colab Safe Execution Mode"
echo "========================"

# 1. 最新コードを確実に取得
echo "Fetching latest code..."
git fetch origin performance-optimization
git reset --hard origin/performance-optimization

# 2. 環境変数を設定
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb=256"
export CUDA_LAUNCH_BLOCKING=1

# 3. torch.compileを無効化
python colab_fix_torch_compile.py

# 4. キャッシュをクリア
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# 5. 実行（optimizedフラグなしで）
echo "Running WAN inference without torch.compile..."
python -m vace.vace_wan_inference \
    --ckpt_dir models/Wan2.1-VACE-1.3B/ \
    --model_name vace-1.3B \
    --size 480p \
    --frame_num 81 \
    --sample_steps 50 \
    --offload_model True \
    --save_dir results/colab_safe/

echo "Done!"