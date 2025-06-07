#!/usr/bin/env python
"""
Colab用: torch.compileを完全に無効化するパッチ
"""
import os
import sys

# 環境変数を設定
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# torch.compileをモンキーパッチで無効化
import torch

def dummy_compile(model, *args, **kwargs):
    """torch.compileの代替（何もしない）"""
    print("torch.compile is disabled via monkey patch")
    return model

# torch.compileを上書き
if hasattr(torch, 'compile'):
    torch.compile = dummy_compile
    print("torch.compile has been disabled")

# dynamo も無効化
if hasattr(torch, '_dynamo'):
    torch._dynamo.disable()
    print("torch._dynamo has been disabled")

print("All compilation features disabled successfully")