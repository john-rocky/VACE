#!/usr/bin/env python
"""
Disable torch.compile for environments with limited memory or compatibility issues
"""

import os

# Disable torch.compile globally
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Additional memory optimization settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

# Disable cudnn benchmarking to save memory
os.environ['CUDNN_BENCHMARK'] = '0'

print("torch.compile disabled and memory optimizations applied")
print("You can now run the model without compilation overhead")