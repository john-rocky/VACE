# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from . import modules
from .wan_vace import WanVace, WanVaceMP

# Import optimized versions if available
try:
    from .optimized_wan_vace import OptimizedWanVace, OptimizedWanVaceMP
    from .optimized_pipeline import OptimizedWanPipeline, MemoryEfficientConfig
except ImportError:
    OptimizedWanVace = None
    OptimizedWanVaceMP = None
    OptimizedWanPipeline = None
    MemoryEfficientConfig = None

__all__ = ['modules', 'WanVace', 'WanVaceMP', 'OptimizedWanVace', 'OptimizedWanVaceMP', 
           'OptimizedWanPipeline', 'MemoryEfficientConfig']
