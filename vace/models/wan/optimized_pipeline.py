# -*- coding: utf-8 -*-
# Optimized pipeline for WAN model with CPU-GPU sync improvements
import torch
import logging
from contextlib import contextmanager


class OptimizedWanPipeline:
    """Pipeline optimizations for WAN model focusing on CPU-GPU sync"""
    
    @staticmethod
    def optimize_timesteps(timesteps, device):
        """Pre-convert timesteps to tensors to avoid repeated CPU-GPU sync"""
        if isinstance(timesteps, list):
            # Convert all timesteps to tensors at once
            return torch.tensor(timesteps, device=device, dtype=torch.float32)
        elif isinstance(timesteps, torch.Tensor):
            # Ensure tensor is on correct device
            return timesteps.to(device)
        else:
            return timesteps
    
    @staticmethod
    @contextmanager
    def optimized_inference_mode():
        """Context manager for optimized inference with better memory management"""
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                # Clear cache before inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                yield
                
                # Clear cache after inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    @staticmethod
    def optimize_memory_allocation():
        """Optimize CUDA memory allocation for better performance"""
        if torch.cuda.is_available():
            # Set memory fraction for better allocation
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            # Enable cudnn benchmarking for optimal conv performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    @staticmethod
    def create_optimized_scheduler(scheduler_class, timesteps, device, **kwargs):
        """Create scheduler with pre-allocated tensors"""
        scheduler = scheduler_class(**kwargs)
        
        # Pre-allocate timesteps on device
        if hasattr(scheduler, 'timesteps'):
            scheduler.timesteps = OptimizedWanPipeline.optimize_timesteps(
                scheduler.timesteps, device
            )
        
        return scheduler
    
    @staticmethod
    def batch_tensor_operations(tensors, operation, batch_size=8):
        """Batch tensor operations to reduce kernel launch overhead"""
        results = []
        
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i+batch_size]
            if len(batch) > 1:
                # Stack tensors for batch operation
                stacked = torch.stack(batch)
                result = operation(stacked)
                results.extend(result.unbind(0))
            else:
                # Single tensor, process directly
                results.append(operation(batch[0]))
        
        return results


class MemoryEfficientConfig:
    """Configuration for memory-efficient WAN model execution"""
    
    # Default optimization settings
    DEFAULT_SETTINGS = {
        'enable_flash_attention': True,
        'enable_torch_compile': True,
        'enable_cpu_offload': False,
        'enable_sequential_cpu_offload': False,
        'vae_batch_size': 4,
        'enable_gradient_checkpointing': False,
        'enable_xformers': True,
        'optimize_cuda_allocation': True,
        'clear_cache_steps': 10,  # Clear CUDA cache every N steps
        'enable_frame_skip': False,  # Generate half frames and interpolate
    }
    
    # Memory-optimized settings
    MEMORY_OPTIMIZED = {
        **DEFAULT_SETTINGS,
        'enable_cpu_offload': True,
        'vae_batch_size': 2,
        'enable_gradient_checkpointing': True,
        'clear_cache_steps': 5,
    }
    
    # Speed-optimized settings
    SPEED_OPTIMIZED = {
        **DEFAULT_SETTINGS,
        'enable_cpu_offload': False,
        'vae_batch_size': 8,
        'enable_gradient_checkpointing': False,
        'clear_cache_steps': 20,
        'enable_frame_skip': True,  # Enable frame skipping for 2x speedup
    }
    
    # Balanced settings
    BALANCED = {
        **DEFAULT_SETTINGS,
        'enable_cpu_offload': False,
        'vae_batch_size': 4,
        'enable_gradient_checkpointing': False,
        'clear_cache_steps': 10,
    }
    
    @classmethod
    def get_config(cls, mode='balanced'):
        """Get optimization configuration by mode"""
        configs = {
            'memory': cls.MEMORY_OPTIMIZED,
            'speed': cls.SPEED_OPTIMIZED,
            'balanced': cls.BALANCED,
            'default': cls.DEFAULT_SETTINGS,
        }
        return configs.get(mode, cls.DEFAULT_SETTINGS).copy()