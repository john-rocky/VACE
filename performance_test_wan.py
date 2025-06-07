#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance test script for WAN model optimizations
"""
import time
import torch
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from vace.models.wan import WanVace, OptimizedWanVace, MemoryEfficientConfig
from vace.models.wan.optimized_pipeline import OptimizedWanPipeline


def measure_performance(model, test_config, num_runs=3):
    """Measure performance metrics for a model"""
    
    # Test parameters
    input_prompt = "A beautiful sunset over the ocean with waves"
    size = (832, 480)  # Standard test size
    frame_num = 81
    sampling_steps = test_config.get('sampling_steps', 25)
    
    # Prepare dummy inputs
    device = model.device
    input_frames = [torch.randn(3, frame_num, size[1], size[0], device=device)]
    input_masks = [torch.ones(1, frame_num, size[1], size[0], device=device)]
    input_ref_images = [None]
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        _ = model.generate(
            input_prompt, input_frames, input_masks, input_ref_images,
            size=size, frame_num=frame_num, sampling_steps=5,  # Quick warmup
            seed=42, offload_model=False
        )
    
    # Measure performance
    times = []
    memory_usage = []
    
    for run in range(num_runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        with torch.no_grad():
            _ = model.generate(
                input_prompt, input_frames, input_masks, input_ref_images,
                size=size, frame_num=frame_num, sampling_steps=sampling_steps,
                seed=42, offload_model=test_config.get('offload_model', False)
            )
        
        end_time = time.time()
        
        times.append(end_time - start_time)
        memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)  # GB
        
        print(f"Run {run+1}: {times[-1]:.2f}s, Memory: {memory_usage[-1]:.2f}GB")
    
    avg_time = sum(times) / len(times)
    avg_memory = sum(memory_usage) / len(memory_usage)
    
    return {
        'avg_time': avg_time,
        'avg_memory': avg_memory,
        'times': times,
        'memory_usage': memory_usage
    }


def compare_models(config_path, checkpoint_dir):
    """Compare standard vs optimized WAN models"""
    
    print("="*60)
    print("WAN Model Performance Comparison")
    print("="*60)
    
    # Test configurations
    test_configs = {
        'standard': {
            'sampling_steps': 25,
            'offload_model': False,
        },
        'optimized_speed': {
            'sampling_steps': 25,
            'offload_model': False,
            'optimization_mode': 'speed',
        },
        'optimized_memory': {
            'sampling_steps': 25,
            'offload_model': True,
            'optimization_mode': 'memory',
        }
    }
    
    results = {}
    
    # Load config
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    
    # Test standard model
    print("\n1. Testing Standard WAN Model...")
    print("-"*40)
    
    standard_model = WanVace(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0
    )
    
    results['standard'] = measure_performance(standard_model, test_configs['standard'])
    del standard_model
    torch.cuda.empty_cache()
    
    # Test optimized models
    if OptimizedWanVace is not None:
        # Speed-optimized
        print("\n2. Testing Speed-Optimized WAN Model...")
        print("-"*40)
        
        opt_config = MemoryEfficientConfig.get_config('speed')
        optimized_model = OptimizedWanVace(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=0,
            enable_flash_attn=opt_config['enable_flash_attention'],
            enable_torch_compile=opt_config['enable_torch_compile']
        )
        
        results['optimized_speed'] = measure_performance(
            optimized_model, test_configs['optimized_speed']
        )
        del optimized_model
        torch.cuda.empty_cache()
        
        # Memory-optimized
        print("\n3. Testing Memory-Optimized WAN Model...")
        print("-"*40)
        
        opt_config = MemoryEfficientConfig.get_config('memory')
        memory_model = OptimizedWanVace(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=0,
            enable_flash_attn=opt_config['enable_flash_attention'],
            enable_torch_compile=opt_config['enable_torch_compile']
        )
        
        results['optimized_memory'] = measure_performance(
            memory_model, test_configs['optimized_memory']
        )
        del memory_model
        torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    baseline_time = results['standard']['avg_time']
    baseline_memory = results['standard']['avg_memory']
    
    for name, result in results.items():
        speedup = baseline_time / result['avg_time']
        memory_reduction = (1 - result['avg_memory'] / baseline_memory) * 100
        
        print(f"\n{name.upper()}:")
        print(f"  Average Time: {result['avg_time']:.2f}s")
        print(f"  Average Memory: {result['avg_memory']:.2f}GB")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Memory Reduction: {memory_reduction:+.1f}%")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if 'optimized_speed' in results:
        speed_gain = (baseline_time / results['optimized_speed']['avg_time'] - 1) * 100
        print(f"✓ Speed optimization provides {speed_gain:.1f}% performance improvement")
    
    if 'optimized_memory' in results:
        mem_save = (1 - results['optimized_memory']['avg_memory'] / baseline_memory) * 100
        print(f"✓ Memory optimization saves {mem_save:.1f}% GPU memory")
    
    print("\nTo use optimizations in your pipeline:")
    print("  python -m vace.vace_pipeline --base wan --use_optimized --optimization_mode speed")
    print("  python -m vace.vace_pipeline --base wan --use_optimized --optimization_mode memory")


def main():
    parser = argparse.ArgumentParser(description='Test WAN model performance optimizations')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to WAN config file')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='Number of test runs (default: 3)')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This test requires a GPU.")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Run comparison
    compare_models(args.config, args.checkpoint_dir)


if __name__ == "__main__":
    main()