# -*- coding: utf-8 -*-
# Performance-optimized version of WAN VACE model
import os
import sys
import gc
import math
import time
import random
import types
import logging
import traceback
from contextlib import contextmanager
from functools import partial

from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

from .wan_vace import WanVace, WanVaceMP
from wan.text2video import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps, FlowUniPCMultistepScheduler


class OptimizedWanVace(WanVace):
    """Performance-optimized version of WanVace with Flash Attention 2 and other optimizations"""
    
    def __init__(self, *args, enable_flash_attn=True, enable_torch_compile=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.enable_flash_attn = enable_flash_attn
        self.enable_torch_compile = enable_torch_compile
        
        # Enable Flash Attention 2 if available
        if self.enable_flash_attn:
            self._enable_flash_attention()
        
        # Prepare model for torch.compile if available
        if self.enable_torch_compile and hasattr(torch, 'compile'):
            self._prepare_torch_compile()
    
    def _enable_flash_attention(self):
        """Enable Flash Attention 2 for all attention blocks"""
        try:
            from flash_attn import flash_attn_func
            logging.info("Flash Attention 2 is available, enabling for WAN model")
            
            # Patch attention computation in all blocks
            for block in self.model.blocks:
                if hasattr(block, 'self_attn'):
                    block.self_attn._use_flash_attention_2 = True
                    
            for block in self.model.vace_blocks:
                if hasattr(block, 'self_attn'):
                    block.self_attn._use_flash_attention_2 = True
                    
        except ImportError:
            logging.warning("Flash Attention 2 not available. Install with: pip install flash-attn --no-build-isolation")
    
    def _prepare_torch_compile(self):
        """Prepare model for torch.compile optimization"""
        # Check if torch.compile is disabled via environment variable
        if os.environ.get('TORCH_COMPILE_DISABLE', '0') == '1':
            logging.info("torch.compile disabled via environment variable")
            return
            
        if hasattr(torch, 'compile'):
            try:
                # Only compile if we have enough memory
                if torch.cuda.is_available():
                    free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # GB
                    if free_memory < 8:  # Less than 8GB free
                        logging.warning(f"Insufficient GPU memory ({free_memory:.1f}GB), skipping torch.compile")
                        return
                
                # Compile the model with reduce-overhead mode for best performance
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False,  # Allow for dynamic shapes
                    dynamic=True  # Better for varying input sizes
                )
                logging.info("Model compiled with torch.compile for optimized performance")
            except Exception as e:
                logging.warning(f"Failed to compile model: {e}")
    
    def vace_encode_frames_batch(self, frames, ref_images, masks=None, vae=None, batch_size=4):
        """Optimized batch encoding for VAE with configurable batch size"""
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)
        
        all_latents = []
        
        # Process in batches for better GPU utilization
        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            batch_refs = ref_images[batch_start:batch_end]
            batch_masks = masks[batch_start:batch_end] if masks else None
            
            if batch_masks is None:
                # Simple encoding without masks
                batch_latents = vae.encode(batch_frames)
            else:
                # Optimized mask processing using vectorized operations
                batch_masks_binary = [torch.where(m > 0.5, 1.0, 0.0) for m in batch_masks]
                
                # Vectorized computation for inactive and reactive parts
                inactive = [f * (1 - m) for f, m in zip(batch_frames, batch_masks_binary)]
                reactive = [f * m for f, m in zip(batch_frames, batch_masks_binary)]
                
                # Batch encode
                inactive_latents = vae.encode(inactive)
                reactive_latents = vae.encode(reactive)
                
                # Stack latents
                batch_latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive_latents, reactive_latents)]
            
            # Process reference images if provided
            cat_latents = []
            for idx, (latent, refs) in enumerate(zip(batch_latents, batch_refs)):
                if refs is not None:
                    if batch_masks is None:
                        ref_latent = vae.encode(refs)
                    else:
                        ref_latent = vae.encode(refs)
                        ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                    
                    assert all([x.shape[1] == 1 for x in ref_latent])
                    latent = torch.cat([*ref_latent, latent], dim=1)
                cat_latents.append(latent)
            
            all_latents.extend(cat_latents)
        
        return all_latents
    
    def vace_encode_masks_optimized(self, masks, ref_images=None, vae_stride=None):
        """Optimized mask encoding using vectorized operations"""
        vae_stride = self.vae_stride if vae_stride is None else vae_stride
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)
        
        result_masks = []
        
        # Process all masks in parallel where possible
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // vae_stride[0])
            height = 2 * (int(height) // (vae_stride[1] * 2))
            width = 2 * (int(width) // (vae_stride[2] * 2))
            
            # Vectorized reshape operation
            mask = mask[0]  # Remove channel dimension
            
            # Use einops-style reshape for clarity and performance
            mask_reshaped = mask.view(depth, height, vae_stride[1], width, vae_stride[2])
            mask_reshaped = mask_reshaped.permute(2, 4, 0, 1, 3)
            mask_reshaped = mask_reshaped.reshape(vae_stride[1] * vae_stride[2], depth, height, width)
            
            # Batch interpolation
            mask_interp = F.interpolate(
                mask_reshaped.unsqueeze(0), 
                size=(new_depth, height, width), 
                mode='nearest-exact'
            ).squeeze(0)
            
            # Handle reference images
            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask_interp[:, :length])
                mask_interp = torch.cat((mask_pad, mask_interp), dim=1)
            
            result_masks.append(mask_interp)
        
        return result_masks
    
    def decode_latent_batch(self, zs, ref_images=None, vae=None, batch_size=4):
        """Optimized batch decoding for VAE"""
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(zs)
        else:
            assert len(zs) == len(ref_images)
        
        # Trim latents based on reference images
        trimmed_zs = []
        for z, refs in zip(zs, ref_images):
            if refs is not None:
                z = z[:, len(refs):, :, :]
            trimmed_zs.append(z)
        
        # Batch decode for better GPU utilization
        all_decoded = []
        for batch_start in range(0, len(trimmed_zs), batch_size):
            batch_end = min(batch_start + batch_size, len(trimmed_zs))
            batch_zs = trimmed_zs[batch_start:batch_end]
            
            batch_decoded = vae.decode(batch_zs)
            all_decoded.extend(batch_decoded)
        
        return all_decoded
    
    def generate(self, *args, **kwargs):
        """Override generate method to use optimized encoding/decoding"""
        # Extract relevant arguments
        input_frames = args[1] if len(args) > 1 else kwargs.get('input_frames')
        input_masks = args[2] if len(args) > 2 else kwargs.get('input_masks')
        input_ref_images = args[3] if len(args) > 3 else kwargs.get('input_ref_images')
        
        # Replace with optimized methods
        original_encode = self.vace_encode_frames
        original_encode_masks = self.vace_encode_masks
        original_decode = self.decode_latent
        
        try:
            # Use optimized methods
            self.vace_encode_frames = self.vace_encode_frames_batch
            self.vace_encode_masks = self.vace_encode_masks_optimized
            self.decode_latent = self.decode_latent_batch
            
            # Call parent generate method
            return super().generate(*args, **kwargs)
            
        finally:
            # Restore original methods
            self.vace_encode_frames = original_encode
            self.vace_encode_masks = original_encode_masks
            self.decode_latent = original_decode


class OptimizedWanVaceMP(WanVaceMP):
    """Optimized multi-GPU version of WanVaceMP"""
    
    def __init__(self, *args, enable_flash_attn=True, enable_torch_compile=True, **kwargs):
        self.enable_flash_attn = enable_flash_attn
        self.enable_torch_compile = enable_torch_compile
        super().__init__(*args, **kwargs)
    
    def mp_worker(self, gpu, gpu_infer, pmi_rank, pmi_world_size, in_q_list, out_q, initialized_events, work_env):
        """Override mp_worker to use optimized model"""
        try:
            # Initialize distributed environment
            world_size = pmi_world_size * gpu_infer
            rank = pmi_rank * gpu_infer + gpu
            
            torch.cuda.set_device(gpu)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=rank,
                world_size=world_size
            )
            
            # Import and initialize xfuser if using USP
            if self.use_usp:
                from xfuser.core.distributed import (initialize_model_parallel,
                                                   init_distributed_environment)
                init_distributed_environment(
                    rank=dist.get_rank(), world_size=dist.get_world_size())
                
                initialize_model_parallel(
                    sequence_parallel_degree=dist.get_world_size(),
                    ring_degree=self.ring_size or 1,
                    ulysses_degree=self.ulysses_size or 1
                )
            
            # Create optimized instance for this worker
            optimized_vace = OptimizedWanVace(
                config=self.config,
                checkpoint_dir=self.checkpoint_dir,
                device_id=gpu,
                rank=rank,
                t5_fsdp=True,
                dit_fsdp=True,
                use_usp=self.use_usp,
                enable_flash_attn=self.enable_flash_attn,
                enable_torch_compile=self.enable_torch_compile
            )
            
            # Signal initialization complete
            event = initialized_events[gpu]
            in_q = in_q_list[gpu]
            event.set()
            
            # Process tasks
            while True:
                item = in_q.get()
                if item is None:  # Shutdown signal
                    break
                
                # Unpack arguments
                (input_prompt, input_frames, input_masks, input_ref_images, size, 
                 frame_num, context_scale, shift, sample_solver, sampling_steps, 
                 guide_scale, n_prompt, seed, offload_model) = item
                
                # Transfer data to GPU
                input_frames = self.transfer_data_to_cuda(input_frames, gpu)
                input_masks = self.transfer_data_to_cuda(input_masks, gpu)
                input_ref_images = self.transfer_data_to_cuda(input_ref_images, gpu)
                
                # Generate using optimized model
                result = optimized_vace.generate(
                    input_prompt, input_frames, input_masks, input_ref_images,
                    size=size, frame_num=frame_num, context_scale=context_scale,
                    shift=shift, sample_solver=sample_solver,
                    sampling_steps=sampling_steps, guide_scale=guide_scale,
                    n_prompt=n_prompt, seed=seed, offload_model=offload_model
                )
                
                if rank == 0 and result is not None:
                    out_q.put(result.cpu())
                    
        except Exception as e:
            trace_info = traceback.format_exc()
            logging.error(f"Worker {gpu} error: {trace_info}")
            logging.error(f"Exception: {e}")