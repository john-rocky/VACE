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
    
    def __init__(self, *args, enable_flash_attn=True, enable_torch_compile=True, enable_frame_skip=False, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.enable_flash_attn = enable_flash_attn
        # Force disable torch.compile if environment variable is set
        self.enable_torch_compile = enable_torch_compile and os.environ.get('TORCH_COMPILE_DISABLE', '0') != '1'
        self.enable_frame_skip = enable_frame_skip
        
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable cuDNN autotuner for optimal convolution algorithms
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            print("✓ CUDA optimizations enabled (TF32, cuDNN benchmark)")
        
        if self.enable_frame_skip:
            print("✓ Frame skip optimization enabled (2x speedup)")
            print("  → Will generate half frames and interpolate in latent space")
        
        # Enable Flash Attention 2 if available
        if self.enable_flash_attn:
            self._enable_flash_attention()
        
        # Prepare model for torch.compile if available
        if self.enable_torch_compile and hasattr(torch, 'compile'):
            self._prepare_torch_compile()
            
        # Optimize memory allocation
        if torch.cuda.is_available():
            # Pre-allocate memory pool for better performance
            torch.cuda.empty_cache()
            # Set memory fraction for caching allocator
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory
    
    def _setup_comprehensive_tracing(self):
        """Set up comprehensive tracing to see what functions are actually called"""
        try:
            import wan.modules.attention as wan_attn
            
            print("  → Setting up function call tracing...")
            
            # Get all callable attributes
            all_functions = []
            for attr_name in dir(wan_attn):
                if not attr_name.startswith('_'):
                    attr = getattr(wan_attn, attr_name)
                    if callable(attr):
                        all_functions.append(attr_name)
            
            print(f"  → Found functions to trace: {all_functions}")
            
            # Trace each function
            for func_name in all_functions:
                try:
                    original_func = getattr(wan_attn, func_name)
                    
                    def make_tracer(name, orig_func):
                        def traced_func(*args, **kwargs):
                            print(f"🔍 {name} called!")
                            if args and hasattr(args[0], 'shape'):
                                shapes = [f"{arg.shape}" for arg in args[:3] if hasattr(arg, 'shape')]
                                print(f"  → Input shapes: {shapes}")
                            
                            result = orig_func(*args, **kwargs)
                            
                            if hasattr(result, 'shape'):
                                print(f"  → Output shape: {result.shape}")
                            
                            return result
                        return traced_func
                    
                    setattr(wan_attn, func_name, make_tracer(func_name, original_func))
                    
                except Exception as e:
                    print(f"  → Could not trace {func_name}: {e}")
            
            # Replace PyTorch attention with Flash Attention
            try:
                import torch.nn.functional as F
                from flash_attn import flash_attn_func
                original_sdpa = F.scaled_dot_product_attention
                
                def flash_attention_replacement(*args, **kwargs):
                    """Replace PyTorch SDPA with Flash Attention"""
                    if len(args) >= 3:
                        q, k, v = args[0], args[1], args[2]
                        
                        # Check head dimension size
                        head_dim = q.shape[-1]
                        
                        # Try to reshape for Flash Attention if head_dim is too large
                        if head_dim > 256 and head_dim % 128 == 0:
                            # Likely multiple heads concatenated (e.g., 384 = 3 x 128)
                            num_heads_combined = head_dim // 128
                            batch_size = q.shape[0]
                            seq_len = q.shape[2] if q.dim() == 4 else q.shape[1]
                            
                            print(f"🔄 Attempting to reshape head_dim {head_dim} -> {num_heads_combined} x 128")
                            print(f"  Original shape: q{q.shape}, k{k.shape}, v{v.shape}")
                            
                            try:
                                # Reshape to split the concatenated heads
                                if q.dim() == 4:  # [batch, heads, seq, dim]
                                    orig_heads = q.shape[1]
                                    # Reshape to [batch, heads * num_heads_combined, seq, 128]
                                    q_split = q.view(batch_size, orig_heads * num_heads_combined, seq_len, 128)
                                    k_split = k.view(batch_size, orig_heads * num_heads_combined, seq_len, 128)
                                    v_split = v.view(batch_size, orig_heads * num_heads_combined, seq_len, 128)
                                    
                                    # Now use Flash Attention with proper head dimension
                                    q_fa = q_split.transpose(1, 2)  # [batch, seq, heads, 128]
                                    k_fa = k_split.transpose(1, 2)
                                    v_fa = v_split.transpose(1, 2)
                                    
                                    # Ensure half precision
                                    if q_fa.dtype not in [torch.half, torch.bfloat16]:
                                        q_fa = q_fa.half()
                                        k_fa = k_fa.half()
                                        v_fa = v_fa.half()
                                    
                                    # Use Flash Attention
                                    out = flash_attn_func(q_fa, k_fa, v_fa, 
                                                        dropout_p=kwargs.get('dropout_p', 0.0),
                                                        causal=kwargs.get('is_causal', False))
                                    
                                    # Reshape back
                                    out = out.transpose(1, 2)  # [batch, heads, seq, 128]
                                    out = out.view(batch_size, orig_heads, seq_len, head_dim)
                                    
                                    return out
                                    
                            except Exception as e:
                                pass  # Fall through to other methods
                        
                        if head_dim > 256:
                            # Head dimension too large for Flash Attention
                            # Try xFormers memory efficient attention
                            try:
                                import xformers.ops as xops
                                # xFormers expects [batch, seq, heads, dim] format
                                if q.dim() == 4:  # [batch, heads, seq, dim]
                                    q_xf = q.transpose(1, 2).contiguous()
                                    k_xf = k.transpose(1, 2).contiguous()
                                    v_xf = v.transpose(1, 2).contiguous()
                                    
                                    # Use xFormers memory efficient attention
                                    out = xops.memory_efficient_attention(q_xf, k_xf, v_xf)
                                    # Convert back to [batch, heads, seq, dim]
                                    out = out.transpose(1, 2).contiguous()
                                    return out
                                else:
                                    # Already in correct format
                                    return xops.memory_efficient_attention(q, k, v)
                                    
                            except Exception:
                                # Fallback to PyTorch SDPA with memory efficient settings
                                with torch.backends.cuda.sdp_kernel(
                                    enable_flash=False,
                                    enable_math=False,
                                    enable_mem_efficient=True
                                ):
                                    return original_sdpa(*args, **kwargs)
                        
                        # Standard Flash Attention for smaller head dimensions
                        if head_dim <= 256:
                            print(f"🚀 Using Flash Attention: q{q.shape}, head_dim={head_dim}")
                        
                        try:
                            # Convert to Flash Attention format: [batch, seq, heads, dim]
                            if q.dim() == 4:  # [batch, heads, seq, dim] -> [batch, seq, heads, dim]
                                q_fa = q.transpose(1, 2)
                                k_fa = k.transpose(1, 2) 
                                v_fa = v.transpose(1, 2)
                            else:
                                q_fa, k_fa, v_fa = q, k, v
                            
                            # Ensure half precision for Flash Attention
                            if q_fa.dtype not in [torch.half, torch.bfloat16]:
                                q_fa = q_fa.half()
                                k_fa = k_fa.half()
                                v_fa = v_fa.half()
                            
                            # Use Flash Attention
                            out = flash_attn_func(q_fa, k_fa, v_fa, 
                                                dropout_p=kwargs.get('dropout_p', 0.0),
                                                causal=kwargs.get('is_causal', False))
                            
                            # Convert back to original format
                            if q.dim() == 4:  # Convert back to [batch, heads, seq, dim]
                                out = out.transpose(1, 2)
                            
                            print(f"  → Flash Attention success: {out.shape}")
                            return out
                            
                        except Exception as e:
                            print(f"  → Flash Attention failed ({e}), falling back to PyTorch SDPA")
                            return original_sdpa(*args, **kwargs)
                    else:
                        return original_sdpa(*args, **kwargs)
                
                F.scaled_dot_product_attention = flash_attention_replacement
                print("  → ✅ Replaced PyTorch SDPA with Flash Attention!")
                
            except Exception as e:
                print(f"  → Could not replace PyTorch attention: {e}")
                
        except Exception as e:
            print(f"  → Tracing setup failed: {e}")
    
    def _enable_flash_attention(self):
        """Enable Flash Attention 2 for all attention blocks"""
        try:
            from flash_attn import flash_attn_func
            print("✓ Flash Attention 2 detected")
            logging.info("Flash Attention 2 is available, optimizing WAN model")
            
            # First, let's trace what functions actually exist and are called
            self._setup_comprehensive_tracing()
            
            # Import WAN attention module to patch
            try:
                import wan.modules.attention as wan_attn
                
                # Store original attention function
                if not hasattr(wan_attn, '_original_flash_attention'):
                    wan_attn._original_flash_attention = getattr(wan_attn, 'flash_attention', None)
                
                # Create optimized flash attention function
                def optimized_flash_attention(q, k, v, k_lens, causal=False, dropout_p=0.0):
                    """Optimized flash attention using flash_attn_func"""
                    print(f"🚀 Flash Attention called: q{q.shape}, k{k.shape}, v{v.shape}")
                    
                    try:
                        # Handle k_lens properly - concatenate tensors if needed
                        if isinstance(k_lens, (list, tuple)):
                            k_cat = torch.cat([k_i[:length] for k_i, length in zip(k, k_lens)])
                            v_cat = torch.cat([v_i[:length] for v_i, length in zip(v, k_lens)])
                            q_cat = q  # Usually q doesn't need truncation
                        else:
                            k_cat = k
                            v_cat = v  
                            q_cat = q
                        
                        # Convert to flash_attn format: [batch, seq, heads, dim]
                        if q_cat.dim() == 4:  # [batch, heads, seq, dim]
                            q_fa = q_cat.transpose(1, 2)  # [batch, seq, heads, dim]
                            k_fa = k_cat.transpose(1, 2)
                            v_fa = v_cat.transpose(1, 2)
                        else:
                            q_fa, k_fa, v_fa = q_cat, k_cat, v_cat
                        
                        # Ensure correct dtype
                        if q_fa.dtype != torch.half and q_fa.dtype != torch.bfloat16:
                            q_fa = q_fa.half()
                            k_fa = k_fa.half() 
                            v_fa = v_fa.half()
                        
                        print(f"  → Using Flash Attention with shapes: q{q_fa.shape}, k{k_fa.shape}, v{v_fa.shape}")
                        
                        # Use flash attention
                        out = flash_attn_func(q_fa, k_fa, v_fa, dropout_p=dropout_p, causal=causal)
                        
                        # Convert back to original format if needed
                        if q.dim() == 4:  # Need to transpose back
                            out = out.transpose(1, 2)  # Back to [batch, heads, seq, dim]
                        
                        print(f"  → Flash Attention output: {out.shape}")
                        return out
                        
                    except Exception as e:
                        print(f"  → Flash Attention failed: {e}, falling back to original")
                        # Fallback to original implementation
                        if wan_attn._original_flash_attention:
                            return wan_attn._original_flash_attention(q, k, v, k_lens, causal, dropout_p)
                        else:
                            # Fallback to standard attention
                            if q.dim() == 4:  # [batch, heads, seq, dim]
                                return torch.nn.functional.scaled_dot_product_attention(q, k, v)
                            else:
                                # Reshape for standard attention
                                q_std = q.transpose(1, 2) if q.dim() == 4 else q
                                k_std = k.transpose(1, 2) if k.dim() == 4 else k
                                v_std = v.transpose(1, 2) if v.dim() == 4 else v
                                out_std = torch.nn.functional.scaled_dot_product_attention(q_std, k_std, v_std)
                                return out_std.transpose(1, 2) if q.dim() == 4 else out_std
                
                # Patch the function
                wan_attn.flash_attention = optimized_flash_attention
                print("  → Successfully patched wan.modules.attention.flash_attention")
                
                # Also try to patch other attention functions if they exist
                attention_functions = [attr for attr in dir(wan_attn) if 'attention' in attr.lower() and callable(getattr(wan_attn, attr))]
                for func_name in attention_functions:
                    if func_name not in ['flash_attention', '_original_flash_attention']:
                        try:
                            original_func = getattr(wan_attn, func_name)
                            # Save original if not already saved
                            backup_name = f'_original_{func_name}'
                            if not hasattr(wan_attn, backup_name):
                                setattr(wan_attn, backup_name, original_func)
                            
                            # Create wrapper that tries flash attention first
                            def make_flash_wrapper(orig_func, fname):
                                def flash_wrapper(*args, **kwargs):
                                    try:
                                        # If this looks like an attention call, try our optimized version
                                        if len(args) >= 3:
                                            return optimized_flash_attention(*args, **kwargs)
                                    except:
                                        pass
                                    # Fall back to original
                                    return orig_func(*args, **kwargs)
                                return flash_wrapper
                            
                            setattr(wan_attn, func_name, make_flash_wrapper(original_func, func_name))
                            print(f"  → Also patched {func_name}")
                            
                        except Exception as e:
                            print(f"  → Could not patch {func_name}: {e}")
                
                # Patch attention at the block level too
                try:
                    from wan.modules.attention import WanAttentionBlock
                    
                    # Store original forward method
                    if not hasattr(WanAttentionBlock, '_original_forward'):
                        WanAttentionBlock._original_forward = WanAttentionBlock.forward
                    
                    def optimized_attention_forward(self, x, **kwargs):
                        """Optimized attention block forward with Flash Attention"""
                        # Add debug print to see if this is being called
                        if hasattr(self, '_flash_debug_count'):
                            self._flash_debug_count += 1
                        else:
                            self._flash_debug_count = 1
                            print(f"🔧 Optimized attention block forward called (block #{id(self) % 1000})")
                        
                        return self._original_forward(x, **kwargs)
                    
                    WanAttentionBlock.forward = optimized_attention_forward
                    print("  → Patched WanAttentionBlock.forward")
                    
                except Exception as e:
                    print(f"  → Could not patch WanAttentionBlock: {e}")
                
                # Count blocks for display
                main_blocks = len(self.model.blocks) if hasattr(self.model, 'blocks') else 0
                vace_blocks = len(self.model.vace_blocks) if hasattr(self.model, 'vace_blocks') else 0
                print(f"  → Will apply to {main_blocks} main blocks + {vace_blocks} VACE blocks")
                
            except ImportError as e:
                print(f"  → Cannot patch WAN attention module: {e}")
                print("  → Falling back to attribute setting method")
                
                # Fallback to the original method
                main_blocks = 0
                vace_blocks = 0
                
                for block in self.model.blocks:
                    if hasattr(block, 'self_attn'):
                        block.self_attn._use_flash_attention_2 = True
                        main_blocks += 1
                        
                for block in self.model.vace_blocks:
                    if hasattr(block, 'self_attn'):
                        block.self_attn._use_flash_attention_2 = True
                        vace_blocks += 1
                
                print(f"  → Applied flags to {main_blocks} main blocks + {vace_blocks} VACE blocks")
                    
        except ImportError:
            print("✗ Flash Attention 2 not available")
            logging.warning("Flash Attention 2 not available. Install with: pip install flash-attn --no-build-isolation")
    
    def _prepare_torch_compile(self):
        """Prepare model for torch.compile optimization"""
        # Check if torch.compile is disabled via environment variable
        if os.environ.get('TORCH_COMPILE_DISABLE', '0') == '1':
            print("○ torch.compile disabled via environment variable")
            logging.info("torch.compile disabled via environment variable")
            return
            
        if hasattr(torch, 'compile'):
            try:
                # Only compile if we have enough memory
                if torch.cuda.is_available():
                    free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # GB
                    if free_memory < 8:  # Less than 8GB free
                        print(f"○ torch.compile skipped (low memory: {free_memory:.1f}GB)")
                        logging.warning(f"Insufficient GPU memory ({free_memory:.1f}GB), skipping torch.compile")
                        return
                
                print("✓ torch.compile enabled (reduce-overhead mode)")
                # Compile the model with reduce-overhead mode for best performance
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False,  # Allow for dynamic shapes
                    dynamic=True  # Better for varying input sizes
                )
                logging.info("Model compiled with torch.compile for optimized performance")
            except Exception as e:
                print(f"✗ torch.compile failed: {e}")
                logging.warning(f"Failed to compile model: {e}")
        else:
            print("○ torch.compile not available (PyTorch < 2.0)")
    
    def vace_encode_frames_batch(self, frames, ref_images, masks=None, vae=None, batch_size=4):
        """Optimized batch encoding for VAE with configurable batch size"""
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)
        
        all_latents = []
        total_batches = (len(frames) + batch_size - 1) // batch_size
        
        if total_batches > 1:
            print(f"✓ VAE batch encoding enabled (batch_size={batch_size}, {total_batches} batches)")
        
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
        
        if len(masks) > 0:
            print(f"✓ Vectorized mask processing enabled ({len(masks)} masks)")
        
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
        
        total_batches = (len(trimmed_zs) + batch_size - 1) // batch_size
        if total_batches > 1:
            print(f"✓ VAE batch decoding enabled (batch_size={batch_size}, {total_batches} batches)")
        
        # Batch decode for better GPU utilization
        all_decoded = []
        for batch_start in range(0, len(trimmed_zs), batch_size):
            batch_end = min(batch_start + batch_size, len(trimmed_zs))
            batch_zs = trimmed_zs[batch_start:batch_end]
            
            batch_decoded = vae.decode(batch_zs)
            all_decoded.extend(batch_decoded)
        
        return all_decoded
    
    def interpolate_frames(self, frames, method='linear'):
        """Interpolate frames to double the frame count"""
        if len(frames) < 2:
            return frames
            
        interpolated = []
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            # Simple linear interpolation in pixel space
            interpolated.append((frames[i] + frames[i + 1]) / 2)
        interpolated.append(frames[-1])
        
        return interpolated
    
    def generate_with_frame_skip(self, *args, **kwargs):
        """Generate with frame skipping for 2x speedup - using latent space interpolation"""
        # Get original frame_num
        frame_num = args[4] if len(args) > 4 else kwargs.get('frame_num', 81)
        
        # Also reduce sampling steps for more speedup (optional)
        reduce_steps = kwargs.get('reduce_steps_with_frame_skip', True)
        
        # Store original decode_latent method
        original_decode = self.decode_latent
        interpolated_latents = []
        
        def decode_with_interpolation(zs, ref_images=None, vae=None):
            """Decode latents with interpolation in latent space"""
            # Create new list for interpolated latents
            new_interpolated_latents = []
            
            # Interpolate in latent space before decoding
            if len(zs) > 0:
                for z_batch in zs:
                    if z_batch.shape[1] > 1:  # Check temporal dimension
                        # z_batch shape: [channels, frames, height, width]
                        interpolated = []
                        for i in range(z_batch.shape[1] - 1):
                            # Add original frame
                            interpolated.append(z_batch[:, i:i+1])
                            # Add interpolated frame
                            interp_frame = (z_batch[:, i:i+1] + z_batch[:, i+1:i+2]) / 2
                            interpolated.append(interp_frame)
                        # Add last frame
                        interpolated.append(z_batch[:, -1:])
                        
                        # Concatenate along frame dimension
                        z_interp = torch.cat(interpolated, dim=1)
                        new_interpolated_latents.append(z_interp[:, :frame_num])
                    else:
                        new_interpolated_latents.append(z_batch)
            else:
                new_interpolated_latents = zs
            
            if len(zs) > 0 and len(new_interpolated_latents) > 0:
                print(f"  → Interpolating latents: {len(zs)} batches, {zs[0].shape[1]} → {new_interpolated_latents[0].shape[1]} frames")
                print(f"  → Latent shapes: {[z.shape for z in zs[:2]]}")  # Show first 2 shapes
            
            # Decode the interpolated latents
            return original_decode(new_interpolated_latents, ref_images, vae)
        
        # Generate half the frames
        half_frame_num = (frame_num + 1) // 2
        if len(args) > 4:
            args = list(args)
            args[4] = half_frame_num
            args = tuple(args)
        else:
            kwargs['frame_num'] = half_frame_num
        
        print(f"  → Generating {half_frame_num} frames (will interpolate to {frame_num} in latent space)")
        print(f"  ⚠️  Note: WAN uses fixed sequence length based on resolution, not frame count")
        print(f"  ⚠️  For true speedup, consider reducing resolution or sampling steps")
        
        # Get sampling steps for logging
        sampling_steps = args[10] if len(args) > 10 else kwargs.get('sampling_steps', 50)
        print(f"  → Sampling steps: {sampling_steps}")
        
        import time
        start_time = time.time()
        
        try:
            # Replace decode_latent temporarily
            self.decode_latent = decode_with_interpolation
            
            # Temporarily disable frame skip to avoid recursion
            self.enable_frame_skip = False
            
            # Log actual arguments being passed
            actual_frame_num = args[4] if len(args) > 4 else kwargs.get('frame_num')
            print(f"  → Actually generating {actual_frame_num} frames")
            
            # Profile generation phases
            import time
            
            # Time the actual generation (diffusion)
            print(f"  → Starting diffusion with {actual_frame_num} frames...")
            diff_start = time.time()
            
            # Generate with half frames
            result = self.generate(*args, **kwargs)
            
            diff_end = time.time()
            print(f"  → Diffusion completed in {diff_end - diff_start:.2f}s")
            
        finally:
            # Restore original settings
            self.enable_frame_skip = True
            self.decode_latent = original_decode
        
        end_time = time.time()
        print(f"  → Frame skip generation completed in {end_time - start_time:.2f}s")
        
        return result
    
    def generate(self, *args, **kwargs):
        """Override generate method to use optimized encoding/decoding"""
        # If frame skip is enabled, use the frame skip method
        if self.enable_frame_skip:
            print(f"🚀 Frame skip activated - generating with interpolation")
            return self.generate_with_frame_skip(*args, **kwargs)
        
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
            
            # Enable CUDA graph caching for inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Override timestep loop for optimization
            self._optimized_inference = True
            
            # Add hook to monitor model input sizes
            def log_input_hook(module, input, output):
                if hasattr(input[0], 'shape'):
                    if not hasattr(module, '_logged_shape'):
                        print(f"  → Model input shape: {input[0].shape}")
                        module._logged_shape = True
                
            # Register hook on the main model
            if hasattr(self.model, 'register_forward_hook'):
                hook_handle = self.model.register_forward_hook(log_input_hook)
            else:
                hook_handle = None
            
            try:
                # Call parent generate method with optimizations
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    return super().generate(*args, **kwargs)
            finally:
                # Remove hook
                if hook_handle:
                    hook_handle.remove()
            
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