# WAN Frame Processing Deep Analysis

## Key Finding: Sequence Length is Fixed Regardless of Frame Count

After analyzing the WAN model's code, I've identified the core issue: **the sequence length (`seq_len`) is calculated based on spatial dimensions only, not temporal dimensions**. This means processing fewer frames provides NO computational speedup.

## 1. Frame to Token/Patch Conversion

### Patch Embedding Process
```python
# From VaceWanModel.__init__
self.patch_embedding = nn.Conv3d(
    self.vae_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
)
```

- **Patch size**: `(1, 2, 2)` for temporal and spatial dimensions
- **Input**: Video tensor of shape `[C, F, H, W]` where:
  - C = channels (16 after VAE encoding)
  - F = frames
  - H = height
  - W = width

### Shape Transformations
1. **VAE Encoding**: `[3, F, H, W]` → `[16, F, H/8, W/8]` (spatial downsampling by 8x)
2. **Patch Embedding**: `[16, F, H/8, W/8]` → `[2048, F/1, H/16, W/16]`
3. **Flattening**: `[2048, F, H/16, W/16]` → `[seq_len, 2048]`

## 2. The Critical Sequence Length Calculation

```python
# From wan_vace.py, lines 355-357
seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                    (self.patch_size[1] * self.patch_size[2]) *
                    target_shape[1] / self.sp_size) * self.sp_size
```

Breaking this down:
- `target_shape[2]` = H/8 (height after VAE)
- `target_shape[3]` = W/8 (width after VAE)
- `self.patch_size[1]` = 2 (spatial patch height)
- `self.patch_size[2]` = 2 (spatial patch width)
- `target_shape[1]` = F (number of frames)
- `self.sp_size` = 1 (sequence parallel size, usually 1)

**Simplified**: `seq_len = (H/8 * W/8) / (2 * 2) * F = (H * W * F) / 256`

## 3. Why Sequence Length is Fixed

The critical observation is in the forward pass:

```python
# From VaceWanModel.forward, lines 192-196
assert seq_lens.max() <= seq_len
x = torch.cat([
    torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
              dim=1) for u in x
])
```

**The model ALWAYS pads the sequence to `seq_len` with zeros!** This means:
- If actual tokens = 1000 and seq_len = 2000, it adds 1000 zero tokens
- The attention mechanism processes ALL 2000 tokens (including zeros)
- No computational savings from having fewer actual frames

## 4. Fixed Sequence Lengths by Resolution

Based on the code:
- **480×832**: `seq_len = 32,760`
- **720×1280**: `seq_len = 75,600`

These are calculated to accommodate the maximum possible frames while maintaining the spatial resolution.

## 5. Why Frame Skipping Won't Help

The model processes frames in a **3D spatiotemporal manner**:
1. All frames are embedded together
2. Attention is computed across the entire padded sequence
3. The transformer blocks process the full `seq_len` tokens regardless of actual content

## 6. Theoretical Ways to Reduce Computation (All Require Model Modification)

### Option 1: Dynamic Sequence Length
Modify the model to:
- Calculate `seq_len` based on actual frames
- Remove zero padding
- Adjust attention masks dynamically

**Problem**: Requires retraining or fine-tuning the model.

### Option 2: Temporal Downsampling Before VAE
- Skip frames before VAE encoding
- Interpolate results after generation

**Problem**: Reduces temporal quality significantly.

### Option 3: Adaptive Computation
- Implement early exit mechanisms
- Skip attention computation for zero-padded regions

**Problem**: Requires significant architectural changes.

## Conclusion

**Without modifying the model architecture, there is NO way to achieve speedup by processing fewer frames.** The WAN model's design fundamentally processes a fixed sequence length based on spatial resolution, padding with zeros as needed. The attention mechanism and all transformer computations operate on this full sequence length regardless of actual frame content.

The only practical speedup options are:
1. Use lower spatial resolution (480×832 vs 720×1280)
2. Reduce sampling steps
3. Use different hardware (better GPU)
4. Implement model parallelism (already partially done with USP)