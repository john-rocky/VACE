"""
Demonstration of WAN's fixed sequence length behavior
This shows why processing fewer frames doesn't speed up computation
"""

import math

class WANSequenceLengthDemo:
    def __init__(self):
        self.vae_stride = (4, 8, 8)  # Temporal and spatial downsampling
        self.patch_size = (1, 2, 2)  # Patch dimensions
        self.sp_size = 1  # Sequence parallel size (usually 1)
        
    def calculate_seq_len(self, video_shape, resolution_name):
        """Calculate the sequence length for a given video shape"""
        channels, frames, height, width = video_shape
        
        # After VAE encoding
        vae_frames = frames // self.vae_stride[0] + 1
        vae_height = height // self.vae_stride[1]
        vae_width = width // self.vae_stride[2]
        
        print(f"\n{resolution_name} Video Analysis:")
        print(f"Input: {frames} frames @ {height}x{width}")
        print(f"After VAE: {vae_frames} frames @ {vae_height}x{vae_width}")
        
        # Calculate sequence length (THIS IS THE KEY FORMULA)
        seq_len = math.ceil((vae_height * vae_width) / 
                           (self.patch_size[1] * self.patch_size[2]) * 
                           vae_frames / self.sp_size) * self.sp_size
        
        # Fixed sequence lengths used by the model
        if height == 480 and width == 832:
            fixed_seq_len = 32760
        elif height == 720 and width == 1280:
            fixed_seq_len = 75600
        else:
            fixed_seq_len = seq_len
            
        print(f"Calculated seq_len: {seq_len:,}")
        print(f"Fixed seq_len used: {fixed_seq_len:,}")
        
        # Show padding behavior
        if seq_len > fixed_seq_len:
            print(f"⚠️  CLIPPED: {seq_len - fixed_seq_len:,} tokens removed")
        elif seq_len < fixed_seq_len:
            print(f"📦 PADDED: {fixed_seq_len - seq_len:,} zero tokens added")
        
        return fixed_seq_len

    def demonstrate_fixed_computation(self):
        """Show that different frame counts result in same computation"""
        print("="*60)
        print("WAN SEQUENCE LENGTH DEMONSTRATION")
        print("="*60)
        
        # Test different frame counts at 480x832
        frame_counts = [10, 30, 81, 100]
        resolution = (480, 832)
        
        print(f"\nTesting resolution: {resolution[0]}x{resolution[1]}")
        print("-"*50)
        
        seq_lens = []
        for frames in frame_counts:
            video_shape = (3, frames, resolution[0], resolution[1])
            seq_len = self.calculate_seq_len(video_shape, f"{frames} frames")
            seq_lens.append(seq_len)
        
        print("\n" + "="*50)
        print("CONCLUSION:")
        print("="*50)
        print(f"All videos process exactly {seq_lens[0]:,} tokens")
        print("Computation time is IDENTICAL regardless of frame count!")
        
        # Show the same for 720x1280
        print("\n" + "="*60)
        print(f"\nTesting resolution: 720x1280")
        print("-"*50)
        
        for frames in [10, 81]:
            video_shape = (3, frames, 720, 1280)
            self.calculate_seq_len(video_shape, f"{frames} frames")

    def show_why_no_speedup(self):
        """Explain why there's no speedup with fewer frames"""
        print("\n" + "="*60)
        print("WHY FEWER FRAMES DON'T SPEED UP PROCESSING:")
        print("="*60)
        
        print("""
1. FIXED SEQUENCE LENGTH:
   - The model always processes a fixed number of tokens
   - This number depends ONLY on spatial resolution, not frame count
   
2. PADDING MECHANISM:
   - If actual tokens < fixed length: zeros are padded
   - If actual tokens > fixed length: tokens are clipped
   
3. ATTENTION COMPUTATION:
   - Self-attention has O(n²) complexity
   - Processes ALL tokens including padding
   - Zero tokens still require computation
   
4. MODEL ARCHITECTURE:
   - Designed for fixed-length sequences
   - Cannot dynamically adjust computation
   - Would require retraining to change this behavior
        """)

if __name__ == "__main__":
    demo = WANSequenceLengthDemo()
    demo.demonstrate_fixed_computation()
    demo.show_why_no_speedup()
    
    print("\n" + "="*60)
    print("PRACTICAL IMPLICATIONS:")
    print("="*60)
    print("""
To reduce computation time, you can only:
1. Use lower spatial resolution (480x832 instead of 720x1280)
   - 32,760 tokens vs 75,600 tokens = 2.3x speedup
   
2. Reduce sampling steps (e.g., 30 instead of 50)
   - Linear speedup with step count
   
3. Use better hardware or model parallelism
   - But this doesn't reduce total computation

Frame reduction strategies that DON'T work:
❌ Processing fewer frames
❌ Frame skipping
❌ Temporal downsampling
    """)