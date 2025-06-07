# VAEバッチ処理の最適化
import torch

def optimized_vae_decode(vae, latents, batch_size=4):
    """複数フレームを同時にデコード"""
    B, C, T, H, W = latents.shape
    decoded_frames = []
    
    # バッチ処理でデコード
    for i in range(0, T, batch_size):
        batch_latents = latents[:, :, i:i+batch_size].permute(0, 2, 1, 3, 4)
        batch_latents = batch_latents.reshape(-1, C, H, W)
        
        with torch.no_grad():
            batch_decoded = vae.decode(batch_latents / vae.config.scaling_factor).sample
        
        decoded_frames.append(batch_decoded)
    
    return torch.cat(decoded_frames, dim=0)