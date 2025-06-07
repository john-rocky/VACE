#!/usr/bin/env python
"""
実際の高速化を測定するスクリプト
"""
import time
import torch
import os

def measure_attention_performance():
    """アテンション計算の性能を測定"""
    print("アテンション性能テスト")
    print("="*40)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device('cuda')
    
    # テストパラメータ（WANモデル相当）
    batch_size = 1
    seq_len = 32760  # WANの典型的なシーケンス長
    num_heads = 16
    head_dim = 64
    
    # テストデータ生成
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.half)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.half)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.half)
    
    print(f"Test shape: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, dim={head_dim}")
    
    # ウォームアップ
    for _ in range(3):
        _ = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
    torch.cuda.synchronize()
    
    # 標準アテンション測定
    print("\n1. 標準アテンション測定...")
    start_time = time.time()
    for _ in range(5):
        with torch.no_grad():
            out_standard = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
    torch.cuda.synchronize()
    standard_time = (time.time() - start_time) / 5
    print(f"  平均時間: {standard_time:.4f}s")
    
    # Flash Attention測定
    try:
        from flash_attn import flash_attn_func
        print("\n2. Flash Attention測定...")
        
        # ウォームアップ
        for _ in range(3):
            _ = flash_attn_func(q, k, v)
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(5):
            with torch.no_grad():
                out_flash = flash_attn_func(q, k, v)
        torch.cuda.synchronize()
        flash_time = (time.time() - start_time) / 5
        print(f"  平均時間: {flash_time:.4f}s")
        
        speedup = standard_time / flash_time
        print(f"\n高速化率: {speedup:.2f}x")
        
        # 結果の比較
        diff = torch.max(torch.abs(out_standard.transpose(1, 2) - out_flash))
        print(f"最大差分: {diff:.6f}")
        
    except ImportError:
        print("\n2. Flash Attention not available")
    except Exception as e:
        print(f"\n2. Flash Attention error: {e}")

def measure_vae_batch_performance():
    """VAEバッチ処理の性能測定"""
    print("\n" + "="*40)
    print("VAEバッチ処理性能テスト")
    print("="*40)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device('cuda')
    
    # テストパラメータ
    batch_sizes = [1, 2, 4, 8]
    channels = 16
    frames = 21  # 81フレームの1/4
    height, width = 60, 104  # 480x832の1/8
    
    for batch_size in batch_sizes:
        print(f"\nバッチサイズ {batch_size}:")
        
        # ダミーVAEデコード関数
        def dummy_vae_decode(latents):
            # 実際のVAE処理をシミュレート
            return torch.nn.functional.interpolate(
                latents, scale_factor=8, mode='trilinear'
            )
        
        # 個別処理
        latents_list = [torch.randn(1, channels, frames, height, width, device=device) 
                       for _ in range(batch_size)]
        
        torch.cuda.synchronize()
        start_time = time.time()
        results_individual = []
        for latent in latents_list:
            with torch.no_grad():
                result = dummy_vae_decode(latent)
                results_individual.append(result)
        torch.cuda.synchronize()
        individual_time = time.time() - start_time
        
        # バッチ処理
        latents_batch = torch.cat(latents_list, dim=0)
        
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            result_batch = dummy_vae_decode(latents_batch)
        torch.cuda.synchronize()
        batch_time = time.time() - start_time
        
        speedup = individual_time / batch_time
        print(f"  個別処理: {individual_time:.4f}s")
        print(f"  バッチ処理: {batch_time:.4f}s")
        print(f"  高速化率: {speedup:.2f}x")

def check_optimization_status():
    """現在の最適化状況を確認"""
    print("\n" + "="*40)
    print("最適化状況確認")
    print("="*40)
    
    # Flash Attention
    try:
        from flash_attn import flash_attn_func
        print("✓ Flash Attention 2 available")
    except ImportError:
        print("✗ Flash Attention 2 not available")
    
    # xFormers
    try:
        import xformers
        print("✓ xFormers available")
    except ImportError:
        print("✗ xFormers not available")
    
    # torch.compile
    if hasattr(torch, 'compile'):
        if os.environ.get('TORCH_COMPILE_DISABLE', '0') == '1':
            print("○ torch.compile available but disabled")
        else:
            print("✓ torch.compile available and enabled")
    else:
        print("✗ torch.compile not available")
    
    # CUDA情報
    if torch.cuda.is_available():
        print(f"✓ CUDA: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("✗ CUDA not available")

if __name__ == "__main__":
    check_optimization_status()
    measure_attention_performance()
    measure_vae_batch_performance()