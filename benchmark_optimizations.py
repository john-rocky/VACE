#!/usr/bin/env python
"""
最適化のベンチマークスクリプト
実際の動画生成での性能差を測定
"""
import time
import torch
import os
import sys
from contextlib import contextmanager

# プロジェクトルートを追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@contextmanager
def timer(description):
    """実行時間を測定するコンテキストマネージャー"""
    start = time.time()
    yield
    end = time.time()
    print(f"{description}: {end - start:.2f}秒")

def benchmark_wan_models():
    """標準モデルと最適化モデルの性能比較"""
    print("WAN モデル性能ベンチマーク")
    print("="*50)
    
    try:
        from vace.models.wan import WanVace, OptimizedWanVace, MemoryEfficientConfig
        from vace.models.wan.configs import WAN_CONFIGS
        
        config = WAN_CONFIGS["vace-1.3B"]
        checkpoint_dir = "models/Wan2.1-VACE-1.3B/"  # 実際のパスに調整
        
        if not os.path.exists(checkpoint_dir):
            print(f"チェックポイントが見つかりません: {checkpoint_dir}")
            print("ダミーデータでテストを続行...")
            return benchmark_dummy_operations()
        
        print("1. 標準WANモデルをロード中...")
        with timer("標準モデルロード時間"):
            standard_model = WanVace(
                config=config,
                checkpoint_dir=checkpoint_dir,
                device_id=0
            )
        
        print("\n2. 最適化WANモデルをロード中...")
        os.environ['TORCH_COMPILE_DISABLE'] = '1'  # コンパイルは無効化
        
        with timer("最適化モデルロード時間"):
            optimized_model = OptimizedWanVace(
                config=config,
                checkpoint_dir=checkpoint_dir,
                device_id=0,
                enable_flash_attn=True,
                enable_torch_compile=False
            )
        
        # ダミー入力データの準備
        device = torch.device('cuda:0')
        
        # 小さなテストデータ（実際の推論をシミュレート）
        test_frames = [torch.randn(3, 21, 240, 416, device=device)]  # 小さなフレーム
        test_masks = [torch.ones(1, 21, 240, 416, device=device)]
        test_ref_images = [None]
        
        print("\n3. 標準モデルでの推論テスト...")
        with timer("標準モデル推論時間"):
            try:
                # エンコード処理のみテスト（フル推論は時間がかかりすぎる）
                _ = standard_model.vace_encode_frames(test_frames, test_ref_images, masks=test_masks)
            except Exception as e:
                print(f"標準モデルテストエラー: {e}")
        
        print("\n4. 最適化モデルでの推論テスト...")
        with timer("最適化モデル推論時間"):
            try:
                _ = optimized_model.vace_encode_frames_batch(test_frames, test_ref_images, masks=test_masks)
            except Exception as e:
                print(f"最適化モデルテストエラー: {e}")
        
    except ImportError as e:
        print(f"モデルインポートエラー: {e}")
        print("ダミーテストを実行...")
        return benchmark_dummy_operations()
    except Exception as e:
        print(f"ベンチマークエラー: {e}")
        return benchmark_dummy_operations()

def benchmark_dummy_operations():
    """実際のモデルが利用できない場合のダミー操作ベンチマーク"""
    print("\nダミー操作ベンチマーク")
    print("="*30)
    
    if not torch.cuda.is_available():
        print("CUDA利用不可 - CPUでテスト")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    # Flash Attentionのテスト
    print("\n1. Flash Attention性能テスト")
    batch, heads, seq_len, dim = 1, 16, 4096, 64
    
    q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=torch.half)
    k = torch.randn(batch, seq_len, heads, dim, device=device, dtype=torch.half)
    v = torch.randn(batch, seq_len, heads, dim, device=device, dtype=torch.half)
    
    # 標準アテンション
    print("標準アテンション:")
    with timer("標準アテンション時間"):
        for _ in range(10):
            with torch.no_grad():
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                )
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Flash Attention（利用可能な場合）
    try:
        from flash_attn import flash_attn_func
        print("Flash Attention:")
        with timer("Flash Attention時間"):
            for _ in range(10):
                with torch.no_grad():
                    _ = flash_attn_func(q, k, v)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    except ImportError:
        print("Flash Attention利用不可")
    
    # VAEバッチ処理のテスト
    print("\n2. VAEバッチ処理性能テスト")
    channels, frames, height, width = 16, 21, 60, 104
    
    # 個別処理
    latents_list = [torch.randn(1, channels, frames, height, width, device=device) for _ in range(4)]
    
    print("個別VAE処理:")
    with timer("個別処理時間"):
        for latent in latents_list:
            with torch.no_grad():
                # ダミーVAE処理（アップサンプリング）
                _ = torch.nn.functional.interpolate(latent, scale_factor=2, mode='trilinear')
    
    # バッチ処理
    latents_batch = torch.cat(latents_list, dim=0)
    
    print("バッチVAE処理:")
    with timer("バッチ処理時間"):
        with torch.no_grad():
            _ = torch.nn.functional.interpolate(latents_batch, scale_factor=2, mode='trilinear')

def check_environment():
    """実行環境の確認"""
    print("実行環境確認")
    print("="*20)
    
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA: 利用不可")
    
    # 最適化ライブラリの確認
    try:
        import flash_attn
        print(f"Flash Attention: ✓ (version: {flash_attn.__version__})")
    except ImportError:
        print("Flash Attention: ✗")
    
    try:
        import xformers
        print(f"xFormers: ✓ (version: {xformers.__version__})")
    except ImportError:
        print("xFormers: ✗")
    
    print(f"torch.compile: {'✓' if hasattr(torch, 'compile') else '✗'}")

if __name__ == "__main__":
    check_environment()
    print("\n")
    benchmark_wan_models()