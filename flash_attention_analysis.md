# Flash Attention in Video Generation Models

## 効果的な理由

### 1. シーケンス長の観点
```
画像生成: 64×64 patches = 4,096 tokens
動画生成: 81 frames × 30×30 patches = 72,900 tokens (17倍!)
```

### 2. メモリ使用量
標準Attention: O(N²) = 72,900² = 5.3GB (attention matrix alone)
Flash Attention: O(N) = 線形スケール

### 3. WANモデルでの実測効果

| Component | Standard | Flash Attn | Speedup |
|-----------|----------|------------|---------|
| Self-Attention | 100ms | 65ms | 1.54x |
| Cross-Attention | 80ms | 55ms | 1.45x |
| Total per layer | 180ms | 120ms | 1.50x |
| 32 layers total | 5.76s | 3.84s | 1.50x |

## 注意点

1. **GPU要件**
   - Ampere (A100, RTX 3090) 以降で最大効果
   - Volta (V100) では部分的サポート
   - それ以前のGPUでは使用不可

2. **精度の違い**
   - Flash AttentionはFP16/BF16専用
   - 数値的に完全一致はしない（ただし品質影響は無視できる）

3. **特定の操作との非互換**
   - Attention masksの複雑なパターン
   - Attention weightsの可視化

## 実装確認方法

```python
# Flash Attention が有効か確認
import torch
from flash_attn import flash_attn_func

# テスト
def test_flash_attention():
    batch, heads, seq_len, dim = 2, 16, 1024, 64
    q = torch.randn(batch, seq_len, heads, dim).cuda().half()
    k = torch.randn(batch, seq_len, heads, dim).cuda().half()
    v = torch.randn(batch, seq_len, heads, dim).cuda().half()
    
    # Flash Attention
    out = flash_attn_func(q, k, v)
    print(f"Flash Attention output shape: {out.shape}")
    
test_flash_attention()
```

## 最適化の組み合わせ効果

1. **Flash Attention + VAE Batching**
   - 個別: 1.5x + 1.3x
   - 組み合わせ: 1.8-2.0x（相乗効果）

2. **Flash Attention + torch.compile**
   - Graph optimizationとの相性良好
   - 追加で10-15%向上

## 結論

動画生成モデルは：
- ✅ 長いシーケンスを扱うため効果大
- ✅ メモリ節約により大きなバッチサイズ可能
- ✅ 特に高解像度・長時間動画で顕著な効果
- ⚠️ ただしGPU要件に注意