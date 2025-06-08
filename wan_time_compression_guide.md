# WAN時間圧縮ガイド - 同じシーンを少ないフレームで実現

## 問題
- 81フレームで車が蛇行しながら奥から手前に来るシーン
- 同じ内容を41フレームで実現したい

## 解決策

### 1. プロンプトエンジニアリング

#### 元のプロンプト（81フレーム用）
```
"A car slowly winds its way from the distant background, following a serpentine path as it approaches the camera"
```

#### 調整後のプロンプト（41フレーム用）
```
"A car rapidly winds its way from the distant background, quickly following a serpentine path as it approaches the camera"
```

### 2. 時間圧縮のキーワード

**動きの速度を上げる修飾語：**
- slowly → quickly / rapidly
- walks → runs / rushes
- moves → speeds / races
- gradual → swift / rapid
- leisurely → brisk / fast

### 3. 実装方法

```python
# 時間圧縮モードを使用
from vace.models.wan import OptimizedWanVace

model = OptimizedWanVace(...)

# 自動プロンプト調整付きで生成
result = model.generate_compressed_scene(
    prompt="A car moves from background to foreground",
    frame_num=41,  # 81から41フレームに圧縮
    auto_adjust_prompt=True  # プロンプトを自動調整
)
```

### 4. フレーム数とシーンの関係

| フレーム数 | 適した動き | プロンプト例 |
|-----------|-----------|------------|
| 81 | ゆっくり、詳細な動き | "slowly", "gradually", "leisurely" |
| 41 | 標準〜速い動き | "quickly", "rapidly", "briskly" |
| 21 | 非常に速い動き | "very fast", "rushing", "speeding" |

### 5. 実際の使用例

```bash
# 41フレームで高速な動きを生成
python -m vace.vace_wan_inference \
    --prompt "A car rapidly winds through a serpentine path approaching camera" \
    --frame_num 41 \
    --size 480p \
    --sample_steps 25
```

## 重要な注意点

1. **物理的な制約**: WANは解像度ベースの固定シーケンス長を使用するため、計算時間は変わりません
2. **品質の考慮**: 動きが速すぎると不自然になる可能性があります
3. **シーンの複雑さ**: 複雑な動きは多くのフレームが必要です

## まとめ

同じシーンを少ないフレームで実現するには：
1. プロンプトで動きの速度を調整
2. 適切なキーワードを使用
3. シーンの内容に応じてフレーム数を選択

これにより、81フレームの内容を41フレームで表現できますが、生成時間の短縮にはなりません。真の高速化には解像度やステップ数の削減が必要です。