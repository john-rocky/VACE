# 品質を保ちながら高速化する設定ガイド
# Quality-preserving optimization configurations

# 方法1: スケジューラーの最適化
OPTIMIZED_SCHEDULER_CONFIG = {
    "base": "wan",
    "solver": "dpm++",  # unipcより高速収束
    "num_inference_steps": 30,
    "guidance_scale": 5.5,  # 少し上げて品質補償
}

# 方法2: 段階的生成（プレビュー→本番）
PROGRESSIVE_GENERATION = {
    # ステップ1: 高速プレビュー
    "preview": {
        "num_inference_steps": 15,
        "resolution": "360p",  # 低解像度で確認
    },
    # ステップ2: 本番生成
    "final": {
        "num_inference_steps": 40,
        "resolution": "720p",  # フル解像度
    }
}

# 方法3: 適応的ステップ数（コンテンツによる調整）
ADAPTIVE_STEPS = {
    "simple_scene": {  # 単純なシーン（静的背景など）
        "num_inference_steps": 25,
        "guidance_scale": 4.0,
    },
    "complex_scene": {  # 複雑なシーン（動きが多い）
        "num_inference_steps": 40,
        "guidance_scale": 5.0,
    },
    "fine_detail": {  # 細部重視（顔のクローズアップなど）
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
    }
}

# 方法4: ハイブリッド最適化
HYBRID_OPTIMIZATION = {
    "num_inference_steps": 30,
    "guidance_scale": 5.5,  # 高めに設定
    "stg_scale": 1.2,  # 時空間ガイダンスを強化
    "context_scale": 1.1,  # コンディショニング強化
    "precision": "mixed_precision",
    "enable_xformers": True,
}