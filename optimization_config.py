# 最適化設定ファイル
# Video generation optimization configurations

# 速度優先設定
SPEED_OPTIMIZED = {
    "precision": "mixed_precision",
    "num_inference_steps": 25,
    "offload_to_cpu": False,
    "decode_timestep": 0.0,
    "decode_noise_scale": 0.0,
    "enable_xformers": True,
}

# メモリ優先設定
MEMORY_OPTIMIZED = {
    "precision": "mixed_precision", 
    "num_inference_steps": 30,
    "offload_to_cpu": True,
    "enable_tiling": True,
    "enable_slicing": True,
    "sequential_cpu_offload": True,
}

# バランス設定（推奨）
BALANCED = {
    "precision": "mixed_precision",
    "num_inference_steps": 30,
    "offload_to_cpu": True,
    "enable_xformers": True,
    "chunk_size": 4,  # 一度に処理するフレーム数
}