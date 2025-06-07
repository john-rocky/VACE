import torch
from vace.models.ltx.ltx_vace import LTXVace

class OptimizedLTXVace(LTXVace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Flash Attention 2を有効化
        if hasattr(self.transformer, 'enable_flash_attention_2'):
            self.transformer.enable_flash_attention_2()
        
    def generate(self, *args, **kwargs):
        # torch.compileで最適化（PyTorch 2.0+）
        if hasattr(torch, 'compile'):
            self.transformer = torch.compile(
                self.transformer, 
                mode="reduce-overhead",
                fullgraph=True
            )
        return super().generate(*args, **kwargs)