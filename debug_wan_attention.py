#!/usr/bin/env python
"""
WANモデルのアテンション構造を調査するデバッグスクリプト
"""
import torch
import logging
from vace.models.wan import WanVace
from vace.models.wan.configs import WAN_CONFIGS

def debug_wan_attention_structure():
    """WANモデルのアテンション構造を詳しく調査"""
    
    print("WANモデルのアテンション構造調査")
    print("="*50)
    
    # 簡単な設定でモデルをロード
    config = WAN_CONFIGS["vace-1.3B"]
    
    try:
        # モデル構造を調査（実際のチェックポイントなしで）
        from vace.models.wan.modules.model import VaceWanModel
        
        model = VaceWanModel.from_config(config.model)
        
        print(f"Model type: {type(model)}")
        print(f"Has blocks: {hasattr(model, 'blocks')}")
        print(f"Has vace_blocks: {hasattr(model, 'vace_blocks')}")
        
        if hasattr(model, 'blocks'):
            print(f"Number of main blocks: {len(model.blocks)}")
            if len(model.blocks) > 0:
                block = model.blocks[0]
                print(f"Block type: {type(block)}")
                print(f"Block attributes: {[attr for attr in dir(block) if not attr.startswith('_')]}")
                
                # アテンション層を探す
                attention_attrs = [attr for attr in dir(block) if 'attn' in attr.lower()]
                print(f"Attention attributes: {attention_attrs}")
                
                if hasattr(block, 'self_attn'):
                    attn = block.self_attn
                    print(f"Self attention type: {type(attn)}")
                    print(f"Self attention attributes: {[attr for attr in dir(attn) if not attr.startswith('_')]}")
        
        if hasattr(model, 'vace_blocks'):
            print(f"Number of VACE blocks: {len(model.vace_blocks)}")
            if len(model.vace_blocks) > 0:
                vblock = model.vace_blocks[0]
                print(f"VACE block type: {type(vblock)}")
                print(f"VACE block attributes: {[attr for attr in dir(vblock) if not attr.startswith('_')]}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # 代替: WANライブラリから直接調査
        try:
            import wan
            from wan.modules.attention import WanAttentionBlock
            
            print("\nWAN attention block analysis:")
            print(f"WanAttentionBlock: {WanAttentionBlock}")
            
            # アテンションブロックの構造を調査
            dummy_block = WanAttentionBlock(
                cross_attn_type="text",
                dim=1024,
                ffn_dim=4096,
                num_heads=16
            )
            
            print(f"Attention block attributes: {[attr for attr in dir(dummy_block) if not attr.startswith('_')]}")
            
            # Flash Attentionの使用可能性を確認
            if hasattr(dummy_block, 'self_attn'):
                print(f"Has self_attn: True")
                print(f"Self attention type: {type(dummy_block.self_attn)}")
            
        except Exception as e2:
            print(f"Error with WAN analysis: {e2}")

def check_flash_attention_compatibility():
    """Flash Attentionの互換性を確認"""
    print("\nFlash Attention 互換性チェック")
    print("="*50)
    
    try:
        from flash_attn import flash_attn_func
        print("✓ flash_attn_func available")
        
        # 簡単なテスト
        batch, heads, seq_len, dim = 1, 8, 128, 64
        q = torch.randn(batch, seq_len, heads, dim).cuda().half()
        k = torch.randn(batch, seq_len, heads, dim).cuda().half()
        v = torch.randn(batch, seq_len, heads, dim).cuda().half()
        
        out = flash_attn_func(q, k, v)
        print(f"✓ Flash Attention test successful: {out.shape}")
        
    except ImportError:
        print("✗ Flash Attention not available")
    except Exception as e:
        print(f"✗ Flash Attention test failed: {e}")

def investigate_wan_attention_usage():
    """WANモデルでの実際のアテンション使用を調査"""
    print("\nWAN アテンション使用調査")
    print("="*50)
    
    try:
        import wan.modules.attention as wan_attn
        
        # WANのアテンション実装を調査
        print(f"WAN attention module: {wan_attn}")
        
        # アテンション関数を探す
        attention_functions = [attr for attr in dir(wan_attn) if 'attention' in attr.lower()]
        print(f"Attention functions: {attention_functions}")
        
        if hasattr(wan_attn, 'flash_attention'):
            print("✓ WAN has flash_attention function")
        else:
            print("✗ WAN does not have flash_attention function")
            
    except Exception as e:
        print(f"Error investigating WAN attention: {e}")

if __name__ == "__main__":
    debug_wan_attention_structure()
    check_flash_attention_compatibility()
    investigate_wan_attention_usage()