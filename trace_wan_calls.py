#!/usr/bin/env python
"""
WANモデルで実際に呼び出される関数を追跡
"""
import torch
import sys
import os

def trace_all_wan_functions():
    """WANモジュールの全関数にトレースを追加"""
    
    try:
        import wan.modules.attention as wan_attn
        
        print("WANアテンションモジュールの全関数をトレース中...")
        print("="*50)
        
        # モジュール内の全ての関数を取得
        all_functions = []
        for attr_name in dir(wan_attn):
            attr = getattr(wan_attn, attr_name)
            if callable(attr) and not attr_name.startswith('_'):
                all_functions.append((attr_name, attr))
        
        print(f"発見した関数: {[name for name, _ in all_functions]}")
        
        # 各関数にトレースを追加
        original_functions = {}
        for func_name, func in all_functions:
            try:
                # 元の関数を保存
                original_functions[func_name] = func
                
                # トレース付きラッパーを作成
                def make_tracer(name, original_func):
                    def traced_func(*args, **kwargs):
                        print(f"🔍 WAN function called: {name}")
                        if args:
                            print(f"  → args count: {len(args)}")
                            for i, arg in enumerate(args[:3]):  # 最初の3つの引数のみ表示
                                if hasattr(arg, 'shape'):
                                    print(f"    arg[{i}] shape: {arg.shape}")
                                else:
                                    print(f"    arg[{i}] type: {type(arg)}")
                        if kwargs:
                            print(f"  → kwargs: {list(kwargs.keys())}")
                        
                        result = original_func(*args, **kwargs)
                        
                        if hasattr(result, 'shape'):
                            print(f"  → result shape: {result.shape}")
                        
                        return result
                    return traced_func
                
                # 関数を置き換え
                setattr(wan_attn, func_name, make_tracer(func_name, func))
                print(f"✓ Traced: {func_name}")
                
            except Exception as e:
                print(f"✗ Failed to trace {func_name}: {e}")
        
        return original_functions
        
    except ImportError as e:
        print(f"Cannot import WAN attention module: {e}")
        return {}
    except Exception as e:
        print(f"Error setting up traces: {e}")
        return {}

def trace_torch_attention():
    """PyTorchの標準アテンション関数もトレース"""
    
    print("\nPyTorchアテンション関数をトレース中...")
    print("="*40)
    
    try:
        # scaled_dot_product_attention をトレース
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        
        def traced_sdpa(*args, **kwargs):
            print("🔍 PyTorch scaled_dot_product_attention called")
            if args:
                print(f"  → q shape: {args[0].shape}")
                print(f"  → k shape: {args[1].shape}")
                print(f"  → v shape: {args[2].shape}")
            
            result = original_sdpa(*args, **kwargs)
            print(f"  → output shape: {result.shape}")
            return result
        
        torch.nn.functional.scaled_dot_product_attention = traced_sdpa
        print("✓ Traced: torch.nn.functional.scaled_dot_product_attention")
        
    except Exception as e:
        print(f"Failed to trace PyTorch attention: {e}")

def trace_flash_attn():
    """Flash Attentionの関数もトレース"""
    
    print("\nFlash Attention関数をトレース中...")
    print("="*30)
    
    try:
        from flash_attn import flash_attn_func
        import flash_attn
        
        # flash_attn_func をトレース
        original_flash_func = flash_attn_func
        
        def traced_flash_func(*args, **kwargs):
            print("🚀 flash_attn_func called!")
            if args:
                print(f"  → q shape: {args[0].shape}")
                print(f"  → k shape: {args[1].shape}")
                print(f"  → v shape: {args[2].shape}")
            
            result = original_flash_func(*args, **kwargs)
            print(f"  → output shape: {result.shape}")
            return result
        
        # モンキーパッチ
        flash_attn.flash_attn_func = traced_flash_func
        
        print("✓ Traced: flash_attn.flash_attn_func")
        
    except ImportError:
        print("✗ Flash Attention not available")
    except Exception as e:
        print(f"Failed to trace Flash Attention: {e}")

def setup_comprehensive_tracing():
    """包括的なトレースを設定"""
    
    print("包括的トレース設定中...")
    print("="*60)
    
    # WANの関数をトレース
    wan_originals = trace_all_wan_functions()
    
    # PyTorchの関数をトレース
    trace_torch_attention()
    
    # Flash Attentionをトレース
    trace_flash_attn()
    
    print(f"\n✓ トレース設定完了 ({len(wan_originals)} WAN functions traced)")
    print("これで実際に呼び出される関数がすべて表示されます")
    
    return wan_originals

def inspect_wan_model_structure():
    """WANモデルの構造を詳しく調査"""
    
    print("\nWANモデル構造調査...")
    print("="*30)
    
    try:
        # WANのAttentionBlockクラスを直接確認
        import wan.modules.model as wan_model
        
        print("wan.modules.model で利用可能なクラス:")
        model_classes = [attr for attr in dir(wan_model) if not attr.startswith('_') and isinstance(getattr(wan_model, attr), type)]
        for cls_name in model_classes:
            cls = getattr(wan_model, cls_name)
            print(f"  - {cls_name}: {cls}")
            
            # Attentionに関連するメソッドを探す
            attention_methods = [method for method in dir(cls) if 'attn' in method.lower() and not method.startswith('_')]
            if attention_methods:
                print(f"    Attention methods: {attention_methods}")
        
        # attention.py の詳細を調査
        import wan.modules.attention as wan_attn
        print(f"\nwan.modules.attention で利用可能な要素:")
        for attr in dir(wan_attn):
            if not attr.startswith('_'):
                obj = getattr(wan_attn, attr)
                print(f"  - {attr}: {type(obj)}")
        
    except Exception as e:
        print(f"モデル構造調査エラー: {e}")

if __name__ == "__main__":
    # 構造調査
    inspect_wan_model_structure()
    
    # 包括的トレース設定
    setup_comprehensive_tracing()
    
    print("\n" + "="*60)
    print("トレース設定完了 - 実際のモデル実行でアテンション呼び出しを確認できます")
    print("="*60)