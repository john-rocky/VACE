#!/usr/bin/env python
"""
WANモデルの実際のアテンション呼び出しを調査
"""
import torch
import sys
import os

def hook_wan_attention():
    """WANのアテンション関数にフックを追加して実際の使用を監視"""
    
    try:
        import wan.modules.attention as wan_attn
        
        # 元の関数を保存
        original_functions = {}
        
        # flash_attention関数をフック
        if hasattr(wan_attn, 'flash_attention'):
            original_functions['flash_attention'] = wan_attn.flash_attention
            
            def hooked_flash_attention(*args, **kwargs):
                print(f"🔍 flash_attention called with args: {len(args)}, kwargs: {list(kwargs.keys())}")
                if len(args) >= 3:
                    q, k, v = args[0], args[1], args[2]
                    print(f"  → q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
                return original_functions['flash_attention'](*args, **kwargs)
            
            wan_attn.flash_attention = hooked_flash_attention
            print("✓ Hooked wan.modules.attention.flash_attention")
        
        # その他のアテンション関数も探す
        attention_functions = [attr for attr in dir(wan_attn) if 'attention' in attr.lower() and callable(getattr(wan_attn, attr))]
        print(f"Available attention functions: {attention_functions}")
        
        for func_name in attention_functions:
            if func_name != 'flash_attention':
                func = getattr(wan_attn, func_name)
                original_functions[func_name] = func
                
                def make_hooked_func(name, original_func):
                    def hooked_func(*args, **kwargs):
                        print(f"🔍 {name} called")
                        return original_func(*args, **kwargs)
                    return hooked_func
                
                setattr(wan_attn, func_name, make_hooked_func(func_name, func))
                print(f"✓ Hooked {func_name}")
        
        return original_functions
        
    except Exception as e:
        print(f"Error hooking WAN attention: {e}")
        return {}

def test_wan_model_attention():
    """WANモデルでの実際のアテンション使用をテスト"""
    
    print("WANモデルアテンション使用テスト")
    print("="*40)
    
    # フックを設定
    hooks = hook_wan_attention()
    
    try:
        from vace.models.wan import OptimizedWanVace
        from vace.models.wan.configs import WAN_CONFIGS
        
        config = WAN_CONFIGS["vace-1.3B"]
        
        # 実際のチェックポイントが必要なので、ダミーテストを実行
        print("\nダミーアテンション操作をテスト...")
        
        # WANアテンション関数を直接呼び出してテスト
        import wan.modules.attention as wan_attn
        
        if hasattr(wan_attn, 'flash_attention'):
            print("flash_attention関数を直接呼び出し...")
            
            # ダミーデータでテスト
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # WANの典型的な形状
            batch, heads, seq_len, dim = 1, 16, 1024, 64
            
            q = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.half)
            k = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.half)
            v = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.half)
            k_lens = torch.tensor([seq_len], device=device)
            
            try:
                result = wan_attn.flash_attention(q, k, v, k_lens)
                print(f"✓ flash_attention successful, output shape: {result.shape}")
            except Exception as e:
                print(f"✗ flash_attention failed: {e}")
        
    except Exception as e:
        print(f"Model test error: {e}")
    
    # フックを元に戻す
    try:
        import wan.modules.attention as wan_attn
        for func_name, original_func in hooks.items():
            setattr(wan_attn, func_name, original_func)
        print(f"\n✓ Restored {len(hooks)} hooked functions")
    except:
        pass

def analyze_wan_source():
    """WANのソースコードを分析してアテンション使用箇所を特定"""
    
    print("\nWANソースコード分析")
    print("="*30)
    
    try:
        import wan.modules.attention as wan_attn
        import inspect
        
        # flash_attention関数のソースを確認
        if hasattr(wan_attn, 'flash_attention'):
            try:
                source = inspect.getsource(wan_attn.flash_attention)
                print("flash_attention function source:")
                print("-" * 20)
                print(source[:500] + "..." if len(source) > 500 else source)
            except:
                print("Cannot get flash_attention source")
        
        # WanAttentionBlockの分析
        try:
            from wan.modules.attention import WanAttentionBlock
            attention_block_source = inspect.getsource(WanAttentionBlock)
            
            # flash_attentionの呼び出し箇所を探す
            if 'flash_attention' in attention_block_source:
                print("\n✓ WanAttentionBlock uses flash_attention")
                
                # 呼び出し箇所の周辺を表示
                lines = attention_block_source.split('\n')
                for i, line in enumerate(lines):
                    if 'flash_attention' in line:
                        start = max(0, i-2)
                        end = min(len(lines), i+3)
                        print(f"Lines {start+1}-{end}:")
                        for j in range(start, end):
                            prefix = ">>> " if j == i else "    "
                            print(f"{prefix}{lines[j]}")
                        print()
            else:
                print("\n✗ WanAttentionBlock does not use flash_attention")
                
        except Exception as e:
            print(f"Cannot analyze WanAttentionBlock: {e}")
    
    except Exception as e:
        print(f"Source analysis error: {e}")

def check_actual_flash_usage():
    """実際にFlash Attentionが使用されているかを確認"""
    
    print("\nFlash Attention使用確認")
    print("="*30)
    
    # フラグを設定してFlash Attentionの呼び出しを監視
    flash_call_count = 0
    standard_call_count = 0
    
    try:
        from flash_attn import flash_attn_func
        
        # flash_attn_funcにフックを追加
        original_flash_attn_func = flash_attn_func
        
        def hooked_flash_attn_func(*args, **kwargs):
            nonlocal flash_call_count
            flash_call_count += 1
            print(f"🚀 flash_attn_func called #{flash_call_count}")
            return original_flash_attn_func(*args, **kwargs)
        
        # モンキーパッチ
        import flash_attn
        flash_attn.flash_attn_func = hooked_flash_attn_func
        
        print("✓ Hooked flash_attn_func for monitoring")
        
    except ImportError:
        print("✗ flash_attn not available for hooking")
    
    # 標準アテンションもフック
    try:
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        
        def hooked_sdpa(*args, **kwargs):
            nonlocal standard_call_count
            standard_call_count += 1
            if standard_call_count <= 5:  # 最初の5回だけ表示
                print(f"📊 scaled_dot_product_attention called #{standard_call_count}")
            return original_sdpa(*args, **kwargs)
        
        torch.nn.functional.scaled_dot_product_attention = hooked_sdpa
        print("✓ Hooked scaled_dot_product_attention for monitoring")
        
    except:
        print("✗ Cannot hook scaled_dot_product_attention")
    
    return flash_call_count, standard_call_count

if __name__ == "__main__":
    test_wan_model_attention()
    analyze_wan_source()
    check_actual_flash_usage()
    
    print("\n" + "="*50)
    print("調査完了 - 実際のモデル実行でこのスクリプトの出力を確認してください")
    print("="*50)