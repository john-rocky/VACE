#!/usr/bin/env python3
"""Test script for VACEWrapper."""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))

from vace_wrapper import VACEWrapper

def test_wrapper():
    """Test VACEWrapper functionality."""
    print("Testing VACEWrapper...")
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, this may not work properly")
        return False
    
    try:
        wrapper = VACEWrapper(model_name="vace-1.3B")
        print("✓ VACEWrapper initialized successfully")
        
        assert hasattr(wrapper, 'create_simple_video'), "create_simple_video method missing"
        assert hasattr(wrapper, 'create_referenced_video'), "create_referenced_video method missing"
        print("✓ Required methods exist")
        
        print("VACEWrapper tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_wrapper()
    sys.exit(0 if success else 1)
