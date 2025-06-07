#!/bin/bash
# Performance optimization libraries installation script

echo "Installing performance optimization libraries for VACE..."

# Check CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -c2-5)
echo "Detected CUDA version: ${CUDA_VERSION:-'Not found'}"

# Install Flash Attention 2 (compile from source for best compatibility)
echo "Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

# Install xFormers based on CUDA version
if [[ "$CUDA_VERSION" == "11.8" ]]; then
    echo "Installing xFormers for CUDA 11.8..."
    pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12.1" ]]; then
    echo "Installing xFormers for CUDA 12.1..."
    pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing xFormers (auto-detect CUDA version)..."
    pip install xformers==0.0.23
fi

# Install additional performance libraries
echo "Installing additional performance libraries..."
pip install triton>=2.1.0
pip install ninja
pip install psutil

# Verify installations
echo "Verifying installations..."
python -c "
try:
    from flash_attn import flash_attn_func
    print('✓ Flash Attention 2 installed successfully')
except ImportError as e:
    print('✗ Flash Attention 2 installation failed:', e)

try:
    import xformers
    print('✓ xFormers installed successfully')
except ImportError as e:
    print('✗ xFormers installation failed:', e)

try:
    import triton
    print('✓ Triton installed successfully')  
except ImportError as e:
    print('✗ Triton installation failed:', e)
"

echo "Installation complete!"
echo ""
echo "To use optimizations:"
echo "  python -m vace.vace_wan_inference --use_optimized --optimization_mode speed"
echo ""
echo "To disable torch.compile if needed:"
echo "  export TORCH_COMPILE_DISABLE=1"