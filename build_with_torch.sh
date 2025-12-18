#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Building torch-inference with PyTorch Support            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if libtorch exists
if [ -d "./libtorch" ]; then
    echo "✓ LibTorch directory found"
    export LIBTORCH=$(pwd)/libtorch
    export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
    export DYLD_LIBRARY_PATH=${LIBTORCH}/lib:$DYLD_LIBRARY_PATH
else
    echo "⚠ LibTorch not found - build script will attempt auto-download"
fi

echo ""
echo "Building with torch feature..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cargo build --release --features torch 2>&1 | tee build_torch.log

BUILD_STATUS=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $BUILD_STATUS -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "Binary: ./target/release/torch-inference-server"
    echo ""
    
    # Check if binary was created
    if [ -f "./target/release/torch-inference-server" ]; then
        SIZE=$(ls -lh ./target/release/torch-inference-server | awk '{print $5}')
        echo "Binary size: $SIZE"
        echo ""
        
        echo "To run the server:"
        echo "  ./target/release/torch-inference-server"
        echo ""
        echo "Or test with:"
        echo "  ./test_final_report.sh"
    fi
else
    echo "❌ Build failed!"
    echo ""
    echo "Check build_torch.log for details"
    echo ""
    echo "Common issues:"
    echo "  1. LibTorch not found - download with: ./download_libtorch.sh"
    echo "  2. CUDA mismatch - ensure CUDA version matches libtorch"
    echo "  3. Missing dependencies - check build.rs requirements"
    echo ""
    
    # Show last 20 lines of errors
    echo "Last errors:"
    tail -20 build_torch.log | grep -i "error"
fi

exit $BUILD_STATUS
