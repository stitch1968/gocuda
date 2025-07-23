#!/bin/bash

# Build script for gocuda library
# Usage: ./build.sh [cuda|nocuda] [demo]

set -e

MODE=${1:-nocuda}
RUN_DEMO=${2:-false}

echo "Building gocuda library..."

case $MODE in
    cuda)
        echo "Building with CUDA support..."
        echo "Requirements:"
        echo "  - NVIDIA CUDA Toolkit installed"
        echo "  - CGO enabled"
        echo ""
        
        # Check if CUDA is available
        if command -v nvcc &> /dev/null; then
            echo "✓ CUDA compiler found: $(nvcc --version | grep "release")"
        else
            echo "✗ CUDA compiler (nvcc) not found"
            echo "Please install NVIDIA CUDA Toolkit"
            exit 1
        fi
        
        # Build with CUDA support
        echo "Building with CUDA tags..."
        CGO_ENABLED=1 go build -tags cuda -v ./...
        
        # Run tests with CUDA
        echo "Running tests with CUDA support..."
        CGO_ENABLED=1 go test -tags cuda -v ./...
        ;;
        
    nocuda)
        echo "Building without CUDA support (simulation mode only)..."
        echo "This mode works on any system without CUDA installation."
        echo ""
        
        # Build without CUDA support
        echo "Building with simulation-only mode..."
        go build -v ./...
        
        # Run tests in simulation mode
        echo "Running tests in simulation mode..."
        go test -v ./...
        ;;
        
    *)
        echo "Usage: $0 [cuda|nocuda] [demo]"
        echo ""
        echo "Options:"
        echo "  cuda    - Build with real CUDA support (requires CUDA toolkit)"
        echo "  nocuda  - Build with CPU simulation only (default)"
        echo "  demo    - Run comprehensive demo after successful build"
        exit 1
        ;;
esac

echo ""
echo "Build completed successfully!"
echo ""

# Run demo if requested
if [[ $RUN_DEMO == "demo" ]] || [[ $2 == "demo" ]]; then
    echo "Running comprehensive demo..."
    echo ""
    cd demos/missing_features && go run main.go
    echo ""
fi

# Show build info
echo "=== Build Information ==="
echo "Mode: $MODE"
echo "Go version: $(go version)"
if [[ $MODE == "cuda" ]]; then
    echo "CUDA version: $(nvcc --version | grep "release" || echo "Unknown")"
    echo "CGO: Enabled"
else
    echo "CUDA: Not required"
    echo "CGO: Disabled"
fi
echo "========================="
