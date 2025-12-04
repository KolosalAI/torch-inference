#!/usr/bin/env bash

# Build script for torch-inference-rs
# Supports multiple build configurations

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Rust
    if ! command -v rustc &> /dev/null; then
        print_error "Rust is not installed. Please install from https://rustup.rs/"
        exit 1
    fi
    
    RUST_VERSION=$(rustc --version | awk '{print $2}')
    print_success "Rust version: $RUST_VERSION"
    
    # Check cargo
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo is not installed"
        exit 1
    fi
    
    # Check CUDA (optional)
    if command -v nvidia-smi &> /dev/null; then
        print_success "CUDA detected"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n 1
        CUDA_AVAILABLE=true
    else
        print_warning "CUDA not detected - will build CPU-only version"
        CUDA_AVAILABLE=false
    fi
}

# Build configurations
build_cpu() {
    print_info "Building CPU-only version..."
    cargo build --release
    print_success "CPU build completed"
}

build_cuda() {
    if [ "$CUDA_AVAILABLE" = false ]; then
        print_error "CUDA not available, cannot build CUDA version"
        print_info "Falling back to CPU build..."
        build_cpu
        return
    fi
    
    print_info "Building with CUDA support..."
    cargo build --release --features cuda
    print_success "CUDA build completed"
}

build_onnx() {
    print_info "Building with ONNX Runtime support..."
    cargo build --release --features onnx
    print_success "ONNX build completed"
}

build_all() {
    if [ "$CUDA_AVAILABLE" = true ]; then
        print_info "Building with all backends (CUDA + ONNX + Torch)..."
        cargo build --release --features all-backends
    else
        print_info "Building with all CPU backends (ONNX + Torch)..."
        cargo build --release --features onnx,torch
    fi
    print_success "Full build completed"
}

# Development build
build_dev() {
    print_info "Building development version (with debug symbols)..."
    cargo build
    print_success "Development build completed"
}

# Clean build
clean_build() {
    print_info "Cleaning previous builds..."
    cargo clean
    print_success "Clean completed"
}

# Run tests
run_tests() {
    print_info "Running tests..."
    
    if [ "$CUDA_AVAILABLE" = true ]; then
        cargo test --features cuda -- --nocapture
    else
        cargo test -- --nocapture
    fi
    
    print_success "Tests completed"
}

# Format code
format_code() {
    print_info "Formatting code..."
    cargo fmt
    print_success "Code formatted"
}

# Run linter
run_linter() {
    print_info "Running clippy..."
    cargo clippy -- -D warnings
    print_success "Linting completed"
}

# Check code without building
check_code() {
    print_info "Checking code..."
    cargo check
    print_success "Check completed"
}

# Install binary
install_binary() {
    print_info "Installing binary to ~/.cargo/bin/..."
    cargo install --path .
    print_success "Binary installed: torch-inference-server"
}

# Create distribution package
create_package() {
    print_info "Creating distribution package..."
    
    PACKAGE_NAME="torch-inference-rs-$(uname -s)-$(uname -m)"
    PACKAGE_DIR="target/package/$PACKAGE_NAME"
    
    mkdir -p "$PACKAGE_DIR"
    
    # Copy binary
    cp target/release/torch-inference-server "$PACKAGE_DIR/"
    
    # Copy documentation
    cp README_RUST_FEATURES.md "$PACKAGE_DIR/README.md"
    cp MIGRATION_GUIDE.md "$PACKAGE_DIR/"
    
    # Create archive
    cd target/package
    tar -czf "$PACKAGE_NAME.tar.gz" "$PACKAGE_NAME"
    cd ../..
    
    print_success "Package created: target/package/$PACKAGE_NAME.tar.gz"
}

# Show binary info
show_info() {
    if [ -f "target/release/torch-inference-server" ]; then
        print_info "Binary information:"
        ls -lh target/release/torch-inference-server
        
        print_info "Binary features:"
        ./target/release/torch-inference-server --version || true
    else
        print_warning "Binary not found. Run build first."
    fi
}

# Print usage
usage() {
    cat << EOF
Usage: $0 [COMMAND]

Commands:
    cpu         Build CPU-only version (default)
    cuda        Build with CUDA support
    onnx        Build with ONNX Runtime support
    all         Build with all backends
    dev         Build development version (with debug)
    
    clean       Clean build artifacts
    test        Run tests
    check       Check code without building
    fmt         Format code
    lint        Run clippy linter
    
    install     Install binary to system
    package     Create distribution package
    info        Show binary information
    
    help        Show this help message

Examples:
    $0 cpu          # Build CPU version
    $0 cuda         # Build with CUDA
    $0 all          # Build with all features
    $0 test         # Run tests
    $0 install      # Install to system

Environment Variables:
    RUST_LOG        Set log level (trace, debug, info, warn, error)
    CUDA_VISIBLE_DEVICES    Select GPU devices
    
EOF
}

# Main script
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║   PyTorch Inference Framework - Build Script    ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""
    
    # Check dependencies first
    check_dependencies
    echo ""
    
    # Parse command
    COMMAND=${1:-cpu}
    
    case $COMMAND in
        cpu)
            build_cpu
            show_info
            ;;
        cuda)
            build_cuda
            show_info
            ;;
        onnx)
            build_onnx
            show_info
            ;;
        all)
            build_all
            show_info
            ;;
        dev)
            build_dev
            ;;
        clean)
            clean_build
            ;;
        test)
            run_tests
            ;;
        check)
            check_code
            ;;
        fmt|format)
            format_code
            ;;
        lint)
            run_linter
            ;;
        install)
            install_binary
            ;;
        package)
            create_package
            ;;
        info)
            show_info
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo ""
            usage
            exit 1
            ;;
    esac
    
    echo ""
    print_success "Done!"
}

# Run main
main "$@"
