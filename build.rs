// Build script for torch-inference
// Handles automatic PyTorch library detection and setup

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // Check if torch feature is enabled
    if !cfg!(feature = "torch") {
        println!("cargo:warning=PyTorch feature not enabled. Skipping libtorch setup.");
        return;
    }

    println!("cargo:warning=Checking for PyTorch (libtorch) installation...");

    // Try to find libtorch
    if let Some(libtorch_path) = find_libtorch() {
        println!("cargo:warning=Found libtorch at: {:?}", libtorch_path);
        setup_libtorch_paths(&libtorch_path);
    } else {
        println!("cargo:warning=");
        println!("cargo:warning=╔════════════════════════════════════════════════════════╗");
        println!("cargo:warning=║  PyTorch (libtorch) Not Found                          ║");
        println!("cargo:warning=╚════════════════════════════════════════════════════════╝");
        println!("cargo:warning=");
        println!("cargo:warning=PyTorch will be auto-downloaded on first server startup.");
        println!("cargo:warning=Or you can manually install it:");
        println!("cargo:warning=");
        println!("cargo:warning=1. Download from: https://pytorch.org/get-started/locally/");
        println!("cargo:warning=2. Extract to ./libtorch/");
        println!("cargo:warning=3. Set LIBTORCH environment variable");
        println!("cargo:warning=");
        
        // Set a flag to indicate libtorch is not available at build time
        println!("cargo:rustc-cfg=libtorch_unavailable");
    }
}

fn find_libtorch() -> Option<PathBuf> {
    // Method 1: Check LIBTORCH environment variable
    if let Ok(libtorch) = env::var("LIBTORCH") {
        let path = PathBuf::from(libtorch);
        if validate_libtorch(&path) {
            return Some(path);
        }
    }

    // Method 2: Check LIBTORCH_DIR environment variable
    if let Ok(libtorch_dir) = env::var("LIBTORCH_DIR") {
        let path = PathBuf::from(libtorch_dir);
        if validate_libtorch(&path) {
            return Some(path);
        }
    }

    // Method 3: Check default installation directory
    let default_path = PathBuf::from("./libtorch");
    if validate_libtorch(&default_path) {
        return Some(default_path);
    }

    // Method 4: Check common installation paths
    let common_paths = vec![
        "/usr/local/libtorch",
        "/opt/libtorch",
        "C:\\libtorch",
        "C:\\Program Files\\libtorch",
    ];

    for path_str in common_paths {
        let path = PathBuf::from(path_str);
        if validate_libtorch(&path) {
            return Some(path);
        }
    }

    None
}

fn validate_libtorch(path: &PathBuf) -> bool {
    if !path.exists() {
        return false;
    }

    let lib_dir = path.join("lib");
    let include_dir = path.join("include");

    if !lib_dir.exists() || !include_dir.exists() {
        return false;
    }

    // Check for torch library
    #[cfg(target_os = "windows")]
    let torch_lib = lib_dir.join("torch.lib");
    
    #[cfg(target_os = "linux")]
    let torch_lib = lib_dir.join("libtorch.so");
    
    #[cfg(target_os = "macos")]
    let torch_lib = lib_dir.join("libtorch.dylib");

    torch_lib.exists()
}

fn setup_libtorch_paths(libtorch_path: &PathBuf) {
    let lib_path = libtorch_path.join("lib");
    
    // Set linker paths
    println!("cargo:rustc-link-search=native={}", lib_path.display());
    
    // Link libraries
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    // Check for CUDA libraries
    #[cfg(target_os = "windows")]
    {
        if lib_path.join("torch_cuda.dll").exists() {
            println!("cargo:rustc-link-lib=dylib=torch_cuda");
            println!("cargo:warning=CUDA support detected in libtorch");
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if lib_path.join("libtorch_cuda.so").exists() {
            println!("cargo:rustc-link-lib=dylib=torch_cuda");
            println!("cargo:warning=CUDA support detected in libtorch");
        }
    }

    // Set runtime library path
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-env=LD_LIBRARY_PATH={}", lib_path.display());
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-env=DYLD_LIBRARY_PATH={}", lib_path.display());
    }

    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-env=PATH={}", lib_path.display());
    }
}
