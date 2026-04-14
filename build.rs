// Build script for torch-inference
// Handles automatic PyTorch library detection, download, and setup
// Integrates functionality from build.sh for comprehensive build automation

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH");

    print_header();

    // Detect system configuration
    let system_info = detect_system();
    print_system_info(&system_info);

    // Check if torch feature is enabled
    if !cfg!(feature = "torch") {
        println!("cargo:warning=[INFO] PyTorch feature not enabled. Skipping libtorch setup.");
        return;
    }

    println!("cargo:warning=[INFO] Checking for PyTorch (libtorch) installation...");

    // Try to find existing libtorch
    if let Some(libtorch_path) = find_libtorch() {
        println!("cargo:warning=[OK] Found libtorch at: {:?}", libtorch_path);
        setup_libtorch_paths(&libtorch_path, &system_info);
    } else {
        println!("cargo:warning=[WARN] LibTorch not found - attempting auto-download...");

        // Try to auto-download libtorch
        match auto_download_libtorch(&system_info) {
            Ok(libtorch_path) => {
                println!(
                    "cargo:warning=[OK] Successfully downloaded libtorch to: {:?}",
                    libtorch_path
                );
                setup_libtorch_paths(&libtorch_path, &system_info);
            }
            Err(e) => {
                println!(
                    "cargo:warning=[WARN] Failed to auto-download libtorch: {}",
                    e
                );
                print_manual_instructions();
                println!("cargo:rustc-cfg=libtorch_unavailable");
            }
        }
    }

    // Check for additional features
    check_cuda_support(&system_info);
    check_metal_support(&system_info);
    check_onnx_support();
}

fn print_header() {
    println!("cargo:warning=");
    println!("cargo:warning=╔══════════════════════════════════════════════════════════╗");
    println!("cargo:warning=║    PyTorch Inference Framework - Build Configuration    ║");
    println!("cargo:warning=╚══════════════════════════════════════════════════════════╝");
    println!("cargo:warning=");
}

struct SystemInfo {
    os: String,
    arch: String,
    has_cuda: bool,
    has_metal: bool,
    cuda_version: Option<String>,
}

fn detect_system() -> SystemInfo {
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "unknown".to_string());
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "unknown".to_string());

    // Check for CUDA
    let has_cuda = check_cuda_available();
    let cuda_version = if has_cuda { get_cuda_version() } else { None };

    // Check for Metal (Apple Silicon)
    let has_metal = os == "macos" && (arch == "aarch64" || arch == "arm64");

    SystemInfo {
        os,
        arch,
        has_cuda,
        has_metal,
        cuda_version,
    }
}

fn print_system_info(info: &SystemInfo) {
    println!("cargo:warning=[INFO] System Configuration:");
    println!("cargo:warning=   OS: {}", info.os);
    println!("cargo:warning=   Architecture: {}", info.arch);
    println!(
        "cargo:warning=   CUDA: {}",
        if info.has_cuda {
            format!(
                "Available ({})",
                info.cuda_version.as_ref().unwrap_or(&"unknown".to_string())
            )
        } else {
            "Not Available".to_string()
        }
    );
    println!(
        "cargo:warning=   Metal: {}",
        if info.has_metal {
            "Available (Apple Silicon)"
        } else {
            "Not Available"
        }
    );
    println!("cargo:warning=");
}

fn check_cuda_available() -> bool {
    Command::new("nvidia-smi")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn get_cuda_version() -> Option<String> {
    Command::new("nvidia-smi")
        .arg("--query-gpu=driver_version")
        .arg("--format=csv,noheader")
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .ok()
                .map(|s| s.trim().to_string())
        })
}

fn auto_download_libtorch(system_info: &SystemInfo) -> Result<PathBuf, String> {
    println!("cargo:warning=[INFO] Auto-downloading LibTorch...");

    // Determine download URL based on system
    let download_url = get_libtorch_download_url(system_info)?;

    // Determine cache directory
    let cache_dir = get_cache_dir()?;

    // For local directory, we extract directly to it
    let is_local = cache_dir == Path::new(".");
    let libtorch_dir = if is_local {
        PathBuf::from("./libtorch")
    } else {
        cache_dir.join("libtorch")
    };

    // Check if already downloaded
    if validate_libtorch(&libtorch_dir) {
        println!(
            "cargo:warning=[OK] LibTorch already cached at: {:?}",
            libtorch_dir
        );
        return Ok(libtorch_dir);
    }

    println!("cargo:warning=[INFO] Downloading from: {}", download_url);
    println!(
        "cargo:warning=[INFO] Target directory: {:?}",
        if is_local {
            PathBuf::from("./")
        } else {
            cache_dir.clone()
        }
    );

    // Create cache directory
    fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("Failed to create cache directory: {}", e))?;

    // Download using curl or wget
    let archive_path = cache_dir.join("libtorch.zip");

    let download_success = if Command::new("curl").arg("--version").output().is_ok() {
        println!("cargo:warning=[INFO] Using curl for download...");
        Command::new("curl")
            .arg("-L")
            .arg("-o")
            .arg(&archive_path)
            .arg(&download_url)
            .arg("--progress-bar")
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    } else if Command::new("wget").arg("--version").output().is_ok() {
        println!("cargo:warning=[INFO] Using wget for download...");
        Command::new("wget")
            .arg("-O")
            .arg(&archive_path)
            .arg(&download_url)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    } else {
        return Err("Neither curl nor wget found. Please install one of them.".to_string());
    };

    if !download_success {
        return Err("Download failed".to_string());
    }

    println!("cargo:warning=[INFO] Extracting LibTorch...");

    // Extract using appropriate tool based on OS
    let extract_success = if cfg!(target_os = "windows") {
        // On Windows, try PowerShell Expand-Archive
        Command::new("powershell")
            .arg("-Command")
            .arg(format!(
                "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                archive_path.display(),
                cache_dir.display()
            ))
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    } else {
        // On Unix-like systems, use unzip
        Command::new("unzip")
            .arg("-q")
            .arg(&archive_path)
            .arg("-d")
            .arg(&cache_dir)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };

    if !extract_success {
        return Err("Extraction failed. On Windows, ensure PowerShell is available. On Unix, install unzip.".to_string());
    }

    // Clean up archive
    let _ = fs::remove_file(&archive_path);

    // Verify extraction
    if validate_libtorch(&libtorch_dir) {
        println!("cargo:warning=[OK] LibTorch installed successfully");
        Ok(libtorch_dir)
    } else {
        Err("LibTorch extraction validation failed".to_string())
    }
}

fn get_libtorch_download_url(system_info: &SystemInfo) -> Result<String, String> {
    let version = "2.1.0";

    let url = match (
        system_info.os.as_str(),
        system_info.arch.as_str(),
        system_info.has_cuda,
    ) {
        // macOS (Intel and Apple Silicon use universal binary)
        ("macos", _, _) => {
            format!(
                "https://download.pytorch.org/libtorch/cpu/libtorch-macos-{}.zip",
                version
            )
        }

        // Linux with CUDA
        ("linux", "x86_64", true) => {
            let cuda_ver = system_info
                .cuda_version
                .as_ref()
                .and_then(|v| {
                    // Extract major.minor from version like "535.104.05"
                    v.split('.')
                        .take(2)
                        .collect::<Vec<_>>()
                        .join(".")
                        .parse::<f32>()
                        .ok()
                        .map(|ver| {
                            if ver >= 12.0 {
                                "cu121"
                            } else if ver >= 11.8 {
                                "cu118"
                            } else {
                                "cu117"
                            }
                        })
                })
                .unwrap_or("cu118");

            format!("https://download.pytorch.org/libtorch/{}/libtorch-cxx11-abi-shared-with-deps-{}%2B{}.zip", 
                    cuda_ver, version, cuda_ver)
        }

        // Linux CPU
        ("linux", "x86_64", false) => {
            format!("https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-{}%2Bcpu.zip", version)
        }

        // Windows with CUDA
        ("windows", "x86_64", true) => {
            let cuda_ver = system_info
                .cuda_version
                .as_ref()
                .and_then(|v| {
                    v.split('.')
                        .take(2)
                        .collect::<Vec<_>>()
                        .join(".")
                        .parse::<f32>()
                        .ok()
                        .map(|ver| {
                            if ver >= 12.0 {
                                "cu121"
                            } else if ver >= 11.8 {
                                "cu118"
                            } else {
                                "cu117"
                            }
                        })
                })
                .unwrap_or("cu118");

            format!("https://download.pytorch.org/libtorch/{}/libtorch-win-shared-with-deps-{}%2B{}.zip", 
                    cuda_ver, version, cuda_ver)
        }

        // Windows CPU
        ("windows", "x86_64", false) => {
            format!("https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-{}%2Bcpu.zip", version)
        }

        _ => {
            return Err(format!(
                "Unsupported platform: {} {}",
                system_info.os, system_info.arch
            ));
        }
    };

    Ok(url)
}

fn get_cache_dir() -> Result<PathBuf, String> {
    // Check for explicit LIBTORCH env var
    if let Ok(dir) = env::var("LIBTORCH") {
        return Ok(PathBuf::from(dir)
            .parent()
            .unwrap_or(&PathBuf::from("."))
            .to_path_buf());
    }

    // Priority 1: Use local ./libtorch directory in project root
    let local_dir = PathBuf::from("./libtorch");
    if local_dir.exists() && validate_libtorch(&local_dir) {
        return Ok(PathBuf::from("."));
    }

    // Priority 2: Try to create local directory
    // Only use system cache if we can't use local directory
    let use_local = env::var("LIBTORCH_LOCAL").is_ok() || env::var("LIBTORCH_USE_LOCAL").is_ok();

    if use_local || !local_dir.exists() {
        // Prefer local directory for easier project portability
        return Ok(PathBuf::from("."));
    }

    // Priority 3: Use standard cache directory based on OS
    let cache_dir = if cfg!(target_os = "windows") {
        // Windows: Use %LOCALAPPDATA%\libtorch or %USERPROFILE%\.cache\libtorch
        if let Ok(local_app_data) = env::var("LOCALAPPDATA") {
            PathBuf::from(local_app_data).join("libtorch")
        } else if let Ok(userprofile) = env::var("USERPROFILE") {
            PathBuf::from(userprofile).join(".cache").join("libtorch")
        } else {
            PathBuf::from(".")
        }
    } else if cfg!(target_os = "macos") {
        // macOS: Use ~/Library/Caches/libtorch or ~/.cache/libtorch
        if let Ok(home) = env::var("HOME") {
            let library_cache = PathBuf::from(&home)
                .join("Library")
                .join("Caches")
                .join("libtorch");
            if library_cache.parent().map(|p| p.exists()).unwrap_or(false) {
                library_cache
            } else {
                PathBuf::from(home).join(".cache").join("libtorch")
            }
        } else {
            PathBuf::from(".")
        }
    } else {
        // Linux: Use ~/.cache/libtorch
        if let Ok(home) = env::var("HOME") {
            PathBuf::from(home).join(".cache").join("libtorch")
        } else {
            PathBuf::from(".")
        }
    };

    Ok(cache_dir)
}

fn find_libtorch() -> Option<PathBuf> {
    // Method 1: Check LIBTORCH environment variable
    if let Ok(libtorch) = env::var("LIBTORCH") {
        let path = PathBuf::from(&libtorch);
        if validate_libtorch(&path) {
            println!("cargo:warning=[OK] Found via LIBTORCH env: {}", libtorch);
            return Some(path);
        }
    }

    // Method 2: Check LIBTORCH_DIR environment variable
    if let Ok(libtorch_dir) = env::var("LIBTORCH_DIR") {
        let path = PathBuf::from(&libtorch_dir);
        if validate_libtorch(&path) {
            println!(
                "cargo:warning=[OK] Found via LIBTORCH_DIR env: {}",
                libtorch_dir
            );
            return Some(path);
        }
    }

    // Method 3: Check cache directory
    if let Ok(cache_dir) = get_cache_dir() {
        let cached_path = cache_dir.join("libtorch");
        if validate_libtorch(&cached_path) {
            println!("cargo:warning=[OK] Found in cache: {:?}", cached_path);
            return Some(cached_path);
        }
    }

    // Method 4: Check default installation directory
    let default_path = PathBuf::from("./libtorch");
    if validate_libtorch(&default_path) {
        println!("cargo:warning=[OK] Found in local directory");
        return Some(default_path);
    }

    // Method 5: Check common installation paths
    let common_paths = vec![
        "/usr/local/libtorch",
        "/opt/libtorch",
        "C:\\libtorch",
        "C:\\Program Files\\libtorch",
    ];

    for path_str in common_paths {
        let path = PathBuf::from(path_str);
        if validate_libtorch(&path) {
            println!("cargo:warning=[OK] Found at: {}", path_str);
            return Some(path);
        }
    }

    // Method 6: Try to use PyTorch's libtorch (if LIBTORCH_USE_PYTORCH is set)
    if env::var("LIBTORCH_USE_PYTORCH").is_ok() {
        if let Some(path) = find_pytorch_libtorch() {
            println!("cargo:warning=[OK] Using PyTorch's libtorch");
            return Some(path);
        }
    }

    None
}

fn find_pytorch_libtorch() -> Option<PathBuf> {
    // Try to find PyTorch installation and get libtorch path
    let output = Command::new("python3")
        .args(["-c", "import torch; print(torch.__path__[0])"])
        .output()
        .ok()?;

    if output.status.success() {
        let torch_path = String::from_utf8(output.stdout).ok()?;
        let libtorch_path = PathBuf::from(torch_path.trim()).join("lib");

        if validate_libtorch(&libtorch_path) {
            return Some(libtorch_path);
        }
    }

    None
}

fn validate_libtorch(path: &Path) -> bool {
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

fn setup_libtorch_paths(libtorch_path: &Path, system_info: &SystemInfo) {
    let lib_path = libtorch_path.join("lib");

    println!("cargo:warning=[INFO] Configuring LibTorch paths...");

    // Set LIBTORCH environment variable for torch-sys
    println!("cargo:rustc-env=LIBTORCH={}", libtorch_path.display());

    // Set linker paths
    println!("cargo:rustc-link-search=native={}", lib_path.display());

    // Link core libraries
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    println!("cargo:warning=[OK] Core libraries linked");

    // Check for CUDA libraries
    let has_cuda_lib = if cfg!(target_os = "windows") {
        lib_path.join("torch_cuda.dll").exists()
    } else {
        lib_path.join("libtorch_cuda.so").exists() || lib_path.join("libtorch_cuda.dylib").exists()
    };

    if has_cuda_lib {
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
        println!("cargo:rustc-cfg=has_cuda");
        println!("cargo:warning=[OK] CUDA support enabled in libtorch");
    }

    // Check for Metal/MPS support (macOS)
    if system_info.has_metal {
        let has_mps = lib_path.join("libtorch_mps.dylib").exists();
        if has_mps {
            println!("cargo:rustc-cfg=has_metal");
            println!("cargo:warning=[OK] Metal (MPS) support enabled");
        }
    }

    // Set runtime library paths
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-env=LD_LIBRARY_PATH={}", lib_path.display());
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-env=DYLD_LIBRARY_PATH={}", lib_path.display());
        println!(
            "cargo:rustc-env=DYLD_FALLBACK_LIBRARY_PATH={}",
            lib_path.display()
        );
    }

    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-env=PATH={}", lib_path.display());
    }

    // Output library information
    println!("cargo:warning=[INFO] LibTorch configuration:");
    println!("cargo:warning=   Library path: {}", lib_path.display());
    println!(
        "cargo:warning=   CUDA: {}",
        if has_cuda_lib { "Enabled" } else { "Disabled" }
    );
    println!(
        "cargo:warning=   Metal: {}",
        if system_info.has_metal {
            "Available"
        } else {
            "Not Available"
        }
    );
}

fn check_cuda_support(system_info: &SystemInfo) {
    if cfg!(feature = "cuda") {
        if system_info.has_cuda {
            println!("cargo:rustc-cfg=cuda_enabled");
            println!("cargo:warning=[OK] CUDA feature enabled");
        } else {
            println!("cargo:warning=[WARN] CUDA feature enabled but CUDA not found");
        }
    }
}

fn check_metal_support(system_info: &SystemInfo) {
    if system_info.has_metal {
        println!("cargo:rustc-cfg=metal_available");
        println!("cargo:warning=[OK] Metal GPU acceleration available");
    }
}

fn check_onnx_support() {
    if cfg!(feature = "onnx") {
        println!("cargo:warning=[INFO] ONNX Runtime feature enabled");
    }
}

fn print_manual_instructions() {
    println!("cargo:warning=");
    println!("cargo:warning=╔════════════════════════════════════════════════════════╗");
    println!("cargo:warning=║  Manual LibTorch Installation Required                ║");
    println!("cargo:warning=╚════════════════════════════════════════════════════════╝");
    println!("cargo:warning=");
    println!("cargo:warning=Auto-download failed. Please install LibTorch manually:");
    println!("cargo:warning=");
    println!("cargo:warning=1. Download LibTorch from:");
    println!("cargo:warning=   https://pytorch.org/get-started/locally/");
    println!("cargo:warning=");
    println!("cargo:warning=2. Extract to one of these locations:");
    println!("cargo:warning=   - ~/.cache/libtorch/");
    println!("cargo:warning=   - ./libtorch/");
    println!("cargo:warning=");
    println!("cargo:warning=3. Or set environment variable:");
    println!("cargo:warning=   export LIBTORCH=/path/to/libtorch");
    println!("cargo:warning=");
    println!("cargo:warning=4. Then rebuild:");
    println!("cargo:warning=   cargo clean && cargo build --release --features torch");
    println!("cargo:warning=");
}
