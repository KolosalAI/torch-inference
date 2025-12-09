/// Auto-detection and download for PyTorch (libtorch) libraries
/// Handles CUDA detection and automatic library installation

use anyhow::{Result, Context, bail};
use std::path::{Path, PathBuf};
use std::env;
use std::fs;

#[derive(Debug, Clone, PartialEq)]
pub enum TorchBackend {
    Cuda(String),  // CUDA version (e.g., "12.1")
    Cpu,
}

#[derive(Debug, Clone)]
pub struct TorchConfig {
    pub backend: TorchBackend,
    pub libtorch_path: PathBuf,
    pub version: String,
}

pub struct TorchLibAutoDetect {
    cuda_available: bool,
    cuda_version: Option<String>,
    libtorch_dir: PathBuf,
}

impl TorchLibAutoDetect {
    pub fn new() -> Self {
        let libtorch_dir = env::var("LIBTORCH_DIR")
            .or_else(|_| env::var("LIBTORCH"))
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./libtorch"));

        Self {
            cuda_available: false,
            cuda_version: None,
            libtorch_dir,
        }
    }

    /// Detect CUDA availability and version
    pub fn detect_cuda(&mut self) -> Result<()> {
        log::info!("Detecting CUDA installation...");

        // Try multiple methods to detect CUDA
        
        // Method 1: Check CUDA_PATH environment variable
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            let cuda_path = PathBuf::from(&cuda_path);
            if cuda_path.exists() {
                if let Some(version) = self.extract_cuda_version_from_path(&cuda_path) {
                    log::info!("✅ CUDA detected via CUDA_PATH: {}", version);
                    self.cuda_available = true;
                    self.cuda_version = Some(version);
                    return Ok(());
                }
            }
        }

        // Method 2: Check for versioned CUDA_PATH variables
        for major in (10..=13).rev() {
            for minor in (0..=9).rev() {
                let var_name = format!("CUDA_PATH_V{}_{}", major, minor);
                if let Ok(cuda_path) = env::var(&var_name) {
                    let cuda_path = PathBuf::from(&cuda_path);
                    if cuda_path.exists() {
                        let version = format!("{}.{}", major, minor);
                        log::info!("✅ CUDA detected via {}: {}", var_name, version);
                        self.cuda_available = true;
                        self.cuda_version = Some(version);
                        // Set CUDA_PATH for consistency
                        env::set_var("CUDA_PATH", &cuda_path);
                        return Ok(());
                    }
                }
            }
        }

        // Method 3: Check common installation paths (Windows)
        #[cfg(target_os = "windows")]
        {
            let base_path = PathBuf::from("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA");
            if base_path.exists() {
                if let Ok(entries) = fs::read_dir(&base_path) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_dir() {
                            if let Some(version) = self.extract_cuda_version_from_path(&path) {
                                log::info!("✅ CUDA detected at: {:?} ({})", path, version);
                                self.cuda_available = true;
                                self.cuda_version = Some(version);
                                env::set_var("CUDA_PATH", &path);
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }

        // Method 4: Check for nvidia-smi (UNIX/Linux)
        #[cfg(not(target_os = "windows"))]
        {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=driver_version")
                .arg("--format=csv,noheader")
                .output()
            {
                if output.status.success() {
                    log::info!("✅ CUDA detected via nvidia-smi");
                    self.cuda_available = true;
                    self.cuda_version = Some("11.8".to_string()); // Default for compatibility
                    return Ok(());
                }
            }
        }

        log::warn!("❌ CUDA not detected - will use CPU backend");
        self.cuda_available = false;
        self.cuda_version = None;
        Ok(())
    }

    fn extract_cuda_version_from_path(&self, path: &Path) -> Option<String> {
        path.file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| {
                if name.starts_with("v") {
                    Some(name[1..].to_string())
                } else {
                    None
                }
            })
    }

    /// Check if libtorch is already installed
    pub fn check_existing_installation(&self) -> Option<TorchConfig> {
        log::info!("Checking for existing libtorch installation...");

        // Check LIBTORCH environment variable
        if let Ok(libtorch_path) = env::var("LIBTORCH") {
            let path = PathBuf::from(&libtorch_path);
            if self.validate_libtorch_installation(&path) {
                let backend = self.detect_backend(&path);
                log::info!("✅ Found libtorch at: {:?} (backend: {:?})", path, backend);
                return Some(TorchConfig {
                    backend,
                    libtorch_path: path,
                    version: "2.2.0".to_string(), // Default version
                });
            }
        }

        // Check default installation directory
        if self.libtorch_dir.exists() && self.validate_libtorch_installation(&self.libtorch_dir) {
            let backend = self.detect_backend(&self.libtorch_dir);
            log::info!("✅ Found libtorch at: {:?} (backend: {:?})", self.libtorch_dir, backend);
            return Some(TorchConfig {
                backend,
                libtorch_path: self.libtorch_dir.clone(),
                version: "2.2.0".to_string(),
            });
        }

        log::info!("❌ No existing libtorch installation found");
        None
    }

    fn validate_libtorch_installation(&self, path: &Path) -> bool {
        // Check for required files
        let lib_dir = path.join("lib");
        let include_dir = path.join("include");

        if !lib_dir.exists() || !include_dir.exists() {
            return false;
        }

        // Check for torch library
        #[cfg(target_os = "windows")]
        let torch_lib = lib_dir.join("torch.lib");
        #[cfg(not(target_os = "windows"))]
        let torch_lib = lib_dir.join("libtorch.so");

        torch_lib.exists()
    }

    fn detect_backend(&self, path: &Path) -> TorchBackend {
        let lib_dir = path.join("lib");

        // Check for CUDA libraries
        #[cfg(target_os = "windows")]
        {
            if lib_dir.join("torch_cuda.dll").exists() {
                return TorchBackend::Cuda(self.cuda_version.clone().unwrap_or("11.8".to_string()));
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            if lib_dir.join("libtorch_cuda.so").exists() {
                return TorchBackend::Cuda(self.cuda_version.clone().unwrap_or("11.8".to_string()));
            }
        }

        TorchBackend::Cpu
    }

    /// Get download URL for libtorch
    pub fn get_download_url(&self, backend: &TorchBackend) -> String {
        let base_url = "https://download.pytorch.org/libtorch";
        let version = "2.2.0"; // Latest stable version

        #[cfg(target_os = "windows")]
        {
            match backend {
                TorchBackend::Cuda(cuda_ver) => {
                    // Map CUDA version to PyTorch compatible version
                    let cuda_suffix = if cuda_ver.starts_with("12") {
                        "cu121"
                    } else if cuda_ver.starts_with("11.8") {
                        "cu118"
                    } else {
                        "cu118" // Default to 11.8
                    };
                    format!("{}/cpu/libtorch-win-shared-with-deps-{}.zip", base_url, version)
                    // Note: CUDA builds require manual download from PyTorch website
                }
                TorchBackend::Cpu => {
                    format!("{}/cpu/libtorch-win-shared-with-deps-{}.zip", base_url, version)
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            match backend {
                TorchBackend::Cuda(cuda_ver) => {
                    let cuda_suffix = if cuda_ver.starts_with("12") {
                        "cu121"
                    } else {
                        "cu118"
                    };
                    format!("{}/{}/libtorch-cxx11-abi-shared-with-deps-{}.zip", 
                        base_url, cuda_suffix, version)
                }
                TorchBackend::Cpu => {
                    format!("{}/cpu/libtorch-cxx11-abi-shared-with-deps-{}.zip", 
                        base_url, version)
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            format!("{}/cpu/libtorch-macos-{}.zip", base_url, version)
        }
    }

    /// Download and install libtorch
    pub async fn download_libtorch(&self, backend: &TorchBackend) -> Result<PathBuf> {
        let url = self.get_download_url(backend);
        let filename = url.split('/').last().unwrap_or("libtorch.zip");
        let download_path = self.libtorch_dir.parent()
            .unwrap_or(Path::new("."))
            .join(filename);

        log::info!("📥 Downloading libtorch from: {}", url);
        log::info!("   Destination: {:?}", download_path);
        log::info!("   Backend: {:?}", backend);

        // Create parent directory
        if let Some(parent) = download_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Download file
        let response = reqwest::get(&url).await
            .context("Failed to download libtorch")?;

        if !response.status().is_success() {
            bail!("Failed to download libtorch: HTTP {}", response.status());
        }

        let total_size = response.content_length().unwrap_or(0);
        log::info!("   Size: {} MB", total_size / 1024 / 1024);

        let bytes = response.bytes().await
            .context("Failed to read download response")?;

        fs::write(&download_path, &bytes)
            .context("Failed to write downloaded file")?;

        log::info!("✅ Downloaded successfully");

        // Extract archive
        log::info!("📦 Extracting archive...");
        self.extract_archive(&download_path)?;

        // Clean up downloaded file
        fs::remove_file(&download_path)?;

        log::info!("✅ Libtorch installed successfully at: {:?}", self.libtorch_dir);

        // Set environment variable
        env::set_var("LIBTORCH", &self.libtorch_dir);
        env::set_var("LD_LIBRARY_PATH", self.libtorch_dir.join("lib"));

        Ok(self.libtorch_dir.clone())
    }

    fn extract_archive(&self, archive_path: &Path) -> Result<()> {
        let file = fs::File::open(archive_path)?;
        let mut archive = zip::ZipArchive::new(file)
            .context("Failed to open archive")?;

        let extract_dir = self.libtorch_dir.parent()
            .unwrap_or(Path::new("."));

        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let outpath = match file.enclosed_name() {
                Some(path) => extract_dir.join(path),
                None => continue,
            };

            if file.name().ends_with('/') {
                fs::create_dir_all(&outpath)?;
            } else {
                if let Some(p) = outpath.parent() {
                    fs::create_dir_all(p)?;
                }
                let mut outfile = fs::File::create(&outpath)?;
                std::io::copy(&mut file, &mut outfile)?;
            }

            // Set permissions on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Some(mode) = file.unix_mode() {
                    fs::set_permissions(&outpath, fs::Permissions::from_mode(mode))?;
                }
            }
        }

        Ok(())
    }

    /// Auto-detect and setup PyTorch
    pub async fn auto_setup(&mut self) -> Result<TorchConfig> {
        log::info!("╔════════════════════════════════════════════╗");
        log::info!("║   PyTorch Auto-Detection and Setup        ║");
        log::info!("╚════════════════════════════════════════════╝");

        // Step 1: Detect CUDA
        self.detect_cuda()?;

        // Step 2: Check for existing installation
        if let Some(config) = self.check_existing_installation() {
            log::info!("✅ Using existing PyTorch installation");
            return Ok(config);
        }

        // Step 3: Determine backend
        let backend = if self.cuda_available {
            log::info!("🎮 CUDA available - will download CUDA-enabled PyTorch");
            TorchBackend::Cuda(self.cuda_version.clone().unwrap_or("11.8".to_string()))
        } else {
            log::info!("💻 CUDA not available - will download CPU-only PyTorch");
            TorchBackend::Cpu
        };

        // Step 4: Download and install
        let libtorch_path = self.download_libtorch(&backend).await?;

        Ok(TorchConfig {
            backend,
            libtorch_path,
            version: "2.2.0".to_string(),
        })
    }

    /// Get current configuration
    pub fn get_config(&self) -> Result<TorchConfig> {
        if let Some(config) = self.check_existing_installation() {
            Ok(config)
        } else {
            bail!("PyTorch not installed. Run auto_setup() first.");
        }
    }
}

/// Initialize PyTorch with auto-detection
pub async fn initialize_torch() -> Result<TorchConfig> {
    let mut detector = TorchLibAutoDetect::new();
    detector.auto_setup().await
}

/// Check if PyTorch is available
pub fn is_torch_available() -> bool {
    let detector = TorchLibAutoDetect::new();
    detector.check_existing_installation().is_some()
}

/// Get PyTorch info without downloading
pub fn get_torch_info() -> Option<TorchConfig> {
    let mut detector = TorchLibAutoDetect::new();
    detector.detect_cuda().ok()?;
    detector.check_existing_installation()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_detection() {
        let mut detector = TorchLibAutoDetect::new();
        assert!(detector.detect_cuda().is_ok());
    }

    #[tokio::test]
    async fn test_torch_info() {
        let info = get_torch_info();
        println!("PyTorch info: {:?}", info);
    }
}
