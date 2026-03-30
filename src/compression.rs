#![allow(dead_code)]
use flate2::Compression;
use flate2::write::GzEncoder;
use std::io::Write;
use std::io::Read;
use log::debug;

pub struct CompressionService {
    level: Compression,
}

impl CompressionService {
    pub fn new(level: u32) -> Self {
        Self {
            level: Compression::new(level),
        }
    }
    
    /// Compress data using gzip
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
        let mut encoder = GzEncoder::new(Vec::new(), self.level);
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;
        
        let ratio = if data.len() > 0 {
            (compressed.len() as f64 / data.len() as f64) * 100.0
        } else {
            100.0
        };
        
        debug!(
            "Compressed {} bytes to {} bytes ({:.1}%)",
            data.len(),
            compressed.len(),
            ratio
        );
        
        Ok(compressed)
    }
    
    /// Decompress gzip data
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
        use flate2::read::GzDecoder as GzReadDecoder;
        
        let mut decoder = GzReadDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        
        debug!(
            "Decompressed {} bytes to {} bytes",
            data.len(),
            decompressed.len()
        );
        
        Ok(decompressed)
    }
    
    /// Check if compression is worthwhile
    pub fn should_compress(&self, data: &[u8], threshold_bytes: usize) -> bool {
        data.len() >= threshold_bytes
    }
}

impl Default for CompressionService {
    fn default() -> Self {
        Self::new(6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_roundtrip() {
        let service = CompressionService::new(6);
        let original = b"Hello, World! This is a test of compression.";
        
        let compressed = service.compress(original).unwrap();
        let decompressed = service.decompress(&compressed).unwrap();
        
        assert_eq!(original.to_vec(), decompressed);
    }

    #[test]
    fn test_compress_reduces_size() {
        let service = CompressionService::new(6);
        // Highly compressible data
        let original = vec![b'A'; 1000];
        
        let compressed = service.compress(&original).unwrap();
        assert!(compressed.len() < original.len());
    }

    #[test]
    fn test_should_compress_threshold() {
        let service = CompressionService::new(6);
        
        let small_data = vec![0u8; 100];
        let large_data = vec![0u8; 2000];
        
        assert!(!service.should_compress(&small_data, 1000));
        assert!(service.should_compress(&large_data, 1000));
    }

    #[test]
    fn test_different_compression_levels() {
        let data = vec![b'X'; 1000];
        
        let service_fast = CompressionService::new(1);
        let service_slow = CompressionService::new(9);
        
        let fast = service_fast.compress(&data).unwrap();
        let slow = service_slow.compress(&data).unwrap();
        
        // Higher compression should produce smaller output
        assert!(slow.len() <= fast.len());
    }

    #[test]
    fn test_empty_data() {
        let service = CompressionService::new(6);
        let empty: &[u8] = &[];
        
        let compressed = service.compress(empty).unwrap();
        let decompressed = service.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed.len(), 0);
    }

    #[test]
    fn test_large_data() {
        let service = CompressionService::new(6);
        let large = vec![0u8; 1_000_000];
        
        let compressed = service.compress(&large).unwrap();
        let decompressed = service.decompress(&compressed).unwrap();
        
        assert_eq!(large, decompressed);
        assert!(compressed.len() < large.len());
    }

    #[test]
    fn test_binary_data() {
        let service = CompressionService::new(6);
        let binary: Vec<u8> = (0..=255).cycle().take(1000).collect();
        
        let compressed = service.compress(&binary).unwrap();
        let decompressed = service.decompress(&compressed).unwrap();
        
        assert_eq!(binary, decompressed);
    }

    #[test]
    fn test_default_construction() {
        let service = CompressionService::default();
        let data = b"Test data for default compression";

        let compressed = service.compress(data).unwrap();
        let decompressed = service.decompress(&compressed).unwrap();

        assert_eq!(data.to_vec(), decompressed);
    }

    // ── Additional coverage tests ──────────────────────────────────────────────

    #[test]
    fn test_compress_empty_data_ratio_else_branch() {
        // Exercises the `else { 100.0 }` branch in compress() (data.len() == 0)
        let service = CompressionService::new(6);
        let empty: &[u8] = &[];

        // Compress empty slice — ratio branch takes the `else` path
        let compressed = service.compress(empty).unwrap();
        // Decompress to verify correctness
        let decompressed = service.decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_compress_single_byte() {
        let service = CompressionService::new(6);
        let data = &[42u8];
        let compressed = service.compress(data).unwrap();
        let decompressed = service.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data.to_vec());
    }

    #[test]
    fn test_should_compress_at_exact_threshold() {
        let service = CompressionService::new(6);
        // Exactly at threshold — should return true (>=)
        let data = vec![0u8; 1000];
        assert!(service.should_compress(&data, 1000));
    }

    #[test]
    fn test_should_compress_just_below_threshold() {
        let service = CompressionService::new(6);
        let data = vec![0u8; 999];
        assert!(!service.should_compress(&data, 1000));
    }

    #[test]
    fn test_compression_level_zero() {
        // Level 0 = no compression (store only)
        let service = CompressionService::new(0);
        let data = b"Store without compression";
        let compressed = service.compress(data).unwrap();
        let decompressed = service.decompress(&compressed).unwrap();
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_decompress_produces_correct_size() {
        let service = CompressionService::new(6);
        let original: Vec<u8> = (0..=255u8).cycle().take(500).collect();
        let compressed = service.compress(&original).unwrap();
        let decompressed = service.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 500);
        assert_eq!(decompressed, original);
    }

    // ── Edge-case coverage tests ───────────────────────────────────────────────

    #[test]
    fn test_should_compress_empty_data_below_positive_threshold() {
        // Empty slice length (0) is always below any positive threshold
        let service = CompressionService::new(6);
        assert!(!service.should_compress(&[], 1));
        assert!(!service.should_compress(&[], 100));
    }

    #[test]
    fn test_should_compress_zero_threshold_always_true() {
        // data.len() >= 0 is always true for any slice, including empty
        let service = CompressionService::new(6);
        assert!(service.should_compress(&[], 0));
        assert!(service.should_compress(b"x", 0));
        assert!(service.should_compress(b"hello", 0));
    }

    #[test]
    fn test_compress_all_zeros() {
        // All-zero data is highly compressible; exercises the ratio > 0 branch
        let service = CompressionService::new(6);
        let data = vec![0u8; 256];
        let compressed = service.compress(&data).unwrap();
        let decompressed = service.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
        // Compression ratio should be < 100%
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_compress_incompressible_data() {
        // Random-looking data (pseudo-random bytes) may compress to larger size;
        // exercises the ratio-logging branch with ratio > 100.
        let service = CompressionService::new(1);
        // Use a pseudo-random sequence that is difficult to compress
        let data: Vec<u8> = (0u8..=255).collect(); // 256 unique bytes
        let compressed = service.compress(&data).unwrap();
        let decompressed = service.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_new_with_various_levels() {
        // Exercises CompressionService::new with a variety of levels
        for level in [0u32, 1, 3, 6, 9] {
            let service = CompressionService::new(level);
            let data = b"hello world";
            let compressed = service.compress(data).unwrap();
            let decompressed = service.decompress(&compressed).unwrap();
            assert_eq!(decompressed, data.to_vec());
        }
    }

    // ── Logger-enabled tests to cover log::debug! argument lines (31-33, 49-51) ──
    //
    // The `log` crate's debug!/info! macros only evaluate their arguments when
    // a logger with a matching level is installed.  Without one the arguments
    // (lines 32-33 and 50-51) are never evaluated and appear uncovered.
    // We use `env_logger::Builder` to install a per-test logger at TRACE level
    // so the macro bodies are executed.  `try_init` is used so the call is safe
    // even if another test has already installed a global logger.

    fn init_logger() {
        let _ = env_logger::Builder::new()
            .filter_level(log::LevelFilter::Trace)
            .is_test(true)
            .try_init();
    }

    #[test]
    fn test_compress_debug_log_lines_covered() {
        init_logger();
        let service = CompressionService::new(6);
        // Lines 31-33: debug! arguments evaluated only when logger at DEBUG level.
        let compressed = service.compress(b"cover debug macro args").unwrap();
        // Lines 49-51: debug! arguments in decompress.
        let decompressed = service.decompress(&compressed).unwrap();
        assert_eq!(decompressed, b"cover debug macro args");
    }

    #[test]
    fn test_compress_debug_log_empty_data() {
        init_logger();
        let service = CompressionService::new(6);
        // Empty data: exercises the `else { 100.0 }` ratio branch AND the debug! args.
        let compressed = service.compress(&[]).unwrap();
        let decompressed = service.decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_compress_debug_log_large_compressible_data() {
        init_logger();
        let service = CompressionService::new(6);
        let data: Vec<u8> = b"abcdef".iter().cloned().cycle().take(4096).collect();
        let compressed = service.compress(&data).unwrap();
        let decompressed = service.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }
}
