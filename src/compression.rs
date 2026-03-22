use flate2::Compression;
use flate2::write::{GzEncoder, GzDecoder};
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
}
