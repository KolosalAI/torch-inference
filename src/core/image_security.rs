use serde::{Deserialize, Serialize};
use image::{DynamicImage, ImageBuffer, Rgba};
use anyhow::{Result, Context, bail};
use sha2::{Sha256, Digest};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    Low,
    Medium,
    High,
    Maximum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSecurityResult {
    pub is_safe: bool,
    pub security_level: SecurityLevel,
    pub threats_detected: Vec<ThreatInfo>,
    pub confidence: f32,
    pub sanitized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatInfo {
    pub threat_type: ThreatType,
    pub severity: Severity,
    pub description: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatType {
    AdversarialPattern,
    MaliciousPayload,
    Steganography,
    SuspiciousMetadata,
    ExcessiveSize,
    InvalidFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageStats {
    pub width: u32,
    pub height: u32,
    pub format: String,
    pub size_bytes: usize,
    pub has_alpha: bool,
    pub color_depth: u8,
}

pub struct ImageSecurityValidator {
    max_dimension: u32,
    max_file_size: usize,
    enable_steganography_check: bool,
}

impl ImageSecurityValidator {
    pub fn new() -> Self {
        Self {
            max_dimension: 8192,
            max_file_size: 50 * 1024 * 1024, // 50MB
            enable_steganography_check: true,
        }
    }

    pub fn with_limits(max_dimension: u32, max_file_size: usize) -> Self {
        Self {
            max_dimension,
            max_file_size,
            enable_steganography_check: true,
        }
    }

    /// Validate image security
    pub fn validate(&self, image_data: &[u8], security_level: SecurityLevel) -> Result<ImageSecurityResult> {
        let mut threats = Vec::new();

        // Check file size
        if image_data.len() > self.max_file_size {
            threats.push(ThreatInfo {
                threat_type: ThreatType::ExcessiveSize,
                severity: Severity::High,
                description: format!("File size {} exceeds limit {}", image_data.len(), self.max_file_size),
                confidence: 1.0,
            });
        }

        // Load image
        let img = match image::load_from_memory(image_data) {
            Ok(img) => img,
            Err(e) => {
                threats.push(ThreatInfo {
                    threat_type: ThreatType::InvalidFormat,
                    severity: Severity::Critical,
                    description: format!("Invalid image format: {}", e),
                    confidence: 1.0,
                });
                return Ok(ImageSecurityResult {
                    is_safe: false,
                    security_level,
                    threats_detected: threats,
                    confidence: 1.0,
                    sanitized: false,
                });
            }
        };

        // Check dimensions
        if img.width() > self.max_dimension || img.height() > self.max_dimension {
            threats.push(ThreatInfo {
                threat_type: ThreatType::ExcessiveSize,
                severity: Severity::Medium,
                description: format!("Image dimensions {}x{} exceed limit {}", img.width(), img.height(), self.max_dimension),
                confidence: 1.0,
            });
        }

        // Check for adversarial patterns
        if let Some(threat) = self.check_adversarial_patterns(&img, &security_level) {
            threats.push(threat);
        }

        // Check for steganography
        if self.enable_steganography_check {
            if let Some(threat) = self.check_steganography(&img, &security_level) {
                threats.push(threat);
            }
        }

        // Check metadata
        if let Some(threat) = self.check_metadata(image_data) {
            threats.push(threat);
        }

        let is_safe = threats.iter().all(|t| !matches!(t.severity, Severity::Critical | Severity::High));
        let confidence = if threats.is_empty() {
            1.0
        } else {
            threats.iter().map(|t| t.confidence).sum::<f32>() / threats.len() as f32
        };

        Ok(ImageSecurityResult {
            is_safe,
            security_level,
            threats_detected: threats,
            confidence,
            sanitized: false,
        })
    }

    /// Sanitize image
    pub fn sanitize(&self, img: &DynamicImage, security_level: SecurityLevel) -> Result<DynamicImage> {
        let mut sanitized = img.clone();

        match security_level {
            SecurityLevel::Low => {
                // Basic sanitization: remove alpha channel
                sanitized = DynamicImage::ImageRgb8(sanitized.to_rgb8());
            }
            SecurityLevel::Medium => {
                // Medium: remove alpha, normalize colors
                let rgb = sanitized.to_rgb8();
                sanitized = DynamicImage::ImageRgb8(rgb);
            }
            SecurityLevel::High => {
                // High: remove alpha, normalize, add noise
                let mut rgb = sanitized.to_rgb8();
                self.add_noise(&mut rgb, 0.01);
                sanitized = DynamicImage::ImageRgb8(rgb);
            }
            SecurityLevel::Maximum => {
                // Maximum: full sanitization with strong noise
                let mut rgb = sanitized.to_rgb8();
                self.add_noise(&mut rgb, 0.02);
                // Resize if too large
                if rgb.width() > 2048 || rgb.height() > 2048 {
                    sanitized = sanitized.resize(2048, 2048, image::imageops::FilterType::Lanczos3);
                } else {
                    sanitized = DynamicImage::ImageRgb8(rgb);
                }
            }
        }

        Ok(sanitized)
    }

    fn check_adversarial_patterns(&self, img: &DynamicImage, level: &SecurityLevel) -> Option<ThreatInfo> {
        // Simple adversarial pattern detection based on frequency analysis
        let rgb = img.to_rgb8();
        let pixels = rgb.as_raw();

        // Calculate variance in pixel values
        let mean: f32 = pixels.iter().map(|&p| p as f32).sum::<f32>() / pixels.len() as f32;
        let variance: f32 = pixels.iter()
            .map(|&p| {
                let diff = p as f32 - mean;
                diff * diff
            })
            .sum::<f32>() / pixels.len() as f32;

        // High variance might indicate adversarial patterns
        let threshold = match level {
            SecurityLevel::Low => 100000.0,
            SecurityLevel::Medium => 80000.0,
            SecurityLevel::High => 60000.0,
            SecurityLevel::Maximum => 40000.0,
        };

        if variance > threshold {
            Some(ThreatInfo {
                threat_type: ThreatType::AdversarialPattern,
                severity: Severity::Medium,
                description: format!("Suspicious pattern detected (variance: {:.2})", variance),
                confidence: 0.6,
            })
        } else {
            None
        }
    }

    fn check_steganography(&self, img: &DynamicImage, level: &SecurityLevel) -> Option<ThreatInfo> {
        // LSB (Least Significant Bit) analysis
        let rgb = img.to_rgb8();
        let pixels = rgb.as_raw();

        let mut lsb_ones = 0;
        for &pixel in pixels {
            if pixel & 1 == 1 {
                lsb_ones += 1;
            }
        }

        let lsb_ratio = lsb_ones as f32 / pixels.len() as f32;
        
        // Random data should have ~50% LSB as 1
        // Significant deviation might indicate steganography
        let deviation = (lsb_ratio - 0.5).abs();
        
        let threshold = match level {
            SecurityLevel::Low => 0.15,
            SecurityLevel::Medium => 0.12,
            SecurityLevel::High => 0.10,
            SecurityLevel::Maximum => 0.08,
        };

        if deviation > threshold {
            Some(ThreatInfo {
                threat_type: ThreatType::Steganography,
                severity: Severity::Low,
                description: format!("Possible steganography detected (LSB deviation: {:.4})", deviation),
                confidence: 0.4,
            })
        } else {
            None
        }
    }

    fn check_metadata(&self, image_data: &[u8]) -> Option<ThreatInfo> {
        // Check for excessive metadata or suspicious patterns in header
        // This is a simplified check - real implementation would parse EXIF, etc.
        
        // Look for common metadata markers
        let has_exif = image_data.windows(4).any(|w| w == b"Exif");
        let has_xmp = image_data.windows(3).any(|w| w == b"XMP" || w == b"xmp");
        
        if has_exif || has_xmp {
            Some(ThreatInfo {
                threat_type: ThreatType::SuspiciousMetadata,
                severity: Severity::Low,
                description: "Image contains metadata that may include sensitive information".to_string(),
                confidence: 0.8,
            })
        } else {
            None
        }
    }

    fn add_noise(&self, img: &mut ImageBuffer<image::Rgb<u8>, Vec<u8>>, intensity: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for pixel in img.pixels_mut() {
            for channel in pixel.0.iter_mut() {
                let noise: f32 = rng.gen_range(-intensity..intensity);
                let new_val = (*channel as f32 + noise * 255.0).clamp(0.0, 255.0);
                *channel = new_val as u8;
            }
        }
    }

    pub fn get_image_stats(&self, img: &DynamicImage) -> ImageStats {
        let has_alpha = matches!(img, DynamicImage::ImageRgba8(_) | DynamicImage::ImageRgba16(_));
        let color_depth = match img {
            DynamicImage::ImageRgb8(_) | DynamicImage::ImageRgba8(_) => 8,
            DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgba16(_) => 16,
            _ => 8,
        };

        ImageStats {
            width: img.width(),
            height: img.height(),
            format: format!("{:?}", img.color()),
            size_bytes: (img.width() * img.height() * (color_depth as u32 / 8) * if has_alpha { 4 } else { 3 }) as usize,
            has_alpha,
            color_depth,
        }
    }

    pub fn compute_hash(&self, img: &DynamicImage) -> String {
        let mut hasher = Sha256::new();
        hasher.update(img.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

impl Default for ImageSecurityValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = ImageSecurityValidator::new();
        assert_eq!(validator.max_dimension, 8192);
        assert_eq!(validator.max_file_size, 50 * 1024 * 1024);
    }

    #[test]
    fn test_security_level_variants() {
        let levels = vec![
            SecurityLevel::Low,
            SecurityLevel::Medium,
            SecurityLevel::High,
            SecurityLevel::Maximum,
        ];
        assert_eq!(levels.len(), 4);
    }
}
