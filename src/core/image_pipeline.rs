#![allow(dead_code)]
/// SIMD-fused decode → resize → normalize pipeline for image preprocessing.
///
/// Hot path (feature `simd-image`):
///   JPEG bytes  → zune-jpeg (SIMD, 2× faster decode) → HWC u8
///   Other bytes → image-rs decode                    → HWC u8
///   HWC u8      → image-rs Lanczos3 resize           → HWC u8
///   HWC u8      → NCHW f32, per-channel wide::f32x8 normalize
///
/// Scalar fallback (no `simd-image`):
///   Any bytes   → image-rs decode + resize → scalar normalize
///
/// Output: `ndarray::Array4<f32>` in NCHW layout, ready for ORT.
use anyhow::{bail, Result};
use ndarray::Array4;
use std::sync::OnceLock;

use crate::tensor_pool::BufferPool;

/// Module-level pool for the mutable source buffer required by fast_image_resize.
/// Eliminates the per-request `src.to_vec()` allocation (~1.2 MB for 640×640 inputs).
static RESIZE_SRC_POOL: OnceLock<BufferPool> = OnceLock::new();

fn resize_src_pool() -> &'static BufferPool {
    RESIZE_SRC_POOL.get_or_init(|| BufferPool::new(8))
}

// ── Configuration ─────────────────────────────────────────────────────────

/// Per-channel normalization statistics + target spatial size.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Target width after resize.
    pub width: u32,
    /// Target height after resize.
    pub height: u32,
    /// Per-channel mean (R, G, B) in [0, 1].
    pub mean: [f32; 3],
    /// Per-channel std  (R, G, B) in [0, 1].
    pub std: [f32; 3],
}

impl PreprocessConfig {
    /// Standard ImageNet statistics (mean/std used by ResNet, ViT, EfficientNet …).
    pub fn imagenet(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    /// Zero mean, unit std — useful for models that apply their own normalisation.
    pub fn identity(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        }
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────

/// Stateless image preprocessing pipeline.
///
/// Create once per model and call [`preprocess_bytes`] per image.
pub struct ImagePipeline {
    cfg: PreprocessConfig,
}

impl ImagePipeline {
    pub fn new(cfg: PreprocessConfig) -> Self {
        Self { cfg }
    }

    /// Decode `data`, resize to `cfg.width × cfg.height`, and normalise.
    ///
    /// Accepts JPEG, PNG, BMP, WebP, TIFF — any format supported by image-rs.
    /// JPEG is additionally accelerated by zune-jpeg when the `simd-image`
    /// feature is enabled.
    ///
    /// Returns `Array4<f32>` with shape `[1, 3, height, width]`.
    pub fn preprocess_bytes(&self, data: &[u8]) -> Result<Array4<f32>> {
        if data.is_empty() {
            bail!("empty image data");
        }

        let (hwc_bytes, src_w, src_h) = decode_to_hwc_rgb(data)?;

        // Resize only when necessary.
        let (hwc_resized, rw, rh) = if src_w != self.cfg.width || src_h != self.cfg.height {
            resize_hwc(&hwc_bytes, src_w, src_h, self.cfg.width, self.cfg.height)?
        } else {
            (hwc_bytes, src_w, src_h)
        };

        Ok(hwc_u8_to_nchw_f32(
            &hwc_resized,
            rh as usize,
            rw as usize,
            self.cfg.mean,
            self.cfg.std,
        ))
    }

    /// Convenience wrapper that reads a file from disk.
    pub fn preprocess_path(&self, path: &std::path::Path) -> Result<Array4<f32>> {
        let data = std::fs::read(path)?;
        self.preprocess_bytes(&data)
    }

    /// Preprocess a batch of images and stack them into `[N, 3, H, W]`.
    ///
    /// Each image is decoded + resized + normalized on a rayon worker thread in
    /// parallel, then assembled into the output tensor sequentially (O(N) memory
    /// copy only).
    pub fn preprocess_batch(&self, images: &[Vec<u8>]) -> Result<Array4<f32>> {
        if images.is_empty() {
            bail!("empty batch");
        }
        use rayon::prelude::*;
        let h = self.cfg.height as usize;
        let w = self.cfg.width as usize;

        // Parallel decode + resize + normalize.
        let results: Vec<Result<Array4<f32>>> =
            images.par_iter().map(|data| self.preprocess_bytes(data)).collect();

        let mut out = Array4::<f32>::zeros((images.len(), 3, h, w));
        for (i, res) in results.into_iter().enumerate() {
            let arr = res?;
            out.slice_mut(ndarray::s![i, .., .., ..])
                .assign(&arr.slice(ndarray::s![0, .., .., ..]));
        }
        Ok(out)
    }
}

// ── Decode ────────────────────────────────────────────────────────────────

/// Decode `data` to raw HWC-RGB u8.  Returns `(bytes, width, height)`.
fn decode_to_hwc_rgb(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    // Detect JPEG by magic bytes (FF D8 FF).
    let is_jpeg = data.len() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF;

    #[cfg(feature = "simd-image")]
    if is_jpeg {
        return decode_jpeg_zune(data);
    }
    #[cfg(not(feature = "simd-image"))]
    let _ = is_jpeg; // suppress unused warning

    decode_generic(data)
}

/// zune-jpeg fast JPEG decode (SIMD, ~2× vs image-rs libjpeg).
#[cfg(feature = "simd-image")]
fn decode_jpeg_zune(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    use zune_jpeg::JpegDecoder;
    let mut dec = JpegDecoder::new(data);
    let pixels = dec
        .decode()
        .map_err(|e| anyhow::anyhow!("zune-jpeg decode error: {:?}", e))?;
    let (w, h) = dec
        .dimensions()
        .ok_or_else(|| anyhow::anyhow!("zune-jpeg: no dimensions after decode"))?;
    Ok((pixels, w as u32, h as u32))
}

/// image-rs decode for non-JPEG or when simd-image is not enabled.
fn decode_generic(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    use image::GenericImageView;
    let img = image::load_from_memory(data)?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    Ok((rgb.into_raw(), w, h))
}

// ── Resize ────────────────────────────────────────────────────────────────

/// Resize `src` (HWC RGB u8) from `src_w × src_h` to `dst_w × dst_h`.
///
/// Uses `fast_image_resize` with CatmullRom filtering — SIMD-accelerated
/// (SSE4.1 / AVX2 / NEON), ~4× faster than image-rs Lanczos3 for typical
/// ML model input sizes with visually equivalent quality.
pub(crate) fn resize_hwc(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<(Vec<u8>, u32, u32)> {
    use fast_image_resize::{
        images::Image, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
    };

    if src_w == 0 || src_h == 0 {
        anyhow::bail!("resize: zero src dimension");
    }
    if dst_w == 0 || dst_h == 0 {
        anyhow::bail!("resize: zero dst dimension");
    }

    // Acquire a pooled buffer large enough for the source pixels.
    // fast_image_resize requires owning Vec<u8> in 5.x, so we still copy src
    // into a reusable buffer to avoid allocating a fresh Vec per request.
    let src_len = src.len();
    let mut src_buf = resize_src_pool().acquire(src_len);
    src_buf[..src_len].copy_from_slice(src);

    let src_img = Image::from_vec_u8(src_w, src_h, src_buf[..src_len].to_vec(), PixelType::U8x3)
        .map_err(|e| anyhow::anyhow!("fast_image_resize src: {e}"))?;
    let mut dst_img = Image::new(dst_w, dst_h, PixelType::U8x3);

    let mut resizer = Resizer::new();
    let opts = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
    let resize_result = resizer
        .resize(&src_img, &mut dst_img, &opts)
        .map_err(|e| anyhow::anyhow!("fast_image_resize resize: {e}"));

    // Release src_buf before propagating any error.
    drop(src_img);
    resize_src_pool().release(src_buf);
    resize_result?;

    Ok((dst_img.into_vec(), dst_w, dst_h))
}

// ── Normalize ─────────────────────────────────────────────────────────────

/// Convert HWC u8 RGB → NCHW f32 with per-channel normalisation.
///
/// Uses `wide::f32x8` (AVX2 / NEON 8-wide SIMD) unconditionally for the
/// normalize step.  Step 1 (HWC scatter → NCHW) remains scalar because the
/// random-access pattern doesn't vectorise cleanly.
fn hwc_u8_to_nchw_f32(
    src: &[u8],
    height: usize,
    width: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Array4<f32> {
    let mut out = Array4::<f32>::zeros((1, 3, height, width));

    // Step 1: scatter HWC → NCHW, dividing by 255.
    for h in 0..height {
        for w in 0..width {
            for c in 0..3usize {
                let px = src[h * width * 3 + w * 3 + c] as f32 / 255.0;
                out[[0, c, h, w]] = px;
            }
        }
    }

    // Step 2: normalise each channel's contiguous NCHW slice with SIMD.
    for c in 0..3usize {
        if let Some(slice) = out.slice_mut(ndarray::s![0, c, .., ..]).as_slice_mut() {
            normalize_channel_simd(slice, mean[c], std[c]);
        } else {
            // Non-contiguous fallback (should not occur for fresh zeros Array4).
            for h in 0..height {
                for w in 0..width {
                    out[[0, c, h, w]] = (out[[0, c, h, w]] - mean[c]) / std[c];
                }
            }
        }
    }

    out
}

/// Normalise a contiguous f32 slice in-place using `wide::f32x8` (8-wide SIMD).
pub(crate) fn normalize_channel_simd(channel: &mut [f32], mean: f32, std: f32) {
    use wide::f32x8;
    let mean_v = f32x8::splat(mean);
    let std_v = f32x8::splat(std);

    let (head, middle, tail) = bytemuck::pod_align_to_mut::<f32, f32x8>(channel);

    for v in head.iter_mut() {
        *v = (*v - mean) / std;
    }
    for v in middle.iter_mut() {
        *v = (*v - mean_v) / std_v;
    }
    for v in tail.iter_mut() {
        *v = (*v - mean) / std;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────

    /// Build a solid-color PNG of size `w × h`.
    fn solid_png(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        use image::{DynamicImage, ImageBuffer, Rgb};
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(w, h, Rgb([r, g, b])));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    /// Build a solid-color JPEG of size `w × h`.
    fn solid_jpeg(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        use image::{DynamicImage, ImageBuffer, Rgb};
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(w, h, Rgb([r, g, b])));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
        buf.into_inner()
    }

    // ── PreprocessConfig ──────────────────────────────────────────────────

    #[test]
    fn test_imagenet_config_mean_std() {
        let cfg = PreprocessConfig::imagenet(224, 224);
        assert_eq!(cfg.width, 224);
        assert_eq!(cfg.height, 224);
        assert!((cfg.mean[0] - 0.485).abs() < 1e-4);
        assert!((cfg.mean[1] - 0.456).abs() < 1e-4);
        assert!((cfg.mean[2] - 0.406).abs() < 1e-4);
        assert!((cfg.std[0] - 0.229).abs() < 1e-4);
        assert!((cfg.std[1] - 0.224).abs() < 1e-4);
        assert!((cfg.std[2] - 0.225).abs() < 1e-4);
    }

    #[test]
    fn test_identity_config_zero_mean_unit_std() {
        let cfg = PreprocessConfig::identity(64, 64);
        for c in 0..3 {
            assert_eq!(cfg.mean[c], 0.0);
            assert_eq!(cfg.std[c], 1.0);
        }
    }

    // ── Output shape ──────────────────────────────────────────────────────

    #[test]
    fn test_output_shape_nchw_224() {
        let pipeline = ImagePipeline::new(PreprocessConfig::imagenet(224, 224));
        let data = solid_png(32, 32, 128, 64, 32);
        let out = pipeline.preprocess_bytes(&data).unwrap();
        assert_eq!(out.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_output_shape_nchw_no_resize_needed() {
        let pipeline = ImagePipeline::new(PreprocessConfig::imagenet(16, 16));
        let data = solid_png(16, 16, 0, 0, 0);
        let out = pipeline.preprocess_bytes(&data).unwrap();
        assert_eq!(out.shape(), &[1, 3, 16, 16]);
    }

    // ── Normalisation correctness ─────────────────────────────────────────

    #[test]
    fn test_white_pixel_normalized() {
        // White pixel: R=G=B=255, value = 1.0 before normalise.
        // With ImageNet: R = (1.0 - 0.485) / 0.229 ≈ 2.249
        let pipeline = ImagePipeline::new(PreprocessConfig::imagenet(1, 1));
        let data = solid_png(1, 1, 255, 255, 255);
        let out = pipeline.preprocess_bytes(&data).unwrap();
        let expected_r = (1.0_f32 - 0.485) / 0.229;
        assert!(
            (out[[0, 0, 0, 0]] - expected_r).abs() < 0.02,
            "R={}, expected≈{}",
            out[[0, 0, 0, 0]],
            expected_r
        );
    }

    #[test]
    fn test_black_pixel_normalized() {
        // Black pixel: R=G=B=0, value = 0.0 before normalise.
        // R = (0.0 - 0.485) / 0.229 ≈ -2.118
        let pipeline = ImagePipeline::new(PreprocessConfig::imagenet(1, 1));
        let data = solid_png(1, 1, 0, 0, 0);
        let out = pipeline.preprocess_bytes(&data).unwrap();
        let expected_r = (0.0_f32 - 0.485) / 0.229;
        assert!(
            (out[[0, 0, 0, 0]] - expected_r).abs() < 0.02,
            "R={}, expected≈{}",
            out[[0, 0, 0, 0]],
            expected_r
        );
    }

    #[test]
    fn test_identity_config_preserves_pixel_value() {
        // identity: (px/255 - 0) / 1 = px/255
        let pipeline = ImagePipeline::new(PreprocessConfig::identity(1, 1));
        let data = solid_png(1, 1, 128, 0, 255);
        let out = pipeline.preprocess_bytes(&data).unwrap();
        assert!(
            (out[[0, 0, 0, 0]] - 128.0 / 255.0).abs() < 0.01,
            "R channel mismatch"
        );
        assert!(
            (out[[0, 1, 0, 0]] - 0.0).abs() < 0.01,
            "G channel should be 0"
        );
        assert!(
            (out[[0, 2, 0, 0]] - 1.0).abs() < 0.01,
            "B channel should be 1"
        );
    }

    // ── JPEG path ─────────────────────────────────────────────────────────

    #[test]
    fn test_jpeg_produces_same_shape_as_png() {
        let pipeline = ImagePipeline::new(PreprocessConfig::imagenet(32, 32));
        let png = solid_png(64, 64, 200, 100, 50);
        let jpg = solid_jpeg(64, 64, 200, 100, 50);
        let out_png = pipeline.preprocess_bytes(&png).unwrap();
        let out_jpg = pipeline.preprocess_bytes(&jpg).unwrap();
        assert_eq!(out_png.shape(), out_jpg.shape());
    }

    #[test]
    fn test_jpeg_pixel_values_close_to_png() {
        // JPEG is lossy, so allow up to 0.1 difference after normalization.
        let pipeline = ImagePipeline::new(PreprocessConfig::identity(1, 1));
        let png = solid_png(1, 1, 200, 100, 50);
        let jpg = solid_jpeg(1, 1, 200, 100, 50);
        let out_png = pipeline.preprocess_bytes(&png).unwrap();
        let out_jpg = pipeline.preprocess_bytes(&jpg).unwrap();
        for c in 0..3 {
            let diff = (out_png[[0, c, 0, 0]] - out_jpg[[0, c, 0, 0]]).abs();
            assert!(diff < 0.1, "channel {} diff too large: {}", c, diff);
        }
    }

    // ── Error paths ───────────────────────────────────────────────────────

    #[test]
    fn test_empty_data_returns_error() {
        let pipeline = ImagePipeline::new(PreprocessConfig::imagenet(224, 224));
        assert!(pipeline.preprocess_bytes(&[]).is_err());
    }

    #[test]
    fn test_invalid_data_returns_error() {
        let pipeline = ImagePipeline::new(PreprocessConfig::imagenet(224, 224));
        assert!(pipeline.preprocess_bytes(b"not an image").is_err());
    }

    // ── Batch ─────────────────────────────────────────────────────────────

    #[test]
    fn test_batch_shape_is_n_c_h_w() {
        let pipeline = ImagePipeline::new(PreprocessConfig::imagenet(16, 16));
        let images = vec![
            solid_png(8, 8, 255, 0, 0),
            solid_png(8, 8, 0, 255, 0),
            solid_png(8, 8, 0, 0, 255),
        ];
        let out = pipeline.preprocess_batch(&images).unwrap();
        assert_eq!(out.shape(), &[3, 3, 16, 16]);
    }

    #[test]
    fn test_batch_empty_returns_error() {
        let pipeline = ImagePipeline::new(PreprocessConfig::imagenet(16, 16));
        assert!(pipeline.preprocess_batch(&[]).is_err());
    }

    #[test]
    fn test_batch_each_image_normalized_independently() {
        let pipeline = ImagePipeline::new(PreprocessConfig::identity(1, 1));
        let images = vec![
            solid_png(1, 1, 255, 0, 0), // R=1, G=0, B=0
            solid_png(1, 1, 0, 255, 0), // R=0, G=1, B=0
        ];
        let out = pipeline.preprocess_batch(&images).unwrap();
        // image 0: R channel ≈ 1.0
        assert!((out[[0, 0, 0, 0]] - 1.0).abs() < 0.01);
        // image 1: G channel ≈ 1.0
        assert!((out[[1, 1, 0, 0]] - 1.0).abs() < 0.01);
    }

    // ── hwc_u8_to_nchw_f32 unit tests ────────────────────────────────────

    #[test]
    fn test_hwc_to_nchw_single_pixel() {
        // 1×1 image, pixel = [100, 150, 200]
        let src = [100u8, 150u8, 200u8];
        let mean = [0.0f32; 3];
        let std = [1.0f32; 3];
        let out = hwc_u8_to_nchw_f32(&src, 1, 1, mean, std);
        assert_eq!(out.shape(), &[1, 3, 1, 1]);
        assert!((out[[0, 0, 0, 0]] - 100.0 / 255.0).abs() < 1e-5);
        assert!((out[[0, 1, 0, 0]] - 150.0 / 255.0).abs() < 1e-5);
        assert!((out[[0, 2, 0, 0]] - 200.0 / 255.0).abs() < 1e-5);
    }

    #[test]
    fn test_hwc_to_nchw_channel_order() {
        // 1×2 image: pixel[0]=[10,20,30], pixel[1]=[40,50,60]
        // NCHW: [0,0,0,0]=10/255, [0,0,0,1]=40/255, [0,1,0,0]=20/255, ...
        let src = [10u8, 20, 30, 40, 50, 60];
        let out = hwc_u8_to_nchw_f32(&src, 1, 2, [0.0; 3], [1.0; 3]);
        assert!((out[[0, 0, 0, 0]] - 10.0 / 255.0).abs() < 1e-5, "R[0]");
        assert!((out[[0, 0, 0, 1]] - 40.0 / 255.0).abs() < 1e-5, "R[1]");
        assert!((out[[0, 1, 0, 0]] - 20.0 / 255.0).abs() < 1e-5, "G[0]");
        assert!((out[[0, 2, 0, 0]] - 30.0 / 255.0).abs() < 1e-5, "B[0]");
    }

    // ── resize_hwc ────────────────────────────────────────────────────────

    #[test]
    fn test_resize_hwc_output_size() {
        let src = vec![128u8; 4 * 4 * 3]; // 4×4 RGB
        let (out, w, h) = resize_hwc(&src, 4, 4, 2, 2).unwrap();
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(out.len(), 2 * 2 * 3);
    }

    #[test]
    fn test_resize_hwc_upscale() {
        let src = vec![255u8; 1 * 1 * 3]; // 1×1 RGB white
        let (out, w, h) = resize_hwc(&src, 1, 1, 4, 4).unwrap();
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    // ── SIMD normalize (only tested when feature is active) ───────────────

    #[cfg(feature = "simd-image")]
    #[test]
    fn test_normalize_channel_simd_known_values() {
        let mut data: Vec<f32> = (0..32).map(|i| i as f32 / 31.0).collect();
        let expected: Vec<f32> = data.iter().map(|v| (v - 0.5) / 0.25).collect();
        normalize_channel_simd(&mut data, 0.5, 0.25);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{} vs {}", a, b);
        }
    }

    #[cfg(feature = "simd-image")]
    #[test]
    fn test_normalize_channel_simd_non_multiple_of_8() {
        let mut data = vec![0.5_f32; 13]; // 13 is not a multiple of 8
        let expected = vec![(0.5_f32 - 0.485) / 0.229; 13];
        normalize_channel_simd(&mut data, 0.485, 0.229);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{} vs {}", a, b);
        }
    }
}
