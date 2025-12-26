//! Image processing with parallel batch support
//! 
//! Provides image preprocessing with:
//! - Parallel processing using rayon
//! - Bounded concurrency control
//! - Efficient batch operations
//! - Memory-efficient tensor conversion

use image::{RgbImage, ImageBuffer, imageops};
use rayon::prelude::*;
use std::sync::Arc;
use parking_lot::Mutex;
use std::collections::VecDeque;
use crate::concurrency_limiter::ConcurrencyLimiter;

/// Memory pool for tensor reuse
struct TensorPool {
    pool: Mutex<VecDeque<Vec<f32>>>,
    capacity: usize,
    max_size: usize,
}

impl TensorPool {
    fn new(capacity: usize, initial_size: usize) -> Self {
        let mut pool = VecDeque::with_capacity(initial_size);
        
        for _ in 0..initial_size {
            let mut tensor = Vec::with_capacity(capacity);
            unsafe {
                tensor.set_len(capacity);
                tensor.set_len(0);
            }
            pool.push_back(tensor);
        }
        
        Self {
            pool: Mutex::new(pool),
            capacity,
            max_size: initial_size * 2,
        }
    }
    
    #[inline]
    fn acquire(&self) -> Vec<f32> {
        if let Some(mut tensor) = self.pool.lock().pop_front() {
            tensor.clear();
            return tensor;
        }
        
        let mut tensor = Vec::with_capacity(self.capacity);
        unsafe {
            tensor.set_len(self.capacity);
            tensor.set_len(0);
        }
        tensor
    }
    
    #[inline]
    fn release(&self, mut tensor: Vec<f32>) {
        let mut pool = self.pool.lock();
        if pool.len() < self.max_size && tensor.capacity() >= self.capacity {
            tensor.clear();
            pool.push_back(tensor);
        }
    }
}

/// Image processor with bounded concurrency
pub struct ImageProcessor {
    limiter: Arc<ConcurrencyLimiter>,
    pool_224: Arc<TensorPool>,
    pool_384: Arc<TensorPool>,
    pool_512: Arc<TensorPool>,
    thread_pool: rayon::ThreadPool,
}

impl ImageProcessor {
    /// Create a new image processor
    /// 
    /// # Arguments
    /// * `max_concurrent` - Maximum concurrent operations
    pub fn new(max_concurrent: usize) -> Self {
        let num_threads = num_cpus::get();
        
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("img-worker-{}", i))
            .stack_size(2 * 1024 * 1024)
            .build()
            .expect("Failed to build thread pool");
        
        let pool_capacity = num_threads * 8;
        
        Self {
            limiter: Arc::new(ConcurrencyLimiter::new(max_concurrent)),
            pool_224: Arc::new(TensorPool::new(224 * 224 * 3, pool_capacity)),
            pool_384: Arc::new(TensorPool::new(384 * 384 * 3, pool_capacity / 2)),
            pool_512: Arc::new(TensorPool::new(512 * 512 * 3, pool_capacity / 2)),
            thread_pool,
        }
    }
    
    fn get_pool(&self, size: (u32, u32)) -> Option<&Arc<TensorPool>> {
        match size {
            (224, 224) => Some(&self.pool_224),
            (384, 384) => Some(&self.pool_384),
            (512, 512) => Some(&self.pool_512),
            _ => None,
        }
    }
    
    /// Preprocess a single image (async with concurrency limiting)
    /// 
    /// # Arguments
    /// * `img` - Input image
    /// * `target_size` - Target dimensions (width, height)
    /// 
    /// # Returns
    /// Preprocessed tensor as Vec<f32>
    pub async fn preprocess_async(
        &self,
        img: RgbImage,
        target_size: (u32, u32),
    ) -> Vec<f32> {
        self.limiter.execute(move || {
            Self::preprocess_sync(&img, target_size)
        }).await
    }
    
    /// Preprocess image with pooled memory
    #[inline]
    fn preprocess_with_pool(
        &self,
        img: &RgbImage,
        target_size: (u32, u32),
    ) -> Vec<f32> {
        let resized = imageops::resize(
            img,
            target_size.0,
            target_size.1,
            imageops::FilterType::Triangle,
        );
        
        let pool = self.get_pool(target_size);
        let mut tensor = pool.map(|p| p.acquire())
            .unwrap_or_else(|| {
                let capacity = (target_size.0 * target_size.1 * 3) as usize;
                let mut t = Vec::with_capacity(capacity);
                unsafe {
                    t.set_len(capacity);
                    t.set_len(0);
                }
                t
            });
        
        tensor.clear();
        
        let pixels = resized.as_raw();
        const INV_SCALE: f32 = 1.0 / 127.5;
        
        for chunk in pixels.chunks_exact(3) {
            unsafe {
                let r = (*chunk.get_unchecked(0) as f32 - 127.5) * INV_SCALE;
                let g = (*chunk.get_unchecked(1) as f32 - 127.5) * INV_SCALE;
                let b = (*chunk.get_unchecked(2) as f32 - 127.5) * INV_SCALE;
                tensor.push(r);
                tensor.push(g);
                tensor.push(b);
            }
        }
        
        tensor
    }
    
    /// Preprocess image synchronously
    /// 
    /// # Arguments
    /// * `img` - Input image
    /// * `target_size` - Target dimensions (width, height)
    /// 
    /// # Returns
    /// Preprocessed tensor as Vec<f32>
    #[inline]
    pub fn preprocess_sync(
        img: &RgbImage,
        target_size: (u32, u32),
    ) -> Vec<f32> {
        let resized = imageops::resize(
            img,
            target_size.0,
            target_size.1,
            imageops::FilterType::Triangle,
        );
        
        let capacity = (target_size.0 * target_size.1 * 3) as usize;
        let mut tensor = Vec::with_capacity(capacity);
        
        let pixels = resized.as_raw();
        const INV_SCALE: f32 = 1.0 / 127.5;
        
        for chunk in pixels.chunks_exact(3) {
            unsafe {
                let r = (*chunk.get_unchecked(0) as f32 - 127.5) * INV_SCALE;
                let g = (*chunk.get_unchecked(1) as f32 - 127.5) * INV_SCALE;
                let b = (*chunk.get_unchecked(2) as f32 - 127.5) * INV_SCALE;
                tensor.push(r);
                tensor.push(g);
                tensor.push(b);
            }
        }
        
        tensor
    }
    
    /// Preprocess batch of images in parallel
    /// 
    /// # Arguments
    /// * `images` - Batch of input images
    /// * `target_size` - Target dimensions for all images
    /// 
    /// # Returns
    /// Vector of preprocessed tensors
    pub fn preprocess_batch(
        &self,
        images: &[RgbImage],
        target_size: (u32, u32),
    ) -> Vec<Vec<f32>> {
        if images.len() <= 2 {
            return images.iter()
                .map(|img| self.preprocess_with_pool(img, target_size))
                .collect();
        }
        
        let chunk_size = if images.len() < 32 {
            1
        } else if images.len() < 128 {
            2
        } else {
            4
        };
        
        self.thread_pool.install(|| {
            images.par_chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk.iter()
                        .map(|img| Self::preprocess_sync(img, target_size))
                        .collect::<Vec<_>>()
                })
                .collect()
        })
    }
    
    /// Preprocess batch of images in parallel (static method)
    #[inline]
    pub fn preprocess_batch_parallel(
        images: &[RgbImage],
        target_size: (u32, u32),
    ) -> Vec<Vec<f32>> {
        let chunk_size = if images.len() < 32 {
            1
        } else if images.len() < 128 {
            4
        } else {
            8
        };
        
        images.par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.iter()
                    .map(|img| Self::preprocess_sync(img, target_size))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
    
    /// Preprocess batch with specific chunk size
    pub fn preprocess_batch_chunked(
        images: &[RgbImage],
        target_size: (u32, u32),
        chunk_size: usize,
    ) -> Vec<Vec<f32>> {
        images.par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.iter()
                    .map(|img| Self::preprocess_sync(img, target_size))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
    
    /// Get concurrency limiter metrics
    pub fn metrics(&self) -> (usize, usize, usize) {
        (
            self.limiter.max_concurrent(),
            self.limiter.available(),
            self.limiter.active(),
        )
    }
}

/// Create a test image for benchmarking
pub fn create_test_image(width: u32, height: u32) -> RgbImage {
    ImageBuffer::from_fn(width, height, |x, y| {
        image::Rgb([
            ((x * 255) / width) as u8,
            ((y * 255) / height) as u8,
            128,
        ])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_preprocess_async() {
        let processor = ImageProcessor::new(4);
        let img = create_test_image(1920, 1080);
        
        let result = processor.preprocess_async(img, (224, 224)).await;
        
        assert_eq!(result.len(), 224 * 224 * 3);
        
        // Check normalization range [-1, 1]
        assert!(result.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }
    
    #[test]
    fn test_preprocess_batch_parallel() {
        let images: Vec<_> = (0..4)
            .map(|_| create_test_image(1920, 1080))
            .collect();
        
        let results = ImageProcessor::preprocess_batch_parallel(&images, (224, 224));
        
        assert_eq!(results.len(), 4);
        assert!(results.iter().all(|r| r.len() == 224 * 224 * 3));
    }
    
    #[test]
    fn test_preprocess_sync() {
        let img = create_test_image(1920, 1080);
        let result = ImageProcessor::preprocess_sync(&img, (224, 224));
        
        assert_eq!(result.len(), 224 * 224 * 3);
    }
}
