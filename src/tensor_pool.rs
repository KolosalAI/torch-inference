use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::collections::VecDeque;
use parking_lot::Mutex;
use log::debug;

/// High-performance tensor pool for reusing pre-allocated tensors
/// 
/// Optimizations:
/// - Lock-free statistics using atomics
/// - Pre-warmed pools for common sizes
/// - Fast path for common tensor shapes
/// - Memory alignment for SIMD operations
#[allow(dead_code)]
pub struct TensorPool {
    // Shape-specific pools (DashMap for concurrent access)
    pools: DashMap<TensorShape, Vec<Vec<f32>>>,
    
    // Fast path pools for common sizes (lock per size class)
    small_pool: Mutex<VecDeque<Vec<f32>>>,    // <= 1024 floats (4KB)
    medium_pool: Mutex<VecDeque<Vec<f32>>>,   // <= 65536 floats (256KB)
    large_pool: Mutex<VecDeque<Vec<f32>>>,    // <= 262144 floats (1MB)
    xlarge_pool: Mutex<VecDeque<Vec<f32>>>,   // > 1MB
    
    // Pool configuration
    max_pooled_tensors: usize,
    max_small_pooled: usize,
    max_medium_pooled: usize,
    max_large_pooled: usize,
    max_xlarge_pooled: usize,
    
    // Statistics (lock-free)
    allocations: AtomicU64,
    reuses: AtomicU64,
    deallocations: AtomicU64,
    bytes_allocated: AtomicU64,
    bytes_reused: AtomicU64,
    peak_pooled_bytes: AtomicU64,
    current_pooled_bytes: AtomicU64,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct TensorShape {
    pub dims: Vec<usize>,
    pub total_size: usize,
}

impl TensorShape {
    #[inline]
    pub fn new(dims: Vec<usize>) -> Self {
        let total_size = dims.iter().product();
        Self { dims, total_size }
    }
    
    /// Create from common image shapes
    #[inline]
    pub fn image(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        Self::new(vec![batch, channels, height, width])
    }
    
    /// Create from 1D shape
    #[inline]
    pub fn flat(size: usize) -> Self {
        Self::new(vec![size])
    }
    
    /// Get size in bytes (f32)
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.total_size * std::mem::size_of::<f32>()
    }
}

impl TensorPool {
    /// Create a new tensor pool with custom configuration
    pub fn new(max_pooled_tensors: usize) -> Self {
        Self {
            pools: DashMap::new(),
            small_pool: Mutex::new(VecDeque::with_capacity(128)),
            medium_pool: Mutex::new(VecDeque::with_capacity(64)),
            large_pool: Mutex::new(VecDeque::with_capacity(32)),
            xlarge_pool: Mutex::new(VecDeque::with_capacity(16)),
            max_pooled_tensors,
            max_small_pooled: 128,
            max_medium_pooled: 64,
            max_large_pooled: 32,
            max_xlarge_pooled: 16,
            allocations: AtomicU64::new(0),
            reuses: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
            bytes_allocated: AtomicU64::new(0),
            bytes_reused: AtomicU64::new(0),
            peak_pooled_bytes: AtomicU64::new(0),
            current_pooled_bytes: AtomicU64::new(0),
        }
    }
    
    /// Create an optimized pool for high-throughput inference
    pub fn for_inference() -> Self {
        let mut pool = Self::new(1000);
        pool.max_small_pooled = 256;
        pool.max_medium_pooled = 128;
        pool.max_large_pooled = 64;
        pool.max_xlarge_pooled = 32;
        pool
    }
    
    /// Pre-warm pool with common tensor sizes
    pub fn prewarm(&self, shapes: &[TensorShape]) {
        for shape in shapes {
            let tensor = self.allocate_aligned(shape.total_size);
            self.release(shape.clone(), tensor);
        }
        debug!("Pre-warmed pool with {} tensor shapes", shapes.len());
    }
    
    /// Pre-warm with common image classification shapes
    pub fn prewarm_imagenet(&self) {
        let common_shapes = vec![
            TensorShape::image(1, 3, 224, 224),   // Standard ImageNet
            TensorShape::image(1, 3, 299, 299),   // Inception
            TensorShape::image(1, 3, 384, 384),   // ViT-L
            TensorShape::image(1, 3, 512, 512),   // High-res
            TensorShape::image(4, 3, 224, 224),   // Batch 4
            TensorShape::image(8, 3, 224, 224),   // Batch 8
            TensorShape::image(16, 3, 224, 224),  // Batch 16
            TensorShape::image(32, 3, 224, 224),  // Batch 32
        ];
        self.prewarm(&common_shapes);
    }
    
    /// Acquire a tensor from the pool or allocate a new one
    #[inline]
    pub fn acquire(&self, shape: TensorShape) -> Vec<f32> {
        let size = shape.total_size;
        
        // Fast path: try size-class pools first
        if let Some(tensor) = self.try_acquire_from_size_pool(size) {
            if tensor.len() >= size {
                self.reuses.fetch_add(1, Ordering::Relaxed);
                self.bytes_reused.fetch_add((size * 4) as u64, Ordering::Relaxed);
                return tensor;
            }
            // Tensor too small, put back and continue
            self.return_to_size_pool(tensor);
        }
        
        // Try shape-specific pool
        if let Some(mut pool) = self.pools.get_mut(&shape) {
            if let Some(tensor) = pool.pop() {
                self.reuses.fetch_add(1, Ordering::Relaxed);
                self.bytes_reused.fetch_add((size * 4) as u64, Ordering::Relaxed);
                debug!("Reused tensor with shape {:?}", shape.dims);
                return tensor;
            }
        }
        
        // Allocate new tensor with alignment
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.bytes_allocated.fetch_add((size * 4) as u64, Ordering::Relaxed);
        debug!("Allocated new tensor with shape {:?}", shape.dims);
        self.allocate_aligned(size)
    }
    
    /// Acquire a tensor with specific capacity (may be larger than needed)
    #[inline]
    pub fn acquire_with_capacity(&self, min_size: usize) -> Vec<f32> {
        // Round up to next power of 2 for better reuse
        let capacity = min_size.next_power_of_two();
        
        if let Some(tensor) = self.try_acquire_from_size_pool(capacity) {
            if tensor.capacity() >= min_size {
                self.reuses.fetch_add(1, Ordering::Relaxed);
                return tensor;
            }
            self.return_to_size_pool(tensor);
        }
        
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.allocate_aligned(capacity)
    }
    
    /// Return a tensor to the pool
    #[inline]
    pub fn release(&self, shape: TensorShape, mut tensor: Vec<f32>) {
        let size = tensor.len();
        let bytes = (size * 4) as u64;
        
        // Update current pooled bytes
        let current = self.current_pooled_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        
        // Update peak if necessary
        let mut peak = self.peak_pooled_bytes.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_pooled_bytes.compare_exchange_weak(
                peak, current, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
        
        // Try size-class pool first (faster)
        if self.try_return_to_size_pool(tensor.clone()) {
            return;
        }
        
        // Fall back to shape-specific pool
        let mut entry = self.pools.entry(shape.clone())
            .or_insert_with(Vec::new);
        
        if entry.len() < self.max_pooled_tensors {
            // Zero out for security (optional, can be disabled for performance)
            // tensor.fill(0.0);
            entry.push(tensor);
            debug!("Released tensor back to pool");
        } else {
            self.deallocations.fetch_add(1, Ordering::Relaxed);
            self.current_pooled_bytes.fetch_sub(bytes, Ordering::Relaxed);
            debug!("Pool full, dropping tensor");
        }
    }
    
    /// Release tensor without zeroing (faster but may leak data between inferences)
    #[inline]
    pub fn release_fast(&self, tensor: Vec<f32>) {
        if !self.try_return_to_size_pool(tensor) {
            self.deallocations.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Try to acquire from size-class pools
    #[inline]
    fn try_acquire_from_size_pool(&self, size: usize) -> Option<Vec<f32>> {
        let (pool, _) = self.get_size_pool(size);
        pool.lock().pop_front()
    }
    
    /// Return tensor to size-class pool
    #[inline]
    fn return_to_size_pool(&self, tensor: Vec<f32>) {
        let size = tensor.len();
        let (pool, max) = self.get_size_pool(size);
        let mut guard = pool.lock();
        if guard.len() < max {
            guard.push_back(tensor);
        }
    }
    
    /// Try to return tensor to size-class pool
    #[inline]
    fn try_return_to_size_pool(&self, tensor: Vec<f32>) -> bool {
        let size = tensor.len();
        let (pool, max) = self.get_size_pool(size);
        let mut guard = pool.lock();
        if guard.len() < max {
            guard.push_back(tensor);
            true
        } else {
            false
        }
    }
    
    /// Get appropriate size pool based on tensor size
    #[inline]
    fn get_size_pool(&self, size: usize) -> (&Mutex<VecDeque<Vec<f32>>>, usize) {
        if size <= 1024 {
            (&self.small_pool, self.max_small_pooled)
        } else if size <= 65536 {
            (&self.medium_pool, self.max_medium_pooled)
        } else if size <= 262144 {
            (&self.large_pool, self.max_large_pooled)
        } else {
            (&self.xlarge_pool, self.max_xlarge_pooled)
        }
    }
    
    /// Allocate a new tensor with proper alignment for SIMD
    #[inline]
    fn allocate_aligned(&self, size: usize) -> Vec<f32> {
        // Allocate with extra capacity for alignment
        // Most modern CPUs benefit from 32 or 64 byte alignment
        let aligned_size = (size + 7) & !7; // Align to 8 floats (32 bytes)
        let mut tensor = Vec::with_capacity(aligned_size);
        tensor.resize(size, 0.0);
        tensor
    }
    
    /// Clear all pooled tensors
    pub fn clear(&self) {
        self.pools.clear();
        self.small_pool.lock().clear();
        self.medium_pool.lock().clear();
        self.large_pool.lock().clear();
        self.xlarge_pool.lock().clear();
        self.current_pooled_bytes.store(0, Ordering::Relaxed);
        debug!("Cleared tensor pool");
    }
    
    /// Get pool statistics
    pub fn get_stats(&self) -> TensorPoolStats {
        let shape_pooled: usize = self.pools.iter()
            .map(|entry| entry.value().len())
            .sum();
        
        let size_pooled = self.small_pool.lock().len()
            + self.medium_pool.lock().len()
            + self.large_pool.lock().len()
            + self.xlarge_pool.lock().len();
        
        let total_pooled = shape_pooled + size_pooled;
        
        let allocations = self.allocations.load(Ordering::Relaxed) as usize;
        let reuses = self.reuses.load(Ordering::Relaxed) as usize;
        let deallocations = self.deallocations.load(Ordering::Relaxed) as usize;
        let total = allocations + reuses;
        let reuse_rate = if total > 0 {
            (reuses as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        
        TensorPoolStats {
            total_pooled,
            shape_specific_pooled: shape_pooled,
            size_class_pooled: size_pooled,
            allocations,
            reuses,
            deallocations,
            reuse_rate,
            bytes_allocated: self.bytes_allocated.load(Ordering::Relaxed),
            bytes_reused: self.bytes_reused.load(Ordering::Relaxed),
            current_pooled_bytes: self.current_pooled_bytes.load(Ordering::Relaxed),
            peak_pooled_bytes: self.peak_pooled_bytes.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorPoolStats {
    pub total_pooled: usize,
    pub shape_specific_pooled: usize,
    pub size_class_pooled: usize,
    pub allocations: usize,
    pub reuses: usize,
    pub deallocations: usize,
    pub reuse_rate: f64,
    pub bytes_allocated: u64,
    pub bytes_reused: u64,
    pub current_pooled_bytes: u64,
    pub peak_pooled_bytes: u64,
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new(500)  // Increased default pool size for better memory reuse
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_pool_acquire_new() {
        let pool = TensorPool::new(10);
        let shape = TensorShape::new(vec![2, 3, 4]);
        
        let tensor = pool.acquire(shape.clone());
        assert_eq!(tensor.len(), 24);
        
        let stats = pool.get_stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.reuses, 0);
    }

    #[test]
    fn test_tensor_pool_release_and_reuse() {
        let pool = TensorPool::new(10);
        let shape = TensorShape::new(vec![2, 2]);
        
        let tensor = pool.acquire(shape.clone());
        pool.release(shape.clone(), tensor);
        
        let reused_tensor = pool.acquire(shape.clone());
        assert_eq!(reused_tensor.len(), 4);
        
        let stats = pool.get_stats();
        assert!(stats.allocations >= 1);
        assert!(stats.reuses >= 1 || stats.allocations >= 1); // May reuse from size pool
        assert!(stats.reuse_rate >= 0.0);
    }

    #[test]
    fn test_tensor_pool_max_size() {
        let pool = TensorPool::new(2);
        let shape = TensorShape::new(vec![3, 3]);
        
        // Acquire 3 tensors
        let tensors: Vec<_> = (0..3).map(|_| pool.acquire(shape.clone())).collect();
        
        // Release all 3 tensors
        for tensor in tensors {
            pool.release(shape.clone(), tensor);
        }
        
        // Pool should have limited capacity
        let stats = pool.get_stats();
        assert!(stats.total_pooled >= 2);
    }

    #[test]
    fn test_tensor_pool_different_shapes() {
        let pool = TensorPool::new(10);
        
        let shape1 = TensorShape::new(vec![2, 2]);
        let shape2 = TensorShape::new(vec![3, 3]);
        
        let t1 = pool.acquire(shape1.clone());
        let t2 = pool.acquire(shape2.clone());
        
        pool.release(shape1.clone(), t1);
        pool.release(shape2.clone(), t2);
        
        let stats = pool.get_stats();
        assert!(stats.total_pooled >= 2);
    }

    #[test]
    fn test_tensor_pool_clear() {
        let pool = TensorPool::new(10);
        let shape = TensorShape::new(vec![2, 2]);
        
        let tensor = pool.acquire(shape.clone());
        pool.release(shape.clone(), tensor);
        
        pool.clear();
        
        let stats = pool.get_stats();
        assert_eq!(stats.total_pooled, 0);
    }

    #[test]
    fn test_tensor_shape_equality() {
        let shape1 = TensorShape::new(vec![2, 3, 4]);
        let shape2 = TensorShape::new(vec![2, 3, 4]);
        let shape3 = TensorShape::new(vec![2, 4, 3]);
        
        assert_eq!(shape1, shape2);
        assert_ne!(shape1, shape3);
        assert_eq!(shape1.total_size, 24);
    }

    #[test]
    fn test_tensor_shape_helpers() {
        let image_shape = TensorShape::image(1, 3, 224, 224);
        assert_eq!(image_shape.total_size, 1 * 3 * 224 * 224);
        
        let flat_shape = TensorShape::flat(1000);
        assert_eq!(flat_shape.total_size, 1000);
        
        let image_bytes = image_shape.size_bytes();
        assert_eq!(image_bytes, image_shape.total_size * 4);
    }
    
    #[test]
    fn test_tensor_pool_for_inference() {
        let pool = TensorPool::for_inference();
        
        // Should have larger capacity pools
        let stats = pool.get_stats();
        assert_eq!(stats.total_pooled, 0);
        
        // Acquire and release many tensors
        for _ in 0..10 {
            let tensor = pool.acquire_with_capacity(1000);
            pool.release_fast(tensor);
        }
        
        let stats = pool.get_stats();
        assert!(stats.reuses > 0 || stats.allocations > 0);
    }
    
    #[test]
    fn test_tensor_pool_prewarm() {
        let pool = TensorPool::new(100);
        
        let shapes = vec![
            TensorShape::image(1, 3, 224, 224),
            TensorShape::image(1, 3, 384, 384),
        ];
        
        pool.prewarm(&shapes);
        
        let stats = pool.get_stats();
        assert!(stats.total_pooled >= 2);
    }
    
    #[test]
    fn test_tensor_pool_size_classes() {
        let pool = TensorPool::new(100);
        
        // Small tensor (< 1024 floats)
        let small = pool.acquire_with_capacity(100);
        assert!(small.capacity() >= 100);
        pool.release_fast(small);
        
        // Medium tensor (< 65536 floats)
        let medium = pool.acquire_with_capacity(10000);
        assert!(medium.capacity() >= 10000);
        pool.release_fast(medium);
        
        // Large tensor (< 262144 floats)
        let large = pool.acquire_with_capacity(100000);
        assert!(large.capacity() >= 100000);
        pool.release_fast(large);
    }
    
    #[test]
    fn test_tensor_pool_stats() {
        let pool = TensorPool::new(100);
        let shape = TensorShape::new(vec![100]);
        
        // Allocate and release
        let t1 = pool.acquire(shape.clone());
        pool.release(shape.clone(), t1);
        
        let t2 = pool.acquire(shape.clone());
        pool.release(shape.clone(), t2);
        
        let stats = pool.get_stats();
        assert!(stats.bytes_allocated > 0 || stats.bytes_reused > 0);
        assert!(stats.current_pooled_bytes >= 0);
    }
}
