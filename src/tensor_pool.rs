#![allow(dead_code)]
use dashmap::DashMap;
use log::debug;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tensor pool for reusing pre-allocated tensors
pub struct TensorPool {
    pools: DashMap<TensorShape, Vec<Vec<f32>>>,
    max_pooled_tensors: usize,
    allocations: AtomicUsize,
    reuses: AtomicUsize,
}

/// Stack-resident compact representation of up to 8 tensor dimensions.
///
/// `dims` is kept as `Vec<usize>` for display/debug compatibility; all
/// hashing and equality checks use the inline `key` field which stores each
/// dimension packed as `u32` in a fixed `[u32; 8]` array (zero-padded).
/// This eliminates the heap allocation that the old `derive(Hash)` caused on
/// every `DashMap::get_mut` / `entry` call.
#[derive(Debug, Clone)]
pub struct TensorShape {
    pub dims: Vec<usize>,
    pub total_size: usize,
    /// Compact inline key: `key[i] = dims[i] as u32` for `i < rank`,
    /// zero for `i >= rank`.  Hashing this is a single stack-resident scan
    /// with no heap indirection.
    key: [u32; 8],
}

impl TensorShape {
    pub fn new(dims: Vec<usize>) -> Self {
        let total_size = dims.iter().product();
        let mut key = [0u32; 8];
        for (i, &d) in dims.iter().enumerate().take(8) {
            key[i] = d as u32;
        }
        Self { dims, total_size, key }
    }
}

impl PartialEq for TensorShape {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for TensorShape {}

impl Hash for TensorShape {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl TensorPool {
    pub fn new(max_pooled_tensors: usize) -> Self {
        Self {
            pools: DashMap::new(),
            max_pooled_tensors,
            allocations: AtomicUsize::new(0),
            reuses: AtomicUsize::new(0),
        }
    }

    /// Acquire a tensor from the pool or allocate a new one
    pub fn acquire(&self, shape: TensorShape) -> Vec<f32> {
        if let Some(mut pool) = self.pools.get_mut(&shape) {
            if let Some(tensor) = pool.pop() {
                self.reuses.fetch_add(1, Ordering::Relaxed);
                debug!("Reused tensor with shape {:?}", shape.dims);
                return tensor;
            }
        }

        // Allocate new tensor
        self.allocations.fetch_add(1, Ordering::Relaxed);
        debug!("Allocated new tensor with shape {:?}", shape.dims);
        vec![0.0; shape.total_size]
    }

    /// Return a tensor to the pool.
    ///
    /// **Content is unspecified on reuse** — callers must fully overwrite the
    /// tensor before reading it.  The previous `fill(0.0)` on every release
    /// was unnecessary because all callers (image preprocessing, ORT output
    /// copy) write every element before reading.  Removing it saves O(size)
    /// writes per release (~4 µs for a 224×224×3 tensor).
    pub fn release(&self, shape: TensorShape, tensor: Vec<f32>) {
        let mut entry = self.pools.entry(shape.clone()).or_insert_with(Vec::new);

        if entry.len() < self.max_pooled_tensors {
            entry.push(tensor);
            debug!("Released tensor back to pool");
        } else {
            debug!("Pool full, dropping tensor");
        }
    }

    /// Clear all pooled tensors
    pub fn clear(&self) {
        self.pools.clear();
        debug!("Cleared tensor pool");
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> TensorPoolStats {
        let total_pooled: usize = self.pools.iter().map(|entry| entry.value().len()).sum();

        let allocations = self.allocations.load(Ordering::Relaxed);
        let reuses = self.reuses.load(Ordering::Relaxed);
        let total = allocations + reuses;
        let reuse_rate = if total > 0 {
            (reuses as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        TensorPoolStats {
            total_pooled,
            allocations,
            reuses,
            reuse_rate,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorPoolStats {
    pub total_pooled: usize,
    pub allocations: usize,
    pub reuses: usize,
    /// Percentage of acquires satisfied from the pool (0.0–100.0).
    pub reuse_rate: f64,
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new(100)
    }
}

// ── Image buffer pool ─────────────────────────────────────────────────────

/// Size-class buckets for `BufferPool`.
const BUFFER_BUCKETS: &[usize] = &[1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576];

fn buffer_bucket_for(min_size: usize) -> usize {
    BUFFER_BUCKETS
        .iter()
        .copied()
        .find(|&b| b >= min_size)
        .unwrap_or(min_size) // oversized allocation falls through
}

/// Pool of `Vec<u8>` scratch buffers for image preprocessing.
///
/// Reuses buffers across requests to eliminate per-request heap allocations
/// in the JPEG decode → resize → normalize pipeline.
pub struct BufferPool {
    pub(crate) buckets: DashMap<usize, Vec<Vec<u8>>>,
    max_per_bucket: usize,
    allocations: AtomicUsize,
    reuses: AtomicUsize,
}

#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    pub allocations: usize,
    pub reuses: usize,
    /// Percentage of acquires satisfied from the pool (0.0–100.0).
    pub reuse_rate: f64,
}

impl BufferPool {
    pub fn new(max_per_bucket: usize) -> Self {
        Self {
            buckets: DashMap::new(),
            max_per_bucket,
            allocations: AtomicUsize::new(0),
            reuses: AtomicUsize::new(0),
        }
    }

    /// Acquire a buffer of at least `min_size` bytes from the pool.
    /// Returns a buffer from the smallest fitting bucket, or allocates fresh.
    ///
    /// # Content guarantee
    /// Buffer contents are **unspecified** on reuse — callers must write before
    /// reading. Image pipeline callers (JPEG decode, resize) always overwrite
    /// the full buffer, making zeroing unnecessary overhead.
    pub fn acquire(&self, min_size: usize) -> Vec<u8> {
        let bucket = buffer_bucket_for(min_size);
        if let Some(mut pool) = self.buckets.get_mut(&bucket) {
            if let Some(buf) = pool.pop() {
                self.reuses.fetch_add(1, Ordering::Relaxed);
                return buf;
            }
        }
        self.allocations.fetch_add(1, Ordering::Relaxed);
        vec![0u8; bucket]
    }

    /// Return a buffer to the pool. Dropped if the bucket is already at capacity.
    pub fn release(&self, buf: Vec<u8>) {
        let bucket = buffer_bucket_for(buf.capacity());
        let mut pool = self.buckets.entry(bucket).or_default();
        if pool.len() < self.max_per_bucket {
            pool.push(buf);
        }
        // else: drop buf — avoids unbounded growth
    }

    pub fn get_stats(&self) -> BufferPoolStats {
        let allocs = self.allocations.load(Ordering::Relaxed);
        let reuses = self.reuses.load(Ordering::Relaxed);
        let total = allocs + reuses;
        BufferPoolStats {
            allocations: allocs,
            reuses,
            reuse_rate: if total > 0 {
                reuses as f64 / total as f64 * 100.0
            } else {
                0.0
            },
        }
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new(32)
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
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.reuses, 1);
        assert!(stats.reuse_rate > 0.0);
    }

    #[test]
    fn test_tensor_pool_max_size() {
        let pool = TensorPool::new(2);
        let shape = TensorShape::new(vec![3, 3]);

        // Acquire 3 tensors
        let tensors: Vec<_> = (0..3).map(|_| pool.acquire(shape.clone())).collect();

        // Release all 3 tensors, but pool max is 2
        for tensor in tensors {
            pool.release(shape.clone(), tensor);
        }

        let stats = pool.get_stats();
        assert_eq!(stats.total_pooled, 2);
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
        assert_eq!(stats.total_pooled, 2);
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
    fn test_tensor_pool_stats_zero_reuse_rate() {
        // Fresh pool with no operations: total = 0, so reuse_rate should be 0.0 (line 86)
        let pool = TensorPool::new(10);
        let stats = pool.get_stats();
        assert_eq!(stats.allocations, 0);
        assert_eq!(stats.reuses, 0);
        assert_eq!(stats.reuse_rate, 0.0);
    }

    #[test]
    fn test_tensor_pool_default() {
        // Exercises TensorPool::default() (lines 107-108)
        let pool = TensorPool::default();
        let shape = TensorShape::new(vec![2]);
        let t = pool.acquire(shape.clone());
        assert_eq!(t.len(), 2);
        let stats = pool.get_stats();
        assert_eq!(stats.allocations, 1);
    }

    #[test]
    fn test_tensor_pool_reuse_no_zeroing() {
        // Pool does NOT zero tensors on release (callers overwrite before read).
        // Verify that a released tensor is reused (pool reuse path exercised).
        let pool = TensorPool::new(10);
        let shape = TensorShape::new(vec![3]);

        let mut tensor = pool.acquire(shape.clone());
        tensor[0] = 1.0;
        tensor[1] = 2.0;
        tensor[2] = 3.0;

        pool.release(shape.clone(), tensor);

        // Reuse should come from pool (reuse counter increments).
        let _reused = pool.acquire(shape.clone());
        let stats = pool.get_stats();
        assert_eq!(stats.reuses, 1);
    }

    mod buffer_pool_tests {
        use super::*;

        #[test]
        fn buffer_pool_reuses_released_buffer() {
            let pool = BufferPool::new(4);
            let buf = pool.acquire(1000); // gets 1024-bucket
            assert_eq!(buf.capacity(), 1024);
            pool.release(buf);

            assert_eq!(pool.get_stats().allocations, 1);
            assert_eq!(pool.get_stats().reuses, 0);

            let buf2 = pool.acquire(500); // should reuse the 1024-bucket buffer
            assert_eq!(pool.get_stats().reuses, 1);
            drop(buf2);
        }

        #[test]
        fn buffer_pool_respects_max_depth() {
            let pool = BufferPool::new(2);
            let b1 = pool.acquire(100);
            let b2 = pool.acquire(100);
            let b3 = pool.acquire(100);
            pool.release(b1);
            pool.release(b2);
            pool.release(b3); // pool is at max (2), this should be dropped

            let bucket_depth = pool.buckets.get(&1024).map(|v| v.len()).unwrap_or(0);
            assert_eq!(
                bucket_depth, 2,
                "pool should hold at most max_per_bucket buffers"
            );
        }

        #[test]
        fn buffer_bucket_for_returns_correct_size() {
            assert_eq!(buffer_bucket_for(0), 1_024);
            assert_eq!(buffer_bucket_for(1_000), 1_024);
            assert_eq!(buffer_bucket_for(1_024), 1_024);
            assert_eq!(buffer_bucket_for(1_025), 4_096);
            assert_eq!(buffer_bucket_for(65_536), 65_536);
            assert_eq!(buffer_bucket_for(65_537), 262_144);
            assert_eq!(buffer_bucket_for(2_000_000), 2_000_000); // oversized fallthrough
        }
    }
}
