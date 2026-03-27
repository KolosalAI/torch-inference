use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use log::debug;

/// Tensor pool for reusing pre-allocated tensors
pub struct TensorPool {
    pools: DashMap<TensorShape, Vec<Vec<f32>>>,
    max_pooled_tensors: usize,
    allocations: AtomicUsize,
    reuses: AtomicUsize,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct TensorShape {
    pub dims: Vec<usize>,
    pub total_size: usize,
}

impl TensorShape {
    pub fn new(dims: Vec<usize>) -> Self {
        let total_size = dims.iter().product();
        Self { dims, total_size }
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
    
    /// Return a tensor to the pool
    pub fn release(&self, shape: TensorShape, mut tensor: Vec<f32>) {
        let mut entry = self.pools.entry(shape.clone())
            .or_insert_with(Vec::new);
        
        if entry.len() < self.max_pooled_tensors {
            // Zero out the tensor before returning to pool
            tensor.fill(0.0);
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
        let total_pooled: usize = self.pools.iter()
            .map(|entry| entry.value().len())
            .sum();
        
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
    pub reuse_rate: f64,
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new(100)
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
    fn test_tensor_pool_zeroing() {
        let pool = TensorPool::new(10);
        let shape = TensorShape::new(vec![3]);
        
        let mut tensor = pool.acquire(shape.clone());
        tensor[0] = 1.0;
        tensor[1] = 2.0;
        tensor[2] = 3.0;
        
        pool.release(shape.clone(), tensor);
        
        let reused = pool.acquire(shape.clone());
        assert_eq!(reused[0], 0.0);
        assert_eq!(reused[1], 0.0);
        assert_eq!(reused[2], 0.0);
    }
}
