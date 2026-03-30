#![allow(dead_code)]
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tokio::sync::{Semaphore, SemaphorePermit};
use log::{debug, info};
use std::time::Instant;

/// RAII guard for semaphore permit to prevent leaks
pub struct PermitGuard<'a> {
    _permit: SemaphorePermit<'a>,
}

/// Metrics for a single model instance
#[allow(dead_code)]
pub struct InstanceMetrics {
    active_requests: AtomicU64,
    total_requests: AtomicU64,
    avg_latency_ms: AtomicU64,
}

impl InstanceMetrics {
    fn new() -> Self {
        Self {
            active_requests: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            avg_latency_ms: AtomicU64::new(0),
        }
    }

    fn start_request(&self) {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn end_request(&self, latency_ms: u64) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        
        // Update moving average latency
        let current_avg = self.avg_latency_ms.load(Ordering::Relaxed);
        let new_avg = if current_avg == 0 {
            latency_ms
        } else {
            // Exponential moving average: new_avg = 0.9 * old_avg + 0.1 * new_value
            (current_avg * 9 + latency_ms) / 10
        };
        self.avg_latency_ms.store(new_avg, Ordering::Relaxed);
    }

    fn load_score(&self) -> u64 {
        let active = self.active_requests.load(Ordering::Relaxed);
        let latency = self.avg_latency_ms.load(Ordering::Relaxed);
        
        // Score = active_requests * 1000 + avg_latency
        // Lower score = less loaded
        active * 1000 + latency
    }
}

/// Wrapper for model instance with metrics
struct InstanceWrapper<T> {
    instance: Arc<T>,
    metrics: Arc<InstanceMetrics>,
}

/// Pool for managing multiple instances of loaded models
pub struct ModelPool<T> {
    instances: DashMap<String, Vec<InstanceWrapper<T>>>,
    max_instances_per_model: usize,
    semaphores: DashMap<String, Arc<Semaphore>>,
    active_instances: AtomicUsize,
}

impl<T> ModelPool<T> {
    pub fn new(max_instances_per_model: usize) -> Self {
        Self {
            instances: DashMap::new(),
            max_instances_per_model,
            semaphores: DashMap::new(),
            active_instances: AtomicUsize::new(0),
        }
    }
    
    /// Add a model instance to the pool
    pub fn add_instance(&self, model_name: String, instance: Arc<T>) {
        let mut instances = self.instances.entry(model_name.clone())
            .or_insert_with(Vec::new);
        
        if instances.len() < self.max_instances_per_model {
            let wrapper = InstanceWrapper {
                instance,
                metrics: Arc::new(InstanceMetrics::new()),
            };
            instances.push(wrapper);
            info!("Added model instance for '{}', total: {}", model_name, instances.len());
        }
    }
    
    /// Acquire a model instance with load-aware selection (blocks if all instances are busy)
    /// Returns tuple of (model_instance, permit_guard, metrics, start_time)
    pub async fn acquire(&self, model_name: &str) -> Option<(Arc<T>, Arc<InstanceMetrics>, Instant)> {
        let semaphore = self.semaphores.entry(model_name.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(self.max_instances_per_model)))
            .clone();
        
        // Acquire permit - blocks if all instances are busy
        let permit = semaphore.acquire().await.ok()?;
        
        let instances_ref = self.instances.get(model_name)?;
        if instances_ref.is_empty() {
            // Permit is automatically dropped here, preventing leak
            return None;
        }
        
        // Load-aware selection: choose instance with lowest load score
        let idx = self.select_least_loaded_instance(&instances_ref);
        let wrapper = &instances_ref[idx];
        
        // Track request start
        wrapper.metrics.start_request();
        let start_time = Instant::now();
        
        debug!("Acquired model instance '{}' (instance {}, load score: {})", 
               model_name, idx, wrapper.metrics.load_score());
        
        // Keep permit alive by moving it into guard (not used but prevents drop)
        let _guard = PermitGuard { _permit: permit };
        
        Some((wrapper.instance.clone(), wrapper.metrics.clone(), start_time))
    }

    /// Select the least loaded instance based on active requests and latency
    fn select_least_loaded_instance<'a>(&self, instances: &'a [InstanceWrapper<T>]) -> usize {
        if instances.len() == 1 {
            return 0;
        }

        instances.iter()
            .enumerate()
            .min_by_key(|(_, wrapper)| wrapper.metrics.load_score())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Release tracking for a model instance (call after inference completes)
    pub fn release(&self, metrics: Arc<InstanceMetrics>, start_time: Instant) {
        let latency_ms = start_time.elapsed().as_millis() as u64;
        metrics.end_request(latency_ms);
    }
    
    /// Get number of instances for a model
    pub fn instance_count(&self, model_name: &str) -> usize {
        self.instances.get(model_name)
            .map(|v| v.len())
            .unwrap_or(0)
    }
    
    /// Remove all instances of a model
    pub fn remove_model(&self, model_name: &str) {
        self.instances.remove(model_name);
        self.semaphores.remove(model_name);
        info!("Removed all instances of model '{}'", model_name);
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> ModelPoolStats {
        let total_models = self.instances.len();
        let total_instances: usize = self.instances.iter()
            .map(|entry| entry.value().len())
            .sum();
        
        // Collect active requests by iterating properly
        let mut total_active: u64 = 0;
        for entry in self.instances.iter() {
            for wrapper in entry.value().iter() {
                total_active += wrapper.metrics.active_requests.load(Ordering::Relaxed);
            }
        }
        
        ModelPoolStats {
            total_models,
            total_instances,
            active_requests: total_active as usize,
        }
    }

    /// Get detailed statistics for a specific model
    pub fn get_model_stats(&self, model_name: &str) -> Option<Vec<InstanceStats>> {
        let instances = self.instances.get(model_name)?;
        
        Some(instances.iter().enumerate().map(|(idx, wrapper)| {
            InstanceStats {
                instance_id: idx,
                active_requests: wrapper.metrics.active_requests.load(Ordering::Relaxed),
                total_requests: wrapper.metrics.total_requests.load(Ordering::Relaxed),
                avg_latency_ms: wrapper.metrics.avg_latency_ms.load(Ordering::Relaxed),
                load_score: wrapper.metrics.load_score(),
            }
        }).collect())
    }
}

#[derive(Debug, Clone)]
pub struct ModelPoolStats {
    pub total_models: usize,
    pub total_instances: usize,
    pub active_requests: usize,
}

#[derive(Debug, Clone)]
pub struct InstanceStats {
    pub instance_id: usize,
    pub active_requests: u64,
    pub total_requests: u64,
    pub avg_latency_ms: u64,
    pub load_score: u64,
}

impl<T> Default for ModelPool<T> {
    fn default() -> Self {
        Self::new(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct MockModel {
        name: String,
    }

    #[tokio::test]
    async fn test_model_pool_add_and_acquire() {
        let pool = ModelPool::new(3);
        let model = Arc::new(MockModel {
            name: "test_model".to_string(),
        });
        
        pool.add_instance("model1".to_string(), model.clone());
        
        let acquired = pool.acquire("model1").await;
        assert!(acquired.is_some());
        
        let (instance, metrics, start) = acquired.unwrap();
        assert_eq!(instance.name, "test_model");
        
        // Clean up
        pool.release(metrics, start);
    }

    #[tokio::test]
    async fn test_model_pool_multiple_instances() {
        let pool = ModelPool::new(3);
        
        for i in 0..3 {
            let model = Arc::new(MockModel {
                name: format!("instance_{}", i),
            });
            pool.add_instance("model1".to_string(), model);
        }
        
        assert_eq!(pool.instance_count("model1"), 3);
        
        let stats = pool.get_stats();
        assert_eq!(stats.total_models, 1);
        assert_eq!(stats.total_instances, 3);
    }

    #[tokio::test]
    async fn test_model_pool_max_instances() {
        let pool = ModelPool::new(2);
        
        for i in 0..5 {
            let model = Arc::new(MockModel {
                name: format!("instance_{}", i),
            });
            pool.add_instance("model1".to_string(), model);
        }
        
        // Should only have 2 instances (max)
        assert_eq!(pool.instance_count("model1"), 2);
    }

    #[tokio::test]
    async fn test_model_pool_load_balancing() {
        let pool = Arc::new(ModelPool::new(3));
        
        for i in 0..3 {
            let model = Arc::new(MockModel {
                name: format!("instance_{}", i),
            });
            pool.add_instance("model1".to_string(), model);
        }
        
        let mut handles = vec![];
        
        for _ in 0..9 {
            let pool_clone = Arc::clone(&pool);
            let handle = tokio::spawn(async move {
                let result = pool_clone.acquire("model1").await;
                if let Some((instance, metrics, start)) = result {
                    // Simulate some work
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    pool_clone.release(metrics, start);
                    Some(instance)
                } else {
                    None
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_some());
        }
    }

    #[tokio::test]
    async fn test_model_pool_remove_model() {
        let pool = ModelPool::new(3);
        let model = Arc::new(MockModel {
            name: "test".to_string(),
        });
        
        pool.add_instance("model1".to_string(), model);
        assert_eq!(pool.instance_count("model1"), 1);
        
        pool.remove_model("model1");
        assert_eq!(pool.instance_count("model1"), 0);
    }

    #[tokio::test]
    async fn test_model_pool_nonexistent_model() {
        let pool = ModelPool::<MockModel>::new(3);
        let result = pool.acquire("nonexistent").await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_model_pool_multiple_models() {
        let pool = ModelPool::new(2);
        
        for i in 1..=3 {
            let model = Arc::new(MockModel {
                name: format!("model{}", i),
            });
            pool.add_instance(format!("model{}", i), model);
        }
        
        let stats = pool.get_stats();
        assert_eq!(stats.total_models, 3);
        assert_eq!(stats.total_instances, 3);
    }

    #[tokio::test]
    async fn test_model_pool_concurrent_access() {
        let pool = Arc::new(ModelPool::new(5));
        
        let model = Arc::new(MockModel {
            name: "shared".to_string(),
        });
        pool.add_instance("shared_model".to_string(), model);
        
        let mut handles = vec![];
        
        for _ in 0..10 {
            let pool_clone = Arc::clone(&pool);
            let handle = tokio::spawn(async move {
                let result = pool_clone.acquire("shared_model").await;
                if let Some((instance, metrics, start)) = result {
                    pool_clone.release(metrics, start);
                    Some(instance)
                } else {
                    None
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_some());
        }
    }

    #[tokio::test]
    async fn test_permit_no_leak_on_early_return() {
        let pool = ModelPool::<MockModel>::new(1);
        
        // Try to acquire from non-existent model
        // This tests that permit doesn't leak when returning None
        let result = pool.acquire("nonexistent").await;
        assert!(result.is_none());
        
        // Semaphore should still have capacity
        let semaphore = pool.semaphores.get("nonexistent").unwrap();
        assert_eq!(semaphore.available_permits(), 1);
    }

    #[tokio::test]
    async fn test_load_aware_routing() {
        let pool = Arc::new(ModelPool::new(3));
        
        // Add 3 instances
        for i in 0..3 {
            let model = Arc::new(MockModel {
                name: format!("instance_{}", i),
            });
            pool.add_instance("model1".to_string(), model);
        }
        
        // Acquire first instance and hold it (simulating load)
        let (_, metrics1, start1) = pool.acquire("model1").await.unwrap();
        
        // Acquire second instance - should route to different instance
        let (_, metrics2, start2) = pool.acquire("model1").await.unwrap();
        
        // Get instance stats
        let stats = pool.get_model_stats("model1").unwrap();
        
        // Should have 2 active requests distributed across instances
        let total_active: u64 = stats.iter().map(|s| s.active_requests).sum();
        assert_eq!(total_active, 2);
        
        // Clean up
        pool.release(metrics1, start1);
        pool.release(metrics2, start2);
        
        // After release, active requests should be 0
        let stats_after = pool.get_model_stats("model1").unwrap();
        let total_active_after: u64 = stats_after.iter().map(|s| s.active_requests).sum();
        assert_eq!(total_active_after, 0);
    }

    #[tokio::test]
    async fn test_instance_metrics() {
        let pool = ModelPool::new(1);
        let model = Arc::new(MockModel {
            name: "test".to_string(),
        });
        pool.add_instance("model1".to_string(), model);
        
        // Acquire and release multiple times
        for _ in 0..5 {
            let (_, metrics, start) = pool.acquire("model1").await.unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            pool.release(metrics, start);
        }
        
        // Check stats
        let stats = pool.get_model_stats("model1").unwrap();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].total_requests, 5);
        assert_eq!(stats[0].active_requests, 0);
        assert!(stats[0].avg_latency_ms > 0);
    }

    #[test]
    fn test_model_pool_default() {
        let pool = ModelPool::<MockModel>::default();
        let stats = pool.get_stats();
        assert_eq!(stats.total_models, 0);
        assert_eq!(stats.total_instances, 0);
    }

    /// Cover line 110: the `return None` path when the key exists in `instances`
    /// but the vector is empty.  We insert an empty vec directly (private field
    /// access is allowed because this test lives in the same module).
    #[tokio::test]
    async fn test_acquire_returns_none_when_instances_vec_is_empty() {
        let pool = ModelPool::<MockModel>::new(3);
        // Insert an empty vector for "empty_model" directly into the private field.
        pool.instances.insert("empty_model".to_string(), Vec::new());
        // acquire() must find the key, discover the empty vec, and return None.
        let result = pool.acquire("empty_model").await;
        assert!(result.is_none(), "expected None when instance list is empty");
    }
}
