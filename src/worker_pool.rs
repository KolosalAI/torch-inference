use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore, Notify};
use tokio::task::JoinHandle;
use log::{info, debug, warn, error};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};

/// Worker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    Idle,
    Processing,
    Paused,
    Stopping,
    Stopped,
}

/// Worker statistics
#[derive(Debug, Clone)]
pub struct WorkerStats {
    pub worker_id: usize,
    pub state: WorkerState,
    pub tasks_processed: u64,
    pub total_processing_time_ms: u64,
    pub avg_processing_time_ms: u64,
    pub last_active: Option<Instant>,
    pub uptime_secs: u64,
}

/// Individual worker
pub struct Worker {
    pub id: usize,
    state: Arc<RwLock<WorkerState>>,
    tasks_processed: AtomicU64,
    total_processing_time_ms: AtomicU64,
    start_time: Instant,
    last_active: Arc<RwLock<Option<Instant>>>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            state: Arc::new(RwLock::new(WorkerState::Idle)),
            tasks_processed: AtomicU64::new(0),
            total_processing_time_ms: AtomicU64::new(0),
            start_time: Instant::now(),
            last_active: Arc::new(RwLock::new(None)),
            handle: None,
        }
    }
    
    pub async fn get_state(&self) -> WorkerState {
        *self.state.read().await
    }
    
    pub async fn set_state(&self, new_state: WorkerState) {
        *self.state.write().await = new_state;
        debug!("Worker {} state changed to {:?}", self.id, new_state);
    }
    
    pub fn record_task(&self, processing_time_ms: u64) {
        self.tasks_processed.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time_ms.fetch_add(processing_time_ms, Ordering::Relaxed);
    }
    
    pub async fn update_last_active(&self) {
        *self.last_active.write().await = Some(Instant::now());
    }
    
    pub async fn get_stats(&self) -> WorkerStats {
        let tasks = self.tasks_processed.load(Ordering::Relaxed);
        let total_time = self.total_processing_time_ms.load(Ordering::Relaxed);
        let avg_time = if tasks > 0 { total_time / tasks } else { 0 };
        
        WorkerStats {
            worker_id: self.id,
            state: *self.state.read().await,
            tasks_processed: tasks,
            total_processing_time_ms: total_time,
            avg_processing_time_ms: avg_time,
            last_active: *self.last_active.read().await,
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }
}

/// Worker pool configuration
#[derive(Debug, Clone)]
pub struct WorkerPoolConfig {
    pub min_workers: usize,
    pub max_workers: usize,
    pub scale_up_threshold: usize,
    pub scale_down_threshold: usize,
    pub worker_timeout_secs: u64,
    pub health_check_interval_secs: u64,
    pub enable_auto_scaling: bool,
    pub enable_zero_scaling: bool,
}

impl Default for WorkerPoolConfig {
    fn default() -> Self {
        Self {
            min_workers: 2,
            max_workers: 16,
            scale_up_threshold: 10,
            scale_down_threshold: 2,
            worker_timeout_secs: 300,
            health_check_interval_secs: 30,
            enable_auto_scaling: true,
            enable_zero_scaling: false,
        }
    }
}

/// Worker pool manager
pub struct WorkerPool {
    config: WorkerPoolConfig,
    workers: Arc<RwLock<Vec<Arc<Worker>>>>,
    active_workers: AtomicUsize,
    total_tasks: AtomicU64,
    failed_tasks: AtomicU64,
    queue_size: AtomicUsize,
    shutdown: Arc<AtomicBool>,
    scale_notify: Arc<Notify>,
    semaphore: Arc<Semaphore>,
    idle_workers: Arc<std::sync::Mutex<Vec<usize>>>,
}

impl WorkerPool {
    pub fn new(config: WorkerPoolConfig) -> Arc<Self> {
        Arc::new(Self {
            config: config.clone(),
            workers: Arc::new(RwLock::new(Vec::new())),
            active_workers: AtomicUsize::new(0),
            total_tasks: AtomicU64::new(0),
            failed_tasks: AtomicU64::new(0),
            queue_size: AtomicUsize::new(0),
            shutdown: Arc::new(AtomicBool::new(false)),
            scale_notify: Arc::new(Notify::new()),
            semaphore: Arc::new(Semaphore::new(config.max_workers)),
            idle_workers: Arc::new(std::sync::Mutex::new(Vec::new())),
        })
    }
    
    /// Initialize the worker pool
    pub async fn initialize(self: &Arc<Self>) -> Result<(), String> {
        let initial_workers = if self.config.enable_zero_scaling { 0 } else { self.config.min_workers };
        info!("Initializing worker pool with {} workers", initial_workers);
        
        for _ in 0..initial_workers {
            self.add_worker().await?;
        }
        
        // Start auto-scaling monitor if enabled
        if self.config.enable_auto_scaling {
            self.start_auto_scaler();
        }
        
        // Start health checker
        self.start_health_checker();
        
        info!("Worker pool initialized successfully");
        Ok(())
    }
    
    /// Add a new worker
    async fn add_worker(&self) -> Result<(), String> {
        let mut workers = self.workers.write().await;
        let worker_id = workers.len();
        
        if worker_id >= self.config.max_workers {
            return Err("Maximum workers reached".to_string());
        }
        
        let worker = Arc::new(Worker::new(worker_id));
        workers.push(worker.clone());
        
        // Add to idle list
        self.idle_workers.lock().unwrap().push(worker_id);
        
        self.active_workers.fetch_add(1, Ordering::Relaxed);
        
        info!("Added worker {} (total: {})", worker_id, workers.len());
        Ok(())
    }
    
    /// Remove idle workers
    async fn remove_idle_worker(&self) -> Result<(), String> {
        let mut workers = self.workers.write().await;
        
        let min_limit = if self.config.enable_zero_scaling { 0 } else { self.config.min_workers };
        
        if workers.len() <= min_limit {
            return Err("Minimum workers reached".to_string());
        }
        
        // Try to pop from idle list first (O(1))
        let idle_id = {
            let mut idle = self.idle_workers.lock().unwrap();
            idle.pop()
        };
        
        if let Some(id) = idle_id {
            // Verify worker exists and is actually idle (double check)
            if id < workers.len() {
                let worker = &workers[id];
                if worker.get_state().await == WorkerState::Idle {
                    worker.set_state(WorkerState::Stopping).await;
                    // Note: Removing from middle of vector invalidates indices > id
                    // This is problematic for our index-based approach.
                    // Instead of removing, we should mark as stopped/inactive or use a map.
                    // For now, we'll just mark it as stopped and keep it in the vector but ignore it.
                    // Or better: swap_remove if we track ID mapping.
                    
                    // SIMPLIFICATION: For this optimization, we'll stick to just marking it stopped
                    // and not actually removing from the Vec to avoid index invalidation issues
                    // until a full refactor to HashMap<usize, Worker> is done.
                    
                    // Actually, let's just return OK and let the health checker clean up or 
                    // accept that "removing" just means "stopping" for now.
                    
                    // But wait, the original code did `workers.remove(i)`.
                    // If we change to index-based idle list, we MUST handle index stability.
                    // Strategy: Only remove from the END of the list? No, idle worker might be anywhere.
                    
                    // Strategy: Use swap_remove to replace the removed worker with the last one.
                    // Then we need to update the index of the moved worker in the idle list.
                    // This is getting complex for a quick optimization.
                    
                    // ALTERNATIVE: Just pop from idle list. If we successfully popped, we know it's idle.
                    // But we need to remove it from `workers` vector too.
                    
                    // Let's revert to scanning for removal (it's rare event), but optimize acquisition (frequent event).
                    // So for removal, we scan, find an idle one, remove it, and ALSO remove it from idle_workers list.
                }
            }
        }
        
        // Fallback to scanning if idle list was empty or inconsistent
        // Find an idle worker
        for i in (0..workers.len()).rev() {
            let state = workers[i].get_state().await;
            if state == WorkerState::Idle {
                workers[i].set_state(WorkerState::Stopping).await;
                workers.remove(i);
                
                // Also remove from idle list if present
                let mut idle = self.idle_workers.lock().unwrap();
                if let Some(pos) = idle.iter().position(|&x| x == i) {
                    idle.remove(pos);
                }
                // Adjust indices for all workers > i
                for val in idle.iter_mut() {
                    if *val > i {
                        *val -= 1;
                    }
                }
                
                self.active_workers.fetch_sub(1, Ordering::Relaxed);
                info!("Removed idle worker {} (total: {})", i, workers.len());
                return Ok(());
            }
        }
        
        Err("No idle workers to remove".to_string())
    }
    
    /// Get an available worker
    pub async fn acquire_worker(&self) -> Option<Arc<Worker>> {
        // Fast path: try to get from idle queue
        let idle_id = {
            let mut idle = self.idle_workers.lock().unwrap();
            idle.pop()
        };
        
        if let Some(id) = idle_id {
            let workers = self.workers.read().await;
            if let Some(worker) = workers.get(id) {
                worker.set_state(WorkerState::Processing).await;
                worker.update_last_active().await;
                return Some(worker.clone());
            }
        }
        
        // Fallback (should be rare/impossible if logic is perfect)
        None
    }
    
    /// Release a worker back to the pool
    pub async fn release_worker(&self, worker: Arc<Worker>, processing_time_ms: u64) {
        worker.record_task(processing_time_ms);
        worker.set_state(WorkerState::Idle).await;
        worker.update_last_active().await;
        
        // Add back to idle queue
        // We need the worker's ID. The Worker struct has it.
        // But wait, Worker struct definition:
        // pub struct Worker { id: usize, ... }
        // We need to access it. It's private in the struct definition above?
        // Let's check Worker definition.
        
        self.idle_workers.lock().unwrap().push(worker.id);
        
        self.total_tasks.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record a failed task
    pub fn record_failure(&self) {
        self.failed_tasks.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Update queue size
    pub fn set_queue_size(&self, size: usize) {
        self.queue_size.store(size, Ordering::Relaxed);
        
        // Notify auto-scaler of queue change
        if self.config.enable_auto_scaling {
            self.scale_notify.notify_one();
        }
    }
    
    /// Get current worker count
    pub async fn worker_count(&self) -> usize {
        self.workers.read().await.len()
    }
    
    /// Get active worker count
    pub async fn active_worker_count(&self) -> usize {
        let workers = self.workers.read().await;
        let mut count = 0;
        
        for worker in workers.iter() {
            if worker.get_state().await == WorkerState::Processing {
                count += 1;
            }
        }
        
        count
    }
    
    /// Start auto-scaling monitor
    fn start_auto_scaler(self: &Arc<Self>) {
        let workers = self.workers.clone();
        let config = self.config.clone();
        let queue_size = Arc::new(AtomicUsize::new(self.queue_size.load(Ordering::Relaxed)));
        let shutdown = self.shutdown.clone();
        let scale_notify = self.scale_notify.clone();
        let pool_self = self.clone();
        
        tokio::spawn(async move {
            info!("Auto-scaler started");
            
            while !shutdown.load(Ordering::Relaxed) {
                tokio::select! {
                    _ = scale_notify.notified() => {},
                    _ = tokio::time::sleep(Duration::from_secs(5)) => {},
                }
                
                let queue = queue_size.load(Ordering::Relaxed);
                let worker_count = workers.read().await.len();
                
                // Scale up if queue is large
                if queue > config.scale_up_threshold && worker_count < config.max_workers {
                    info!("Scaling up: queue={}, workers={}", queue, worker_count);
                    if let Err(e) = pool_self.add_worker().await {
                        warn!("Failed to scale up: {}", e);
                    }
                }
                
                // Scale down if queue is small
                let min_limit = if config.enable_zero_scaling { 0 } else { config.min_workers };
                
                if queue < config.scale_down_threshold && worker_count > min_limit {
                    let active_count = pool_self.active_worker_count().await;
                    
                    if worker_count - active_count > 1 {
                        info!("Scaling down: queue={}, workers={}, active={}", queue, worker_count, active_count);
                        if let Err(e) = pool_self.remove_idle_worker().await {
                            debug!("Failed to scale down: {}", e);
                        }
                    }
                }
            }
            
            info!("Auto-scaler stopped");
        });
    }
    
    /// Start health checker
    fn start_health_checker(&self) {
        let workers = self.workers.clone();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();
        
        tokio::spawn(async move {
            info!("Health checker started");
            
            while !shutdown.load(Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_secs(config.health_check_interval_secs)).await;
                
                let workers_read = workers.read().await;
                let mut unhealthy = Vec::new();
                
                for (idx, worker) in workers_read.iter().enumerate() {
                    let stats = worker.get_stats().await;
                    
                    // Check if worker is stuck
                    if let Some(last_active) = stats.last_active {
                        let idle_time = last_active.elapsed().as_secs();
                        
                        if stats.state == WorkerState::Processing && idle_time > config.worker_timeout_secs {
                            warn!("Worker {} appears stuck (idle for {}s)", idx, idle_time);
                            unhealthy.push(idx);
                        }
                    }
                }
                
                if !unhealthy.is_empty() {
                    warn!("Found {} unhealthy workers", unhealthy.len());
                    // In production, we could restart these workers
                }
                
                debug!("Health check completed: {} workers checked", workers_read.len());
            }
            
            info!("Health checker stopped");
        });
    }
    
    /// Get pool statistics
    pub async fn get_stats(&self) -> WorkerPoolStats {
        let workers = self.workers.read().await;
        let mut worker_stats = Vec::new();
        
        for worker in workers.iter() {
            worker_stats.push(worker.get_stats().await);
        }
        
        let active_count = worker_stats.iter()
            .filter(|s| s.state == WorkerState::Processing)
            .count();
        
        let total_tasks = self.total_tasks.load(Ordering::Relaxed);
        let failed_tasks = self.failed_tasks.load(Ordering::Relaxed);
        let success_rate = if total_tasks > 0 {
            ((total_tasks - failed_tasks) as f64 / total_tasks as f64) * 100.0
        } else {
            100.0
        };
        
        WorkerPoolStats {
            total_workers: workers.len(),
            active_workers: active_count,
            idle_workers: workers.len() - active_count,
            total_tasks,
            failed_tasks,
            success_rate,
            queue_size: self.queue_size.load(Ordering::Relaxed),
            worker_stats,
        }
    }
    
    /// Pause all workers
    pub async fn pause_all(&self) {
        info!("Pausing all workers");
        let workers = self.workers.read().await;
        
        for worker in workers.iter() {
            if worker.get_state().await == WorkerState::Idle {
                worker.set_state(WorkerState::Paused).await;
            }
        }
    }
    
    /// Resume all workers
    pub async fn resume_all(&self) {
        info!("Resuming all workers");
        let workers = self.workers.read().await;
        
        for worker in workers.iter() {
            if worker.get_state().await == WorkerState::Paused {
                worker.set_state(WorkerState::Idle).await;
            }
        }
    }
    
    /// Shutdown the worker pool
    pub async fn shutdown(&self) {
        info!("Shutting down worker pool");
        self.shutdown.store(true, Ordering::Relaxed);
        
        let workers = self.workers.read().await;
        for worker in workers.iter() {
            worker.set_state(WorkerState::Stopped).await;
        }
        
        info!("Worker pool shutdown complete");
    }
}

#[derive(Debug, Clone)]
pub struct WorkerPoolStats {
    pub total_workers: usize,
    pub active_workers: usize,
    pub idle_workers: usize,
    pub total_tasks: u64,
    pub failed_tasks: u64,
    pub success_rate: f64,
    pub queue_size: usize,
    pub worker_stats: Vec<WorkerStats>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_worker_creation() {
        let worker = Worker::new(0);
        assert_eq!(worker.get_state().await, WorkerState::Idle);
        assert_eq!(worker.tasks_processed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_worker_state_change() {
        let worker = Worker::new(0);
        worker.set_state(WorkerState::Processing).await;
        assert_eq!(worker.get_state().await, WorkerState::Processing);
    }

    #[tokio::test]
    async fn test_worker_record_task() {
        let worker = Worker::new(0);
        worker.record_task(100);
        worker.record_task(200);
        
        assert_eq!(worker.tasks_processed.load(Ordering::Relaxed), 2);
        assert_eq!(worker.total_processing_time_ms.load(Ordering::Relaxed), 300);
    }

    #[tokio::test]
    async fn test_worker_stats() {
        let worker = Worker::new(42);
        worker.record_task(100);
        worker.record_task(200);
        
        let stats = worker.get_stats().await;
        assert_eq!(stats.worker_id, 42);
        assert_eq!(stats.tasks_processed, 2);
        assert_eq!(stats.avg_processing_time_ms, 150);
    }

    #[tokio::test]
    async fn test_worker_pool_initialization() {
        let config = WorkerPoolConfig {
            min_workers: 3,
            max_workers: 10,
            enable_auto_scaling: false,
            ..Default::default()
        };
        
        let pool = WorkerPool::new(config);
        pool.initialize().await.unwrap();
        
        assert_eq!(pool.worker_count().await, 3);
    }

    #[tokio::test]
    async fn test_worker_pool_acquire_release() {
        let config = WorkerPoolConfig {
            min_workers: 2,
            enable_auto_scaling: false,
            ..Default::default()
        };
        
        let pool = WorkerPool::new(config);
        pool.initialize().await.unwrap();
        
        let worker = pool.acquire_worker().await.unwrap();
        assert_eq!(worker.get_state().await, WorkerState::Processing);
        
        pool.release_worker(worker, 100).await;
    }

    #[tokio::test]
    async fn test_worker_pool_stats() {
        let config = WorkerPoolConfig {
            min_workers: 2,
            enable_auto_scaling: false,
            ..Default::default()
        };
        
        let pool = WorkerPool::new(config);
        pool.initialize().await.unwrap();
        
        let stats = pool.get_stats().await;
        assert_eq!(stats.total_workers, 2);
        assert_eq!(stats.idle_workers, 2);
    }

    #[tokio::test]
    async fn test_worker_pool_scaling() {
        let config = WorkerPoolConfig {
            min_workers: 2,
            max_workers: 5,
            enable_auto_scaling: false,
            ..Default::default()
        };
        
        let pool = WorkerPool::new(config);
        pool.initialize().await.unwrap();
        
        // Add worker
        pool.add_worker().await.unwrap();
        assert_eq!(pool.worker_count().await, 3);
        
        // Remove idle worker
        pool.remove_idle_worker().await.unwrap();
        assert_eq!(pool.worker_count().await, 2);
    }

    #[tokio::test]
    async fn test_worker_pool_pause_resume() {
        let config = WorkerPoolConfig {
            min_workers: 2,
            enable_auto_scaling: false,
            ..Default::default()
        };
        
        let pool = WorkerPool::new(config);
        pool.initialize().await.unwrap();
        
        pool.pause_all().await;
        let stats = pool.get_stats().await;
        assert!(stats.worker_stats.iter().all(|s| s.state == WorkerState::Paused));
        
        pool.resume_all().await;
        let stats = pool.get_stats().await;
        assert!(stats.worker_stats.iter().all(|s| s.state == WorkerState::Idle));
    }

    #[tokio::test]
    async fn test_worker_pool_failure_tracking() {
        let pool = WorkerPool::new(WorkerPoolConfig::default());
        pool.initialize().await.unwrap();
        
        pool.record_failure();
        pool.record_failure();
        
        assert_eq!(pool.failed_tasks.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_worker_pool_queue_size() {
        let pool = WorkerPool::new(WorkerPoolConfig::default());
        pool.initialize().await.unwrap();
        
        pool.set_queue_size(50);
        let stats = pool.get_stats().await;
        assert_eq!(stats.queue_size, 50);
    }

    #[tokio::test]
    async fn test_worker_pool_default() {
        let pool = WorkerPool::new(WorkerPoolConfig::default());
        assert_eq!(pool.config.min_workers, 2);
        assert_eq!(pool.config.max_workers, 16);
    }
}
