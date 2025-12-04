use std::sync::Arc;
use tokio::sync::Semaphore;
use log::{debug, warn};

pub struct BulkheadConfig {
    pub max_concurrent: usize,
    pub queue_size: usize,
}

pub struct Bulkhead {
    semaphore: Arc<Semaphore>,
    config: BulkheadConfig,
}

impl Bulkhead {
    pub fn new(config: BulkheadConfig) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
            config,
        }
    }

    pub async fn acquire_permit(&self) -> Result<tokio::sync::SemaphorePermit<'static>, String> {
        match self.semaphore.try_acquire() {
            Ok(permit) => {
                debug!("Bulkhead permit acquired");
                // SAFETY: We know the semaphore lives for the lifetime of Self
                Ok(unsafe { 
                    std::mem::transmute::<tokio::sync::SemaphorePermit, tokio::sync::SemaphorePermit<'static>>(permit)
                })
            }
            Err(_) => {
                warn!("Bulkhead at capacity");
                Err("Bulkhead at capacity".to_string())
            }
        }
    }

    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    pub fn max_concurrent(&self) -> usize {
        self.config.max_concurrent
    }
}

impl Default for BulkheadConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 100,
            queue_size: 1000,
        }
    }
}
