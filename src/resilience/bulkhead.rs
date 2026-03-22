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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bulkhead_new() {
        let config = BulkheadConfig {
            max_concurrent: 5,
            queue_size: 10,
        };
        let bulkhead = Bulkhead::new(config);
        
        assert_eq!(bulkhead.available_permits(), 5);
        assert_eq!(bulkhead.max_concurrent(), 5);
    }

    #[tokio::test]
    async fn test_bulkhead_acquire_permit() {
        let config = BulkheadConfig {
            max_concurrent: 2,
            queue_size: 10,
        };
        let bulkhead = Bulkhead::new(config);
        
        let permit1 = bulkhead.acquire_permit().await;
        assert!(permit1.is_ok());
        assert_eq!(bulkhead.available_permits(), 1);
    }

    #[tokio::test]
    async fn test_bulkhead_at_capacity() {
        let config = BulkheadConfig {
            max_concurrent: 2,
            queue_size: 10,
        };
        let bulkhead = Bulkhead::new(config);
        
        let _permit1 = bulkhead.acquire_permit().await.unwrap();
        let _permit2 = bulkhead.acquire_permit().await.unwrap();
        
        let result = bulkhead.acquire_permit().await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Bulkhead at capacity");
    }

    #[tokio::test]
    async fn test_bulkhead_permit_release() {
        let config = BulkheadConfig {
            max_concurrent: 2,
            queue_size: 10,
        };
        let bulkhead = Bulkhead::new(config);
        
        {
            let _permit1 = bulkhead.acquire_permit().await.unwrap();
            let _permit2 = bulkhead.acquire_permit().await.unwrap();
            assert_eq!(bulkhead.available_permits(), 0);
        }
        
        // Permits should be released
        assert_eq!(bulkhead.available_permits(), 2);
    }

    #[tokio::test]
    async fn test_bulkhead_available_permits() {
        let config = BulkheadConfig {
            max_concurrent: 5,
            queue_size: 10,
        };
        let bulkhead = Bulkhead::new(config);
        
        assert_eq!(bulkhead.available_permits(), 5);
        
        let _permit = bulkhead.acquire_permit().await.unwrap();
        assert_eq!(bulkhead.available_permits(), 4);
    }

    #[tokio::test]
    async fn test_bulkhead_config_default() {
        let config = BulkheadConfig::default();
        assert_eq!(config.max_concurrent, 100);
        assert_eq!(config.queue_size, 1000);
    }
}
