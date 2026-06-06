//! Global semaphore in front of every ONNX call (chat, planner, reflect).
//! Cap is `limits.engine.max_concurrent` (default 1). Acquire is async.
//!
//! With a 1-permit lease, peak ONNX memory cannot multiply across concurrent
//! HTTP requests — the second caller awaits the permit until the first drops.

use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone)]
pub struct EngineLease {
    sem: Arc<Semaphore>,
}

impl EngineLease {
    pub fn new(permits: usize) -> Self {
        Self { sem: Arc::new(Semaphore::new(permits.max(1))) }
    }

    pub async fn acquire(&self) -> OwnedSemaphorePermit {
        // Semaphore::acquire_owned only fails if the semaphore is closed,
        // which we never do.
        self.sem.clone().acquire_owned().await.expect("engine lease semaphore closed")
    }

    /// Visible permit count for tests/observability.
    pub fn available(&self) -> usize { self.sem.available_permits() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn one_permit_serializes_two_calls() {
        let lease = EngineLease::new(1);
        let l1 = lease.clone();
        let l2 = lease.clone();

        let h1 = tokio::spawn(async move {
            let _p = l1.acquire().await;
            tokio::time::sleep(Duration::from_millis(50)).await;
            std::time::Instant::now()
        });
        // Give h1 a head start so it owns the permit.
        tokio::time::sleep(Duration::from_millis(5)).await;
        let h2 = tokio::spawn(async move {
            let _p = l2.acquire().await;
            std::time::Instant::now()
        });

        let t1 = h1.await.unwrap();
        let t2 = h2.await.unwrap();
        assert!(t2 >= t1, "second acquire must occur after first drops");
    }

    #[tokio::test]
    async fn available_drops_then_recovers() {
        let lease = EngineLease::new(2);
        assert_eq!(lease.available(), 2);
        let p = lease.acquire().await;
        assert_eq!(lease.available(), 1);
        drop(p);
        // Drop notification is async; yield to let it propagate.
        tokio::task::yield_now().await;
        assert_eq!(lease.available(), 2);
    }
}
