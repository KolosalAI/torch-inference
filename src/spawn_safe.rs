//! Spawn helpers that don't silently lose panics.
//!
//! `tokio::spawn(future)` returns a `JoinHandle` whose `Drop` discards any
//! panic the spawned task raised. Most of our background tasks are
//! fire-and-forget (downloads, SSE writers, model warmups) — when one
//! panics, the process just keeps going with no log line, which makes
//! debugging incidents nearly impossible.
//!
//! `spawn_logged(name, fut)` wraps the future in `AssertUnwindSafe(...)
//! .catch_unwind()` and logs panics + errors before dropping.

use futures::FutureExt;
use std::future::Future;
use tokio::task::JoinHandle;

/// Spawn `fut` on the runtime; if it panics, log the panic message before
/// the join handle is dropped. `name` is included in the log to make
/// it possible to identify the originating call site.
pub fn spawn_logged<F, T>(name: &'static str, fut: F) -> JoinHandle<()>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    tokio::spawn(async move {
        let result = std::panic::AssertUnwindSafe(fut).catch_unwind().await;
        if let Err(payload) = result {
            let msg = panic_message(&payload);
            tracing::error!(task = name, panic = %msg, "background task panicked");
        }
    })
}

/// Spawn `fut` and additionally log any `Err(_)` it returns. Distinct
/// from `spawn_logged` so callers can opt into result-logging only when
/// the future actually returns a `Result`.
pub fn spawn_logged_result<F, T, E>(name: &'static str, fut: F) -> JoinHandle<()>
where
    F: Future<Output = Result<T, E>> + Send + 'static,
    T: Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    tokio::spawn(async move {
        let result = std::panic::AssertUnwindSafe(fut).catch_unwind().await;
        match result {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => tracing::error!(task = name, error = %e, "background task failed"),
            Err(payload) => {
                let msg = panic_message(&payload);
                tracing::error!(task = name, panic = %msg, "background task panicked");
            }
        }
    })
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}
