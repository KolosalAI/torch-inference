#![allow(dead_code)]
/// CPU core-affinity utilities for pinning inference worker threads to
/// performance cores (P-cores).
///
/// # Motivation
///
/// Modern CPUs expose two kinds of cores:
/// - **P-cores** (performance): high clock speed, large caches.
/// - **E-cores** (efficiency): low power, but slower for ML workloads.
///
/// Pinning inference threads to P-cores avoids scheduler migration overhead
/// and ensures consistent SIMD latency.
///
/// # macOS (Apple Silicon)
///
/// On M-series chips the performance cores are typically indices 0-3 (M1/M2)
/// or 0-7 / 0-11 depending on the die.  Use [`CoreSet::apple_silicon_p_cores`]
/// for a sensible default.
///
/// # Linux
///
/// On Linux, use `/sys/devices/system/cpu/cpu*/topology/core_cpus_list` to
/// discover physical-core IDs.  [`CoreSet::from_range`] covers most servers.
use anyhow::{anyhow, Result};

// ── CoreSet ───────────────────────────────────────────────────────────────

/// A set of CPU core IDs to pin worker threads to.
#[derive(Debug, Clone)]
pub struct CoreSet {
    pub cores: Vec<usize>,
}

impl CoreSet {
    /// Explicit set of core IDs.
    pub fn new(cores: Vec<usize>) -> Self {
        Self { cores }
    }

    /// Contiguous range `[start, end)`.
    pub fn from_range(start: usize, end: usize) -> Self {
        Self {
            cores: (start..end).collect(),
        }
    }

    /// Apple Silicon P-cores (0-7 on M3 Max 16-core; adjust for other chips).
    ///
    /// Returns the first `n_p_cores` IDs starting at 0.
    pub fn apple_silicon_p_cores(n_p_cores: usize) -> Self {
        Self::from_range(0, n_p_cores)
    }

    /// All logical cores reported by the OS.
    pub fn all() -> Self {
        let n = num_cpus::get();
        Self::from_range(0, n)
    }

    /// How many cores are in this set.
    pub fn len(&self) -> usize {
        self.cores.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cores.is_empty()
    }

    /// Return the core at position `idx % len()`.  Safe even if `idx >= len`.
    pub fn core_for_worker(&self, worker_idx: usize) -> usize {
        if self.cores.is_empty() {
            return 0;
        }
        self.cores[worker_idx % self.cores.len()]
    }
}

// ── Affinity operations ───────────────────────────────────────────────────

/// Pin the **calling thread** to `core_id`.
///
/// Returns `Ok(true)` if pinning succeeded, `Ok(false)` if the core ID was
/// not found (harmless — the thread continues on any core), or `Err` for
/// unexpected errors.
pub fn pin_current_thread(core_id: usize) -> Result<bool> {
    let cores = core_affinity::get_core_ids()
        .ok_or_else(|| anyhow!("core_affinity: could not enumerate cores"))?;

    let target = cores.iter().find(|c| c.id == core_id);
    match target {
        Some(cid) => {
            let ok = core_affinity::set_for_current(*cid);
            Ok(ok)
        }
        None => Ok(false),
    }
}

/// Spawn `n` threads, pinning thread `i` to `core_set.core_for_worker(i)`.
///
/// Each thread runs `f(worker_idx)`.  Returns the `JoinHandle`s.
///
/// If affinity pinning fails (e.g. unsupported platform or missing permission),
/// the thread still runs — pinning failure is non-fatal.
pub fn spawn_affinity_workers<F>(
    n: usize,
    core_set: &CoreSet,
    f: F,
) -> Vec<std::thread::JoinHandle<()>>
where
    F: Fn(usize) + Send + Sync + Clone + 'static,
{
    (0..n)
        .map(|idx| {
            let core_id = core_set.core_for_worker(idx);
            let func = f.clone();
            std::thread::spawn(move || {
                let _ = pin_current_thread(core_id);
                func(idx);
            })
        })
        .collect()
}

/// Returns the core IDs visible to the current process.
pub fn available_cores() -> Vec<usize> {
    core_affinity::get_core_ids()
        .map(|ids| ids.iter().map(|c| c.id).collect())
        .unwrap_or_default()
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // ── CoreSet ───────────────────────────────────────────────────────────

    #[test]
    fn test_core_set_new() {
        let cs = CoreSet::new(vec![0, 2, 4]);
        assert_eq!(cs.len(), 3);
        assert_eq!(cs.cores, vec![0, 2, 4]);
    }

    #[test]
    fn test_core_set_from_range() {
        let cs = CoreSet::from_range(2, 6);
        assert_eq!(cs.cores, vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_core_set_all_nonempty() {
        let cs = CoreSet::all();
        assert!(cs.len() >= 1);
    }

    #[test]
    fn test_core_set_apple_silicon_p_cores() {
        let cs = CoreSet::apple_silicon_p_cores(4);
        assert_eq!(cs.cores, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_core_set_is_empty() {
        let cs = CoreSet::new(vec![]);
        assert!(cs.is_empty());
        let cs2 = CoreSet::new(vec![0]);
        assert!(!cs2.is_empty());
    }

    #[test]
    fn test_core_for_worker_wraps() {
        let cs = CoreSet::new(vec![0, 2, 4]);
        assert_eq!(cs.core_for_worker(0), 0);
        assert_eq!(cs.core_for_worker(1), 2);
        assert_eq!(cs.core_for_worker(2), 4);
        assert_eq!(cs.core_for_worker(3), 0); // wraps
        assert_eq!(cs.core_for_worker(4), 2);
    }

    #[test]
    fn test_core_for_worker_empty_returns_zero() {
        let cs = CoreSet::new(vec![]);
        assert_eq!(cs.core_for_worker(0), 0);
        assert_eq!(cs.core_for_worker(99), 0);
    }

    // ── available_cores ───────────────────────────────────────────────────

    #[test]
    fn test_available_cores_nonempty() {
        let cores = available_cores();
        assert!(!cores.is_empty(), "expected at least one core");
    }

    // ── pin_current_thread ────────────────────────────────────────────────

    #[test]
    fn test_pin_current_thread_valid_core() {
        let cores = available_cores();
        if let Some(&core_id) = cores.first() {
            // Should succeed or return Ok(false) — never Err.
            let result = pin_current_thread(core_id);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_pin_current_thread_invalid_core_returns_false() {
        // Core 99999 almost certainly does not exist.
        let result = pin_current_thread(99999);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    // ── spawn_affinity_workers ────────────────────────────────────────────

    #[test]
    fn test_spawn_affinity_workers_all_run() {
        let counter = Arc::new(Mutex::new(0usize));
        let cs = CoreSet::all();
        let n = 4;
        let c = Arc::clone(&counter);

        let handles = spawn_affinity_workers(n, &cs, move |_idx| {
            let mut g = c.lock().unwrap();
            *g += 1;
        });
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(*counter.lock().unwrap(), n);
    }

    #[test]
    fn test_spawn_affinity_workers_zero_threads() {
        let cs = CoreSet::all();
        let handles = spawn_affinity_workers(0, &cs, |_| {});
        assert!(handles.is_empty());
    }

    #[test]
    fn test_spawn_affinity_workers_receives_correct_idx() {
        let indices = Arc::new(Mutex::new(Vec::new()));
        let cs = CoreSet::from_range(0, 4);
        let idxs = Arc::clone(&indices);

        let handles = spawn_affinity_workers(4, &cs, move |idx| {
            idxs.lock().unwrap().push(idx);
        });
        for h in handles {
            h.join().unwrap();
        }
        let mut got = indices.lock().unwrap().clone();
        got.sort();
        assert_eq!(got, vec![0, 1, 2, 3]);
    }

    // ── CoreSet clone + debug ─────────────────────────────────────────────

    #[test]
    fn test_core_set_clone_and_debug() {
        let cs = CoreSet::new(vec![0, 1]);
        let cs2 = cs.clone();
        assert_eq!(cs2.cores, cs.cores);
        let s = format!("{:?}", cs);
        assert!(s.contains("CoreSet"));
    }
}
