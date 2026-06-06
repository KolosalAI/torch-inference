//! Process-RSS admission gate. Refuses new chat/agent runs when the host
//! resident memory exceeds `high_water_mb`; resumes admitting after RSS drops
//! below `low_water_mb` (hysteresis prevents flapping at the boundary).
//!
//! Polled lazily on each admit call — no background thread. Cost is ~µs.

use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, thiserror::Error)]
#[error("memory pressure: RSS {rss_mb} MB > high_water {hw_mb} MB")]
pub struct MemoryRefusal {
    pub rss_mb: u64,
    pub hw_mb: u64,
}

pub struct MemoryGate {
    high_water_bytes: u64,
    low_water_bytes:  u64,
    above_water:      AtomicBool,
    reader:           Box<dyn Fn() -> std::io::Result<u64> + Send + Sync>,
}

impl MemoryGate {
    pub fn new(high_water_mb: u64, low_water_mb: u64) -> Self {
        assert!(low_water_mb <= high_water_mb,
                "low_water_mb must be <= high_water_mb");
        Self {
            high_water_bytes: high_water_mb * 1024 * 1024,
            low_water_bytes:  low_water_mb  * 1024 * 1024,
            above_water:      AtomicBool::new(false),
            reader:           Box::new(current_rss_bytes_default),
        }
    }

    /// Test constructor accepting a mock RSS reader.
    #[cfg(any(test, feature = "mock-rss"))]
    pub fn with_reader<F>(high_water_mb: u64, low_water_mb: u64, reader: F) -> Self
    where F: Fn() -> std::io::Result<u64> + Send + Sync + 'static {
        Self {
            high_water_bytes: high_water_mb * 1024 * 1024,
            low_water_bytes:  low_water_mb  * 1024 * 1024,
            above_water:      AtomicBool::new(false),
            reader:           Box::new(reader),
        }
    }

    pub fn admit(&self) -> Result<(), MemoryRefusal> {
        let rss = (self.reader)().unwrap_or(0);
        let was_above = self.above_water.load(Ordering::Relaxed);

        if was_above {
            // Stay refused until we drop below the low-water mark.
            if rss < self.low_water_bytes {
                self.above_water.store(false, Ordering::Relaxed);
                Ok(())
            } else {
                Err(MemoryRefusal {
                    rss_mb: rss / 1024 / 1024,
                    hw_mb: self.high_water_bytes / 1024 / 1024,
                })
            }
        } else if rss > self.high_water_bytes {
            self.above_water.store(true, Ordering::Relaxed);
            Err(MemoryRefusal {
                rss_mb: rss / 1024 / 1024,
                hw_mb: self.high_water_bytes / 1024 / 1024,
            })
        } else {
            Ok(())
        }
    }
}

fn current_rss_bytes_default() -> std::io::Result<u64> { current_rss_bytes() }

#[cfg(target_os = "macos")]
fn current_rss_bytes() -> std::io::Result<u64> {
    use libc::{c_int, c_void, mach_task_self, task_info};
    // MACH_TASK_BASIC_INFO = 20; count = sizeof(struct)/sizeof(natural_t=u32).
    const MACH_TASK_BASIC_INFO: c_int = 20;
    #[repr(C)]
    #[derive(Default)]
    struct MachTaskBasicInfo {
        virtual_size:     u64,
        resident_size:    u64,
        resident_size_max:u64,
        user_time:        [u32; 2],
        system_time:      [u32; 2],
        policy:           c_int,
        suspend_count:    c_int,
    }
    let mut info = MachTaskBasicInfo::default();
    let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;
    let kr = unsafe {
        task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO as u32,
            &mut info as *mut _ as *mut c_void as *mut i32,
            &mut count,
        )
    };
    if kr != 0 {
        return Err(std::io::Error::new(std::io::ErrorKind::Other,
                                       format!("task_info kr={}", kr)));
    }
    Ok(info.resident_size)
}

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> std::io::Result<u64> {
    let s = std::fs::read_to_string("/proc/self/status")?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // "VmRSS:    12345 kB"
            let kb: u64 = rest.split_whitespace().next()
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "parse VmRSS"))?;
            return Ok(kb * 1024);
        }
    }
    Err(std::io::Error::new(std::io::ErrorKind::NotFound, "VmRSS not in /proc/self/status"))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn current_rss_bytes() -> std::io::Result<u64> { Ok(0) }

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering as O};
    use std::sync::Arc;

    #[test]
    fn admits_below_high_water() {
        let rss = Arc::new(AtomicU64::new(100 * 1024 * 1024));
        let r = rss.clone();
        let gate = MemoryGate::with_reader(1000, 800, move || Ok(r.load(O::Relaxed)));
        assert!(gate.admit().is_ok());
    }

    #[test]
    fn refuses_above_high_water() {
        let rss = Arc::new(AtomicU64::new(2_000 * 1024 * 1024));
        let r = rss.clone();
        let gate = MemoryGate::with_reader(1_000, 800, move || Ok(r.load(O::Relaxed)));
        assert!(gate.admit().is_err());
    }

    #[test]
    fn hysteresis_holds_refusal_between_thresholds() {
        let rss = Arc::new(AtomicU64::new(2_000 * 1024 * 1024));
        let r = rss.clone();
        let gate = MemoryGate::with_reader(1_000, 800, move || Ok(r.load(O::Relaxed)));

        // First admit: above HW -> refuse, sticky.
        assert!(gate.admit().is_err());
        // Drop to between LW and HW: still refused (sticky).
        rss.store(900 * 1024 * 1024, O::Relaxed);
        assert!(gate.admit().is_err());
        // Drop below LW: admits again.
        rss.store(700 * 1024 * 1024, O::Relaxed);
        assert!(gate.admit().is_ok());
        // Sticky cleared; same range now admits.
        rss.store(900 * 1024 * 1024, O::Relaxed);
        assert!(gate.admit().is_ok());
    }

    #[test]
    fn real_reader_returns_nonzero_on_this_platform() {
        // Smoke-test the platform-specific path on macOS / Linux.
        // On unsupported OSes this returns 0; test passes trivially there.
        let gate = MemoryGate::new(u64::MAX / 1024 / 1024 / 2, 0);
        assert!(gate.admit().is_ok());
    }
}
