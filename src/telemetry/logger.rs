use log::{info, LevelFilter};
use chrono::Local;
use std::io::Write;

/// Setup enhanced logging with colors and better formatting
pub fn setup_logging() {
    let log_level = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| "info".to_string())
        .parse::<LevelFilter>()
        .unwrap_or(LevelFilter::Info);
    
    env_logger::builder()
        .format(|buf, record| {
            let level_string = match record.level() {
                log::Level::Error => "\x1b[1;31mERROR\x1b[0m",  // Bold red
                log::Level::Warn  => "\x1b[1;33mWARN \x1b[0m",  // Bold yellow
                log::Level::Info  => "\x1b[1;32mINFO \x1b[0m",  // Bold green
                log::Level::Debug => "\x1b[1;36mDEBUG\x1b[0m",  // Bold cyan
                log::Level::Trace => "\x1b[1;35mTRACE\x1b[0m",  // Bold magenta
            };
            
            let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
            let target = record.target();
            
            // Shorten target for readability
            let short_target = if target.starts_with("torch_inference::") {
                target.strip_prefix("torch_inference::").unwrap_or(target)
            } else {
                target
            };
            
            writeln!(
                buf,
                "\x1b[90m{}\x1b[0m [{}] \x1b[90m{}\x1b[0m - {}",
                timestamp,
                level_string,
                short_target,
                record.args()
            )
        })
        .filter_level(log_level)
        .try_init()
        .ok();
    
    info!("[START] Logging system initialized (level: {})", log_level);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Calling setup_logging multiple times must not panic.
    /// env_logger uses try_init() with .ok(), so duplicate calls are silently
    /// swallowed.
    #[test]
    fn test_setup_logging_does_not_panic() {
        // First call – may or may not succeed depending on test ordering.
        setup_logging();
        // Second call – must also be a no-op, not a panic.
        setup_logging();
    }

    /// Verify that an explicit RUST_LOG value is honoured without panicking.
    #[test]
    fn test_setup_logging_respects_rust_log_env() {
        std::env::set_var("RUST_LOG", "debug");
        setup_logging();
        // Reset so other tests are unaffected.
        std::env::remove_var("RUST_LOG");
    }

    /// Verify that an invalid RUST_LOG value falls back to Info without panicking.
    #[test]
    fn test_setup_logging_invalid_rust_log_fallback() {
        std::env::set_var("RUST_LOG", "not_a_valid_level");
        setup_logging();
        std::env::remove_var("RUST_LOG");
    }

    /// Verify that RUST_LOG=trace does not panic.
    #[test]
    fn test_setup_logging_trace_level() {
        std::env::set_var("RUST_LOG", "trace");
        setup_logging();
        std::env::remove_var("RUST_LOG");
    }

    /// Verify that RUST_LOG=warn does not panic.
    #[test]
    fn test_setup_logging_warn_level() {
        std::env::set_var("RUST_LOG", "warn");
        setup_logging();
        std::env::remove_var("RUST_LOG");
    }

    /// Verify that RUST_LOG=error does not panic.
    #[test]
    fn test_setup_logging_error_level() {
        std::env::set_var("RUST_LOG", "error");
        setup_logging();
        std::env::remove_var("RUST_LOG");
    }

    /// Verify that RUST_LOG=off does not panic.
    #[test]
    fn test_setup_logging_off_level() {
        std::env::set_var("RUST_LOG", "off");
        setup_logging();
        std::env::remove_var("RUST_LOG");
    }
}
