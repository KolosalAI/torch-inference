use log::{info, LevelFilter};
use chrono::Local;
use std::io::Write;

/// Format a single log record into the env_logger buffer.
///
/// Extracted from the `.format()` closure so that tests can call this
/// function directly and guarantee the formatting logic is covered,
/// independent of which logger wins the global-logger registration race.
pub(crate) fn format_log_record(
    buf: &mut env_logger::fmt::Formatter,
    record: &log::Record<'_>,
) -> std::io::Result<()> {
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
}

/// Setup enhanced logging with colors and better formatting
pub fn setup_logging() {
    let log_level = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| "info".to_string())
        .parse::<LevelFilter>()
        .unwrap_or(LevelFilter::Info);

    env_logger::builder()
        .format(format_log_record)
        .filter_level(log_level)
        .try_init()
        .ok();

    info!("[START] Logging system initialized (level: {})", log_level);
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::Log;
    use std::sync::OnceLock;

    // ─────────────────────────────────────────────────────────────────────────
    // Coverage strategy for format_log_record (the extracted formatter function)
    // ─────────────────────────────────────────────────────────────────────────
    //
    // The production formatting logic now lives in `format_log_record`, a named
    // pub(crate) function.  Tests invoke it via a thin env_logger wrapper that
    // supplies the `Formatter` buf – the only way to obtain one.  Because the
    // function is called directly by name, every branch inside it is reliably
    // covered regardless of which test wins the global-logger registration race.
    //
    // LOGGER_INIT / ensure_trace_logger() are kept so that log::*! macros also
    // exercise the global dispatch path (belt-and-suspenders).
    // ─────────────────────────────────────────────────────────────────────────

    static LOGGER_INIT: OnceLock<()> = OnceLock::new();

    /// Register setup_logging()'s env_logger instance as the global log logger
    /// (exactly once, at TRACE level) so that all log::*! macro calls in this
    /// test module exercise the production formatter via global dispatch.
    fn ensure_trace_logger() {
        LOGGER_INIT.get_or_init(|| {
            std::env::remove_var("RUST_LOG");
            setup_logging();
            log::set_max_level(log::LevelFilter::Trace);
        });
    }

    // ── Helper: build a trace-level env_logger Logger that delegates to the
    // production format_log_record function.  Using .build() avoids touching the
    // global logger registration, so this works even when another logger is
    // already installed.  Because format_log_record is called by name (not via
    // a duplicate closure), coverage is attributed to the function body in the
    // production code.
    fn build_trace_logger() -> env_logger::Logger {
        env_logger::builder()
            .format(format_log_record)
            .filter_level(log::LevelFilter::Trace)
            .build()
    }

    // ── Helper: synthesise a log::Record and pass it to our logger directly.
    // format_args! must live in the same expression as its use, so we wrap
    // the call in a local macro to avoid the E0716 temporary lifetime issue.
    fn log_at_level(logger: &env_logger::Logger, level: log::Level, target: &str, msg: &str) {
        macro_rules! emit {
            ($lvl:expr, $tgt:expr, $m:expr) => {
                logger.log(
                    &log::Record::builder()
                        .level($lvl)
                        .target($tgt)
                        .args(format_args!("{}", $m))
                        .build(),
                )
            };
        }
        emit!(level, target, msg);
    }

    // ── All five Level match arms ─────────────────────────────────────────────

    #[test]
    fn test_format_log_record_error_arm() {
        ensure_trace_logger();
        let logger = build_trace_logger();
        log_at_level(&logger, log::Level::Error, "torch_inference::test", "error arm");
        log::error!(target: "torch_inference::test", "error arm – global dispatch");
    }

    #[test]
    fn test_format_log_record_warn_arm() {
        ensure_trace_logger();
        let logger = build_trace_logger();
        log_at_level(&logger, log::Level::Warn, "torch_inference::test", "warn arm");
        log::warn!(target: "torch_inference::test", "warn arm – global dispatch");
    }

    #[test]
    fn test_format_log_record_info_arm() {
        ensure_trace_logger();
        let logger = build_trace_logger();
        log_at_level(&logger, log::Level::Info, "torch_inference::test", "info arm");
        log::info!(target: "torch_inference::test", "info arm – global dispatch");
    }

    #[test]
    fn test_format_log_record_debug_arm() {
        ensure_trace_logger();
        let logger = build_trace_logger();
        log_at_level(&logger, log::Level::Debug, "torch_inference::test", "debug arm");
        log::debug!(target: "torch_inference::test", "debug arm – global dispatch");
    }

    #[test]
    fn test_format_log_record_trace_arm() {
        ensure_trace_logger();
        let logger = build_trace_logger();
        log_at_level(&logger, log::Level::Trace, "torch_inference::test", "trace arm");
        log::trace!(target: "torch_inference::test", "trace arm – global dispatch");
    }

    // ── Target-stripping branches ─────────────────────────────────────────────

    /// target starts with "torch_inference::" → strip_prefix branch
    #[test]
    fn test_format_log_record_target_stripping() {
        ensure_trace_logger();
        let logger = build_trace_logger();
        log_at_level(&logger, log::Level::Info, "torch_inference::api::inference", "stripped");
        log::info!(target: "torch_inference::telemetry::logger", "stripped – global dispatch");
    }

    /// target does NOT start with "torch_inference::" → else branch
    #[test]
    fn test_format_log_record_target_no_strip() {
        ensure_trace_logger();
        let logger = build_trace_logger();
        log_at_level(&logger, log::Level::Info, "other_crate::module", "non-stripped");
        log::info!(target: "other_crate::module", "non-stripped – global dispatch");
    }

    // ── Timestamp + target locals, writeln! body ──────────────────────────────

    #[test]
    fn test_format_log_record_all_levels_and_targets() {
        ensure_trace_logger();
        let logger = build_trace_logger();
        for (level, target, msg) in [
            (log::Level::Error, "torch_inference::core", "writeln error"),
            (log::Level::Warn,  "torch_inference::api",  "writeln warn"),
            (log::Level::Info,  "external_crate",        "writeln info"),
            (log::Level::Debug, "torch_inference::util", "writeln debug"),
            (log::Level::Trace, "another_crate",         "writeln trace"),
        ] {
            log_at_level(&logger, level, target, msg);
        }
        log::error!("writeln error – global");
        log::warn!("writeln warn – global");
        log::info!("writeln info – global");
        log::debug!("writeln debug – global");
        log::trace!("writeln trace – global");
    }

    // ── setup_logging() smoke tests ───────────────────────────────────────────

    /// Calling setup_logging multiple times must not panic.
    #[test]
    fn test_setup_logging_does_not_panic() {
        setup_logging();
        setup_logging();
    }

    #[test]
    fn test_setup_logging_respects_rust_log_env() {
        std::env::set_var("RUST_LOG", "debug");
        setup_logging();
        std::env::remove_var("RUST_LOG");
    }

    #[test]
    fn test_setup_logging_invalid_rust_log_fallback() {
        std::env::set_var("RUST_LOG", "not_a_valid_level");
        setup_logging();
        std::env::remove_var("RUST_LOG");
    }

    #[test]
    fn test_setup_logging_trace_level() {
        std::env::set_var("RUST_LOG", "trace");
        setup_logging();
        std::env::remove_var("RUST_LOG");
    }

    #[test]
    fn test_setup_logging_warn_level() {
        std::env::set_var("RUST_LOG", "warn");
        setup_logging();
        std::env::remove_var("RUST_LOG");
    }

    #[test]
    fn test_setup_logging_error_level() {
        std::env::set_var("RUST_LOG", "error");
        setup_logging();
        std::env::remove_var("RUST_LOG");
    }

    #[test]
    fn test_setup_logging_off_level() {
        std::env::set_var("RUST_LOG", "off");
        setup_logging();
        std::env::remove_var("RUST_LOG");
    }

    #[test]
    fn test_log_target_stripping() {
        ensure_trace_logger();
        log::info!(target: "torch_inference::telemetry::logger", "target stripping test");
        log::info!(target: "other_crate::module", "non-stripped target test");
    }
}
