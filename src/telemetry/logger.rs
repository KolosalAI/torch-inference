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
    
    info!("🚀 Logging system initialized (level: {})", log_level);
}
