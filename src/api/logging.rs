/// Logging management for the inference server
use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::fs;
use std::io::{BufRead, BufReader};
use chrono::Utc;
use crate::error::ApiError;

#[derive(Debug, Serialize)]
pub struct LoggingInfo {
    pub log_directory: String,
    pub log_level: String,
    pub available_log_files: Vec<LogFileInfo>,
    pub total_log_size_mb: f64,
}

#[derive(Debug, Serialize)]
pub struct LogFileInfo {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub size_mb: f64,
    pub line_count: usize,
    pub modified: String,
}

#[derive(Debug, Serialize)]
pub struct LogFileContent {
    pub file_name: String,
    pub content: String,
    pub line_count: usize,
    pub total_lines: usize,
    pub from_end: bool,
}

#[derive(Debug, Serialize)]
pub struct ClearLogResponse {
    pub success: bool,
    pub message: String,
    pub original_size_bytes: u64,
    pub original_size_mb: f64,
}

/// Get logging information and statistics
pub async fn get_logging_info() -> Result<HttpResponse, ApiError> {
    log::info!("[ENDPOINT] Logging information requested");

    let log_dir = PathBuf::from("logs");
    let log_level = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());

    let mut log_info = LoggingInfo {
        log_directory: log_dir.display().to_string(),
        log_level,
        available_log_files: Vec::new(),
        total_log_size_mb: 0.0,
    };

    if log_dir.exists() && log_dir.is_dir() {
        let mut total_size: u64 = 0;

        if let Ok(entries) = fs::read_dir(&log_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                
                if path.extension().and_then(|s| s.to_str()) == Some("log") {
                    if let Ok(metadata) = fs::metadata(&path) {
                        let size = metadata.len();
                        total_size += size;

                        // Count lines
                        let line_count = count_lines(&path).unwrap_or(0);

                        // Get modified time
                        let modified = metadata.modified()
                            .ok()
                            .and_then(|t| chrono::DateTime::<Utc>::from(t).format("%Y-%m-%d %H:%M:%S").to_string().into())
                            .unwrap_or_else(|| "Unknown".to_string());

                        log_info.available_log_files.push(LogFileInfo {
                            name: path.file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("unknown")
                                .to_string(),
                            path: path.display().to_string(),
                            size_bytes: size,
                            size_mb: size as f64 / (1024.0 * 1024.0),
                            line_count,
                            modified,
                        });
                    }
                }
            }
        }

        log_info.total_log_size_mb = total_size as f64 / (1024.0 * 1024.0);
    }

    // Sort by name
    log_info.available_log_files.sort_by(|a, b| a.name.cmp(&b.name));

    log::info!("[ENDPOINT] Found {} log files", log_info.available_log_files.len());

    Ok(HttpResponse::Ok().json(log_info))
}

/// Get specific log file content
pub async fn get_log_file(
    log_file: web::Path<String>,
    query: web::Query<LogFileQuery>,
) -> Result<HttpResponse, ApiError> {
    let log_file = log_file.into_inner();
    let lines = query.lines.unwrap_or(100);
    let from_end = query.from_end.unwrap_or(true);

    log::info!("[ENDPOINT] Log file requested: {}, lines: {}, from_end: {}", 
        log_file, lines, from_end);

    // Validate log file name to prevent directory traversal
    if !is_valid_log_filename(&log_file) {
        return Err(ApiError::BadRequest("Invalid log file name".to_string()));
    }

    let log_path = PathBuf::from("logs").join(&log_file);

    if !log_path.exists() {
        return Err(ApiError::NotFound(format!("Log file not found: {}", log_file)));
    }

    let content = if lines <= 0 {
        // Return entire file
        fs::read_to_string(&log_path)
            .map_err(|e| ApiError::InternalError(format!("Failed to read log file: {}", e)))?
    } else {
        // Read specific number of lines
        read_lines_from_file(&log_path, lines, from_end)?
    };

    let total_lines = count_lines(&log_path).unwrap_or(0);
    let line_count = content.lines().count();

    log::info!("[ENDPOINT] Log file {} retrieved successfully", log_file);

    Ok(HttpResponse::Ok().json(LogFileContent {
        file_name: log_file,
        content,
        line_count,
        total_lines,
        from_end,
    }))
}

/// Clear specific log file
pub async fn clear_log_file(
    log_file: web::Path<String>,
) -> Result<HttpResponse, ApiError> {
    let log_file = log_file.into_inner();
    
    log::info!("[ENDPOINT] Log file clear requested: {}", log_file);

    // Validate log file name
    if !is_valid_log_filename(&log_file) {
        return Err(ApiError::BadRequest("Invalid log file name".to_string()));
    }

    let log_path = PathBuf::from("logs").join(&log_file);

    if !log_path.exists() {
        return Err(ApiError::NotFound(format!("Log file not found: {}", log_file)));
    }

    // Get file size before clearing
    let original_size = fs::metadata(&log_path)
        .map_err(|e| ApiError::InternalError(format!("Failed to get file metadata: {}", e)))?
        .len();

    // Clear the file
    let clear_message = format!("# Log file cleared at {}\n", Utc::now().format("%Y-%m-%d %H:%M:%S"));
    fs::write(&log_path, clear_message)
        .map_err(|e| ApiError::InternalError(format!("Failed to clear log file: {}", e)))?;

    log::info!("[ENDPOINT] Log file {} cleared successfully (was {} bytes)", 
        log_file, original_size);

    Ok(HttpResponse::Ok().json(ClearLogResponse {
        success: true,
        message: format!("Log file {} cleared successfully", log_file),
        original_size_bytes: original_size,
        original_size_mb: original_size as f64 / (1024.0 * 1024.0),
    }))
}

#[derive(Debug, Deserialize)]
pub struct LogFileQuery {
    pub lines: Option<i32>,
    pub from_end: Option<bool>,
}

/// Validate log filename to prevent directory traversal
fn is_valid_log_filename(filename: &str) -> bool {
    filename.ends_with(".log") 
        && !filename.contains('/') 
        && !filename.contains('\\')
        && !filename.contains("..")
}

/// Count lines in a file
fn count_lines(path: &Path) -> std::io::Result<usize> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    Ok(reader.lines().count())
}

/// Read specific number of lines from file
fn read_lines_from_file(path: &Path, lines: i32, from_end: bool) -> Result<String, ApiError> {
    let file = fs::File::open(path)
        .map_err(|e| ApiError::InternalError(format!("Failed to open file: {}", e)))?;
    
    let reader = BufReader::new(file);
    let all_lines: Vec<String> = reader.lines()
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|e| ApiError::InternalError(format!("Failed to read lines: {}", e)))?;

    let lines = lines as usize;
    let selected_lines = if from_end {
        // Get last N lines
        if all_lines.len() > lines {
            &all_lines[all_lines.len() - lines..]
        } else {
            &all_lines[..]
        }
    } else {
        // Get first N lines
        if all_lines.len() > lines {
            &all_lines[..lines]
        } else {
            &all_lines[..]
        }
    };

    Ok(selected_lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_log_filename() {
        assert!(is_valid_log_filename("server.log"));
        assert!(is_valid_log_filename("app_2024.log"));
        assert!(!is_valid_log_filename("../etc/passwd"));
        assert!(!is_valid_log_filename("logs/server.log"));
        assert!(!is_valid_log_filename("server.txt"));
    }
}
