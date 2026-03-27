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
    use std::io::Write;

    #[test]
    fn test_is_valid_log_filename() {
        assert!(is_valid_log_filename("server.log"));
        assert!(is_valid_log_filename("app_2024.log"));
        assert!(!is_valid_log_filename("../etc/passwd"));
        assert!(!is_valid_log_filename("logs/server.log"));
        assert!(!is_valid_log_filename("server.txt"));
    }

    #[test]
    fn test_is_valid_log_filename_backslash() {
        assert!(!is_valid_log_filename("foo\\bar.log"));
    }

    #[test]
    fn test_is_valid_log_filename_dotdot_in_name() {
        assert!(!is_valid_log_filename("foo..bar.log"));
    }

    #[test]
    fn test_is_valid_log_filename_only_extension() {
        assert!(is_valid_log_filename(".log"));
    }

    // Helper to create a temp file with given content
    fn temp_log_file(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("failed to create temp file");
        write!(f, "{}", content).expect("failed to write temp file");
        f
    }

    #[test]
    fn test_count_lines_empty_file() {
        let f = temp_log_file("");
        let count = count_lines(f.path()).expect("count_lines failed");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_lines_single_line() {
        let f = temp_log_file("hello world\n");
        let count = count_lines(f.path()).expect("count_lines failed");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_count_lines_multiple_lines() {
        let f = temp_log_file("line1\nline2\nline3\n");
        let count = count_lines(f.path()).expect("count_lines failed");
        assert_eq!(count, 3);
    }

    #[test]
    fn test_count_lines_no_trailing_newline() {
        let f = temp_log_file("line1\nline2\nline3");
        let count = count_lines(f.path()).expect("count_lines failed");
        assert_eq!(count, 3);
    }

    #[test]
    fn test_read_lines_from_file_from_end() {
        let content = "line1\nline2\nline3\nline4\nline5\n";
        let f = temp_log_file(content);
        let result = read_lines_from_file(f.path(), 3, true).expect("read_lines failed");
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "line3");
        assert_eq!(lines[2], "line5");
    }

    #[test]
    fn test_read_lines_from_file_from_start() {
        let content = "line1\nline2\nline3\nline4\nline5\n";
        let f = temp_log_file(content);
        let result = read_lines_from_file(f.path(), 2, false).expect("read_lines failed");
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "line1");
        assert_eq!(lines[1], "line2");
    }

    #[test]
    fn test_read_lines_from_file_more_than_available_from_end() {
        let content = "line1\nline2\n";
        let f = temp_log_file(content);
        let result = read_lines_from_file(f.path(), 100, true).expect("read_lines failed");
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_read_lines_from_file_more_than_available_from_start() {
        let content = "line1\nline2\n";
        let f = temp_log_file(content);
        let result = read_lines_from_file(f.path(), 100, false).expect("read_lines failed");
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_read_lines_from_file_single_line() {
        let f = temp_log_file("only one line\n");
        let result_end = read_lines_from_file(f.path(), 1, true).expect("read_lines failed");
        assert_eq!(result_end, "only one line");
        let result_start = read_lines_from_file(f.path(), 1, false).expect("read_lines failed");
        assert_eq!(result_start, "only one line");
    }

    #[test]
    fn test_logging_info_struct() {
        let info = LoggingInfo {
            log_directory: "logs".to_string(),
            log_level: "info".to_string(),
            available_log_files: vec![],
            total_log_size_mb: 0.0,
        };
        assert_eq!(info.log_directory, "logs");
        assert_eq!(info.log_level, "info");
        assert!(info.available_log_files.is_empty());
    }

    #[test]
    fn test_log_file_info_struct() {
        let info = LogFileInfo {
            name: "server.log".to_string(),
            path: "logs/server.log".to_string(),
            size_bytes: 2048,
            size_mb: 2048.0 / (1024.0 * 1024.0),
            line_count: 42,
            modified: "2024-01-01 00:00:00".to_string(),
        };
        assert_eq!(info.name, "server.log");
        assert_eq!(info.size_bytes, 2048);
        assert_eq!(info.line_count, 42);
    }

    #[test]
    fn test_log_file_content_struct() {
        let content = LogFileContent {
            file_name: "app.log".to_string(),
            content: "log line 1\nlog line 2".to_string(),
            line_count: 2,
            total_lines: 10,
            from_end: true,
        };
        assert_eq!(content.file_name, "app.log");
        assert_eq!(content.line_count, 2);
        assert!(content.from_end);
    }

    #[test]
    fn test_clear_log_response_struct() {
        let resp = ClearLogResponse {
            success: true,
            message: "Cleared".to_string(),
            original_size_bytes: 4096,
            original_size_mb: 4096.0 / (1024.0 * 1024.0),
        };
        assert!(resp.success);
        assert_eq!(resp.original_size_bytes, 4096);
    }
}
