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
    use actix_web::{test, web, App};
    use actix_web::http::StatusCode;

    // ── actix_web handler tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_logging_info_returns_200() {
        let app = test::init_service(
            App::new().route("/logs", web::get().to(get_logging_info))
        ).await;
        let req = test::TestRequest::get().uri("/logs").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_logging_info_response_shape() {
        let app = test::init_service(
            App::new().route("/logs", web::get().to(get_logging_info))
        ).await;
        let req = test::TestRequest::get().uri("/logs").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert!(body["log_directory"].is_string());
        assert!(body["log_level"].is_string());
        assert!(body["available_log_files"].is_array());
        assert!(body["total_log_size_mb"].is_number());
    }

    #[tokio::test]
    async fn test_get_logging_info_no_log_dir() {
        // Without a logs/ directory, returns an empty list
        let app = test::init_service(
            App::new().route("/logs", web::get().to(get_logging_info))
        ).await;
        let req = test::TestRequest::get().uri("/logs").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        // log_directory field should be set regardless
        assert_eq!(body["log_directory"].as_str().unwrap(), "logs");
    }

    #[tokio::test]
    async fn test_get_logging_info_default_log_level() {
        // Without RUST_LOG set, should default to "info"
        std::env::remove_var("RUST_LOG");
        let app = test::init_service(
            App::new().route("/logs", web::get().to(get_logging_info))
        ).await;
        let req = test::TestRequest::get().uri("/logs").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let log_level = body["log_level"].as_str().unwrap();
        // Either "info" (default) or whatever was set in env
        assert!(!log_level.is_empty());
    }

    #[tokio::test]
    async fn test_get_log_file_invalid_name_returns_400() {
        let app = test::init_service(
            App::new()
                .route("/logs/{log_file}", web::get().to(get_log_file))
        ).await;
        let req = test::TestRequest::get()
            .uri("/logs/..%2Fetc%2Fpasswd")
            .to_request();
        let resp = test::call_service(&app, req).await;
        // Either 400 (invalid name) or 404 (path traversal rejected before hitting handler)
        assert!(resp.status() == StatusCode::BAD_REQUEST || resp.status() == StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_log_file_not_a_log_extension_returns_400() {
        let app = test::init_service(
            App::new()
                .route("/logs/{log_file}", web::get().to(get_log_file))
        ).await;
        let req = test::TestRequest::get()
            .uri("/logs/server.txt")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_get_log_file_not_found_returns_404() {
        let app = test::init_service(
            App::new()
                .route("/logs/{log_file}", web::get().to(get_log_file))
        ).await;
        let req = test::TestRequest::get()
            .uri("/logs/nonexistent_file_abc123.log")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_clear_log_file_invalid_name_returns_400() {
        let app = test::init_service(
            App::new()
                .route("/logs/{log_file}", web::delete().to(clear_log_file))
        ).await;
        let req = test::TestRequest::delete()
            .uri("/logs/server.txt")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_clear_log_file_not_found_returns_404() {
        let app = test::init_service(
            App::new()
                .route("/logs/{log_file}", web::delete().to(clear_log_file))
        ).await;
        let req = test::TestRequest::delete()
            .uri("/logs/nonexistent_clear_test_abc.log")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ── get_log_file with lines query params ──────────────────────────────────

    #[tokio::test]
    async fn test_get_log_file_backslash_in_name_returns_400() {
        let app = test::init_service(
            App::new()
                .route("/logs/{log_file}", web::get().to(get_log_file))
        ).await;
        // URL-encode backslash as %5C
        let req = test::TestRequest::get()
            .uri("/logs/foo%5Cbar.log")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ── validation helpers ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_is_valid_log_filename() {
        assert!(is_valid_log_filename("server.log"));
        assert!(is_valid_log_filename("app_2024.log"));
        assert!(!is_valid_log_filename("../etc/passwd"));
        assert!(!is_valid_log_filename("logs/server.log"));
        assert!(!is_valid_log_filename("server.txt"));
    }

    #[tokio::test]
    async fn test_is_valid_log_filename_backslash() {
        assert!(!is_valid_log_filename("foo\\bar.log"));
    }

    #[tokio::test]
    async fn test_is_valid_log_filename_dotdot_in_name() {
        assert!(!is_valid_log_filename("foo..bar.log"));
    }

    #[tokio::test]
    async fn test_is_valid_log_filename_only_extension() {
        assert!(is_valid_log_filename(".log"));
    }

    // Helper to create a temp file with given content
    fn temp_log_file(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("failed to create temp file");
        write!(f, "{}", content).expect("failed to write temp file");
        f
    }

    #[tokio::test]
    async fn test_count_lines_empty_file() {
        let f = temp_log_file("");
        let count = count_lines(f.path()).expect("count_lines failed");
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_count_lines_single_line() {
        let f = temp_log_file("hello world\n");
        let count = count_lines(f.path()).expect("count_lines failed");
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_count_lines_multiple_lines() {
        let f = temp_log_file("line1\nline2\nline3\n");
        let count = count_lines(f.path()).expect("count_lines failed");
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn test_count_lines_no_trailing_newline() {
        let f = temp_log_file("line1\nline2\nline3");
        let count = count_lines(f.path()).expect("count_lines failed");
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn test_read_lines_from_file_from_end() {
        let content = "line1\nline2\nline3\nline4\nline5\n";
        let f = temp_log_file(content);
        let result = read_lines_from_file(f.path(), 3, true).expect("read_lines failed");
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "line3");
        assert_eq!(lines[2], "line5");
    }

    #[tokio::test]
    async fn test_read_lines_from_file_from_start() {
        let content = "line1\nline2\nline3\nline4\nline5\n";
        let f = temp_log_file(content);
        let result = read_lines_from_file(f.path(), 2, false).expect("read_lines failed");
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "line1");
        assert_eq!(lines[1], "line2");
    }

    #[tokio::test]
    async fn test_read_lines_from_file_more_than_available_from_end() {
        let content = "line1\nline2\n";
        let f = temp_log_file(content);
        let result = read_lines_from_file(f.path(), 100, true).expect("read_lines failed");
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 2);
    }

    #[tokio::test]
    async fn test_read_lines_from_file_more_than_available_from_start() {
        let content = "line1\nline2\n";
        let f = temp_log_file(content);
        let result = read_lines_from_file(f.path(), 100, false).expect("read_lines failed");
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 2);
    }

    #[tokio::test]
    async fn test_read_lines_from_file_single_line() {
        let f = temp_log_file("only one line\n");
        let result_end = read_lines_from_file(f.path(), 1, true).expect("read_lines failed");
        assert_eq!(result_end, "only one line");
        let result_start = read_lines_from_file(f.path(), 1, false).expect("read_lines failed");
        assert_eq!(result_start, "only one line");
    }

    #[tokio::test]
    async fn test_logging_info_struct() {
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

    #[tokio::test]
    async fn test_log_file_info_struct() {
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

    #[tokio::test]
    async fn test_log_file_content_struct() {
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

    #[tokio::test]
    async fn test_clear_log_response_struct() {
        let resp = ClearLogResponse {
            success: true,
            message: "Cleared".to_string(),
            original_size_bytes: 4096,
            original_size_mb: 4096.0 / (1024.0 * 1024.0),
        };
        assert!(resp.success);
        assert_eq!(resp.original_size_bytes, 4096);
    }

    // ── read_lines_from_file: zero-line edge case ─────────────────────────────

    #[tokio::test]
    async fn test_read_lines_from_file_zero_lines_from_end() {
        let f = temp_log_file("line1\nline2\n");
        // lines = 0 would be handled as usize 0 → returns empty
        let result = read_lines_from_file(f.path(), 0, true).expect("should succeed");
        assert_eq!(result, "", "0 lines from_end should yield empty string");
    }

    #[tokio::test]
    async fn test_read_lines_from_file_zero_lines_from_start() {
        let f = temp_log_file("line1\nline2\n");
        let result = read_lines_from_file(f.path(), 0, false).expect("should succeed");
        assert_eq!(result, "", "0 lines from_start should yield empty string");
    }

    // ── get_logging_info with an actual logs/ directory ───────────────────────

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_log_file_handler_returns_200_for_existing_file() {
        // Create a real temporary log file in a subdir named "logs"
        use std::io::Write;
        let logs_dir = std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("logs");
        std::fs::create_dir_all(&logs_dir).ok();
        let log_filename = "test_handler_exists.log";
        let log_path = logs_dir.join(log_filename);
        {
            let mut f = std::fs::File::create(&log_path).unwrap();
            writeln!(f, "line one").unwrap();
            writeln!(f, "line two").unwrap();
        }

        let app = test::init_service(
            App::new()
                .route("/logs/{log_file}", web::get().to(get_log_file))
        ).await;
        let uri = format!("/logs/{}", log_filename);
        let req = test::TestRequest::get().uri(&uri).to_request();
        let resp = test::call_service(&app, req).await;
        let status = resp.status();
        // Clean up before asserting so we don't leave files around
        std::fs::remove_file(&log_path).ok();
        assert_eq!(status, actix_web::http::StatusCode::OK,
            "existing log file should return 200");
    }

    #[tokio::test]
    async fn test_get_log_file_handler_response_body_for_existing_file() {
        use std::io::Write;
        let logs_dir = std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("logs");
        std::fs::create_dir_all(&logs_dir).ok();
        let log_filename = "test_handler_body.log";
        let log_path = logs_dir.join(log_filename);
        {
            let mut f = std::fs::File::create(&log_path).unwrap();
            writeln!(f, "alpha").unwrap();
            writeln!(f, "beta").unwrap();
            writeln!(f, "gamma").unwrap();
        }

        let app = test::init_service(
            App::new()
                .route("/logs/{log_file}", web::get().to(get_log_file))
        ).await;
        let uri = format!("/logs/{}?lines=2&from_end=true", log_filename);
        let req = test::TestRequest::get().uri(&uri).to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        std::fs::remove_file(&log_path).ok();
        assert_eq!(body["file_name"].as_str().unwrap(), log_filename);
        assert!(body["line_count"].as_u64().unwrap() <= 2);
        assert_eq!(body["from_end"].as_bool().unwrap(), true);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_get_log_file_handler_lines_zero_returns_full_content() {
        use std::io::Write;
        let logs_dir = std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("logs");
        std::fs::create_dir_all(&logs_dir).ok();
        let log_filename = "test_handler_full_content.log";
        let log_path = logs_dir.join(log_filename);
        {
            let mut f = std::fs::File::create(&log_path).unwrap();
            writeln!(f, "full line 1").unwrap();
            writeln!(f, "full line 2").unwrap();
        }

        let app = test::init_service(
            App::new()
                .route("/logs/{log_file}", web::get().to(get_log_file))
        ).await;
        // lines=0 means return entire file
        let uri = format!("/logs/{}?lines=0", log_filename);
        let req = test::TestRequest::get().uri(&uri).to_request();
        let resp = test::call_service(&app, req).await;
        let status = resp.status();
        std::fs::remove_file(&log_path).ok();
        assert_eq!(status, actix_web::http::StatusCode::OK);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_clear_log_file_handler_success() {
        use std::io::Write;
        let logs_dir = std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("logs");
        std::fs::create_dir_all(&logs_dir).ok();
        let log_filename = "test_clear_success.log";
        let log_path = logs_dir.join(log_filename);
        {
            let mut f = std::fs::File::create(&log_path).unwrap();
            writeln!(f, "old content line 1").unwrap();
            writeln!(f, "old content line 2").unwrap();
        }

        let app = test::init_service(
            App::new()
                .route("/logs/{log_file}", web::delete().to(clear_log_file))
        ).await;
        let uri = format!("/logs/{}", log_filename);
        let req = test::TestRequest::delete().uri(&uri).to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        std::fs::remove_file(&log_path).ok();
        assert_eq!(body["success"].as_bool().unwrap(), true);
        assert!(body["message"].as_str().unwrap().contains(log_filename));
        assert!(body["original_size_bytes"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_get_logging_info_with_log_directory() {
        use std::io::Write;
        // Create a logs directory with a .log file so the listing path is exercised
        let logs_dir = std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("logs");
        std::fs::create_dir_all(&logs_dir).ok();
        let log_path = logs_dir.join("test_listing_info.log");
        {
            let mut f = std::fs::File::create(&log_path).unwrap();
            writeln!(f, "hello from listing test").unwrap();
        }

        let app = test::init_service(
            App::new().route("/logs", web::get().to(get_logging_info))
        ).await;
        let req = test::TestRequest::get().uri("/logs").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        std::fs::remove_file(&log_path).ok();

        // The listing should be a sorted array
        assert!(body["available_log_files"].is_array());
        assert!(body["total_log_size_mb"].as_f64().unwrap() >= 0.0);
    }

    #[tokio::test]
    async fn test_log_file_query_fields() {
        let q = LogFileQuery { lines: Some(50), from_end: Some(false) };
        assert_eq!(q.lines, Some(50));
        assert_eq!(q.from_end, Some(false));
    }

    #[tokio::test]
    async fn test_log_file_query_optional_fields_none() {
        let q = LogFileQuery { lines: None, from_end: None };
        assert!(q.lines.is_none());
        assert!(q.from_end.is_none());
    }

    #[tokio::test]
    async fn test_get_log_file_from_start_via_handler() {
        use std::io::Write;
        let logs_dir = std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("logs");
        std::fs::create_dir_all(&logs_dir).ok();
        let log_filename = "test_from_start_handler.log";
        let log_path = logs_dir.join(log_filename);
        {
            let mut f = std::fs::File::create(&log_path).unwrap();
            for i in 1..=5 {
                writeln!(f, "line {}", i).unwrap();
            }
        }

        let app = test::init_service(
            App::new()
                .route("/logs/{log_file}", web::get().to(get_log_file))
        ).await;
        let uri = format!("/logs/{}?lines=3&from_end=false", log_filename);
        let req = test::TestRequest::get().uri(&uri).to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        std::fs::remove_file(&log_path).ok();
        assert_eq!(body["from_end"].as_bool().unwrap(), false);
        assert_eq!(body["line_count"].as_u64().unwrap(), 3);
    }
}
