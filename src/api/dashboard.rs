// src/api/dashboard.rs
use actix_web::web::Bytes;
use actix_web::{web, HttpResponse};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tokio_stream::wrappers::ReceiverStream;

use crate::api::model_download::ModelDownloadState;
use crate::api::system::SystemInfoState;
use crate::monitor::Monitor;
use tracing::warn;

#[derive(Debug, Serialize)]
pub struct DashboardMetrics {
    pub uptime_s: u64,
    pub active_req: u64,
    pub total_req: u64,
    pub avg_latency_ms: f64,
    pub error_rate: f64,
    pub throughput_per_s: f64,
    pub cpu_pct: f32,
    pub mem_used_mb: u64,
    pub mem_total_mb: u64,
}

#[derive(Debug, Serialize)]
pub struct DashboardGpu {
    pub name: String,
    pub util_pct: Option<u32>,
    pub temp_c: Option<u32>,
    pub vram_free_mb: u64,
    pub vram_total_mb: u64,
}

#[derive(Debug, Serialize)]
pub struct DashboardDownload {
    pub id: String,
    pub model_name: String,
    pub status: String,
    pub progress: f32,
    pub downloaded_mb: u64,
    pub total_mb: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct DashboardEvent {
    pub metrics: DashboardMetrics,
    pub gpu: Vec<DashboardGpu>,
    pub downloads: Vec<DashboardDownload>,
}

pub async fn dashboard_stream(
    monitor: web::Data<Arc<Monitor>>,
    system_state: web::Data<SystemInfoState>,
    download_state: web::Data<ModelDownloadState>,
) -> HttpResponse {
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);

    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(3));
        // Construct System once; only refresh on each tick.
        let mut sys = sysinfo::System::new_all();
        loop {
            ticker.tick().await;

            // Metrics from Monitor
            let m = monitor.get_metrics();
            let h = monitor.get_health_status();
            let error_rate = if m.total_requests > 0 {
                m.total_errors as f64 / m.total_requests as f64
            } else {
                0.0
            };

            // CPU / RAM from sysinfo
            sys.refresh_cpu_all();
            sys.refresh_memory();
            let cpu_pct = sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
                / sys.cpus().len().max(1) as f32;
            let mem_used_mb = sys.used_memory() / 1024 / 1024;
            let mem_total_mb = sys.total_memory() / 1024 / 1024;

            // GPU from GpuManager
            let gpu = match system_state.gpu_manager.get_info() {
                Ok(info) => info
                    .devices
                    .into_iter()
                    .map(|d| DashboardGpu {
                        name: d.name,
                        util_pct: d.utilization,
                        temp_c: d.temperature,
                        vram_free_mb: d.free_memory / 1024 / 1024,
                        vram_total_mb: d.total_memory / 1024 / 1024,
                    })
                    .collect(),
                Err(_) => vec![],
            };

            // Active/recent downloads
            let downloads = download_state
                .manager
                .list_tasks()
                .into_iter()
                .map(|t| DashboardDownload {
                    id: t.id,
                    model_name: t.model_name,
                    status: serde_json::to_value(&t.status)
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_owned))
                        .unwrap_or_else(|| format!("{:?}", t.status)),
                    progress: t.progress,
                    downloaded_mb: t.downloaded_size / 1024 / 1024,
                    total_mb: t.total_size.map(|s| s / 1024 / 1024),
                })
                .collect();

            let event = DashboardEvent {
                metrics: DashboardMetrics {
                    uptime_s: m.uptime_seconds,
                    active_req: h.active_requests,
                    total_req: m.total_requests,
                    avg_latency_ms: m.avg_latency_ms,
                    error_rate,
                    throughput_per_s: m.throughput_rps,
                    cpu_pct,
                    mem_used_mb,
                    mem_total_mb,
                },
                gpu,
                downloads,
            };

            let json = match serde_json::to_string(&event) {
                Ok(j) => j,
                Err(e) => {
                    warn!("[dashboard] serialization error (skipping tick): {e}");
                    continue;
                }
            };
            let frame = Bytes::from(format!("data: {}\n\n", json));

            if tx.send(Ok(frame)).await.is_err() {
                break; // client disconnected
            }
        }
    });

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(ReceiverStream::new(rx))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(total_req: u64, total_errors: u64) -> DashboardMetrics {
        let error_rate = if total_req > 0 {
            total_errors as f64 / total_req as f64
        } else {
            0.0
        };
        DashboardMetrics {
            uptime_s: 3600,
            active_req: 2,
            total_req,
            avg_latency_ms: 38.5,
            error_rate,
            throughput_per_s: 12.3,
            cpu_pct: 45.0,
            mem_used_mb: 1024,
            mem_total_mb: 8192,
        }
    }

    #[test]
    fn dashboard_event_serializes_all_fields() {
        let event = DashboardEvent {
            metrics: make_metrics(500, 5),
            gpu: vec![DashboardGpu {
                name: "TestGPU".to_string(),
                util_pct: Some(30),
                temp_c: Some(65),
                vram_free_mb: 8000,
                vram_total_mb: 10000,
            }],
            downloads: vec![DashboardDownload {
                id: "abc-123".to_string(),
                model_name: "test/model".to_string(),
                status: "Downloading".to_string(),
                progress: 0.5,
                downloaded_mb: 200,
                total_mb: Some(400),
            }],
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"uptime_s\":3600"));
        assert!(json.contains("\"total_req\":500"));
        assert!(json.contains("\"cpu_pct\":45.0"));
        assert!(json.contains("\"TestGPU\""));
        assert!(json.contains("\"test/model\""));
        assert!(json.contains("\"progress\":0.5"));
    }

    #[test]
    fn error_rate_is_zero_when_no_requests() {
        let m = make_metrics(0, 0);
        assert_eq!(m.error_rate, 0.0);
    }

    #[test]
    fn error_rate_calculated_correctly() {
        let m = make_metrics(100, 10);
        assert!((m.error_rate - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn dashboard_gpu_with_no_optional_fields_serializes() {
        let gpu = DashboardGpu {
            name: "iGPU".to_string(),
            util_pct: None,
            temp_c: None,
            vram_free_mb: 512,
            vram_total_mb: 1024,
        };
        let json = serde_json::to_string(&gpu).unwrap();
        assert!(json.contains("\"iGPU\""));
        assert!(json.contains("\"util_pct\":null"));
        assert!(json.contains("\"temp_c\":null"));
        assert!(json.contains("\"vram_free_mb\":512"));
    }

    #[test]
    fn dashboard_gpu_with_all_fields_serializes() {
        let gpu = DashboardGpu {
            name: "RTX 4090".to_string(),
            util_pct: Some(87),
            temp_c: Some(72),
            vram_free_mb: 4096,
            vram_total_mb: 24576,
        };
        let json = serde_json::to_string(&gpu).unwrap();
        assert!(json.contains("\"util_pct\":87"));
        assert!(json.contains("\"temp_c\":72"));
        assert!(json.contains("\"vram_total_mb\":24576"));
    }

    #[test]
    fn dashboard_download_without_total_mb_serializes() {
        let dl = DashboardDownload {
            id: "dl-1".to_string(),
            model_name: "llama/llama-3".to_string(),
            status: "Pending".to_string(),
            progress: 0.0,
            downloaded_mb: 0,
            total_mb: None,
        };
        let json = serde_json::to_string(&dl).unwrap();
        assert!(json.contains("\"total_mb\":null"));
        assert!(json.contains("\"progress\":0.0"));
    }

    #[test]
    fn dashboard_download_with_total_mb_serializes() {
        let dl = DashboardDownload {
            id: "dl-2".to_string(),
            model_name: "whisper/base".to_string(),
            status: "Completed".to_string(),
            progress: 1.0,
            downloaded_mb: 142,
            total_mb: Some(142),
        };
        let json = serde_json::to_string(&dl).unwrap();
        assert!(json.contains("\"total_mb\":142"));
        assert!(json.contains("\"progress\":1.0"));
        assert!(json.contains("\"Completed\""));
    }

    #[test]
    fn dashboard_event_with_empty_gpu_and_downloads_serializes() {
        let event = DashboardEvent {
            metrics: make_metrics(0, 0),
            gpu: vec![],
            downloads: vec![],
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"gpu\":[]"));
        assert!(json.contains("\"downloads\":[]"));
    }

    #[test]
    fn sse_frame_format_is_correct() {
        let event = DashboardEvent {
            metrics: make_metrics(10, 1),
            gpu: vec![],
            downloads: vec![],
        };
        let json = serde_json::to_string(&event).unwrap();
        let frame = format!("data: {}\n\n", json);
        assert!(frame.starts_with("data: {"));
        assert!(frame.ends_with("\n\n"));
        // Must be valid JSON after stripping the "data: " prefix and trailing newlines
        let payload = frame.trim_start_matches("data: ").trim_end();
        let reparsed: serde_json::Value = serde_json::from_str(payload).unwrap();
        assert!(reparsed["metrics"]["total_req"].is_number());
    }

    #[test]
    fn multiple_gpu_devices_serialize() {
        let event = DashboardEvent {
            metrics: make_metrics(1, 0),
            gpu: vec![
                DashboardGpu {
                    name: "GPU0".to_string(),
                    util_pct: Some(50),
                    temp_c: Some(60),
                    vram_free_mb: 4000,
                    vram_total_mb: 8000,
                },
                DashboardGpu {
                    name: "GPU1".to_string(),
                    util_pct: Some(70),
                    temp_c: Some(65),
                    vram_free_mb: 3500,
                    vram_total_mb: 8000,
                },
            ],
            downloads: vec![],
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"GPU0\""));
        assert!(json.contains("\"GPU1\""));
    }

    #[test]
    fn throughput_and_latency_fields_present() {
        let m = make_metrics(1000, 2);
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("\"throughput_per_s\""));
        assert!(json.contains("\"avg_latency_ms\""));
        assert!(json.contains("\"mem_used_mb\""));
        assert!(json.contains("\"mem_total_mb\""));
    }

    #[test]
    fn dashboard_metrics_debug_format() {
        let m = make_metrics(5, 0);
        let debug = format!("{:?}", m);
        assert!(debug.contains("DashboardMetrics"));
    }

    #[test]
    fn dashboard_gpu_debug_format() {
        let g = DashboardGpu {
            name: "X".to_string(),
            util_pct: None,
            temp_c: None,
            vram_free_mb: 0,
            vram_total_mb: 0,
        };
        let debug = format!("{:?}", g);
        assert!(debug.contains("DashboardGpu"));
    }

    #[test]
    fn dashboard_download_debug_format() {
        let d = DashboardDownload {
            id: "x".to_string(),
            model_name: "y".to_string(),
            status: "z".to_string(),
            progress: 0.0,
            downloaded_mb: 0,
            total_mb: None,
        };
        let debug = format!("{:?}", d);
        assert!(debug.contains("DashboardDownload"));
    }
}
