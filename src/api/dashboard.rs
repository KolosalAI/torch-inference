// src/api/dashboard.rs
use actix_web::{web, HttpResponse};
use actix_web::web::Bytes;
use serde::Serialize;
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::monitor::Monitor;
use crate::api::system::SystemInfoState;
use crate::api::model_download::ModelDownloadState;

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
            let mem_used_mb  = sys.used_memory()  / 1024 / 1024;
            let mem_total_mb = sys.total_memory()  / 1024 / 1024;

            // GPU from GpuManager
            let gpu = match system_state.gpu_manager.get_info() {
                Ok(info) => info
                    .devices
                    .into_iter()
                    .map(|d| DashboardGpu {
                        name:         d.name,
                        util_pct:     d.utilization,
                        temp_c:       d.temperature,
                        vram_free_mb: d.free_memory  / 1024 / 1024,
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
                    id:           t.id,
                    model_name:   t.model_name,
                    status:       serde_json::to_value(&t.status)
                                    .ok()
                                    .and_then(|v| v.as_str().map(str::to_owned))
                                    .unwrap_or_else(|| format!("{:?}", t.status)),
                    progress:     t.progress,
                    downloaded_mb: t.downloaded_size / 1024 / 1024,
                    total_mb:     t.total_size.map(|s| s / 1024 / 1024),
                })
                .collect();

            let event = DashboardEvent {
                metrics: DashboardMetrics {
                    uptime_s:        m.uptime_seconds,
                    active_req:      h.active_requests,
                    total_req:       m.total_requests,
                    avg_latency_ms:  m.avg_latency_ms,
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
                Ok(j)  => j,
                Err(e) => {
                    eprintln!("[dashboard] serialization error (skipping tick): {e}");
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

    #[test]
    fn dashboard_event_serializes_all_fields() {
        let event = DashboardEvent {
            metrics: DashboardMetrics {
                uptime_s: 3600,
                active_req: 2,
                total_req: 500,
                avg_latency_ms: 38.5,
                error_rate: 0.01,
                throughput_per_s: 12.3,
                cpu_pct: 45.0,
                mem_used_mb: 1024,
                mem_total_mb: 8192,
            },
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
}
