use actix_web::{HttpResponse, web};
use crate::telemetry::prometheus;

/// Prometheus metrics endpoint
/// 
/// Returns metrics in Prometheus text format for scraping
pub async fn metrics_handler() -> HttpResponse {
    match prometheus::render_metrics() {
        Ok(metrics_text) => {
            HttpResponse::Ok()
                .content_type("text/plain; version=0.0.4; charset=utf-8")
                .body(metrics_text)
        }
        Err(e) => {
            HttpResponse::InternalServerError()
                .body(format!("Failed to render metrics: {}", e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::test;

    #[actix_web::test]
    async fn test_metrics_endpoint() {
        let resp = metrics_handler().await;
        assert!(resp.status().is_success());
    }
}
