use actix_web::HttpResponse;
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

    #[actix_web::test]
    async fn test_metrics_endpoint_returns_200() {
        let resp = metrics_handler().await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_metrics_endpoint_content_type_is_plain_text() {
        use actix_web::{test as actix_test, App, web};

        let app = actix_test::init_service(
            App::new().route("/metrics", web::get().to(metrics_handler))
        )
        .await;

        let req = actix_test::TestRequest::get()
            .uri("/metrics")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        // Verify the Content-Type header is set for Prometheus scraping
        let content_type = resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            content_type.contains("text/plain"),
            "Expected text/plain content-type, got: {content_type}"
        );
    }

    #[actix_web::test]
    async fn test_metrics_endpoint_body_is_string() {
        use actix_web::{test as actix_test, App, web};

        let app = actix_test::init_service(
            App::new().route("/metrics", web::get().to(metrics_handler))
        )
        .await;

        let req = actix_test::TestRequest::get()
            .uri("/metrics")
            .to_request();
        let body_bytes = actix_test::call_and_read_body(&app, req).await;
        // Body must be valid UTF-8 Prometheus exposition format
        let body_str = std::str::from_utf8(&body_bytes)
            .expect("metrics body should be valid UTF-8");
        // Prometheus output may be empty or contain metric lines; it must not be a failure message
        assert!(
            !body_str.starts_with("Failed to render metrics:"),
            "Unexpected error body: {body_str}"
        );
    }
}
