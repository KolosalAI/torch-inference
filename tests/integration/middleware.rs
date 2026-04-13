use actix_web::{test, web, App, HttpResponse};
use std::sync::Arc;
use torch_inference::middleware::{rate_limit::RateLimiter, CorrelationIdMiddleware};

#[actix_web::test]
async fn response_has_correlation_id_header() {
    let app = test::init_service(
        App::new()
            .wrap(CorrelationIdMiddleware)
            .route(
                "/ping",
                web::get().to(|| async { HttpResponse::Ok().finish() }),
            ),
    )
    .await;
    let req = test::TestRequest::get().uri("/ping").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.headers().contains_key("x-correlation-id"));
}

#[actix_web::test]
async fn request_correlation_id_is_echoed() {
    let app = test::init_service(
        App::new()
            .wrap(CorrelationIdMiddleware)
            .route(
                "/ping",
                web::get().to(|| async { HttpResponse::Ok().finish() }),
            ),
    )
    .await;
    let req = test::TestRequest::get()
        .uri("/ping")
        .insert_header(("x-correlation-id", "test-abc-123"))
        .to_request();
    let resp = test::call_service(&app, req).await;
    let hdr = resp
        .headers()
        .get("x-correlation-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(hdr, "test-abc-123");
}

#[actix_web::test]
async fn generated_correlation_id_is_non_empty() {
    let app = test::init_service(
        App::new()
            .wrap(CorrelationIdMiddleware)
            .route(
                "/ping",
                web::get().to(|| async { HttpResponse::Ok().finish() }),
            ),
    )
    .await;
    // No x-correlation-id header supplied — middleware should generate one
    let req = test::TestRequest::get().uri("/ping").to_request();
    let resp = test::call_service(&app, req).await;
    let hdr = resp
        .headers()
        .get("x-correlation-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(!hdr.is_empty(), "generated correlation id must not be empty");
}

#[actix_web::test]
async fn rate_limiter_allows_under_limit() {
    let limiter = Arc::new(RateLimiter::new(100, 60));
    assert!(limiter.is_allowed("client-a").is_ok());
}

#[actix_web::test]
async fn rate_limiter_rejects_over_limit() {
    let limiter = Arc::new(RateLimiter::new(1, 60));
    let _ = limiter.is_allowed("client-x"); // consume the 1 allowed request
    assert!(limiter.is_allowed("client-x").is_err());
}

#[actix_web::test]
async fn rate_limiter_different_clients_independent() {
    let limiter = Arc::new(RateLimiter::new(1, 60));
    let _ = limiter.is_allowed("client-a");
    // client-b has its own independent counter and should still be allowed
    assert!(limiter.is_allowed("client-b").is_ok());
}

#[actix_web::test]
async fn rate_limiter_zero_limit_always_rejects() {
    let limiter = Arc::new(RateLimiter::new(0, 60));
    assert!(limiter.is_allowed("any-client").is_err());
}
