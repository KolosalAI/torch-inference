use actix_web::{test, web, App};
use torch_inference::api::health::{health, liveness, readiness};
use torch_inference::middleware::CorrelationIdMiddleware;

use super::helpers::monitor;

#[actix_web::test]
async fn get_health_returns_200() {
    let app = test::init_service(
        App::new()
            .app_data(monitor())
            .route("/health", web::get().to(health)),
    )
    .await;
    let req = test::TestRequest::get().uri("/health").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
}

#[actix_web::test]
async fn get_health_body_has_status_field() {
    let app = test::init_service(
        App::new()
            .app_data(monitor())
            .route("/health", web::get().to(health)),
    )
    .await;
    let req = test::TestRequest::get().uri("/health").to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(body.get("status").is_some(), "body: {body}");
}

#[actix_web::test]
async fn get_health_body_has_version_field() {
    let app = test::init_service(
        App::new()
            .app_data(monitor())
            .route("/health", web::get().to(health)),
    )
    .await;
    let req = test::TestRequest::get().uri("/health").to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(body.get("version").is_some(), "body: {body}");
}

#[actix_web::test]
async fn get_liveness_returns_200() {
    let app = test::init_service(
        App::new()
            .app_data(monitor())
            .route("/health/live", web::get().to(liveness)),
    )
    .await;
    let req = test::TestRequest::get().uri("/health/live").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
}

#[actix_web::test]
async fn get_liveness_body_has_status_field() {
    let app = test::init_service(
        App::new()
            .app_data(monitor())
            .route("/health/live", web::get().to(liveness)),
    )
    .await;
    let req = test::TestRequest::get().uri("/health/live").to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(body.get("status").is_some(), "body: {body}");
}

#[actix_web::test]
async fn get_readiness_returns_2xx_or_503() {
    let app = test::init_service(
        App::new()
            .app_data(monitor())
            .route("/health/ready", web::get().to(readiness)),
    )
    .await;
    let req = test::TestRequest::get().uri("/health/ready").to_request();
    let resp = test::call_service(&app, req).await;
    // readiness returns 200 (ready) or 503 (not_ready) depending on monitor state
    assert!(
        resp.status() == 200 || resp.status() == 503,
        "expected 200 or 503, got {}",
        resp.status()
    );
}

#[actix_web::test]
async fn get_readiness_body_has_status_field() {
    let app = test::init_service(
        App::new()
            .app_data(monitor())
            .route("/health/ready", web::get().to(readiness)),
    )
    .await;
    let req = test::TestRequest::get().uri("/health/ready").to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(body.get("status").is_some(), "body: {body}");
}

#[actix_web::test]
async fn health_response_has_correlation_id_header() {
    let app = test::init_service(
        App::new()
            .app_data(monitor())
            .wrap(CorrelationIdMiddleware)
            .route("/health/live", web::get().to(liveness)),
    )
    .await;
    let req = test::TestRequest::get().uri("/health/live").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.headers().contains_key("x-correlation-id"));
}
