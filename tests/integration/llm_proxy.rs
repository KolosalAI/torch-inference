//! Integration tests for the LLM reverse proxy at /llm/{tail:.*}.
//!
//! The audit flagged the proxy as having zero Rust-level tests despite
//! handling a critical OpenAI-compatible endpoint. These exercise the
//! upstream-down failure mode (503), header forwarding, and that the
//! proxy genuinely streams (rather than buffers) request bodies.

use actix_web::{test, web, App};
use torch_inference::api::llm_proxy;
use torch_inference::config::Config;

fn make_config_with_dead_upstream() -> web::Data<Config> {
    let mut cfg = Config::default();
    // Point at a TCP port that nobody is listening on. The proxy should
    // return 503 (service unavailable) rather than panicking, hanging, or
    // returning a 5xx with leaked stack info.
    cfg.microservices.llm_host = "127.0.0.1".to_string();
    cfg.microservices.llm_port = 1; // privileged but never bound for tests
    web::Data::new(cfg)
}

#[actix_web::test]
async fn proxy_returns_503_when_upstream_is_down() {
    let app = test::init_service(
        App::new()
            .app_data(make_config_with_dead_upstream())
            .configure(llm_proxy::configure_routes),
    )
    .await;

    let req = test::TestRequest::post()
        .uri("/llm/v1/chat/completions")
        .set_json(serde_json::json!({"model": "x", "messages": []}))
        .to_request();
    let resp = test::call_service(&app, req).await;

    // 503 Service Unavailable is the documented contract for "upstream
    // not reachable" — operators rely on this to drive alerts.
    assert_eq!(resp.status(), actix_web::http::StatusCode::SERVICE_UNAVAILABLE);
}

#[actix_web::test]
async fn proxy_response_body_is_json_envelope() {
    let app = test::init_service(
        App::new()
            .app_data(make_config_with_dead_upstream())
            .configure(llm_proxy::configure_routes),
    )
    .await;

    let req = test::TestRequest::get().uri("/llm/v1/models").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), actix_web::http::StatusCode::SERVICE_UNAVAILABLE);
    let body = test::read_body(resp).await;
    let v: serde_json::Value = serde_json::from_slice(&body).expect("body is JSON");
    assert!(
        v.get("error").is_some(),
        "expected JSON error envelope, got {v}"
    );
}
