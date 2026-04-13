use actix_web::{test, web, App};
use torch_inference::api::tts::configure_routes;
use torch_inference::config::Config;

use super::helpers::tts_state;

#[actix_web::test]
async fn get_tts_engines_returns_200() {
    let app = test::init_service(
        App::new()
            .app_data(tts_state())
            .configure(configure_routes),
    )
    .await;
    let req = test::TestRequest::get().uri("/tts/engines").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
}

#[actix_web::test]
async fn get_tts_engines_body_has_engines_field() {
    let app = test::init_service(
        App::new()
            .app_data(tts_state())
            .configure(configure_routes),
    )
    .await;
    let req = test::TestRequest::get().uri("/tts/engines").to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(body.get("engines").is_some(), "body: {body}");
}

#[actix_web::test]
async fn get_tts_stats_returns_200() {
    let app = test::init_service(
        App::new()
            .app_data(tts_state())
            .configure(configure_routes),
    )
    .await;
    let req = test::TestRequest::get().uri("/tts/stats").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
}

#[actix_web::test]
async fn get_tts_health_returns_200() {
    let app = test::init_service(
        App::new()
            .app_data(tts_state())
            .configure(configure_routes),
    )
    .await;
    let req = test::TestRequest::get().uri("/tts/health").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
}

#[actix_web::test]
async fn post_tts_synthesize_empty_text_returns_4xx() {
    let app = test::init_service(
        App::new()
            .app_data(tts_state())
            .app_data(web::Data::new(Config::default()))
            .configure(configure_routes),
    )
    .await;
    let req = test::TestRequest::post()
        .uri("/tts/synthesize")
        .set_json(serde_json::json!({"text": ""}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(
        resp.status().is_client_error(),
        "expected 4xx, got {}",
        resp.status()
    );
}

#[actix_web::test]
async fn get_tts_unknown_engine_capabilities_returns_404() {
    let app = test::init_service(
        App::new()
            .app_data(tts_state())
            .configure(configure_routes),
    )
    .await;
    let req = test::TestRequest::get()
        .uri("/tts/engines/nonexistent_engine_xyz/capabilities")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);
}

#[actix_web::test]
async fn get_tts_unknown_engine_voices_returns_404() {
    let app = test::init_service(
        App::new()
            .app_data(tts_state())
            .configure(configure_routes),
    )
    .await;
    let req = test::TestRequest::get()
        .uri("/tts/engines/nonexistent_engine_xyz/voices")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);
}
