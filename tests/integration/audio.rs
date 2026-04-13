use actix_web::{test, web, App};
use std::sync::Arc;
use torch_inference::api::audio::{audio_health, validate_audio, AudioState};
use torch_inference::config::SanitizerConfig;
use torch_inference::core::audio_models::AudioModelManager;
use torch_inference::security::sanitizer::Sanitizer;

fn make_audio_state() -> web::Data<AudioState> {
    web::Data::new(AudioState {
        model_manager: Arc::new(AudioModelManager::new("/tmp")),
        sanitizer: Sanitizer::new(SanitizerConfig::default()),
    })
}

#[actix_web::test]
async fn get_audio_health_returns_200() {
    let app = test::init_service(
        App::new()
            .app_data(make_audio_state())
            .route("/audio/health", web::get().to(audio_health)),
    )
    .await;
    let req = test::TestRequest::get().uri("/audio/health").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
}

#[actix_web::test]
async fn get_audio_health_body_has_status_field() {
    let app = test::init_service(
        App::new()
            .app_data(make_audio_state())
            .route("/audio/health", web::get().to(audio_health)),
    )
    .await;
    let req = test::TestRequest::get().uri("/audio/health").to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(body.get("status").is_some(), "body: {body}");
}

#[actix_web::test]
async fn get_audio_health_body_has_supported_formats() {
    let app = test::init_service(
        App::new()
            .app_data(make_audio_state())
            .route("/audio/health", web::get().to(audio_health)),
    )
    .await;
    let req = test::TestRequest::get().uri("/audio/health").to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(body.get("supported_formats").is_some(), "body: {body}");
}

#[actix_web::test]
async fn post_audio_validate_empty_body_returns_4xx() {
    let app = test::init_service(
        App::new()
            .app_data(make_audio_state())
            .route("/audio/validate", web::post().to(validate_audio)),
    )
    .await;
    // Sending no multipart data should result in a client or server error
    let req = test::TestRequest::post()
        .uri("/audio/validate")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "expected error for empty body, got {}",
        resp.status()
    );
}
