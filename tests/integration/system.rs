use actix_web::{test, web, App};
use torch_inference::api::system::{get_config, get_gpu_stats, get_system_info};

use super::helpers::system_state;

#[actix_web::test]
async fn get_system_info_returns_200() {
    let app = test::init_service(
        App::new()
            .app_data(system_state())
            .route("/system/info", web::get().to(get_system_info)),
    )
    .await;
    let req = test::TestRequest::get().uri("/system/info").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
}

#[actix_web::test]
async fn get_system_info_body_has_system_and_gpu() {
    let app = test::init_service(
        App::new()
            .app_data(system_state())
            .route("/system/info", web::get().to(get_system_info)),
    )
    .await;
    let req = test::TestRequest::get().uri("/system/info").to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(body.get("system").is_some(), "body: {body}");
    assert!(body.get("gpu").is_some(), "body: {body}");
}

#[actix_web::test]
async fn get_system_info_body_has_runtime_field() {
    let app = test::init_service(
        App::new()
            .app_data(system_state())
            .route("/system/info", web::get().to(get_system_info)),
    )
    .await;
    let req = test::TestRequest::get().uri("/system/info").to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(body.get("runtime").is_some(), "body: {body}");
}

#[actix_web::test]
async fn get_system_config_returns_200() {
    let app = test::init_service(
        App::new()
            .app_data(system_state())
            .route("/system/config", web::get().to(get_config)),
    )
    .await;
    let req = test::TestRequest::get().uri("/system/config").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
}

#[actix_web::test]
async fn get_system_config_body_has_server_field() {
    let app = test::init_service(
        App::new()
            .app_data(system_state())
            .route("/system/config", web::get().to(get_config)),
    )
    .await;
    let req = test::TestRequest::get().uri("/system/config").to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(body.get("server").is_some(), "body: {body}");
}

#[actix_web::test]
async fn get_gpu_stats_returns_200() {
    let app = test::init_service(
        App::new()
            .app_data(system_state())
            .route("/system/gpu/stats", web::get().to(get_gpu_stats)),
    )
    .await;
    let req = test::TestRequest::get()
        .uri("/system/gpu/stats")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
}
