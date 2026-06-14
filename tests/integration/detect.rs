use actix_web::{test, web, App};
use std::path::PathBuf;
use torch_inference::api::yolo::{configure, YoloState};

fn make_yolo_state() -> web::Data<YoloState> {
    web::Data::new(YoloState {
        models_dir: PathBuf::from("./models"),
        ort_detector: parking_lot::Mutex::new(None),
    })
}

#[actix_web::test]
async fn get_yolo_models_returns_200() {
    let app = test::init_service(
        App::new()
            .app_data(make_yolo_state())
            .configure(configure),
    )
    .await;
    let req = test::TestRequest::get().uri("/yolo/models").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);
}

#[actix_web::test]
async fn get_yolo_models_body_has_versions_field() {
    let app = test::init_service(
        App::new()
            .app_data(make_yolo_state())
            .configure(configure),
    )
    .await;
    let req = test::TestRequest::get().uri("/yolo/models").to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(body.get("versions").is_some(), "body: {body}");
}

#[actix_web::test]
async fn get_yolo_models_body_has_sizes_field() {
    let app = test::init_service(
        App::new()
            .app_data(make_yolo_state())
            .configure(configure),
    )
    .await;
    let req = test::TestRequest::get().uri("/yolo/models").to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(body.get("sizes").is_some(), "body: {body}");
}

#[actix_web::test]
async fn post_yolo_detect_json_body_returns_error() {
    // YOLO detect expects multipart/form-data; sending plain JSON should produce an error
    use torch_inference::config::Config;
    let app = test::init_service(
        App::new()
            .app_data(make_yolo_state())
            .app_data(web::Data::new(Config::default()))
            .configure(configure),
    )
    .await;
    let req = test::TestRequest::post()
        .uri("/yolo/detect?model_version=v8&model_size=n")
        .set_json(serde_json::json!({}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "expected error status, got {}",
        resp.status()
    );
}

#[actix_web::test]
async fn get_yolo_info_missing_query_returns_error() {
    use torch_inference::config::Config;
    let app = test::init_service(
        App::new()
            .app_data(make_yolo_state())
            .app_data(web::Data::new(Config::default()))
            .configure(configure),
    )
    .await;
    // /yolo/info requires model_version and model_size query params
    let req = test::TestRequest::get().uri("/yolo/info").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "expected error status for missing params, got {}",
        resp.status()
    );
}
