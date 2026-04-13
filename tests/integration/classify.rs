use actix_web::{test, web, App};
use torch_inference::api::classify::configure_routes;
use torch_inference::config::Config;

use super::helpers::classify_state;

#[actix_web::test]
async fn post_classify_empty_images_returns_400() {
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(Config::default()))
            .app_data(classify_state())
            .configure(configure_routes),
    )
    .await;
    let req = test::TestRequest::post()
        .uri("/classify/batch")
        .set_json(serde_json::json!({"images": []}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 400);
}

#[actix_web::test]
async fn post_classify_empty_images_error_body_has_error_field() {
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(Config::default()))
            .app_data(classify_state())
            .configure(configure_routes),
    )
    .await;
    let req = test::TestRequest::post()
        .uri("/classify/batch")
        .set_json(serde_json::json!({"images": []}))
        .to_request();
    let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert!(
        body.get("error").is_some(),
        "400 body should have error field: {body}"
    );
}

#[actix_web::test]
async fn post_classify_invalid_base64_returns_400() {
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(Config::default()))
            .app_data(classify_state())
            .configure(configure_routes),
    )
    .await;
    let req = test::TestRequest::post()
        .uri("/classify/batch")
        .set_json(serde_json::json!({"images": ["not-valid-base64!!!"]}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 400);
}

#[actix_web::test]
async fn post_classify_top_k_zero_returns_400() {
    use base64::Engine as _;
    let img = base64::engine::general_purpose::STANDARD.encode(b"fake");
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(Config::default()))
            .app_data(classify_state())
            .configure(configure_routes),
    )
    .await;
    let req = test::TestRequest::post()
        .uri("/classify/batch")
        .set_json(serde_json::json!({"images": [img], "top_k": 0}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 400);
}

#[actix_web::test]
async fn post_classify_batch_too_large_returns_400() {
    use base64::Engine as _;
    let img = base64::engine::general_purpose::STANDARD.encode(b"fake");
    let images: Vec<_> = (0..129).map(|_| img.clone()).collect();
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(Config::default()))
            .app_data(classify_state())
            .configure(configure_routes),
    )
    .await;
    let req = test::TestRequest::post()
        .uri("/classify/batch")
        .set_json(serde_json::json!({"images": images}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 400);
}
