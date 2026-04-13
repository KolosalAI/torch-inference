use actix_web::{test, web, App, HttpResponse};
use torch_inference::error::ApiError;

#[actix_web::test]
async fn unknown_route_returns_404() {
    let app = test::init_service(
        App::new().route(
            "/exists",
            web::get().to(|| async { HttpResponse::Ok().finish() }),
        ),
    )
    .await;
    let req = test::TestRequest::get().uri("/not-here").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 404);
}

#[actix_web::test]
async fn wrong_method_returns_405() {
    // actix-web returns 405 when the path matches but the method does not,
    // which requires defining multiple methods on the same resource.
    use actix_web::web::resource;
    let app = test::init_service(
        App::new().service(
            resource("/get-only").route(web::get().to(|| async { HttpResponse::Ok().finish() })),
        ),
    )
    .await;
    let req = test::TestRequest::post().uri("/get-only").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 405);
}

#[actix_web::test]
async fn api_error_bad_request_gives_400() {
    use actix_web::ResponseError;
    let err = ApiError::BadRequest("test".to_string());
    let resp: actix_web::HttpResponse = err.error_response();
    assert_eq!(resp.status(), 400);
}

#[actix_web::test]
async fn api_error_not_found_gives_404() {
    use actix_web::ResponseError;
    let err = ApiError::NotFound("thing".to_string());
    let resp: actix_web::HttpResponse = err.error_response();
    assert_eq!(resp.status(), 404);
}

#[actix_web::test]
async fn api_error_internal_gives_500() {
    use actix_web::ResponseError;
    let err = ApiError::InternalError("oops".to_string());
    let resp: actix_web::HttpResponse = err.error_response();
    assert_eq!(resp.status(), 500);
}

#[actix_web::test]
async fn api_error_bad_request_status_code() {
    use actix_web::ResponseError;
    let err = ApiError::BadRequest("bad".to_string());
    assert_eq!(err.status_code(), actix_web::http::StatusCode::BAD_REQUEST);
}

#[actix_web::test]
async fn api_error_not_found_status_code() {
    use actix_web::ResponseError;
    let err = ApiError::NotFound("missing".to_string());
    assert_eq!(err.status_code(), actix_web::http::StatusCode::NOT_FOUND);
}

#[actix_web::test]
async fn api_error_internal_error_status_code() {
    use actix_web::ResponseError;
    let err = ApiError::InternalError("crash".to_string());
    assert_eq!(
        err.status_code(),
        actix_web::http::StatusCode::INTERNAL_SERVER_ERROR
    );
}
