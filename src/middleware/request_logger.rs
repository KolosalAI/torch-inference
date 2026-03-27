use std::task::{Context, Poll};
use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error,
};
use futures::future::{ok, Ready};
use std::pin::Pin;
use std::future::Future;
use std::time::Instant;
use crate::telemetry::{CorrelationId, RequestMetrics};

pub struct RequestLogger;

impl<S, B> Transform<S, ServiceRequest> for RequestLogger
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = RequestLoggerMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ok(RequestLoggerMiddleware { service })
    }
}

pub struct RequestLoggerMiddleware<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for RequestLoggerMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let method = req.method().to_string();
        let path = req.path().to_string();
        let remote_addr = req.connection_info().peer_addr().unwrap_or("unknown").to_string();
        
        // Get or create correlation ID
        let correlation_id = req
            .headers()
            .get("X-Correlation-ID")
            .and_then(|v| v.to_str().ok())
            .map(|s| CorrelationId::from_header(s))
            .unwrap_or_else(CorrelationId::new);

        let metrics = RequestMetrics::new(correlation_id.clone());

        // Log incoming request with structured data
        tracing::info!(
            correlation_id = %correlation_id.as_str(),
            method = %method,
            path = %path,
            remote_addr = %remote_addr,
            event = "request_received",
        );

        let fut = self.service.call(req);

        Box::pin(async move {
            match fut.await {
                Ok(res) => {
                    let status = res.status();
                    
                    // Log successful response
                    tracing::info!(
                        correlation_id = %metrics.correlation_id.as_str(),
                        method = %method,
                        path = %path,
                        status = %status.as_u16(),
                        duration_ms = %metrics.duration_ms(),
                        event = "request_completed",
                    );
                    
                    Ok(res)
                }
                Err(err) => {
                    // Log error response
                    tracing::error!(
                        correlation_id = %metrics.correlation_id.as_str(),
                        method = %method,
                        path = %path,
                        error = %err,
                        duration_ms = %metrics.duration_ms(),
                        event = "request_error",
                    );
                    
                    Err(err)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test as awtest, web, App, HttpResponse};

    // ── Middleware integration tests ──────────────────────────────────────────

    #[actix_web::test]
    async fn test_request_logger_passes_through_200() {
        let app = awtest::init_service(
            App::new()
                .wrap(RequestLogger)
                .route("/", web::get().to(|| async { HttpResponse::Ok().finish() })),
        )
        .await;

        let req = awtest::TestRequest::get().uri("/").to_request();
        let resp = awtest::call_service(&app, req).await;

        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_request_logger_passes_through_404() {
        let app = awtest::init_service(
            App::new()
                .wrap(RequestLogger)
                .route(
                    "/exists",
                    web::get().to(|| async { HttpResponse::Ok().finish() }),
                ),
        )
        .await;

        // Call a path that has no registered route — actix returns 404.
        let req = awtest::TestRequest::get().uri("/missing").to_request();
        let resp = awtest::call_service(&app, req).await;

        assert_eq!(resp.status(), actix_web::http::StatusCode::NOT_FOUND);
    }

    #[actix_web::test]
    async fn test_request_logger_post_method() {
        let app = awtest::init_service(
            App::new()
                .wrap(RequestLogger)
                .route(
                    "/data",
                    web::post().to(|| async { HttpResponse::Created().finish() }),
                ),
        )
        .await;

        let req = awtest::TestRequest::post().uri("/data").to_request();
        let resp = awtest::call_service(&app, req).await;

        assert_eq!(resp.status(), actix_web::http::StatusCode::CREATED);
    }

    #[actix_web::test]
    async fn test_request_logger_delete_method() {
        let app = awtest::init_service(
            App::new()
                .wrap(RequestLogger)
                .route(
                    "/item",
                    web::delete().to(|| async { HttpResponse::NoContent().finish() }),
                ),
        )
        .await;

        let req = awtest::TestRequest::delete().uri("/item").to_request();
        let resp = awtest::call_service(&app, req).await;

        assert_eq!(resp.status(), actix_web::http::StatusCode::NO_CONTENT);
    }

    #[actix_web::test]
    async fn test_request_logger_with_correlation_id_header() {
        let app = awtest::init_service(
            App::new()
                .wrap(RequestLogger)
                .route("/", web::get().to(|| async { HttpResponse::Ok().finish() })),
        )
        .await;

        let req = awtest::TestRequest::get()
            .uri("/")
            .insert_header(("X-Correlation-ID", "trace-abc-999"))
            .to_request();
        let resp = awtest::call_service(&app, req).await;

        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_request_logger_response_body_passes_through() {
        let app = awtest::init_service(
            App::new()
                .wrap(RequestLogger)
                .route(
                    "/hello",
                    web::get().to(|| async { HttpResponse::Ok().body("hello world") }),
                ),
        )
        .await;

        let req = awtest::TestRequest::get().uri("/hello").to_request();
        let resp = awtest::call_service(&app, req).await;

        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body = awtest::read_body(resp).await;
        assert_eq!(body, "hello world");
    }
}
