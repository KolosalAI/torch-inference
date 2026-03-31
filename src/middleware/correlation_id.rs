use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error, HttpMessage,
};
use futures_util::future::LocalBoxFuture;
use std::future::{ready, Ready};
use std::rc::Rc;

use crate::telemetry::CorrelationId;

/// Middleware to add correlation IDs to all requests
pub struct CorrelationIdMiddleware;

impl<S, B> Transform<S, ServiceRequest> for CorrelationIdMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = CorrelationIdMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(CorrelationIdMiddlewareService {
            service: Rc::new(service),
        }))
    }
}

pub struct CorrelationIdMiddlewareService<S> {
    service: Rc<S>,
}

impl<S, B> Service<ServiceRequest> for CorrelationIdMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        // Extract or generate correlation ID
        let correlation_id = req
            .headers()
            .get("X-Correlation-ID")
            .and_then(|h| h.to_str().ok())
            .map(|s| CorrelationId::from_header(s))
            .unwrap_or_else(CorrelationId::new);

        // Store in request extensions
        req.extensions_mut().insert(correlation_id.clone());

        // Add to response headers
        let service = self.service.clone();
        let correlation_id_clone = correlation_id.clone();

        Box::pin(async move {
            let mut res = service.call(req).await?;
            
            res.headers_mut().insert(
                actix_web::http::header::HeaderName::from_static("x-correlation-id"),
                actix_web::http::header::HeaderValue::from_str(correlation_id_clone.as_str())
                    .unwrap_or_else(|_| actix_web::http::header::HeaderValue::from_static("invalid")),
            );

            Ok(res)
        })
    }
}

/// Helper to extract correlation ID from request
pub fn get_correlation_id(req: &actix_web::HttpRequest) -> CorrelationId {
    req.extensions()
        .get::<CorrelationId>()
        .cloned()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test as awtest, web, App, HttpResponse};

    // ── CorrelationId unit tests ──────────────────────────────────────────────

    #[::core::prelude::v1::test]
    fn test_correlation_id_new_is_non_empty() {
        let id = CorrelationId::new();
        assert!(!id.as_str().is_empty());
    }

    #[::core::prelude::v1::test]
    fn test_correlation_id_new_generates_unique_ids() {
        let id1 = CorrelationId::new();
        let id2 = CorrelationId::new();
        assert_ne!(id1.as_str(), id2.as_str());
    }

    #[::core::prelude::v1::test]
    fn test_correlation_id_from_header_preserves_value() {
        let raw = "my-custom-id-42";
        let id = CorrelationId::from_header(raw);
        assert_eq!(id.as_str(), raw);
    }

    #[::core::prelude::v1::test]
    fn test_correlation_id_from_header_empty_string() {
        let id = CorrelationId::from_header("");
        assert_eq!(id.as_str(), "");
    }

    #[::core::prelude::v1::test]
    fn test_correlation_id_clone() {
        let id = CorrelationId::new();
        let cloned = id.clone();
        assert_eq!(id.as_str(), cloned.as_str());
    }

    #[::core::prelude::v1::test]
    fn test_correlation_id_default_is_non_empty() {
        let id = CorrelationId::default();
        assert!(!id.as_str().is_empty());
    }

    // ── get_correlation_id unit test ──────────────────────────────────────────

    #[::core::prelude::v1::test]
    fn test_get_correlation_id_returns_default_when_not_set() {
        // Build a minimal HttpRequest without inserting a CorrelationId extension.
        let req = awtest::TestRequest::get()
            .uri("/")
            .to_http_request();
        // Should not panic and should return a freshly generated (non-empty) ID.
        let id = get_correlation_id(&req);
        assert!(!id.as_str().is_empty());
    }

    // ── Middleware integration tests ──────────────────────────────────────────

    #[actix_web::test]
    async fn test_correlation_id_middleware_adds_header() {
        let app = awtest::init_service(
            App::new()
                .wrap(CorrelationIdMiddleware)
                .route("/", web::get().to(|| async { HttpResponse::Ok().finish() })),
        )
        .await;

        let req = awtest::TestRequest::get().uri("/").to_request();
        let resp = awtest::call_service(&app, req).await;

        assert!(
            resp.headers().contains_key("x-correlation-id"),
            "response must contain x-correlation-id header"
        );
    }

    #[actix_web::test]
    async fn test_correlation_id_middleware_propagates_existing_header() {
        let app = awtest::init_service(
            App::new()
                .wrap(CorrelationIdMiddleware)
                .route("/", web::get().to(|| async { HttpResponse::Ok().finish() })),
        )
        .await;

        let req = awtest::TestRequest::get()
            .uri("/")
            .insert_header(("X-Correlation-ID", "my-id-123"))
            .to_request();
        let resp = awtest::call_service(&app, req).await;

        let header_val = resp
            .headers()
            .get("x-correlation-id")
            .expect("x-correlation-id header missing")
            .to_str()
            .expect("header value is not valid UTF-8");

        assert_eq!(header_val, "my-id-123");
    }

    #[actix_web::test]
    async fn test_correlation_id_middleware_generates_new_id_when_absent() {
        let app = awtest::init_service(
            App::new()
                .wrap(CorrelationIdMiddleware)
                .route("/", web::get().to(|| async { HttpResponse::Ok().finish() })),
        )
        .await;

        let req = awtest::TestRequest::get().uri("/").to_request();
        let resp = awtest::call_service(&app, req).await;

        let header_val = resp
            .headers()
            .get("x-correlation-id")
            .expect("x-correlation-id header missing")
            .to_str()
            .expect("header value is not valid UTF-8");

        // The generated ID must be non-empty and must be a valid UUID v4.
        assert!(!header_val.is_empty());
        assert!(
            uuid::Uuid::parse_str(header_val).is_ok(),
            "auto-generated correlation id should be a valid UUID, got: {header_val}"
        );
    }

    #[actix_web::test]
    async fn test_correlation_id_middleware_preserves_response_status() {
        let app = awtest::init_service(
            App::new()
                .wrap(CorrelationIdMiddleware)
                .route(
                    "/created",
                    web::post().to(|| async { HttpResponse::Created().finish() }),
                ),
        )
        .await;

        let req = awtest::TestRequest::post().uri("/created").to_request();
        let resp = awtest::call_service(&app, req).await;

        assert_eq!(resp.status(), actix_web::http::StatusCode::CREATED);
        assert!(resp.headers().contains_key("x-correlation-id"));
    }
}
