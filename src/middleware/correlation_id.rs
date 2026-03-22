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
