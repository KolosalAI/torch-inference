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

