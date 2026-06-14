//! Security middleware: standard hardening headers + JWT-based auth gate.
//!
//! `SecurityHeaders` is best-effort defence-in-depth (X-Content-Type-Options,
//! X-Frame-Options, Referrer-Policy, CSP for HTML, Permissions-Policy).
//!
//! `AuthMiddleware` enforces a Bearer JWT on every request *except* a small
//! allow-list of public paths (health, metrics, root playground, static
//! assets, dashboard SSE). It is a no-op when `config.auth.enabled` is false,
//! preserving the prior unauthenticated dev experience but giving operators a
//! single switch to lock the surface down.

use actix_web::{
    body::EitherBody,
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    http::header::{HeaderName, HeaderValue},
    Error, HttpResponse,
};
use futures_util::future::LocalBoxFuture;
use std::future::{ready, Ready};
use std::rc::Rc;
use std::sync::Arc;

use crate::auth::JwtHandler;

// ── SecurityHeaders ──────────────────────────────────────────────────────────

pub struct SecurityHeaders;

impl<S, B> Transform<S, ServiceRequest> for SecurityHeaders
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = SecurityHeadersService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(SecurityHeadersService { service: Rc::new(service) }))
    }
}

pub struct SecurityHeadersService<S> {
    service: Rc<S>,
}

impl<S, B> Service<ServiceRequest> for SecurityHeadersService<S>
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
        let service = self.service.clone();
        let path_is_html = req.path() == "/" || req.path() == "/playground";
        Box::pin(async move {
            let mut res = service.call(req).await?;
            let h = res.headers_mut();
            h.insert(
                HeaderName::from_static("x-content-type-options"),
                HeaderValue::from_static("nosniff"),
            );
            h.insert(
                HeaderName::from_static("x-frame-options"),
                HeaderValue::from_static("DENY"),
            );
            h.insert(
                HeaderName::from_static("referrer-policy"),
                HeaderValue::from_static("no-referrer"),
            );
            h.insert(
                HeaderName::from_static("permissions-policy"),
                HeaderValue::from_static("interest-cohort=()"),
            );
            // CSP only for the HTML playground response. JSON/WAV/PCM
            // responses don't render markup so CSP is meaningless and would
            // confuse some clients.
            if path_is_html {
                h.insert(
                    HeaderName::from_static("content-security-policy"),
                    // The playground uses inline event handlers + inline
                    // styles + blob: audio. Strict 'self' for scripts breaks
                    // it — mark inline as unsafe-inline (acknowledged risk;
                    // long-term plan is to extract to /assets/playground.js).
                    HeaderValue::from_static(
                        "default-src 'self'; script-src 'self' 'unsafe-inline'; \
                         style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; \
                         media-src 'self' blob:; connect-src 'self' ws: wss:; \
                         font-src 'self' data:; frame-ancestors 'none'; base-uri 'self'",
                    ),
                );
            }
            Ok(res)
        })
    }
}

// ── AuthMiddleware ───────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AuthMiddleware {
    pub enabled: bool,
    pub jwt: Arc<JwtHandler>,
}

impl AuthMiddleware {
    pub fn new(enabled: bool, secret: &str) -> Self {
        Self {
            enabled,
            jwt: Arc::new(JwtHandler::new(secret)),
        }
    }
}

/// Paths that bypass auth even when enabled. Keep this list small — every
/// route here is reachable unauthenticated from the network.
fn is_public(path: &str, method: &actix_web::http::Method) -> bool {
    if path == "/" || path == "/playground" {
        return true;
    }
    if path.starts_with("/health") {
        return true;
    }
    if path.starts_with("/assets/") {
        return true;
    }
    if path == "/metrics" {
        return true;
    }
    // Only the GET on /v1/models is public by default; chat-completions is
    // protected even though OpenAI clients usually accept either.
    if path == "/v1/models" && method == actix_web::http::Method::GET {
        return true;
    }
    false
}

impl<S, B> Transform<S, ServiceRequest> for AuthMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<EitherBody<B>>;
    type Error = Error;
    type InitError = ();
    type Transform = AuthMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(AuthMiddlewareService {
            service: Rc::new(service),
            enabled: self.enabled,
            jwt: self.jwt.clone(),
        }))
    }
}

pub struct AuthMiddlewareService<S> {
    service: Rc<S>,
    enabled: bool,
    jwt: Arc<JwtHandler>,
}

impl<S, B> Service<ServiceRequest> for AuthMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<EitherBody<B>>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let service = self.service.clone();
        let enabled = self.enabled;
        let jwt = self.jwt.clone();

        if !enabled || is_public(req.path(), req.method()) {
            return Box::pin(async move {
                service
                    .call(req)
                    .await
                    .map(ServiceResponse::map_into_left_body)
            });
        }

        // Extract bearer token.
        let header = req
            .headers()
            .get(actix_web::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        let token = match header.as_deref() {
            Some(s) if s.starts_with("Bearer ") => Some(s["Bearer ".len()..].trim().to_string()),
            _ => None,
        };

        let valid = token.as_deref().map(|t| jwt.verify_token(t).is_ok()).unwrap_or(false);

        if valid {
            Box::pin(async move {
                service
                    .call(req)
                    .await
                    .map(ServiceResponse::map_into_left_body)
            })
        } else {
            Box::pin(async move {
                let resp = HttpResponse::Unauthorized()
                    .insert_header(("WWW-Authenticate", r#"Bearer realm="kolosal""#))
                    .json(serde_json::json!({
                        "error": "authentication required",
                        "status": 401
                    }));
                Ok(req.into_response(resp).map_into_right_body())
            })
        }
    }
}
