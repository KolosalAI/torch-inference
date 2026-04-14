//! Thin reverse proxy: forwards `/llm/{tail:.*}` → `http://<llm_host>:<llm_port>/{tail}`.
//! Returns 503 when the LLM microservice is not reachable.
use actix_web::{web, HttpRequest, HttpResponse};
use bytes::Bytes;
use futures_util::StreamExt;

/// Forward any `/llm/{tail:.*}` request to the LLM microservice.
/// Host and port are read from `[microservices]` in config — no hardcoded values.
pub async fn proxy(
    req: HttpRequest,
    body: Bytes,
    path: web::Path<String>,
    client: web::Data<reqwest::Client>,
    config: web::Data<crate::config::Config>,
) -> HttpResponse {
    let tail = path.into_inner();
    let base = config.microservices.llm_base_url();
    let url = format!("{}/{}", base, tail);

    let url = if let Some(qs) = req.uri().query() {
        format!("{}?{}", url, qs)
    } else {
        url
    };

    let method = reqwest::Method::from_bytes(req.method().as_str().as_bytes())
        .unwrap_or(reqwest::Method::GET);

    let mut rb = client.request(method, &url);
    for (name, value) in req.headers() {
        let lower = name.as_str().to_lowercase();
        // Strip hop-by-hop headers and headers managed by the HTTP client.
        match lower.as_str() {
            "host" | "content-length" | "connection" | "keep-alive"
            | "transfer-encoding" | "te" | "trailers"
            | "proxy-authorization" | "proxy-connection" | "upgrade" => continue,
            _ => {}
        }
        // Validate header name before forwarding — invalid names (e.g. HTTP/2
        // pseudo-headers starting with ':') cause a reqwest builder error.
        let Ok(hname) = reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes()) else { continue };
        if let Ok(v) = reqwest::header::HeaderValue::from_bytes(value.as_bytes()) {
            rb = rb.header(hname, v);
        }
    }
    rb = rb.body(body);

    match rb.send().await {
        Err(e) if e.is_connect() || e.is_timeout() => {
            HttpResponse::ServiceUnavailable().json(
                serde_json::json!({"error": "LLM service unavailable — start it with `make llm-run`"})
            )
        }
        Err(e) => HttpResponse::BadGateway()
            .json(serde_json::json!({"error": e.to_string()})),
        Ok(upstream) => {
            let status = actix_web::http::StatusCode::from_u16(upstream.status().as_u16())
                .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
            let mut resp = HttpResponse::build(status);

            for (name, value) in upstream.headers() {
                let lower = name.as_str().to_lowercase();
                if lower == "transfer-encoding" || lower == "content-length" {
                    continue;
                }
                if let Ok(v) = actix_web::http::header::HeaderValue::from_bytes(value.as_bytes()) {
                    resp.insert_header((name.as_str(), v));
                }
            }

            // Stream the upstream body without buffering — critical for SSE.
            let stream = upstream
                .bytes_stream()
                .map(|r| r.map_err(actix_web::error::ErrorBadGateway));
            resp.streaming(stream)
        }
    }
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/llm/{tail:.*}", web::to(proxy));
}
