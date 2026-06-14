//! Thin reverse proxy: forwards `/llm/{tail:.*}` → `http://<llm_host>:<llm_port>/{tail}`.
//! Returns 503 when the LLM microservice is not reachable.
use actix_web::{web, HttpRequest, HttpResponse};
use bytes::Bytes;
use futures_util::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

/// Forward any `/llm/{tail:.*}` request to the LLM microservice.
/// Host and port are read from `[microservices]` in config — no hardcoded values.
///
/// A fresh reqwest::Client is created per request so it is always bound to the
/// current actix-web worker's tokio `current_thread` runtime. Sharing a client
/// that was built on the main `multi_thread` runtime across worker threads
/// causes the hyper I/O driver to operate on the wrong runtime, which silently
/// fails with connection errors even when the upstream is reachable.
///
/// The request body is **streamed** through reqwest::Body::wrap_stream rather
/// than buffered into memory. Without this, a 1 GiB chat-completion request
/// (image attachments, long prompts) would consume 1 GiB on the proxy worker
/// before forwarding the first byte upstream.
pub async fn proxy(
    req: HttpRequest,
    payload: web::Payload,
    path: web::Path<String>,
    config: web::Data<crate::config::Config>,
) -> HttpResponse {
    // Build client inside the handler — bound to the current worker's runtime.
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(config.server.proxy_timeout_secs))
        .connect_timeout(std::time::Duration::from_secs(2))
        .no_proxy()
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(error = %e, "failed to build reqwest client for LLM proxy");
            return HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": "proxy client build failure"}));
        }
    };

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
    // Stream the request body upstream. `web::Payload` is `!Send` (it
    // owns an `Rc` into the actix worker payload), but reqwest's body
    // stream needs `Send`. Bridge via an mpsc channel: a local actix
    // task drains the payload into the channel; the receiver side is
    // a `Send` Stream that reqwest can consume.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(8);
    let mut payload = payload;
    actix_web::rt::spawn(async move {
        while let Some(chunk) = payload.next().await {
            let mapped = chunk.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()));
            if tx.send(mapped).await.is_err() {
                break; // upstream gone — stop pulling
            }
        }
    });
    let upstream_body = reqwest::Body::wrap_stream(ReceiverStream::new(rx));
    rb = rb.body(upstream_body);

    match rb.send().await {
        Err(e) if e.is_connect() || e.is_timeout() || e.is_builder() || e.is_request() => {
            tracing::warn!(
                url = %url,
                error = %e,
                is_connect = e.is_connect(),
                is_timeout = e.is_timeout(),
                is_builder = e.is_builder(),
                is_request = e.is_request(),
                "LLM proxy error"
            );
            HttpResponse::ServiceUnavailable().json(
                serde_json::json!({"error": "LLM service unavailable — run `make llm-build && make llm-run` to start it"})
            )
        }
        Err(e) => {
            tracing::warn!(url = %url, error = %e, "LLM proxy unexpected error");
            HttpResponse::BadGateway()
                .json(serde_json::json!({"error": format!("LLM proxy error: {}", e)}))
        }
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
