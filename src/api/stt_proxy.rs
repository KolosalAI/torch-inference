//! Thin reverse proxy: forwards `/stt/{tail:.*}` → `http://<stt_host>:<stt_port>/{tail}`.
//! Returns 503 when the STT microservice is not reachable.
use actix_web::{web, HttpRequest, HttpResponse};
use bytes::Bytes;
use futures_util::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

pub async fn proxy(
    req: HttpRequest,
    payload: web::Payload,
    path: web::Path<String>,
    config: web::Data<crate::config::Config>,
) -> HttpResponse {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(config.server.proxy_timeout_secs))
        .connect_timeout(std::time::Duration::from_secs(2))
        .no_proxy()
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(error = %e, "failed to build reqwest client for STT proxy");
            return HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": "proxy client build failure"}));
        }
    };

    let tail = path.into_inner();
    let base = config.microservices.stt_base_url();
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
        match lower.as_str() {
            "host" | "content-length" | "connection" | "keep-alive"
            | "transfer-encoding" | "te" | "trailers"
            | "proxy-authorization" | "proxy-connection" | "upgrade" => continue,
            _ => {}
        }
        let Ok(hname) = reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes()) else { continue };
        if let Ok(v) = reqwest::header::HeaderValue::from_bytes(value.as_bytes()) {
            rb = rb.header(hname, v);
        }
    }
    // web::Payload is !Send (Rc internally) — bridge through an mpsc so
    // reqwest's wrap_stream gets a Send receiver.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(8);
    let mut payload = payload;
    actix_web::rt::spawn(async move {
        while let Some(chunk) = payload.next().await {
            let mapped = chunk.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()));
            if tx.send(mapped).await.is_err() {
                break;
            }
        }
    });
    let upstream_body = reqwest::Body::wrap_stream(ReceiverStream::new(rx));
    rb = rb.body(upstream_body);

    match rb.send().await {
        Err(e) if e.is_connect() || e.is_timeout() || e.is_builder() || e.is_request() => {
            HttpResponse::ServiceUnavailable().json(
                serde_json::json!({"error": "STT service unavailable — run `make stt-build && make stt-run` to start it"})
            )
        }
        Err(e) => HttpResponse::BadGateway()
            .json(serde_json::json!({"error": format!("STT proxy error: {}", e)})),
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

            let stream = upstream
                .bytes_stream()
                .map(|r| r.map_err(actix_web::error::ErrorBadGateway));
            resp.streaming(stream)
        }
    }
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/stt/{tail:.*}", web::to(proxy));
}
