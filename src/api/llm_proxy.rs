//! Thin reverse proxy: forwards `/llm/{tail:.*}` → `http://127.0.0.1:8001/{tail}`.
//! Returns 503 when the LLM microservice is not reachable.
use actix_web::{web, HttpRequest, HttpResponse};
use bytes::Bytes;

static LLM_BASE: &str = "http://127.0.0.1:8001";

/// Forward any `/llm/{tail:.*}` request to the LLM microservice.
pub async fn proxy(
    req: HttpRequest,
    body: Bytes,
    path: web::Path<String>,
) -> HttpResponse {
    let tail = path.into_inner();
    let url = format!("{}/{}", LLM_BASE, tail);

    // Preserve query string
    let url = if let Some(qs) = req.uri().query() {
        format!("{}?{}", url, qs)
    } else {
        url
    };

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": e.to_string()}));
        }
    };

    let method = reqwest::Method::from_bytes(req.method().as_str().as_bytes())
        .unwrap_or(reqwest::Method::GET);

    let mut rb = client.request(method, &url);
    for (name, value) in req.headers() {
        let lower = name.as_str().to_lowercase();
        if lower == "host" || lower == "content-length" {
            continue;
        }
        if let Ok(v) = reqwest::header::HeaderValue::from_bytes(value.as_bytes()) {
            rb = rb.header(name.as_str(), v);
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

            let bytes = match upstream.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    return HttpResponse::BadGateway()
                        .json(serde_json::json!({"error": e.to_string()}));
                }
            };
            resp.body(bytes)
        }
    }
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/llm/{tail:.*}", web::to(proxy));
}
