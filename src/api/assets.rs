use actix_web::{HttpResponse, Responder};
use bytes::Bytes;
use std::sync::OnceLock;

static REMIXICON_CSS: OnceLock<Bytes> = OnceLock::new();
static REMIXICON_WOFF2: OnceLock<Bytes> = OnceLock::new();

pub const REMIXICON_CDN_CSS: &str =
    "https://cdn.jsdelivr.net/npm/remixicon@4.7.0/fonts/remixicon.css";
const REMIXICON_CDN_WOFF2: &str =
    "https://cdn.jsdelivr.net/npm/remixicon@4.7.0/fonts/remixicon.woff2";

/// Rewrite the `url(...)` that references remixicon.woff2 in the @font-face block
/// so it points to our local `/assets/remixicon.woff2` route.
/// Works with both relative (`remixicon.woff2?v=4.7.0`) and absolute CDN URLs.
fn rewrite_woff2_src(css: &str) -> String {
    let marker = "remixicon.woff2";
    if let Some(woff2_pos) = css.find(marker) {
        let before_woff2 = &css[..woff2_pos];
        if let Some(url_start) = before_woff2.rfind("url(") {
            let after_url_open = url_start + 4; // skip past "url("
            if let Some(close_rel) = css[after_url_open..].find(')') {
                let close_abs = after_url_open + close_rel;
                let mut out = String::with_capacity(css.len());
                out.push_str(&css[..url_start]);
                out.push_str(r#"url("/assets/remixicon.woff2")"#);
                out.push_str(&css[close_abs + 1..]);
                return out;
            }
        }
    }
    css.to_owned()
}

/// Fetch Remixicon CSS and woff2 from jsDelivr, rewrite the font URL, and store
/// both into the `OnceLock` statics. Called once at server startup inside a
/// `tokio::spawn`. If the fetch fails the statics stay empty and handlers fall back
/// to redirecting to the CDN — icons still work, offline use degrades gracefully.
pub async fn fetch_remixicon() {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "remixicon: failed to build http client, using CDN fallback");
            return;
        }
    };

    let woff2_bytes = match client.get(REMIXICON_CDN_WOFF2).send().await {
        Ok(resp) => {
            if !resp.status().is_success() {
                tracing::warn!(status = %resp.status(), "remixicon: cdn returned non-2xx for woff2, using CDN fallback");
                return;
            }
            match resp.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(error = %e, "remixicon: failed to read woff2 body, using CDN fallback");
                    return;
                }
            }
        },
        Err(e) => {
            tracing::warn!(error = %e, "remixicon: failed to fetch woff2, using CDN fallback");
            return;
        }
    };

    let css_text = match client.get(REMIXICON_CDN_CSS).send().await {
        Ok(resp) => {
            if !resp.status().is_success() {
                tracing::warn!(status = %resp.status(), "remixicon: cdn returned non-2xx for css, using CDN fallback");
                return;
            }
            match resp.text().await {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(error = %e, "remixicon: failed to read css body, using CDN fallback");
                    return;
                }
            }
        },
        Err(e) => {
            tracing::warn!(error = %e, "remixicon: failed to fetch css, using CDN fallback");
            return;
        }
    };

    let css_rewritten = rewrite_woff2_src(&css_text);

    // OnceLock::set returns Err if already set — safe to ignore (idempotent on restart).
    let _ = REMIXICON_CSS.set(Bytes::from(css_rewritten));
    let _ = REMIXICON_WOFF2.set(woff2_bytes);
    tracing::info!(
        woff2_bytes = REMIXICON_WOFF2.get().map_or(0, |b| b.len()),
        "remixicon assets cached in memory"
    );
}

/// Serve the self-hosted Remixicon CSS with a 1-year immutable cache header.
/// Redirects to CDN if the startup fetch hasn't completed or failed.
pub async fn serve_remixicon_css() -> impl Responder {
    match REMIXICON_CSS.get() {
        Some(css) => HttpResponse::Ok()
            .content_type("text/css; charset=utf-8")
            .insert_header(("Cache-Control", "public, max-age=31536000, immutable"))
            .body(css.clone()),
        None => HttpResponse::TemporaryRedirect()
            .insert_header(("Location", REMIXICON_CDN_CSS))
            .finish(),
    }
}

/// Serve the self-hosted Remixicon woff2 font with a 1-year immutable cache header.
/// Redirects to CDN if the startup fetch hasn't completed or failed.
pub async fn serve_remixicon_woff2() -> impl Responder {
    match REMIXICON_WOFF2.get() {
        Some(woff2) => HttpResponse::Ok()
            .content_type("font/woff2")
            .insert_header(("Cache-Control", "public, max-age=31536000, immutable"))
            .body(woff2.clone()),
        None => HttpResponse::TemporaryRedirect()
            .insert_header(("Location", REMIXICON_CDN_WOFF2))
            .finish(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test as actix_test, web, App};

    #[actix_web::test]
    async fn test_remixicon_css_returns_200_when_cached() {
        // Ensure the static is populated (idempotent — second set is ignored)
        let _ = REMIXICON_CSS.set(Bytes::from_static(b"body{}"));
        // At this point REMIXICON_CSS is Some(_) — test the 200 branch
        let app = actix_test::init_service(
            App::new().route("/assets/remixicon.css", web::get().to(serve_remixicon_css)),
        )
        .await;
        let req = actix_test::TestRequest::get()
            .uri("/assets/remixicon.css")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200, "handler must return 200 when CSS is cached");
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(ct.contains("text/css"), "must be text/css, got: {}", ct);
    }

    #[actix_web::test]
    async fn test_remixicon_css_cache_control_immutable_when_cached() {
        // Ensure static is populated — safe to call multiple times (second set is ignored)
        let _ = REMIXICON_CSS.set(Bytes::from_static(b".ri{}"));
        // REMIXICON_CSS is Some(_) regardless of which test ran first
        let app = actix_test::init_service(
            App::new().route("/assets/remixicon.css", web::get().to(serve_remixicon_css)),
        )
        .await;
        let req = actix_test::TestRequest::get()
            .uri("/assets/remixicon.css")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        // The 200 branch sets this header — only verify cache-control, not content
        let cc = resp
            .headers()
            .get("cache-control")
            .expect("cache-control must be present on cached CSS response")
            .to_str()
            .unwrap();
        assert!(
            cc.contains("max-age=31536000") && cc.contains("immutable"),
            "expected immutable long-cache header, got: {}",
            cc
        );
    }

    #[actix_web::test]
    async fn test_remixicon_woff2_returns_307_when_not_cached() {
        if REMIXICON_WOFF2.get().is_none() {
            let app = actix_test::init_service(
                App::new()
                    .route("/assets/remixicon.woff2", web::get().to(serve_remixicon_woff2)),
            )
            .await;
            let req = actix_test::TestRequest::get()
                .uri("/assets/remixicon.woff2")
                .to_request();
            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(
                resp.status(),
                307,
                "expected 307 redirect to CDN when woff2 not cached"
            );
        }
    }

    #[test]
    fn test_rewrite_woff2_src_relative_url() {
        let input = r#"@font-face { src: url("remixicon.woff2?v=4.7.0") format('woff2'); }"#;
        let output = rewrite_woff2_src(input);
        assert!(
            output.contains(r#"url("/assets/remixicon.woff2")"#),
            "expected local URL, got: {}",
            output
        );
        assert!(
            !output.contains("remixicon.woff2?v="),
            "query string must be replaced"
        );
    }

    #[test]
    fn test_rewrite_woff2_src_absolute_cdn_url() {
        let input = r#"@font-face { src: url("https://cdn.jsdelivr.net/npm/remixicon@4.7.0/fonts/remixicon.woff2") format('woff2'); }"#;
        let output = rewrite_woff2_src(input);
        assert!(
            output.contains(r#"url("/assets/remixicon.woff2")"#),
            "expected local URL, got: {}",
            output
        );
    }

    #[test]
    fn test_rewrite_woff2_src_no_match_returns_unchanged() {
        let input = "body { color: red; }";
        let output = rewrite_woff2_src(input);
        assert_eq!(output, input);
    }
}
