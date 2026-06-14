# Frontend Infrastructure Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce playground.html wire size by ~78%, eliminate all external CDN dependencies, and add HTTP caching so repeat visits cost 0 bytes.

**Architecture:** Four orthogonal changes — system font stack (HTML-only), gzip Compress middleware, SHA-256 ETag conditional responses, and self-hosted Remixicon assets fetched once at startup into `OnceLock<Bytes>` statics.

**Tech Stack:** actix-web 4 `Compress` middleware, `sha2` 0.10 (already in Cargo.toml), `reqwest` 0.12 (already in Cargo.toml), `bytes` 1.5 (already in Cargo.toml), `std::sync::OnceLock`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/api/playground.html` | Modify | Remove Google Fonts links; update font CSS vars; swap CDN Remixicon link for local |
| `src/main.rs` | Modify | Add `Compress::default()` to App builder; spawn `fetch_remixicon()` at startup |
| `src/api/handlers.rs` | Modify | Add ETag + Cache-Control to `root()`; register `/assets/` routes |
| `src/api/assets.rs` | Create | `OnceLock<Bytes>` statics; `fetch_remixicon()` startup task; two actix handlers |
| `src/api/mod.rs` | Modify | Add `pub mod assets;` |

---

## Task 1: System Font Stack

Replace Google Fonts CDN with system fonts. Pure HTML/CSS change — no Rust files touched.

**Files:**
- Modify: `src/api/playground.html` lines 8–10 (remove link tags)
- Modify: `src/api/playground.html` lines 42–44 (update `:root` CSS vars)

- [ ] **Step 1: Remove the three Google Fonts link tags**

In `src/api/playground.html`, delete lines 8–10:
```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&family=Source+Serif+4:ital,opsz,wght@0,8..60,200..900;1,8..60,200..900&display=swap" rel="stylesheet" />
```

Leave line 11 (the Remixicon CDN link) intact — that is handled in Task 4.

- [ ] **Step 2: Update font CSS custom properties**

In the `:root` block (lines 42–44), replace:
```css
    --font:      'Inter', system-ui, sans-serif;
    --serif:     'Source Serif 4', Georgia, serif;
    --mono:      'IBM Plex Mono', monospace;
```
with:
```css
    --font:      -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    --serif:     Georgia, "Times New Roman", serif;
    --mono:      "SF Mono", "Fira Code", "Cascadia Code", "Fira Mono",
                 "Roboto Mono", ui-monospace, monospace;
```

No other CSS changes needed — all usages already go through `var(--font)`, `var(--serif)`, `var(--mono)`.

- [ ] **Step 3: Verify compilation**

```bash
cargo check 2>&1 | tail -5
```

Expected: `Finished` with no errors.

- [ ] **Step 4: Run tests**

```bash
cargo test --lib 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/api/playground.html
git commit -m "feat(frontend): replace Google Fonts CDN with system font stack"
```

---

## Task 2: Gzip Compression Middleware

Add `Compress::default()` to the actix-web `App` builder so all responses ≥ 1 KB are gzip-encoded when the client sends `Accept-Encoding: gzip`. Playground HTML drops from ~300 KB to ~66 KB on the wire.

**Files:**
- Modify: `src/main.rs` (import line 31; add `.wrap()` inside `HttpServer::new`)

- [ ] **Step 1: Add import**

In `src/main.rs`, replace line 31:
```rust
use actix_web::{web, App, HttpServer};
```
with:
```rust
use actix_web::{middleware::Compress, web, App, HttpServer};
```

- [ ] **Step 2: Add `.wrap(Compress::default())` to App builder**

In `src/main.rs`, inside `HttpServer::new(move || { App::new() ... })`, find the two existing `.wrap()` calls:
```rust
            .wrap(CorrelationIdMiddleware)
            .wrap(RequestLogger)
```
Add `Compress` as the last `.wrap()` (last = outermost in actix-web 4 = runs on the final response before bytes leave the server):
```rust
            .wrap(CorrelationIdMiddleware)
            .wrap(RequestLogger)
            .wrap(Compress::default())
```

- [ ] **Step 3: Verify compilation**

```bash
cargo check 2>&1 | tail -5
```

Expected: `Finished` with no errors.

- [ ] **Step 4: Run tests**

```bash
cargo test --lib 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat(server): add actix Compress middleware for gzip encoding"
```

---

## Task 3: ETag + Cache-Control for Playground

Add HTTP conditional-response support to the playground handler. On repeat visits with a matching `If-None-Match` header the server returns `304 Not Modified` (~400 bytes), saving the 300 KB transfer.

**Files:**
- Modify: `src/api/handlers.rs` (top of file for const + static + helper; `root()` function; `#[cfg(test)]` block)

- [ ] **Step 1: Write the failing tests**

In `src/api/handlers.rs`, add these four tests to the existing `#[cfg(test)] mod tests` block:

```rust
    #[actix_web::test]
    async fn test_root_etag_header_present() {
        let app = test::init_service(App::new().route("/", web::get().to(root))).await;
        let req = test::TestRequest::get().uri("/").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
        let etag = resp.headers().get("etag");
        assert!(etag.is_some(), "ETag header must be present on 200 response");
    }

    #[actix_web::test]
    async fn test_root_cache_control_is_no_cache() {
        let app = test::init_service(App::new().route("/", web::get().to(root))).await;
        let req = test::TestRequest::get().uri("/").to_request();
        let resp = test::call_service(&app, req).await;
        let cc = resp
            .headers()
            .get("cache-control")
            .expect("Cache-Control header must be present")
            .to_str()
            .unwrap();
        assert_eq!(cc, "no-cache");
    }

    #[actix_web::test]
    async fn test_root_304_on_matching_if_none_match() {
        let app = test::init_service(App::new().route("/", web::get().to(root))).await;
        // First request — learn the ETag
        let req1 = test::TestRequest::get().uri("/").to_request();
        let resp1 = test::call_service(&app, req1).await;
        let etag = resp1
            .headers()
            .get("etag")
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        // Second request — send matching ETag, expect 304
        let req2 = test::TestRequest::get()
            .uri("/")
            .insert_header(("if-none-match", etag.as_str()))
            .to_request();
        let resp2 = test::call_service(&app, req2).await;
        assert_eq!(resp2.status(), 304);
    }

    #[actix_web::test]
    async fn test_root_200_on_mismatched_if_none_match() {
        let app = test::init_service(App::new().route("/", web::get().to(root))).await;
        let req = test::TestRequest::get()
            .uri("/")
            .insert_header(("if-none-match", "\"stale-00000000\""))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
    }
```

- [ ] **Step 2: Run to verify the tests fail**

```bash
cargo test -p torch-inference-server "handlers::tests::test_root_etag" 2>&1 | tail -15
```

Expected: FAIL — `test_root_etag_header_present` fails because the current `root()` returns no `ETag` header.

- [ ] **Step 3: Implement ETag + conditional response**

In `src/api/handlers.rs`, add the following directly above the existing `root()` function:

```rust
const PLAYGROUND_HTML: &str = include_str!("playground.html");
static PLAYGROUND_ETAG: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn playground_etag() -> &'static str {
    PLAYGROUND_ETAG.get_or_init(|| {
        use sha2::Digest;
        let hash = sha2::Sha256::digest(PLAYGROUND_HTML.as_bytes());
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        format!("\"{}\"", hex)
    })
}
```

Then replace the existing `root()` function:

Old:
```rust
pub async fn root() -> impl Responder {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(include_str!("playground.html"))
}
```

New:
```rust
pub async fn root(req: HttpRequest) -> impl Responder {
    let etag = playground_etag();
    if let Some(inm) = req.headers().get("if-none-match") {
        if inm.to_str().unwrap_or("") == etag {
            return HttpResponse::NotModified()
                .insert_header(("ETag", etag))
                .finish();
        }
    }
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .insert_header(("ETag", etag))
        .insert_header(("Cache-Control", "no-cache"))
        .body(PLAYGROUND_HTML)
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p torch-inference-server "handlers::tests" 2>&1 | tail -25
```

Expected: all handler tests pass, including the four new ETag tests.

- [ ] **Step 5: Commit**

```bash
git add src/api/handlers.rs
git commit -m "feat(handlers): add SHA-256 ETag + Cache-Control: no-cache to playground route"
```

---

## Task 4: Remixicon Self-Hosting

Fetch Remixicon CSS and woff2 once at server startup, rewrite the `@font-face src:` URL to point locally, serve from two `/assets/` routes, and update `playground.html` to load from the local route. A CDN redirect fallback keeps icons working if the startup fetch fails.

**Files:**
- Create: `src/api/assets.rs`
- Modify: `src/api/mod.rs` (add `pub mod assets;`)
- Modify: `src/main.rs` (spawn `fetch_remixicon()` in bootstrap block)
- Modify: `src/api/handlers.rs:configure_routes` (register `/assets/` routes)
- Modify: `src/api/playground.html` (swap CDN link for `/assets/remixicon.css`)

- [ ] **Step 1: Create `src/api/assets.rs` with stub handlers and failing tests**

Create `src/api/assets.rs` with the following content. The stub handlers return 500 so the tests fail — the real implementation replaces them in Step 3.

```rust
use actix_web::{HttpResponse, Responder};
use bytes::Bytes;
use std::sync::OnceLock;

static REMIXICON_CSS: OnceLock<Bytes> = OnceLock::new();
static REMIXICON_WOFF2: OnceLock<Bytes> = OnceLock::new();

pub const REMIXICON_CDN_CSS: &str =
    "https://cdn.jsdelivr.net/npm/remixicon@4.7.0/fonts/remixicon.css";

// Stub — replaced in Step 3
pub async fn serve_remixicon_css() -> impl Responder {
    HttpResponse::InternalServerError().finish()
}

// Stub — replaced in Step 3
pub async fn serve_remixicon_woff2() -> impl Responder {
    HttpResponse::InternalServerError().finish()
}

// Stub — replaced in Step 3
pub async fn fetch_remixicon() {}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, web, App};

    #[actix_web::test]
    async fn test_remixicon_css_returns_200_when_cached() {
        // Pre-populate static to simulate a successful startup fetch.
        // OnceLock::set returns Err if already set — safe to ignore.
        let _ = REMIXICON_CSS.set(Bytes::from_static(b"body{}"));
        let app = test::init_service(
            App::new().route("/assets/remixicon.css", web::get().to(serve_remixicon_css)),
        )
        .await;
        let req = test::TestRequest::get()
            .uri("/assets/remixicon.css")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
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
        let _ = REMIXICON_CSS.set(Bytes::from_static(b".ri{}"));
        let app = test::init_service(
            App::new().route("/assets/remixicon.css", web::get().to(serve_remixicon_css)),
        )
        .await;
        let req = test::TestRequest::get()
            .uri("/assets/remixicon.css")
            .to_request();
        let resp = test::call_service(&app, req).await;
        let cc = resp
            .headers()
            .get("cache-control")
            .expect("cache-control must be present")
            .to_str()
            .unwrap();
        assert!(
            cc.contains("max-age=31536000") && cc.contains("immutable"),
            "expected immutable long-cache header, got: {}",
            cc
        );
    }
}
```

- [ ] **Step 2: Add module declaration and verify failing tests**

Add to `src/api/mod.rs` (append one line at the bottom):
```rust
pub mod assets;
```

Run:
```bash
cargo test -p torch-inference-server "assets::tests" 2>&1 | tail -20
```

Expected: both tests FAIL — stub returns 500, tests expect 200 and a `cache-control` header.

- [ ] **Step 3: Replace stub handlers with full implementation**

Replace the entire content of `src/api/assets.rs` with:

```rust
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
        Ok(resp) => match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, "remixicon: failed to read woff2 body, using CDN fallback");
                return;
            }
        },
        Err(e) => {
            tracing::warn!(error = %e, "remixicon: failed to fetch woff2, using CDN fallback");
            return;
        }
    };

    let css_text = match client.get(REMIXICON_CDN_CSS).send().await {
        Ok(resp) => match resp.text().await {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "remixicon: failed to read css body, using CDN fallback");
                return;
            }
        },
        Err(e) => {
            tracing::warn!(error = %e, "remixicon: failed to fetch css, using CDN fallback");
            return;
        }
    };

    let css_rewritten = rewrite_woff2_src(&css_text);

    // OnceLock::set returns Err if already set — safe to ignore (idempotent on restart).
    let _ = REMIXICON_WOFF2.set(woff2_bytes);
    let _ = REMIXICON_CSS.set(Bytes::from(css_rewritten));
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
    use actix_web::{test, web, App};

    #[actix_web::test]
    async fn test_remixicon_css_returns_200_when_cached() {
        let _ = REMIXICON_CSS.set(Bytes::from_static(b"body{}"));
        let app = test::init_service(
            App::new().route("/assets/remixicon.css", web::get().to(serve_remixicon_css)),
        )
        .await;
        let req = test::TestRequest::get()
            .uri("/assets/remixicon.css")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
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
        let _ = REMIXICON_CSS.set(Bytes::from_static(b".ri{}"));
        let app = test::init_service(
            App::new().route("/assets/remixicon.css", web::get().to(serve_remixicon_css)),
        )
        .await;
        let req = test::TestRequest::get()
            .uri("/assets/remixicon.css")
            .to_request();
        let resp = test::call_service(&app, req).await;
        let cc = resp
            .headers()
            .get("cache-control")
            .expect("cache-control must be present")
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
        // Only runs the assertion when the static is not yet populated by another test.
        if REMIXICON_WOFF2.get().is_none() {
            let app = test::init_service(
                App::new()
                    .route("/assets/remixicon.woff2", web::get().to(serve_remixicon_woff2)),
            )
            .await;
            let req = test::TestRequest::get()
                .uri("/assets/remixicon.woff2")
                .to_request();
            let resp = test::call_service(&app, req).await;
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p torch-inference-server "assets::tests" 2>&1 | tail -20
```

Expected: all 6 asset tests pass.

- [ ] **Step 5: Register `/assets/` routes in `configure_routes`**

In `src/api/handlers.rs`, at the end of the `configure_routes` function, just before the last closing semicolon of the chained `.configure(crate::api::stt_proxy::configure_routes)` call, add:

```rust
        // Self-hosted static assets (fonts, icons)
        .route(
            "/assets/remixicon.css",
            web::get().to(crate::api::assets::serve_remixicon_css),
        )
        .route(
            "/assets/remixicon.woff2",
            web::get().to(crate::api::assets::serve_remixicon_woff2),
        )
```

- [ ] **Step 6: Spawn `fetch_remixicon()` at server startup**

In `src/main.rs`, add a new `tokio::spawn` right after the existing Whisper spawn block (which ends around line 468):

```rust
    // Prefetch Remixicon CSS + woff2 from jsDelivr into memory so /assets/ routes are self-hosted.
    tokio::spawn(crate::api::assets::fetch_remixicon());
```

- [ ] **Step 7: Update playground.html to load Remixicon from local route**

In `src/api/playground.html`, replace the Remixicon CDN link (now line 8 after Task 1 removed the three Google Fonts lines above it):

Old:
```html
<link href="https://cdn.jsdelivr.net/npm/remixicon@4.7.0/fonts/remixicon.css" rel="stylesheet" />
```

New:
```html
<link href="/assets/remixicon.css" rel="stylesheet" />
```

- [ ] **Step 8: Verify full compilation**

```bash
cargo check 2>&1 | tail -10
```

Expected: `Finished` with no errors.

- [ ] **Step 9: Run all tests**

```bash
cargo test 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/api/assets.rs src/api/mod.rs src/main.rs src/api/handlers.rs src/api/playground.html
git commit -m "feat(assets): self-host Remixicon; serve from /assets/ routes with CDN fallback"
```

---

## Success Criteria

| Metric | Before | After |
|--------|--------|-------|
| Cold load wire size | ~300 KB HTML + ~88 KB CDN fonts | ~66 KB HTML (gzip) + ~88 KB Remixicon (first visit only, then 1-year cached) |
| Repeat visit (unchanged HTML) | ~300 KB | ~400 bytes (304 Not Modified) |
| External requests on page load | 5 (fonts.googleapis × 1, fonts.gstatic × 1, jsDelivr CSS × 1, jsDelivr woff2 × 1, redirect × 1) | 0 |
| Offline capable | No | Yes (after first visit) |
