# Frontend Infrastructure Optimization Design

**Goal:** Reduce playground.html wire size by ~78%, eliminate all external CDN dependencies, and add HTTP caching so repeat visits cost 0 bytes.

**Architecture:** Three orthogonal changes to the existing single-binary Rust server — compression middleware, self-hosted Remixicon assets, and a system font stack. No build toolchain added.

**Tech Stack:** actix-web 4 Compress middleware, SHA-256 ETag (via `sha2` crate or stdlib), `reqwest` for startup asset fetch, CSS custom properties.

---

## Section 1 — Compression + HTTP Caching

### Compression

Add `actix_web::middleware::Compress::default()` to the `App` builder in `src/main.rs`. This middleware automatically gzip-encodes any response ≥1KB when the client sends `Accept-Encoding: gzip`. No per-handler changes needed.

Effect: playground.html drops from 300KB to ~66KB on the wire (78% reduction).

### ETag + Conditional Responses

The playground handler (`src/api/handlers.rs`) currently returns the HTML with no caching headers. Add:

- Compute `PLAYGROUND_ETAG: &str` once at compile time or startup as a hex digest of the embedded HTML bytes.
- On each request, check `If-None-Match` header against the ETag.
  - Match → return `304 Not Modified` with empty body (0 bytes transferred).
  - No match → return `200 OK` with full HTML + `ETag` + `Cache-Control: no-cache` headers.

`Cache-Control: no-cache` means: "cache it, but always revalidate." This gives zero-byte repeat visits when the file is unchanged, while ensuring browsers always pick up new deployments immediately.

**Files changed:**
- `src/main.rs` — add `Compress::default()` wrap
- `src/api/handlers.rs` — add ETag computation + conditional response logic

---

## Section 2 — Self-hosted Remixicon

### Problem

On every page load, the browser makes two external requests:
1. `https://cdn.jsdelivr.net/npm/remixicon@4.7.0/fonts/remixicon.css` (~8KB)
2. `https://cdn.jsdelivr.net/npm/remixicon@4.7.0/fonts/remixicon.woff2` (~80KB)

This adds latency, creates a CDN availability dependency, and prevents offline use.

### Solution

At server startup, fetch both resources into memory once via `reqwest`. Serve them from two new actix routes:

- `GET /assets/remixicon.css` — returns CSS with `content-type: text/css`, `Cache-Control: public, max-age=31536000, immutable`
- `GET /assets/remixicon.woff2` — returns font bytes with `content-type: font/woff2`, same cache headers

The CSS served by `/assets/remixicon.css` has its `@font-face` `src:` URL rewritten to point to `/assets/remixicon.woff2` instead of the CDN.

In `playground.html`, replace:
```html
<link href="https://cdn.jsdelivr.net/npm/remixicon@4.7.0/fonts/remixicon.css" rel="stylesheet" />
```
with:
```html
<link href="/assets/remixicon.css" rel="stylesheet" />
```

### Fallback

If the startup fetch fails (no internet, CDN down), the server logs a warning and falls back: `/assets/remixicon.css` returns a redirect to the CDN URL. Icons still work; offline use degrades gracefully.

### Asset Storage

Two `OnceLock<Bytes>` statics hold the CSS and woff2 data after startup. The fetch runs in the existing bootstrap `tokio::spawn` block in `src/main.rs`.

**Files changed:**
- `src/main.rs` — spawn Remixicon fetch into bootstrap block; store in statics
- `src/api/handlers.rs` — add two `/assets/` route handlers
- `src/api/playground.html` — replace CDN `<link>` with `/assets/remixicon.css`

---

## Section 3 — System Font Stack

### Problem

Three Google Fonts resources load on every first visit:
1. `preconnect` to `fonts.googleapis.com`
2. `preconnect` to `fonts.gstatic.com`
3. CSS + woff2 fetches for IBM Plex Mono, Inter, Source Serif 4

These add 2–3 network round trips, block text rendering until resolved, and create an external dependency.

### Solution

Remove the two `<link rel="preconnect">` tags and the Google Fonts `<link>` entirely. Replace the font references in the CSS with a system font stack via CSS custom properties added to the existing `:root` block:

```css
:root {
  --font-sans:  -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                "Helvetica Neue", Arial, sans-serif;
  --font-mono:  "SF Mono", "Fira Code", "Cascadia Code", "Fira Mono",
                "Roboto Mono", ui-monospace, monospace;
  --font-serif: Georgia, "Times New Roman", serif;
}
```

Replace font-family declarations throughout the CSS:
- `"Inter"` → `var(--font-sans)`
- `"IBM Plex Mono"` → `var(--font-mono)`
- `"Source Serif 4"` → `var(--font-serif)`

Platform rendering:
- **macOS:** `-apple-system` (San Francisco) + `SF Mono` — identical visual weight to Inter/IBM Plex Mono
- **Windows:** `Segoe UI` + `Cascadia Code`
- **Linux:** `Roboto` + `Fira Code` (if installed) → `DejaVu Sans Mono` fallback

**Files changed:**
- `src/api/playground.html` — remove 3 `<link>` tags; add `--font-*` vars to `:root`; replace 3 font-family references

---

## Non-Goals

- No JavaScript minification or tree-shaking (no build toolchain)
- No HTTP/2 push
- No service worker / offline PWA
- No brotli (actix-web 4's built-in Compress covers gzip; brotli requires a separate crate and is a future improvement)

---

## Success Criteria

| Metric | Before | After |
|--------|--------|-------|
| Cold load wire size | 300KB HTML + 88KB CDN | ~66KB HTML (gzip) + 88KB Remixicon (first visit, then cached) |
| Repeat visit (unchanged) | 300KB | ~400 bytes (304 response) |
| External requests on load | 5 (fonts.googleapis×1, fonts.gstatic×1, jsDelivr×2, font woff2×1) | 0 |
| Offline capable | No | Yes (after first visit) |
