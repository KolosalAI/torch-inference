/// Batched image classification endpoint.
///
/// POST /classify/batch
///   Body: `{ "images": ["<base64>", ...], "top_k": 5, "model_width": 224, "model_height": 224 }`
///   Response: `{ "results": [[{"label":"..","confidence":0.9},...], ...], "batch_size": N }`
///
/// The handler is decoupled from ORT via the [`ClassificationBackend`] trait so
/// the endpoint can be unit-tested with a mock without a real `.onnx` file.
use actix_web::{web, HttpRequest, HttpResponse};
use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio_stream::wrappers::ReceiverStream;

use crate::config::Config;
use crate::core::image_pipeline::{ImagePipeline, PreprocessConfig};
use crate::error::ApiError;
use crate::middleware::correlation_id::get_correlation_id;
use crate::postprocess::{self, envelope::ResponseMeta, Envelope};

// ── Request / response types ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BatchClassifyRequest {
    /// Base64-encoded images (JPEG or PNG).
    pub images: Vec<String>,
    /// Number of top predictions to return per image (default 5).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Target width for the model (default 224).
    #[serde(default = "default_model_dim")]
    pub model_width: u32,
    /// Target height for the model (default 224).
    #[serde(default = "default_model_dim")]
    pub model_height: u32,
    #[serde(default)]
    pub skip_postprocess: bool,
}

fn default_top_k() -> usize {
    5
}
fn default_model_dim() -> u32 {
    224
}

#[derive(Debug, Serialize, Clone)]
pub struct Prediction {
    pub label: String,
    pub confidence: f32,
    pub class_id: usize,
}

#[derive(Debug, Serialize)]
pub struct BatchClassifyResponse {
    /// One Vec<Prediction> per input image, in submission order.
    pub results: Vec<Vec<Prediction>>,
    pub batch_size: usize,
}

// ── Backend trait ─────────────────────────────────────────────────────────

/// Abstraction over the inference backend.
///
/// Implement this trait on a struct that holds an ORT `Session` to wire
/// real ONNX inference. Use [`MockClassificationBackend`] in tests.
#[async_trait]
pub trait ClassificationBackend: Send + Sync {
    /// Run classification on a pre-normalised NCHW f32 batch.
    ///
    /// `batch` is shaped `[N, 3, H, W]`.  Return one `Vec<Prediction>` per
    /// image, sorted by confidence descending, truncated to `top_k`.
    async fn classify_nchw(
        &self,
        batch: ndarray::Array4<f32>,
        top_k: usize,
    ) -> anyhow::Result<Vec<Vec<Prediction>>>;
}

// ── App state ─────────────────────────────────────────────────────────────

pub struct ClassifyState {
    pub backend: Arc<dyn ClassificationBackend>,
}

// ── Handler ───────────────────────────────────────────────────────────────

/// POST /classify/batch
pub async fn batch_classify(
    req: web::Json<BatchClassifyRequest>,
    state: web::Data<ClassifyState>,
    config: web::Data<Config>,
    http_req: HttpRequest,
) -> Result<HttpResponse, ApiError> {
    let start = Instant::now();

    if req.images.is_empty() {
        return Err(ApiError::BadRequest(
            "images array must not be empty".to_string(),
        ));
    }
    if req.images.len() > 128 {
        return Err(ApiError::BadRequest(
            "batch too large (max 128 images)".to_string(),
        ));
    }
    if req.top_k == 0 {
        return Err(ApiError::BadRequest("top_k must be >= 1".to_string()));
    }
    if req.top_k > 1000 {
        return Err(ApiError::BadRequest("top_k must be <= 1000".to_string()));
    }
    if req.model_width == 0 || req.model_height == 0 {
        return Err(ApiError::BadRequest(
            "model_width and model_height must be >= 1".to_string(),
        ));
    }
    if req.model_width > 4096 || req.model_height > 4096 {
        return Err(ApiError::BadRequest(
            "model_width and model_height must be <= 4096".to_string(),
        ));
    }

    // Decode base64 → raw bytes.
    use base64::Engine as _;
    let raw_images: Vec<Vec<u8>> = req
        .images
        .iter()
        .enumerate()
        .map(|(i, b64)| {
            base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| ApiError::BadRequest(format!("image[{}]: {}", i, e)))
        })
        .collect::<Result<_, _>>()?;

    // Preprocess: decode → resize → normalise → NCHW f32.
    let cfg = PreprocessConfig::imagenet(req.model_width, req.model_height);
    let pipeline = ImagePipeline::new(cfg);
    let batch = pipeline
        .preprocess_batch(&raw_images)
        .map_err(|e| ApiError::BadRequest(format!("preprocess failed: {}", e)))?;

    // Run inference via the backend.
    let results = state
        .backend
        .classify_nchw(batch, req.top_k)
        .await
        .map_err(|e| ApiError::InternalError(format!("inference failed: {}", e)))?;

    // Post-process
    let (results, pp_steps, pp_warnings) = if !req.skip_postprocess {
        let pp = postprocess::classify::process(results, &config.postprocess.classify);
        (pp.predictions, pp.steps, pp.warnings)
    } else {
        (results, vec![], vec![])
    };

    let data = BatchClassifyResponse {
        batch_size: results.len(),
        results,
    };

    let envelope = Envelope::new(
        data,
        ResponseMeta {
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            model_id: "classification-backend".to_string(),
            postprocessing_applied: !req.skip_postprocess && !pp_steps.is_empty(),
            postprocess_steps: pp_steps,
            warnings: pp_warnings,
            version: env!("CARGO_PKG_VERSION"),
            request_id: get_correlation_id(&http_req).as_str().to_string(),
        },
    );

    Ok(HttpResponse::Ok().json(envelope))
}

/// Configure /classify routes.
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/classify")
            .route("/batch", web::post().to(batch_classify))
            .route("/stream", web::post().to(stream_classify)),
    );
}

// ── SSE streaming batch classify ──────────────────────────────────────────────

/// POST /classify/stream
///
/// Accepts the same JSON body as `/classify/batch` but returns
/// `text/event-stream` (SSE), emitting one event per image so the client sees
/// results progressively instead of waiting for the whole batch.
///
/// **Event format (JSON in the `data:` field)**
/// ```text
/// data: {"idx":0,"total":5,"ms":9.1,"predictions":[{"label":"cat","confidence":0.95,"class_id":281},...]}
/// data: {"idx":1,"total":5,"ms":8.6,"predictions":[...]}
/// data: {"type":"done","total":5,"batch_ms":47.2}
/// ```
pub async fn stream_classify(
    req: web::Json<BatchClassifyRequest>,
    state: web::Data<ClassifyState>,
) -> Result<HttpResponse, ApiError> {
    if req.images.is_empty() {
        return Err(ApiError::BadRequest("images must not be empty".to_string()));
    }
    if req.images.len() > 128 {
        return Err(ApiError::BadRequest("batch too large (max 128)".to_string()));
    }
    let top_k    = req.top_k.clamp(1, 1000);
    let width    = req.model_width.clamp(1, 4096);
    let height   = req.model_height.clamp(1, 4096);
    let images   = req.into_inner().images;
    let total    = images.len();
    let backend  = state.into_inner().backend.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(32);

    tokio::spawn(async move {
        use base64::Engine as _;
        let batch_start = Instant::now();

        for (idx, b64) in images.iter().enumerate() {
            let img_start = Instant::now();

            let raw = match base64::engine::general_purpose::STANDARD.decode(b64) {
                Ok(b) => b,
                Err(e) => {
                    let ev = sse_event(&serde_json::json!({
                        "idx": idx, "total": total, "error": e.to_string()
                    }));
                    let _ = tx.send(Ok(ev)).await;
                    continue;
                }
            };

            let cfg = PreprocessConfig::imagenet(width, height);
            let pipeline = ImagePipeline::new(cfg);
            let batch = match pipeline.preprocess_batch(&[raw]) {
                Ok(b) => b,
                Err(e) => {
                    let ev = sse_event(&serde_json::json!({
                        "idx": idx, "total": total, "error": e.to_string()
                    }));
                    let _ = tx.send(Ok(ev)).await;
                    continue;
                }
            };

            let preds = match backend.classify_nchw(batch, top_k).await {
                Ok(mut v) => v.pop().unwrap_or_default(),
                Err(e) => {
                    let ev = sse_event(&serde_json::json!({
                        "idx": idx, "total": total, "error": e.to_string()
                    }));
                    let _ = tx.send(Ok(ev)).await;
                    continue;
                }
            };

            let ms = img_start.elapsed().as_secs_f64() * 1000.0;
            let ev = sse_event(&serde_json::json!({
                "idx": idx,
                "total": total,
                "ms": (ms * 10.0).round() / 10.0,
                "predictions": preds,
            }));
            if tx.send(Ok(ev)).await.is_err() {
                break;
            }
        }

        let batch_ms = batch_start.elapsed().as_secs_f64() * 1000.0;
        let done = sse_event(&serde_json::json!({
            "type": "done",
            "total": total,
            "batch_ms": (batch_ms * 10.0).round() / 10.0,
        }));
        let _ = tx.send(Ok(done)).await;
    });

    let stream = ReceiverStream::new(rx);
    Ok(HttpResponse::Ok()
        .content_type("text/event-stream; charset=utf-8")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(stream))
}

fn sse_event(json: &serde_json::Value) -> Bytes {
    Bytes::from(format!("data: {}\n\n", json))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;
    use actix_web::{test as actix_test, App};

    // ── Mock backend ─────────────────────────────────────────────────────

    /// Returns `top_k` dummy predictions for each image in the batch.
    pub struct MockClassificationBackend {
        pub labels: Vec<String>,
    }

    impl Default for MockClassificationBackend {
        fn default() -> Self {
            Self {
                labels: vec![
                    "cat".into(),
                    "dog".into(),
                    "bird".into(),
                    "fish".into(),
                    "horse".into(),
                    "car".into(),
                ],
            }
        }
    }

    #[async_trait]
    impl ClassificationBackend for MockClassificationBackend {
        async fn classify_nchw(
            &self,
            batch: ndarray::Array4<f32>,
            top_k: usize,
        ) -> anyhow::Result<Vec<Vec<Prediction>>> {
            let n = batch.shape()[0];
            Ok((0..n)
                .map(|_| {
                    self.labels
                        .iter()
                        .enumerate()
                        .take(top_k)
                        .map(|(i, label)| Prediction {
                            label: label.clone(),
                            confidence: 1.0 / (i + 1) as f32,
                            class_id: i,
                        })
                        .collect()
                })
                .collect())
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn make_state() -> web::Data<ClassifyState> {
        web::Data::new(ClassifyState {
            backend: Arc::new(MockClassificationBackend::default()),
        })
    }

    /// 1×1 solid-color PNG as base64.
    fn tiny_b64(r: u8, g: u8, b: u8) -> String {
        use base64::Engine as _;
        use image::{DynamicImage, ImageBuffer, Rgb};
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(1, 1, Rgb([r, g, b])));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        base64::engine::general_purpose::STANDARD.encode(buf.into_inner())
    }

    // ── Request deserialization ──────────────────────────────────────────

    #[test]
    fn test_batch_classify_request_defaults() {
        let json = r#"{"images": ["abc"]}"#;
        let req: BatchClassifyRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.top_k, 5);
        assert_eq!(req.model_width, 224);
        assert_eq!(req.model_height, 224);
    }

    #[test]
    fn test_batch_classify_request_custom_fields() {
        let json = r#"{"images": ["abc"], "top_k": 3, "model_width": 448, "model_height": 448}"#;
        let req: BatchClassifyRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.top_k, 3);
        assert_eq!(req.model_width, 448);
        assert_eq!(req.model_height, 448);
    }

    // ── Response serialization ───────────────────────────────────────────

    #[test]
    fn test_batch_classify_response_serialization() {
        let resp = BatchClassifyResponse {
            results: vec![vec![Prediction {
                label: "cat".into(),
                confidence: 0.95,
                class_id: 0,
            }]],
            batch_size: 1,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["batch_size"], 1);
        assert_eq!(v["results"][0][0]["label"], "cat");
    }

    #[test]
    fn test_prediction_serialization() {
        let p = Prediction {
            label: "dog".into(),
            confidence: 0.8,
            class_id: 1,
        };
        let json = serde_json::to_string(&p).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["label"], "dog");
        assert!((v["confidence"].as_f64().unwrap() - 0.8).abs() < 1e-4);
        assert_eq!(v["class_id"], 1);
    }

    // ── Handler tests ────────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_batch_classify_success() {
        let state = make_state();
        let app = actix_test::init_service(
            App::new()
                .app_data(state)
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;

        let img = tiny_b64(200, 100, 50);
        let payload = serde_json::json!({ "images": [img] });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_batch_classify_response_has_correct_batch_size() {
        let state = make_state();
        let app = actix_test::init_service(
            App::new()
                .app_data(state)
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;

        let img = tiny_b64(128, 64, 32);
        let payload = serde_json::json!({ "images": [img, img.clone()] });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let body: serde_json::Value = actix_test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["data"]["batch_size"], 2);
        assert_eq!(body["data"]["results"].as_array().unwrap().len(), 2);
    }

    #[actix_web::test]
    async fn test_batch_classify_top_k_limits_predictions() {
        let state = make_state();
        let app = actix_test::init_service(
            App::new()
                .app_data(state)
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;

        let img = tiny_b64(100, 100, 100);
        let payload = serde_json::json!({ "images": [img], "top_k": 3 });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let body: serde_json::Value = actix_test::call_and_read_body_json(&app, req).await;
        let preds = body["data"]["results"][0].as_array().unwrap();
        assert_eq!(preds.len(), 3);
    }

    #[actix_web::test]
    async fn test_batch_classify_empty_images_returns_bad_request() {
        let state = make_state();
        let app = actix_test::init_service(
            App::new()
                .app_data(state)
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;

        let payload = serde_json::json!({ "images": [] });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_batch_classify_top_k_zero_returns_bad_request() {
        let state = make_state();
        let app = actix_test::init_service(
            App::new()
                .app_data(state)
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;

        let img = tiny_b64(1, 2, 3);
        let payload = serde_json::json!({ "images": [img], "top_k": 0 });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_batch_classify_invalid_base64_returns_bad_request() {
        let state = make_state();
        let app = actix_test::init_service(
            App::new()
                .app_data(state)
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;

        let payload = serde_json::json!({ "images": ["!!!not_base64!!!"] });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_batch_classify_batch_too_large_returns_bad_request() {
        let state = make_state();
        let app = actix_test::init_service(
            App::new()
                .app_data(state)
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;

        let img = tiny_b64(0, 0, 0);
        let images: Vec<String> = std::iter::repeat(img).take(129).collect();
        let payload = serde_json::json!({ "images": images });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_batch_classify_valid_base64_but_invalid_image_returns_bad_request() {
        let state = make_state();
        let app = actix_test::init_service(
            App::new()
                .app_data(state)
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;

        // Valid base64 but not an image.
        use base64::Engine as _;
        let not_image = base64::engine::general_purpose::STANDARD.encode(b"not an image");
        let payload = serde_json::json!({ "images": [not_image] });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_batch_classify_predictions_have_required_fields() {
        let state = make_state();
        let app = actix_test::init_service(
            App::new()
                .app_data(state)
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;

        let img = tiny_b64(255, 0, 0);
        let payload = serde_json::json!({ "images": [img], "top_k": 2 });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let body: serde_json::Value = actix_test::call_and_read_body_json(&app, req).await;
        let pred = &body["data"]["results"][0][0];
        assert!(pred["label"].is_string());
        assert!(pred["confidence"].is_number());
        assert!(pred["class_id"].is_number());
    }

    #[actix_web::test]
    async fn test_configure_routes_registers_batch_endpoint() {
        let state = make_state();
        let app = actix_test::init_service(
            App::new()
                .app_data(state)
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;

        let img = tiny_b64(50, 100, 150);
        let payload = serde_json::json!({ "images": [img] });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // ── Serde helpers ────────────────────────────────────────────────────

    #[test]
    fn test_default_top_k_is_5() {
        assert_eq!(default_top_k(), 5);
    }

    #[test]
    fn test_default_model_dim_is_224() {
        assert_eq!(default_model_dim(), 224);
    }

    #[test]
    fn test_prediction_clone() {
        let p = Prediction {
            label: "cat".into(),
            confidence: 0.9,
            class_id: 0,
        };
        let p2 = p.clone();
        assert_eq!(p2.label, "cat");
        assert!((p2.confidence - 0.9).abs() < 1e-6);
    }

    // ── Input bounds validation ──────────────────────────────────────────────

    #[actix_web::test]
    async fn test_top_k_over_1000_returns_bad_request() {
        let app = actix_test::init_service(
            App::new()
                .app_data(make_state())
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({
            "images": [tiny_b64(4, 4, 100)],
            "top_k": 1001
        });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_zero_model_dim_returns_bad_request() {
        let app = actix_test::init_service(
            App::new()
                .app_data(make_state())
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({
            "images": [tiny_b64(4, 4, 100)],
            "model_width": 0,
            "model_height": 224
        });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_oversized_model_dim_returns_bad_request() {
        let app = actix_test::init_service(
            App::new()
                .app_data(make_state())
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({
            "images": [tiny_b64(4, 4, 100)],
            "model_width": 5000,
            "model_height": 224
        });
        let req = actix_test::TestRequest::post()
            .uri("/classify/batch")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    // ── Envelope / postprocessing meta ───────────────────────────────────────

    #[actix_web::test]
    async fn test_envelope_meta_has_required_fields() {
        let app = actix_test::init_service(
            App::new()
                .app_data(make_state())
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;
        let body: serde_json::Value = actix_test::call_and_read_body_json(
            &app,
            actix_test::TestRequest::post()
                .uri("/classify/batch")
                .set_json(&serde_json::json!({ "images": [tiny_b64(4, 4, 100)] }))
                .to_request(),
        )
        .await;
        let meta = &body["meta"];
        assert!(meta["latency_ms"].as_f64().is_some(), "latency_ms missing");
        assert!(meta["model_id"].as_str().is_some(), "model_id missing");
        assert!(
            meta["postprocessing_applied"].as_bool().is_some(),
            "postprocessing_applied missing"
        );
        assert!(
            meta["postprocess_steps"].as_array().is_some(),
            "postprocess_steps missing"
        );
        assert!(meta["warnings"].as_array().is_some(), "warnings missing");
        assert!(meta["version"].as_str().is_some(), "version missing");
        assert!(meta["request_id"].as_str().is_some(), "request_id missing");
    }

    #[actix_web::test]
    async fn test_postprocessing_applied_true_in_meta() {
        let app = actix_test::init_service(
            App::new()
                .app_data(make_state())
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;
        let body: serde_json::Value = actix_test::call_and_read_body_json(
            &app,
            actix_test::TestRequest::post()
                .uri("/classify/batch")
                .set_json(&serde_json::json!({ "images": [tiny_b64(4, 4, 100)] }))
                .to_request(),
        )
        .await;
        assert_eq!(body["meta"]["postprocessing_applied"], true);
        assert!(!body["meta"]["postprocess_steps"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[actix_web::test]
    async fn test_skip_postprocess_sets_meta_applied_false() {
        let app = actix_test::init_service(
            App::new()
                .app_data(make_state())
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;
        let body: serde_json::Value = actix_test::call_and_read_body_json(
            &app,
            actix_test::TestRequest::post()
                .uri("/classify/batch")
                .set_json(&serde_json::json!({ "images": [tiny_b64(4, 4, 100)], "skip_postprocess": true }))
                .to_request(),
        ).await;
        assert_eq!(body["meta"]["postprocessing_applied"], false);
        assert!(body["meta"]["postprocess_steps"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    // ── stream_classify tests ────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_stream_classify_empty_images_returns_bad_request() {
        let app = actix_test::init_service(
            App::new()
                .app_data(make_state())
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;
        let req = actix_test::TestRequest::post()
            .uri("/classify/stream")
            .set_json(&serde_json::json!({ "images": [] }))
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_stream_classify_oversized_batch_returns_bad_request() {
        let app = actix_test::init_service(
            App::new()
                .app_data(make_state())
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;
        let images: Vec<String> = std::iter::repeat("abc".to_string()).take(129).collect();
        let req = actix_test::TestRequest::post()
            .uri("/classify/stream")
            .set_json(&serde_json::json!({ "images": images }))
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_stream_classify_valid_image_returns_sse_stream() {
        let app = actix_test::init_service(
            App::new()
                .app_data(make_state())
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;
        let img = tiny_b64(128, 64, 32);
        let req = actix_test::TestRequest::post()
            .uri("/classify/stream")
            .set_json(&serde_json::json!({ "images": [img], "top_k": 2 }))
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let ct = resp.headers().get("content-type").unwrap().to_str().unwrap();
        assert!(ct.contains("text/event-stream"));
        let body = actix_test::read_body(resp).await;
        let body_str = std::str::from_utf8(&body).unwrap();
        // SSE body must contain at least one "data:" line and the done event
        assert!(body_str.contains("data:"), "expected SSE data frames");
        assert!(body_str.contains("\"done\""), "expected done event");
    }

    #[actix_web::test]
    async fn test_stream_classify_invalid_b64_streams_error_event() {
        let app = actix_test::init_service(
            App::new()
                .app_data(make_state())
                .app_data(web::Data::new(crate::config::Config::default()))
                .configure(configure_routes),
        )
        .await;
        let req = actix_test::TestRequest::post()
            .uri("/classify/stream")
            .set_json(&serde_json::json!({ "images": ["!!!not_base64!!!"] }))
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body = actix_test::read_body(resp).await;
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(body_str.contains("\"error\""), "expected error event for bad b64");
    }
}
