/// WebSocket Streaming Inference
///
/// Two endpoints that accept a continuous stream of image frames (JPEG/PNG
/// binary frames) and return inference results as JSON text frames in real time.
/// This enables live camera, video-file playback, and batch image processing with
/// per-frame feedback — covering the gap left by the one-shot REST endpoints.
///
/// ## Endpoints
///
/// | Route | Task |
/// |-------|------|
/// | `GET /ws/detect` | YOLO object detection per frame |
/// | `GET /ws/classify` | Image classification per frame |
///
/// ## Protocol
///
/// **Client → Server (text, optional config)**
/// ```json
/// // Detection
/// {"type":"config","version":"v8","size":"n","conf":0.5,"iou":0.45}
/// // Classification
/// {"type":"config","top_k":5,"width":224,"height":224}
/// ```
///
/// **Client → Server (binary)**
/// Raw JPEG or PNG bytes of the frame to process.  The server assigns a
/// monotonically increasing `frame` counter.
///
/// **Server → Client (text)**
/// ```json
/// {"type":"ready","task":"detect","frame":0}
/// {"type":"detect","frame":1,"ms":14.2,"count":3,
///  "detections":[{"label":"person","conf":0.87,"bbox":[x1,y1,x2,y2]},...]}
/// {"type":"classify","frame":1,"ms":9.1,
///  "predictions":[{"label":"cat","conf":0.95,"class_id":281},...]}
/// {"type":"error","frame":1,"msg":"model not loaded"}
/// ```
use crate::api::classify::{ClassifyState, Prediction};
use crate::api::yolo::YoloState;
use crate::core::image_pipeline::{ImagePipeline, PreprocessConfig};
use actix_web::{web, HttpRequest, HttpResponse};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::time::interval;

// ── Protocol types ────────────────────────────────────────────────────────────

/// Configuration messages the client may send as JSON text.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMsg {
    /// Reconfigure the inference task / parameters at any time.
    Config {
        // ── Detection params ──────────────────────────────────────────────
        #[serde(default)]
        version: Option<String>,
        #[serde(default)]
        size: Option<String>,
        #[serde(default = "default_conf")]
        conf: f32,
        #[serde(default = "default_iou")]
        iou: f32,
        // ── Classification params ─────────────────────────────────────────
        #[serde(default = "default_top_k")]
        top_k: usize,
        #[serde(default = "default_dim")]
        width: u32,
        #[serde(default = "default_dim")]
        height: u32,
    },
}

fn default_conf() -> f32 { 0.5 }
fn default_iou() -> f32 { 0.45 }
fn default_top_k() -> usize { 5 }
fn default_dim() -> u32 { 224 }

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerMsg {
    Ready {
        task: String,
        frame: u64,
    },
    Detect {
        frame: u64,
        ms: f64,
        count: usize,
        detections: Vec<DetectionResult>,
    },
    Classify {
        frame: u64,
        ms: f64,
        predictions: Vec<Prediction>,
    },
    Error {
        frame: u64,
        msg: String,
    },
}

#[derive(Debug, Serialize)]
pub struct DetectionResult {
    pub label: String,
    pub conf: f32,
    /// [x1, y1, x2, y2] in pixels, original image coordinates
    pub bbox: [f32; 4],
}

impl ServerMsg {
    fn to_json(&self) -> String {
        serde_json::to_string(self)
            .unwrap_or_else(|_| r#"{"type":"error","frame":0,"msg":"serialise"}"#.to_string())
    }
}

// ── Runtime config held per-session ──────────────────────────────────────────

struct DetectCfg {
    version: String,
    size: String,
    conf: f32,
    iou: f32,
}

impl Default for DetectCfg {
    fn default() -> Self {
        Self { version: "v8".into(), size: "n".into(), conf: 0.5, iou: 0.45 }
    }
}

#[derive(Clone)]
struct ClassifyCfg {
    top_k: usize,
    width: u32,
    height: u32,
}

impl Default for ClassifyCfg {
    fn default() -> Self {
        Self { top_k: 5, width: 224, height: 224 }
    }
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// `GET /ws/detect` — YOLO detection WebSocket.
pub async fn ws_detect_handler(
    req: HttpRequest,
    stream: web::Payload,
    state: web::Data<YoloState>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, msg_stream) = actix_ws::handle(&req, stream)?;
    let state = state.into_inner();
    actix_web::rt::spawn(run_detect_session(session, msg_stream, state));
    Ok(response)
}

/// `GET /ws/classify` — Classification WebSocket.
pub async fn ws_classify_handler(
    req: HttpRequest,
    stream: web::Payload,
    state: web::Data<ClassifyState>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, msg_stream) = actix_ws::handle(&req, stream)?;
    let state = state.into_inner();
    actix_web::rt::spawn(run_classify_session(session, msg_stream, state));
    Ok(response)
}

// ── Detection session ─────────────────────────────────────────────────────────

async fn run_detect_session(
    mut session: actix_ws::Session,
    mut msg_stream: actix_ws::MessageStream,
    _state: std::sync::Arc<YoloState>,
) {
    let mut cfg = DetectCfg::default();
    let mut frame_id: u64 = 0;
    let mut hb = interval(Duration::from_secs(20));
    hb.tick().await;

    // Cache the detector for the lifetime of this session — avoids the
    // ~140 ms ONNX session load on every frame. Recreated only when the
    // model file changes. Detector is `&self`-only after the post-bench
    // refactor — no inner Mutex is required.
    let mut cached_det: Option<std::sync::Arc<crate::core::ort_yolo::OrtYoloDetector>> = None;
    let mut cached_model: Option<std::path::PathBuf> = None;

    let ready = ServerMsg::Ready { task: "detect".to_string(), frame: 0 };
    if session.text(ready.to_json()).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            _ = hb.tick() => {
                if session.ping(b"hb").await.is_err() { break; }
            }
            msg = msg_stream.next() => {
                match msg {
                    Some(Ok(actix_ws::Message::Text(txt))) => {
                        if let Ok(ClientMsg::Config { version, size, conf, iou, .. }) =
                            serde_json::from_str::<ClientMsg>(&txt)
                        {
                            if let Some(v) = version { cfg.version = v; }
                            if let Some(s) = size   { cfg.size = s; }
                            cfg.conf = conf;
                            cfg.iou  = iou;
                        }
                    }
                    Some(Ok(actix_ws::Message::Binary(data))) => {
                        frame_id += 1;
                        let t = Instant::now();
                        let result = process_detect_frame(
                            &data, &cfg, &mut cached_det, &mut cached_model,
                        ).await;
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        let msg = match result {
                            Ok(dets) => {
                                let count = dets.len();
                                ServerMsg::Detect { frame: frame_id, ms, count, detections: dets }
                            }
                            Err(e) => ServerMsg::Error { frame: frame_id, msg: e },
                        };
                        if session.text(msg.to_json()).await.is_err() { break; }
                    }
                    Some(Ok(actix_ws::Message::Ping(d))) => {
                        if session.pong(&d).await.is_err() { break; }
                    }
                    Some(Ok(actix_ws::Message::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }
    let _ = session.close(None).await;
}

/// Scan `models/yolo/` and return the path to the best available ONNX detection model.
/// Preference order: higher YOLO version > lower; nano size > larger.
/// Tiny/v3 models are excluded as they use an incompatible output format.
fn find_best_yolo_onnx() -> Option<std::path::PathBuf> {
    let yolo_dir = std::path::Path::new("models/yolo");
    if !yolo_dir.is_dir() { return None; }

    let ver_score = |name: &str| -> i32 {
        let n = name.to_lowercase();
        if n.contains("yolo12") || n.contains("yolov12") { 120 }
        else if n.contains("yolo11") || n.contains("yolov11") { 110 }
        else if n.contains("yolov10") || n.contains("yolo10") { 100 }
        else if n.contains("yolov8") || n.contains("yolo8") { 80 }
        else if n.contains("yolov5") || n.contains("yolo5") { 50 }
        else { 10 }
    };

    let size_score = |name: &str| -> i32 {
        let n = name.to_lowercase();
        if n.ends_with("n.onnx") { 5 }
        else if n.ends_with("s.onnx") { 4 }
        else if n.ends_with("m.onnx") { 3 }
        else if n.ends_with("l.onnx") { 2 }
        else { 1 }
    };

    std::fs::read_dir(yolo_dir).ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let p = e.path();
            p.extension().is_some_and(|x| x == "onnx") && {
                let name = e.file_name().to_string_lossy().to_lowercase();
                name.contains("yolo") && !name.starts_with("tiny")
            }
        })
        .max_by_key(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            (ver_score(&name), size_score(&name))
        })
        .map(|e| e.path())
}

async fn process_detect_frame(
    image_bytes: &[u8],
    cfg: &DetectCfg,
    cached_det: &mut Option<std::sync::Arc<crate::core::ort_yolo::OrtYoloDetector>>,
    cached_model: &mut Option<std::path::PathBuf>,
) -> Result<Vec<DetectionResult>, String> {
    use crate::core::yolo::load_coco_names;

    let model_path = find_best_yolo_onnx()
        .ok_or_else(|| "No YOLO ONNX model found in models/yolo/ — download one first".to_string())?;

    // Recreate detector only when model path changes (first frame or model swapped).
    if cached_model.as_ref() != Some(&model_path) {
        tracing::info!(model = ?model_path, "ws/detect loading ONNX model (once per session)");
        let path = model_path.clone();
        let det = tokio::task::spawn_blocking(move || {
            let class_names = load_coco_names();
            crate::core::ort_yolo::OrtYoloDetector::new(&path, class_names)
        })
        .await
        .map_err(|e| format!("task join: {}", e))?
        .map_err(|e| e.to_string())?;
        *cached_det = Some(std::sync::Arc::new(det));
        *cached_model = Some(model_path);
    }

    let det = cached_det.clone().unwrap();
    let conf = cfg.conf;
    let iou  = cfg.iou;
    let bytes = image_bytes.to_vec();

    tokio::task::spawn_blocking(move || {
        // Thresholds at call time — &self only, no inner lock needed.
        let raw = det.detect_bytes(&bytes, conf, iou).map_err(|e| e.to_string())?;
        Ok(raw
            .detections
            .into_iter()
            .map(|d| DetectionResult {
                label: d.class_name,
                conf: d.confidence,
                bbox: [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
            })
            .collect())
    })
    .await
    .map_err(|e| format!("task join: {}", e))?
}

// ── Classification session ────────────────────────────────────────────────────

async fn run_classify_session(
    mut session: actix_ws::Session,
    mut msg_stream: actix_ws::MessageStream,
    state: std::sync::Arc<ClassifyState>,
) {
    let mut cfg = ClassifyCfg::default();
    let mut frame_id: u64 = 0;
    let mut hb = interval(Duration::from_secs(20));
    hb.tick().await;

    let ready = ServerMsg::Ready { task: "classify".to_string(), frame: 0 };
    if session.text(ready.to_json()).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            _ = hb.tick() => {
                if session.ping(b"hb").await.is_err() { break; }
            }
            msg = msg_stream.next() => {
                match msg {
                    Some(Ok(actix_ws::Message::Text(txt))) => {
                        if let Ok(ClientMsg::Config { top_k, width, height, .. }) =
                            serde_json::from_str::<ClientMsg>(&txt)
                        {
                            cfg.top_k  = top_k.clamp(1, 1000);
                            cfg.width  = width.clamp(1, 4096);
                            cfg.height = height.clamp(1, 4096);
                        }
                    }
                    Some(Ok(actix_ws::Message::Binary(data))) => {
                        frame_id += 1;
                        let t = Instant::now();
                        let result = process_classify_frame(&data, &cfg, &state).await;
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        let msg = match result {
                            Ok(preds) => ServerMsg::Classify { frame: frame_id, ms, predictions: preds },
                            Err(e)    => ServerMsg::Error { frame: frame_id, msg: e },
                        };
                        if session.text(msg.to_json()).await.is_err() { break; }
                    }
                    Some(Ok(actix_ws::Message::Ping(d))) => {
                        if session.pong(&d).await.is_err() { break; }
                    }
                    Some(Ok(actix_ws::Message::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }
    let _ = session.close(None).await;
}

async fn process_classify_frame(
    image_bytes: &[u8],
    cfg: &ClassifyCfg,
    state: &ClassifyState,
) -> Result<Vec<Prediction>, String> {
    // Preprocess + ORT inference are CPU-bound; offload to spawn_blocking
    // so the websocket reactor can keep flushing frames.
    let backend = state.backend.clone();
    let owned = image_bytes.to_vec();
    let cfg = cfg.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Prediction>> {
        let pipeline = ImagePipeline::new(PreprocessConfig::imagenet(cfg.width, cfg.height));
        let batch = pipeline.preprocess_batch(&[owned])?;
        let mut v = backend.classify_nchw(batch, cfg.top_k)?;
        Ok(v.pop().unwrap_or_default())
    })
    .await
    .map_err(|e| format!("task join: {e}"))?
    .map_err(|e| e.to_string())
}

// ── Route config ──────────────────────────────────────────────────────────────

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/ws/detect", web::get().to(ws_detect_handler))
        .route("/ws/classify", web::get().to(ws_classify_handler));
}
