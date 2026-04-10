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
    state: std::sync::Arc<YoloState>,
) {
    let mut cfg = DetectCfg::default();
    let mut frame_id: u64 = 0;
    let mut hb = interval(Duration::from_secs(20));
    hb.tick().await;

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
                        let result = process_detect_frame(&data, &cfg, &state);
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

fn process_detect_frame(
    image_bytes: &[u8],
    cfg: &DetectCfg,
    state: &YoloState,
) -> Result<Vec<DetectionResult>, String> {
    use crate::core::yolo::{load_coco_names, YoloSize, YoloVersion};

    let version = YoloVersion::from_str(&cfg.version)
        .ok_or_else(|| format!("invalid version: {}", cfg.version))?;
    let size = YoloSize::from_suffix(&cfg.size)
        .ok_or_else(|| format!("invalid size: {}", cfg.size))?;

    let model_name = format!(
        "yolo{}{}",
        version.as_str().to_lowercase().replace("yolo", ""),
        size.suffix()
    );
    let model_path = state
        .models_dir
        .join(&model_name)
        .join(format!("{}.pt", model_name));

    if !model_path.exists() {
        return Err(format!("model not found: {}  — download it first", model_name));
    }

    // Write frame to a temp file that YoloDetector accepts.
    let tmp = std::env::temp_dir().join(format!("ws_frame_{}.jpg", uuid::Uuid::new_v4()));
    std::fs::write(&tmp, image_bytes).map_err(|e| e.to_string())?;

    let class_names = load_coco_names();

    #[cfg(not(feature = "torch"))]
    {
        let _ = class_names;
        let _ = std::fs::remove_file(&tmp);
        return Err("PyTorch feature not enabled — build with --features torch".to_string());
    }

    #[cfg(feature = "torch")]
    {
        use tch::Device;
        let mut detector =
            crate::core::yolo::YoloDetector::new(&tmp, version, size, class_names, Some(Device::Cpu))
                .map_err(|e| e.to_string())?;
        detector.set_conf_threshold(cfg.conf);
        detector.set_iou_threshold(cfg.iou);
        let raw = detector.detect(&tmp).map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&tmp);
        Ok(raw
            .detections
            .into_iter()
            .map(|d| DetectionResult {
                label: d.class_name.clone(),
                conf: d.confidence,
                bbox: [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
            })
            .collect())
    }
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
                            cfg.top_k  = top_k.max(1).min(1000);
                            cfg.width  = width.max(1).min(4096);
                            cfg.height = height.max(1).min(4096);
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
    let pipeline = ImagePipeline::new(PreprocessConfig::imagenet(cfg.width, cfg.height));
    let batch = pipeline
        .preprocess_batch(&[image_bytes.to_vec()])
        .map_err(|e| e.to_string())?;

    state
        .backend
        .classify_nchw(batch, cfg.top_k)
        .await
        .map_err(|e| e.to_string())
        .map(|mut v| v.pop().unwrap_or_default())
}

// ── Route config ──────────────────────────────────────────────────────────────

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/ws/detect", web::get().to(ws_detect_handler))
        .route("/ws/classify", web::get().to(ws_classify_handler));
}
