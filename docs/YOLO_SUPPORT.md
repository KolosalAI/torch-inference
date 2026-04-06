# YOLO Subsystem — Developer Reference

Internal architecture reference for `torch-inference`'s YOLO object detection subsystem.  
Source: [`src/core/yolo.rs`](../src/core/yolo.rs) · [`src/api/yolo.rs`](../src/api/yolo.rs)

---

## Table of Contents

1. [Pipeline Architecture](#pipeline-architecture)
2. [Type System](#type-system)
3. [Request Flow](#request-flow)
4. [Model Selection Logic](#model-selection-logic)
5. [Supported Models](#supported-models)
6. [NMS Configuration](#nms-configuration)
7. [Adding a New YOLO Variant](#adding-a-new-yolo-variant)
8. [Source File Reference](#source-file-reference)

---

## Pipeline Architecture

Full inference pipeline from HTTP boundary to structured result.

```mermaid
graph TB
    A["POST /yolo/detect\n(multipart/form-data)"] --> B["detect_objects()\nsrc/api/yolo.rs:88"]
    B --> C["Parse YoloDetectRequest\nmodel_version · model_size\nconf_threshold · iou_threshold"]
    C --> D["Read multipart image bytes\nfutures_util::StreamExt"]
    D --> E["YoloDetector::new()\nLoads ONNX session via ort 2.0"]
    E --> F["preprocess_image()\nResize → 640×640\nRGB normalise → [0,1]\nHWC → NCHW tensor"]
    F --> G["ort::Session::run()\nONNX Runtime inference\nCoreML EP on macOS"]
    G --> H{"YoloVersion?"}
    H -- "V5" --> I["postprocess_v5()\nDecode xywh + objectness × class_prob"]
    H -- "V8 / V11 / V12" --> J["postprocess_v8()\nDecode xyxy + class logits, no objectness"]
    H -- "V10" --> K["postprocess_v10()\nNMS-free head, direct top-k"]
    I --> L["non_maximum_suppression()\nIoU-based greedy NMS"]
    J --> L
    K --> M["Collect Vec<Detection>"]
    L --> M
    M --> N["YoloResults\n{ detections, inference_time_ms,\n  preprocessing_time_ms, postprocessing_time_ms }"]
    N --> O{"skip_postprocess?"}
    O -- "false" --> P["EnrichedYoloResults\npostprocess::yolo enrichment"]
    O -- "true" --> Q["Raw YoloDetectResponse JSON"]
    P --> R["Envelope<EnrichedYoloDetectResponse>\nwith ResponseMeta + correlation_id"]
    Q --> R
```

---

## Type System

Core structs shared between `src/core/yolo.rs` and `src/api/yolo.rs`.

```mermaid
classDiagram
    class YoloDetector {
        +version: YoloVersion
        +size: YoloSize
        +conf_threshold: f32
        +iou_threshold: f32
        +input_width: i64
        +input_height: i64
        +class_names: Vec~String~
        +new(version, size, models_dir, class_names) Result~Self~
        +detect(image_path: &Path) Result~YoloResults~
        +detect_bytes(image_bytes: &[u8]) Result~YoloResults~
        +set_conf_threshold(f32)
        +set_iou_threshold(f32)
        +cache_stats() CacheStats
    }

    class YoloVersion {
        <<enum>>
        V5
        V8
        V10
        V11
        V12
        +from_str(s: &str) Option~Self~
        +as_str() &'static str
    }

    class YoloSize {
        <<enum>>
        Nano
        Small
        Medium
        Large
        XLarge
        +suffix() &'static str
        +from_suffix(s: &str) Option~Self~
    }

    class YoloResults {
        +detections: Vec~Detection~
        +inference_time_ms: f64
        +preprocessing_time_ms: f64
        +postprocessing_time_ms: f64
        +total_time_ms: f64
    }

    class Detection {
        +class_id: usize
        +class_name: String
        +confidence: f32
        +bbox: BoundingBox
    }

    class BoundingBox {
        +x1: f32
        +y1: f32
        +x2: f32
        +y2: f32
        +width() f32
        +height() f32
        +center_x() f32
        +center_y() f32
        +area() f32
        +iou(other: &BoundingBox) f32
    }

    class YoloDetectRequest {
        +model_version: String
        +model_size: String
        +conf_threshold: f32
        +iou_threshold: f32
        +skip_postprocess: bool
    }

    YoloDetector --> YoloVersion
    YoloDetector --> YoloSize
    YoloDetector --> YoloResults
    YoloResults --> Detection
    Detection --> BoundingBox
    YoloDetectRequest ..> YoloDetector : constructs
```

---

## Request Flow

Sequence from client to response, including multipart parsing and timing instrumentation.

```mermaid
sequenceDiagram
    participant C as Client
    participant AW as Actix-Web Router
    participant H as detect_objects()<br/>src/api/yolo.rs:88
    participant YD as YoloDetector
    participant ORT as ort::Session<br/>(ONNX Runtime 2.0)
    participant PP as postprocess::yolo

    C->>AW: POST /yolo/detect<br/>Content-Type: multipart/form-data<br/>Fields: model_version, model_size,<br/>conf_threshold, iou_threshold, image

    AW->>H: Multipart stream + form fields
    H->>H: Parse YoloDetectRequest from form fields
    H->>H: Buffer image bytes from stream
    H->>H: t0 = Instant::now()

    H->>YD: YoloDetector::new(version, size, models_dir, coco_names)
    YD->>ORT: Load .onnx model file<br/>SessionBuilder::new()

    H->>YD: detect_bytes(&image_bytes)
    YD->>YD: preprocess_image()<br/>Decode → resize 640×640 → normalize
    YD->>ORT: session.run(inputs![tensor])
    ORT-->>YD: ndarray output tensor

    YD->>YD: postprocess_vN(output)<br/>Decode boxes + scores
    YD->>YD: non_maximum_suppression()<br/>IoU greedy filtering
    YD-->>H: YoloResults { detections, timings }

    alt skip_postprocess == false
        H->>PP: enrich(results)
        PP-->>H: EnrichedYoloResults
        H-->>C: 200 Envelope<EnrichedYoloDetectResponse><br/>{ meta: { correlation_id, latency_ms }, data: {...} }
    else skip_postprocess == true
        H-->>C: 200 YoloDetectResponse<br/>{ success, results, error }
    end
```

---

## Model Selection Logic

How `YoloDetector::new()` resolves the ONNX file path from version + size inputs.

```mermaid
flowchart TD
    A[YoloDetectRequest received] --> B{Parse model_version string}
    B -- "v5 / yolov5" --> C[YoloVersion::V5]
    B -- "v8 / yolov8" --> D[YoloVersion::V8]
    B -- "v10 / yolov10" --> E[YoloVersion::V10]
    B -- "v11 / yolov11 / yolo11" --> F[YoloVersion::V11]
    B -- "v12 / yolov12 / yolo12" --> G[YoloVersion::V12]
    B -- "unknown" --> ERR1[400 Bad Request]

    C & D & E & F & G --> H{Parse model_size suffix}
    H -- "n" --> Nano[YoloSize::Nano]
    H -- "s" --> Small[YoloSize::Small]
    H -- "m" --> Medium[YoloSize::Medium]
    H -- "l" --> Large[YoloSize::Large]
    H -- "x" --> XL[YoloSize::XLarge]
    H -- "unknown" --> ERR2[400 Bad Request]

    Nano & Small & Medium & Large & XL --> I["Build path:\nmodels_dir/yolov{N}{size}.onnx\ne.g. models/yolov8n.onnx"]
    I --> J{File exists?}
    J -- "Yes" --> K[Load ort::Session\nwith CoreML EP on macOS]
    J -- "No" --> L[Return download URL hint\nsrc/api/yolo.rs:292]

    K --> M{version branch}
    M -- "V5" --> N[postprocess_v5\nxywh + obj × cls]
    M -- "V8 / V11 / V12" --> O[postprocess_v8\nxyxy + cls logits]
    M -- "V10" --> P[postprocess_v10\nNMS-free direct]
```

---

## Supported Models

Models pre-configured in `models.json`. ONNX exports from Ultralytics.

| Model      | Version  | Size   | File (ONNX)       | mAP@0.5 (COCO) | Params  |
|------------|----------|--------|-------------------|----------------|---------|
| `yolov5n`  | YOLOv5   | Nano   | `yolov5n.onnx`    | 28.0%          | 1.9 M   |
| `yolov5s`  | YOLOv5   | Small  | `yolov5s.onnx`    | 36.7%          | 7.2 M   |
| `yolov8n`  | YOLOv8   | Nano   | `yolov8n.onnx`    | 37.5%          | 3.2 M   |
| `yolov8s`  | YOLOv8   | Small  | `yolov8s.onnx`    | 44.5%          | 11.2 M  |
| `yolov8m`  | YOLOv8   | Medium | `yolov8m.onnx`    | 50.2%          | 25.9 M  |
| `yolov8l`  | YOLOv8   | Large  | `yolov8l.onnx`    | 52.9%          | 43.7 M  |
| `yolov10n` | YOLOv10  | Nano   | `yolov10n.onnx`   | 38.5%          | 2.3 M   |
| `yolov10s` | YOLOv10  | Small  | `yolov10s.onnx`   | 46.3%          | 7.2 M   |
| `yolo11n`  | YOLO11   | Nano   | `yolo11n.onnx`    | 39.5%          | 2.6 M   |
| `yolo11s`  | YOLO11   | Small  | `yolo11s.onnx`    | 47.0%          | 9.4 M   |

**Download URLs** (resolved at runtime in `src/api/yolo.rs:306`):

```
YOLOv5 : https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5{size}.pt
YOLOv8 : https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8{size}.pt
YOLOv10: https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{size}.pt
YOLO11 : https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11{size}.pt
YOLO12 : https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12{size}.pt
```

> **Note:** Convert to ONNX with `yolo export model=yolov8n.pt format=onnx imgsz=640` before deploying.

---

## NMS Configuration

`non_maximum_suppression()` is implemented in `src/core/yolo.rs:496`. It uses IoU-based greedy filtering (not soft-NMS).

| Parameter         | Field in request     | Default | Range    | Effect                                             |
|-------------------|----------------------|---------|----------|----------------------------------------------------|
| `conf_threshold`  | `conf_threshold`     | `0.25`  | 0.0–1.0  | Boxes below this confidence are discarded pre-NMS  |
| `iou_threshold`   | `iou_threshold`      | `0.45`  | 0.0–1.0  | Overlap threshold; higher = more boxes kept        |
| Input size        | (internal)           | 640×640 | —        | Set via `set_input_size(w, h)` before `detect()`   |

**NMS algorithm** (greedy, class-agnostic):

```
1. Filter detections by conf_threshold
2. Sort remaining by confidence descending
3. For each surviving box:
   - Suppress all lower-confidence boxes where IoU > iou_threshold
4. Return survivors as Vec<Detection>
```

`BoundingBox::iou()` at `src/core/yolo.rs:130` computes intersection-over-union; touching edges return `0.0`.

---

## Adding a New YOLO Variant

### Step 1 — Add the enum variant (`src/core/yolo.rs`)

```rust
pub enum YoloVersion {
    V5, V8, V10, V11, V12,
    V13, // ← new variant
}

impl YoloVersion {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            // existing arms …
            "v13" | "yolov13" | "yolo13" => Some(Self::V13),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            // existing arms …
            Self::V13 => "YOLOv13",
        }
    }
}
```

### Step 2 — Implement a postprocessor (`src/core/yolo.rs`)

Add a method on `YoloDetector`:

```rust
fn postprocess_v13(&self, output: Tensor, detections: &mut Vec<Detection>) -> Result<()> {
    // Inspect the ONNX output shape with: output.size()
    // Typical v8-compatible head: [batch, num_classes+4, num_anchors]
    // Reuse postprocess_v8 if the output format matches.
    self.postprocess_v8(output, detections)
}
```

Wire it into `postprocess()`:

```rust
fn postprocess(&self, output: Tensor) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();
    match self.version {
        // existing arms …
        YoloVersion::V13 => self.postprocess_v13(output, &mut detections)?,
    }
    Ok(self.non_maximum_suppression(detections))
}
```

### Step 3 — Add download URL (`src/api/yolo.rs:306`)

```rust
YoloVersion::V13 => format!(
    "https://github.com/ultralytics/assets/releases/download/v9.0.0/yolov13{}.pt",
    size.suffix()
),
```

### Step 4 — Update `models.json` and `model_registry.json`

Add entries with `"type": "object-detection"` and the ONNX file path.

### Step 5 — Tests

Add unit tests in `src/core/yolo.rs` (see `test_yolo_version_parsing` at line 654 as a template) and integration tests in `tests/integration_test.rs`.

---

## Source File Reference

| File | Purpose |
|------|---------|
| `src/core/yolo.rs` | `YoloDetector`, `YoloVersion`, `YoloSize`, `Detection`, `BoundingBox`, `YoloResults`, NMS, preprocessing |
| `src/api/yolo.rs` | HTTP handlers: `detect_objects`, `get_model_info`, `list_models`, `download_model`; route config via `configure()` |
| `src/postprocess/yolo.rs` | `EnrichedYoloResults` — downstream enrichment applied when `skip_postprocess = false` |
| `src/postprocess/envelope.rs` | `Envelope<T>` + `ResponseMeta` wrapper applied to all enriched responses |
| `src/middleware/correlation_id.rs` | `get_correlation_id()` injected into response metadata |
| `models.json` | Static model registry with download URLs and metadata |
| `model_registry.json` | Runtime registry; loaded by the model management API |

**Routes** (registered by `src/api/yolo.rs:configure()`):

```
POST /yolo/detect    → detect_objects()
GET  /yolo/info      → get_model_info()
GET  /yolo/models    → list_models()
POST /yolo/download  → download_model()
```
