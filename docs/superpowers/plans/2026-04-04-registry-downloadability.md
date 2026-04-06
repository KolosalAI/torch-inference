# Registry Downloadability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two structural bugs that prevent models from downloading correctly: missing `model_type` fields in `model_registry.json`, and a download handler that silently skips non-HuggingFace URLs.

**Architecture:** Three self-contained changes — (1) a data-only JSON patch to add `model_type` to all 65 registry entries, (2) three new `cache_dir` match arms in the Rust handler for STT and NLP model types, and (3) a new 4th download branch that handles direct HTTPS file URLs (GitHub releases, pytorch.org). Changes follow TDD: tests first, implementation second.

**Tech Stack:** Rust (Actix-web, Tokio), `wiremock` for HTTP mocking in tests, Python 3 for the JSON patch script, `jq` for validation.

---

## File Structure

| File | Change |
|------|--------|
| `model_registry.json` | Add `"model_type"` field to all 65 `models.*` entries |
| `src/api/models.rs` | Add 3 `cache_dir` match arms (lines 264–271 area); add 4th download branch (before the final `else` at line 314) |

---

### Task 1: Add `model_type` to all 65 `model_registry.json` entries

**Files:**
- Modify: `model_registry.json`

The `download_model_async` function defaults `model_type` to `"tts"` when the field is missing (`src/api/models.rs:258–262`), so every model regardless of category lands in `models/tts/`. This task patches all 65 entries.

- [ ] **Step 1: Run the patch script**

Run this Python script from the project root. It reads the registry, sets `model_type` on every entry using the canonical mapping, and writes it back with 2-space indentation:

```bash
python3 - <<'PYEOF'
import json

MODEL_TYPES = {
    # TTS (25)
    "windows-sapi": "tts",
    "fish-speech-v1.5": "tts",
    "styletts2": "tts",
    "kokoro-v0.19": "tts",
    "xtts-v2": "tts",
    "piper-lessac": "tts",
    "melotts": "tts",
    "metavoice": "tts",
    "openvoice": "tts",
    "kokoro-v1.0": "tts",
    "kokoro-onnx": "tts",
    "kokoro-onnx-int8": "tts",
    "f5-tts": "tts",
    "parler-tts-mini": "tts",
    "chatterbox": "tts",
    "outetts-0.3-500m": "tts",
    "sesame-csm-1b": "tts",
    "cosyvoice2-0.5b": "tts",
    "index-tts-2": "tts",
    "melotts-english-v3": "tts",
    "bark": "tts",
    "bark-small": "tts",
    "amphion-maskgct": "tts",
    "matcha-tts": "tts",
    "vits-vctk": "tts",
    # Image classification (18)
    "eva02-large-patch14-448": "image-classification",
    "eva-giant-patch14-560": "image-classification",
    "convnextv2-huge-512": "image-classification",
    "convnext-xxlarge-clip": "image-classification",
    "maxvit-xlarge-512": "image-classification",
    "coatnet-3-rw-224": "image-classification",
    "efficientnetv2-xl": "image-classification",
    "mobilenetv4-hybrid-large": "image-classification",
    "vit-giant-patch14-224": "image-classification",
    "beit-large-patch16-512": "image-classification",
    "swin-large-patch4-384": "image-classification",
    "deit3-huge-patch14-224": "image-classification",
    "resnet34": "image-classification",
    "resnet50": "image-classification",
    "resnet101": "image-classification",
    "resnet152": "image-classification",
    "mobilenet-v3-small": "image-classification",
    "mobilenet-v3-large": "image-classification",
    # Speech-to-text (5)
    "whisper-small": "speech-to-text",
    "whisper-medium": "speech-to-text",
    "whisper-large": "speech-to-text",
    "whisper-large-v3": "speech-to-text",
    "whisper-turbo": "speech-to-text",
    # Object detection (12)
    "yolov8x": "object-detection",
    "yolov5m": "object-detection",
    "yolov5l": "object-detection",
    "yolov5x": "object-detection",
    "yolov9c": "object-detection",
    "yolov9e": "object-detection",
    "yolov10n": "object-detection",
    "yolov10s": "object-detection",
    "yolov10m": "object-detection",
    "yolov10b": "object-detection",
    "yolov10l": "object-detection",
    "yolov10x": "object-detection",
    # Text classification (3)
    "bert-base-uncased": "text-classification",
    "roberta-base": "text-classification",
    "bart-large-mnli": "text-classification",
    # Feature extraction (2)
    "all-mpnet-base-v2": "feature-extraction",
    "paraphrase-multilingual-mpnet": "feature-extraction",
}

with open("model_registry.json") as f:
    data = json.load(f)

missing = []
for key, entry in data["models"].items():
    if key in MODEL_TYPES:
        entry["model_type"] = MODEL_TYPES[key]
    else:
        missing.append(key)

if missing:
    print(f"WARNING: no mapping for: {missing}")

with open("model_registry.json", "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print(f"Patched {len(MODEL_TYPES)} entries.")
PYEOF
```

Expected output: `Patched 65 entries.` with no WARNING line.

- [ ] **Step 2: Validate all 65 entries now have `model_type`**

```bash
python3 -c "
import json
with open('model_registry.json') as f:
    data = json.load(f)
missing = [k for k, v in data['models'].items() if not v.get('model_type')]
print(f'Missing model_type: {missing}')
print(f'Total entries: {len(data[\"models\"])}')
"
```

Expected output:
```
Missing model_type: []
Total entries: 65
```

- [ ] **Step 3: Validate JSON is still well-formed**

```bash
python3 -m json.tool model_registry.json > /dev/null && echo "JSON valid"
```

Expected: `JSON valid`

- [ ] **Step 4: Commit**

```bash
git add model_registry.json
git commit -m "fix(registry): add model_type field to all 65 model_registry.json entries"
```

---

### Task 2: Add missing `cache_dir` match arms for STT and NLP types

**Files:**
- Modify: `src/api/models.rs` (lines 264–271 — `cache_dir` match block inside `download_model_async`)

The current match has no arms for `"speech-to-text"`, `"text-classification"`, `"feature-extraction"`, or `"nlp"`. All four fall through to `models/other/` instead of their correct directories.

**Context — current match block** (`src/api/models.rs:264–271`):
```rust
let cache_dir = match model_type {
    "tts" => std::path::Path::new("models/tts").join(model_id),
    "image-classification" => std::path::Path::new("models/classification").join(model_id),
    "object-detection" => std::path::Path::new("models/detection").join(model_id),
    "segmentation" => std::path::Path::new("models/segmentation").join(model_id),
    "neural-network" => std::path::Path::new("models/neural").join(model_id),
    _ => std::path::Path::new("models/other").join(model_id),
};
```

- [ ] **Step 1: Write 4 failing tests**

Add these tests to the `handler_coverage_tests` mod in `src/api/models.rs` (place them after the existing `test_download_model_async_empty_model_type_defaults_to_tts` test, around line 1016):

```rust
#[tokio::test]
#[serial_test::serial]
async fn test_download_model_async_model_type_speech_to_text_creates_stt_dir() {
    let model = ModelInfo {
        name: "Whisper".to_string(),
        score: 0.0,
        rank: 1,
        size: "244MB".to_string(),
        url: "Built-in".to_string(),
        architecture: String::new(),
        voices: String::new(),
        quality: String::new(),
        status: "Active".to_string(),
        note: None,
        model_type: "speech-to-text".to_string(),
        task: "asr".to_string(),
    };
    let result = download_model_async("stt-model", &model).await;
    assert!(result.is_ok());
    assert!(
        std::path::Path::new("models/stt/stt-model").exists(),
        "expected models/stt/stt-model to exist"
    );
    let _ = std::fs::remove_dir_all("models/stt");
}

#[tokio::test]
#[serial_test::serial]
async fn test_download_model_async_model_type_text_classification_creates_nlp_dir() {
    let model = ModelInfo {
        name: "BERT".to_string(),
        score: 0.0,
        rank: 1,
        size: "417MB".to_string(),
        url: "Built-in".to_string(),
        architecture: String::new(),
        voices: String::new(),
        quality: String::new(),
        status: "Active".to_string(),
        note: None,
        model_type: "text-classification".to_string(),
        task: "classification".to_string(),
    };
    let result = download_model_async("nlp-cls-model", &model).await;
    assert!(result.is_ok());
    assert!(
        std::path::Path::new("models/nlp/nlp-cls-model").exists(),
        "expected models/nlp/nlp-cls-model to exist"
    );
    let _ = std::fs::remove_dir_all("models/nlp");
}

#[tokio::test]
#[serial_test::serial]
async fn test_download_model_async_model_type_feature_extraction_creates_nlp_dir() {
    let model = ModelInfo {
        name: "MPNet".to_string(),
        score: 0.0,
        rank: 1,
        size: "438MB".to_string(),
        url: "Built-in".to_string(),
        architecture: String::new(),
        voices: String::new(),
        quality: String::new(),
        status: "Active".to_string(),
        note: None,
        model_type: "feature-extraction".to_string(),
        task: "embedding".to_string(),
    };
    let result = download_model_async("feat-model", &model).await;
    assert!(result.is_ok());
    assert!(
        std::path::Path::new("models/nlp/feat-model").exists(),
        "expected models/nlp/feat-model to exist"
    );
    let _ = std::fs::remove_dir_all("models/nlp");
}

#[tokio::test]
#[serial_test::serial]
async fn test_download_model_async_model_type_nlp_creates_nlp_dir() {
    let model = ModelInfo {
        name: "NLP Direct".to_string(),
        score: 0.0,
        rank: 1,
        size: "100MB".to_string(),
        url: "Built-in".to_string(),
        architecture: String::new(),
        voices: String::new(),
        quality: String::new(),
        status: "Active".to_string(),
        note: None,
        model_type: "nlp".to_string(),
        task: "nlp".to_string(),
    };
    let result = download_model_async("nlp-direct-model", &model).await;
    assert!(result.is_ok());
    assert!(
        std::path::Path::new("models/nlp/nlp-direct-model").exists(),
        "expected models/nlp/nlp-direct-model to exist"
    );
    let _ = std::fs::remove_dir_all("models/nlp");
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test test_download_model_async_model_type_speech_to_text_creates_stt_dir \
          test_download_model_async_model_type_text_classification_creates_nlp_dir \
          test_download_model_async_model_type_feature_extraction_creates_nlp_dir \
          test_download_model_async_model_type_nlp_creates_nlp_dir \
          2>&1 | tail -20
```

Expected: tests fail with assertion `expected models/stt/stt-model to exist` (the model type falls through to `models/other/`).

- [ ] **Step 3: Add the 3 new match arms**

In `src/api/models.rs`, find the `cache_dir` match block (it currently ends with `_ => std::path::Path::new("models/other").join(model_id),`). Replace the match block with:

```rust
let cache_dir = match model_type {
    "tts" => std::path::Path::new("models/tts").join(model_id),
    "image-classification" => std::path::Path::new("models/classification").join(model_id),
    "object-detection" => std::path::Path::new("models/detection").join(model_id),
    "segmentation" => std::path::Path::new("models/segmentation").join(model_id),
    "neural-network" => std::path::Path::new("models/neural").join(model_id),
    "speech-to-text" => std::path::Path::new("models/stt").join(model_id),
    "text-classification" | "feature-extraction" | "nlp" => std::path::Path::new("models/nlp").join(model_id),
    _ => std::path::Path::new("models/other").join(model_id),
};
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test test_download_model_async_model_type_speech_to_text_creates_stt_dir \
          test_download_model_async_model_type_text_classification_creates_nlp_dir \
          test_download_model_async_model_type_feature_extraction_creates_nlp_dir \
          test_download_model_async_model_type_nlp_creates_nlp_dir \
          2>&1 | tail -20
```

Expected: all 4 tests pass.

- [ ] **Step 5: Run the full test suite to catch regressions**

```bash
cargo test 2>&1 | tail -10
```

Expected: `test result: ok.` with 0 failures.

- [ ] **Step 6: Commit**

```bash
git add src/api/models.rs
git commit -m "fix(models): add cache_dir match arms for speech-to-text, text-classification, feature-extraction, nlp"
```

---

### Task 3: Add 4th download branch for direct HTTPS file URLs

**Files:**
- Modify: `src/api/models.rs` (inside `download_model_async`, between the `else if model.url.starts_with("https://huggingface.co/")` block and the final `else`)

GitHub release URLs (YOLO: `https://github.com/ultralytics/...yolov8x.pt`) and pytorch.org URLs (`https://download.pytorch.org/models/resnet50-11ad3fa6.pth`) currently hit the `else` branch, which only logs a warning. This task adds a 4th branch that streams-downloads any URL ending in `.pt`, `.pth`, `.onnx`, `.bin`, `.safetensors`, or `.gguf`.

**Context — current 3-branch URL logic** (`src/api/models.rs:282–316`):
```rust
if model.url == "Built-in" {
    // ...
    return Ok(());
}

if model.url.contains("resolve") {
    // direct file download
} else if model.url.starts_with("https://huggingface.co/") {
    // repo download via HF API
} else {
    log::warn!("{} requires manual download from: {}", model.name, model.url);
}
```

- [ ] **Step 1: Write a failing test**

Add this test to the `handler_coverage_tests` mod in `src/api/models.rs` (after the `test_download_model_async_manual_download_url_logs_and_returns_ok` test, around line 1040):

```rust
#[tokio::test]
async fn test_download_model_async_direct_pt_url_downloads_file() {
    use wiremock::{MockServer, Mock, ResponseTemplate};
    use wiremock::matchers::method;

    let server = MockServer::start().await;
    let content = b"fake pt weights";

    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(content.as_slice()))
        .mount(&server)
        .await;

    // URL ends in .pt — should trigger the new 4th download branch.
    // Mock server uses http:// which the branch accepts (http or https prefix).
    let url = format!("{}/yolov8x.pt", server.uri());
    let model = ModelInfo {
        name: "YOLOv8x".to_string(),
        score: 0.0,
        rank: 1,
        size: "136MB".to_string(),
        url,
        architecture: String::new(),
        voices: String::new(),
        quality: String::new(),
        status: "Available".to_string(),
        note: None,
        model_type: "object-detection".to_string(),
        task: "detection".to_string(),
    };
    let result = download_model_async("yolov8x-direct", &model).await;
    assert!(result.is_ok(), "direct .pt download should return Ok");

    let downloaded = tokio::fs::read("models/detection/yolov8x-direct/model.pt").await;
    assert!(downloaded.is_ok(), "model.pt should have been created");
    assert_eq!(downloaded.unwrap(), content, "file contents should match server response");

    let _ = std::fs::remove_dir_all("models/detection/yolov8x-direct");
}

#[tokio::test]
async fn test_download_model_async_direct_pth_url_downloads_file() {
    use wiremock::{MockServer, Mock, ResponseTemplate};
    use wiremock::matchers::method;

    let server = MockServer::start().await;
    let content = b"fake pth weights";

    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(content.as_slice()))
        .mount(&server)
        .await;

    let url = format!("{}/resnet50-11ad3fa6.pth", server.uri());
    let model = ModelInfo {
        name: "ResNet-50".to_string(),
        score: 0.0,
        rank: 1,
        size: "98MB".to_string(),
        url,
        architecture: String::new(),
        voices: String::new(),
        quality: String::new(),
        status: "Available".to_string(),
        note: None,
        model_type: "image-classification".to_string(),
        task: "classification".to_string(),
    };
    let result = download_model_async("resnet50-direct", &model).await;
    assert!(result.is_ok(), "direct .pth download should return Ok");

    let downloaded = tokio::fs::read("models/classification/resnet50-direct/model.pth").await;
    assert!(downloaded.is_ok(), "model.pth should have been created");
    assert_eq!(downloaded.unwrap(), content);

    let _ = std::fs::remove_dir_all("models/classification/resnet50-direct");
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test test_download_model_async_direct_pt_url_downloads_file \
          test_download_model_async_direct_pth_url_downloads_file \
          2>&1 | tail -20
```

Expected: both tests fail — `model.pt` / `model.pth` does not exist because the URL hits the `else` (warning-only) branch.

- [ ] **Step 3: Add the 4th download branch**

In `src/api/models.rs`, find the 3-branch URL block inside `download_model_async`. Replace:

```rust
    } else {
        log::warn!("{} requires manual download from: {}", model.name, model.url);
    }
```

With:

```rust
    } else if {
        let u = model.url.as_str();
        u.ends_with(".pt") || u.ends_with(".pth") || u.ends_with(".onnx")
            || u.ends_with(".bin") || u.ends_with(".safetensors") || u.ends_with(".gguf")
    } {
        // Direct file download for GitHub releases, pytorch.org, and other CDNs.
        let extension = model.url.rsplit('.').next().unwrap_or("bin");
        let filepath = cache_dir.join(format!("model.{}", extension));
        download_file_streaming(get_http_client(), &model.url, &filepath).await?;
        log::info!("Downloaded {} to {:?}", model.name, filepath);
    } else {
        log::warn!("{} requires manual download from: {}", model.name, model.url);
    }
```

Note: the branch does not filter by `starts_with("https://")` — this keeps it testable with a local mock server (`http://`) while still correctly handling production GitHub and pytorch.org URLs. The branch is guarded by the file extension check, which is the meaningful filter.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test test_download_model_async_direct_pt_url_downloads_file \
          test_download_model_async_direct_pth_url_downloads_file \
          2>&1 | tail -20
```

Expected: both tests pass.

- [ ] **Step 5: Verify the existing manual-download test still passes**

The existing test `test_download_model_async_manual_download_url_logs_and_returns_ok` uses URL `https://some-other-site.example.com/model.zip`. The `.zip` extension is not in the new branch's extension list, so it still falls through to the `else` (warning) branch and returns `Ok(())`. Verify:

```bash
cargo test test_download_model_async_manual_download_url_logs_and_returns_ok 2>&1 | tail -5
```

Expected: `test result: ok. 1 passed`

- [ ] **Step 6: Run the full test suite**

```bash
cargo test 2>&1 | tail -10
```

Expected: `test result: ok.` with 0 failures.

- [ ] **Step 7: Commit**

```bash
git add src/api/models.rs
git commit -m "fix(models): add 4th download branch for direct .pt/.pth/.onnx/.bin/.safetensors/.gguf URLs"
```
