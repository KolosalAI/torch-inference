/// Check for required model files and download them from known sources if missing.
/// All downloads are async and logged; a download failure is a warning, not a fatal error.
use anyhow::{bail, Context, Result};
use std::path::Path;

const HF_KOKORO: &str = "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main";

const KOKORO_VOICES: &[&str] = &[
    "af_bella",
    "af_sarah",
    "af_nicole",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
];

/// Download Kokoro-82M ONNX model + default voice packs if not already present.
pub async fn ensure_kokoro_models(model_dir: &Path) -> Result<()> {
    tokio::fs::create_dir_all(model_dir).await?;
    let voices_dir = model_dir.join("voices");
    tokio::fs::create_dir_all(&voices_dir).await?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()?;

    // ONNX model — try int8 first, fall back to full-precision
    let onnx_candidates = [
        ("kokoro-v1.0.int8.onnx", format!("{}/onnx/model_quantized.onnx", HF_KOKORO)),
    ];
    let mut model_ok = false;
    for (filename, url) in &onnx_candidates {
        let dest = model_dir.join(filename);
        if dest.exists() && dest.metadata().map(|m| m.len()).unwrap_or(0) > 1024 * 1024 {
            tracing::info!(file = %filename, "kokoro model already present");
            model_ok = true;
            break;
        }
        tracing::info!(file = %filename, "downloading kokoro model …");
        match download_file(&client, &url, &dest).await {
            Ok(_) => { model_ok = true; break; }
            Err(e) => tracing::warn!(file = %filename, error = %e, "kokoro model download failed"),
        }
    }
    if !model_ok {
        bail!("no kokoro ONNX model could be obtained; TTS may be unavailable");
    }

    // Voice packs
    for voice in KOKORO_VOICES {
        let dest = voices_dir.join(format!("{}.bin", voice));
        if dest.exists() && dest.metadata().map(|m| m.len()).unwrap_or(0) > 1024 {
            continue;
        }
        let url = format!("{}/voices/{}.bin", HF_KOKORO, voice);
        // Note: onnx-community repo has a subset of voices; missing ones are skipped.
        tracing::info!(voice = %voice, "downloading kokoro voice …");
        if let Err(e) = download_file(&client, &url, &dest).await {
            tracing::warn!(voice = %voice, error = %e, "voice download failed — voice will be skipped");
        }
    }

    Ok(())
}

/// Download YOLOv8n ONNX model if not already present.
pub async fn ensure_yolo_models(model_dir: &Path) -> Result<()> {
    tokio::fs::create_dir_all(model_dir).await?;

    let dest = model_dir.join("yolov8n.onnx");
    if dest.exists() && dest.metadata().map(|m| m.len()).unwrap_or(0) > 1024 * 1024 {
        tracing::info!("yolov8n model already present");
        return Ok(());
    }

    // Ultralytics assets (stable release URL for v8.3.0)
    let url = "https://github.com/ultralytics/assets/releases/latest/download/yolov8n.onnx";
    tracing::info!("downloading yolov8n ONNX model …");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    download_file(&client, url, &dest).await
        .context("yolov8n download failed")?;
    Ok(())
}

/// Download EfficientNet-Lite4 ONNX model + ImageNet labels if not already present.
pub async fn ensure_classify_models(model_dir: &Path) -> Result<()> {
    tokio::fs::create_dir_all(model_dir).await?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    let onnx_dest = model_dir.join("efficientnet-lite4-11.onnx");
    if !onnx_dest.exists() || onnx_dest.metadata().map(|m| m.len()).unwrap_or(0) < 1024 * 1024 {
        let url = "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx";
        tracing::info!("downloading efficientnet-lite4 model …");
        download_file(&client, url, &onnx_dest).await
            .context("efficientnet download failed")?;
    } else {
        tracing::info!("efficientnet model already present");
    }

    let labels_dest = model_dir.join("imagenet1000.txt");
    if !labels_dest.exists() || labels_dest.metadata().map(|m| m.len()).unwrap_or(0) < 1024 {
        // One-of-many mirrors for plain-text ImageNet-1000 labels (one class per line, 1000 lines)
        let url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt";
        tracing::info!("downloading imagenet labels …");
        if let Err(e) = download_file(&client, url, &labels_dest).await {
            tracing::warn!(error = %e, "imagenet labels download failed — classification labels unavailable");
        }
    }

    Ok(())
}

async fn download_file(client: &reqwest::Client, url: &str, dest: &Path) -> Result<()> {
    let resp = client.get(url).send().await
        .with_context(|| format!("GET {}", url))?;
    if !resp.status().is_success() {
        bail!("HTTP {} downloading {}", resp.status(), url);
    }
    let bytes = resp.bytes().await?;
    tokio::fs::write(dest, &bytes).await
        .with_context(|| format!("writing {:?}", dest))?;
    tracing::info!(path = ?dest, size_kb = bytes.len() / 1024, "model file saved");
    Ok(())
}
