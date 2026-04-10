/// ORT-based image classification backend.
///
/// Supports two model flavours, auto-detected from the ONNX input tensor shape:
///
/// NCHW / ImageNet (e.g. MobileNetV2-7, ResNet50):
///   Input  "data"     : [1, 3, H, W] f32, ImageNet-normalised
///   Output [0]        : [1, C] f32 logits   → softmax applied here
///
/// NHWC / 0-255 (e.g. EfficientNet-Lite4-11):
///   Input  "images:0" : [1, H, W, 3] f32, pixel values 0-255
///   Output "Softmax:0": [1, C] f32   → already probabilities
use anyhow::{Context, Result};
use async_trait::async_trait;
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

use crate::api::classify::{ClassificationBackend, Prediction};

/// ImageNet channel statistics used for un-normalisation.
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD:  [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone, Copy)]
enum InputFormat {
    /// NCHW, ImageNet-normalised  (MobileNetV2, ResNet50, …)
    NchwImageNet,
    /// NHWC, raw 0-255 pixels     (EfficientNet-Lite4, …)
    NhwcZeroTo255,
}

pub struct OrtClassificationBackend {
    session: Mutex<Session>,
    labels: Vec<String>,
    input_name: String,
    format: InputFormat,
    /// true when the model already applies softmax (EfficientNet-Lite4 does)
    output_is_prob: bool,
}

impl OrtClassificationBackend {
    /// Load model from `model_path`; labels from a newline-delimited text file.
    /// The input format is auto-detected from the model's first input tensor shape.
    pub fn new(model_path: &Path, labels_path: &Path) -> Result<Self> {
        let labels_text = std::fs::read_to_string(labels_path)
            .with_context(|| format!("reading labels from {:?}", labels_path))?;
        let labels: Vec<String> = labels_text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();

        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?;

        #[cfg(target_os = "macos")]
        {
            builder = builder.with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])?;
        }
        #[cfg(not(target_os = "macos"))]
        {
            builder = builder.with_execution_providers([
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])?;
        }

        let session = builder
            .commit_from_file(model_path)
            .with_context(|| format!("loading ONNX model from {:?}", model_path))?;

        // Auto-detect format from first input tensor name and shape.
        let input_info = session.inputs.first()
            .context("model has no inputs")?;
        let input_name = input_info.name.clone();

        // NHWC models have 4-D input where dim[3] == 3 (channels last).
        // We detect by name — EfficientNet-Lite4 uses "images:0".
        let format = if input_name.contains("images") {
            InputFormat::NhwcZeroTo255
        } else {
            InputFormat::NchwImageNet
        };

        let output_is_prob = session.outputs.first()
            .map(|o| o.name.to_lowercase().contains("softmax") || o.name.to_lowercase().contains("sigmoid"))
            .unwrap_or(false);

        tracing::info!(
            model = ?model_path,
            num_labels = labels.len(),
            input_name = %input_name,
            format = ?format,
            output_is_prob,
            "OrtClassificationBackend loaded"
        );

        Ok(Self { session: Mutex::new(session), labels, input_name, format, output_is_prob })
    }

    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        exp.iter().map(|&x| x / sum).collect()
    }

    fn top_k(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Convert an NCHW ImageNet-normalised slice to NHWC 0-255 pixels.
    fn nchw_imagenet_to_nhwc_255(nchw: &[f32], h: usize, w: usize) -> Vec<f32> {
        let mut nhwc = vec![0f32; 3 * h * w];
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..3 {
                    let nchw_val = nchw[ci * h * w + hi * w + wi];
                    let pixel = (nchw_val * IMAGENET_STD[ci] + IMAGENET_MEAN[ci]) * 255.0;
                    nhwc[hi * w * 3 + wi * 3 + ci] = pixel.clamp(0.0, 255.0);
                }
            }
        }
        nhwc
    }
}

#[async_trait]
impl ClassificationBackend for OrtClassificationBackend {
    async fn classify_nchw(
        &self,
        batch: Array4<f32>,
        top_k: usize,
    ) -> Result<Vec<Vec<Prediction>>> {
        let n = batch.shape()[0];
        let c = batch.shape()[1];
        let h = batch.shape()[2];
        let w = batch.shape()[3];

        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            let img_nchw: Vec<f32> = batch
                .slice(ndarray::s![i, .., .., ..])
                .iter()
                .copied()
                .collect();

            let (input_tensor, input_name) = match self.format {
                InputFormat::NchwImageNet => {
                    let t = Tensor::<f32>::from_array(([1usize, c, h, w], img_nchw))?;
                    (t, self.input_name.as_str())
                }
                InputFormat::NhwcZeroTo255 => {
                    let nhwc = Self::nchw_imagenet_to_nhwc_255(&img_nchw, h, w);
                    let t = Tensor::<f32>::from_array(([1usize, h, w, c], nhwc))?;
                    (t, self.input_name.as_str())
                }
            };

            let mut sess = self.session.lock().unwrap();
            let outputs = sess.run(ort::inputs![input_name => input_tensor])?;

            let (_shape, raw_view) = outputs[0].try_extract_tensor::<f32>()?;
            let raw: Vec<f32> = raw_view.iter().copied().collect();

            let probs = if self.output_is_prob {
                raw
            } else {
                Self::softmax(&raw)
            };

            let top = Self::top_k(&probs, top_k);
            let preds = top
                .into_iter()
                .map(|(class_id, confidence)| Prediction {
                    label: self
                        .labels
                        .get(class_id)
                        .cloned()
                        .unwrap_or_else(|| format!("class_{}", class_id)),
                    confidence,
                    class_id,
                })
                .collect();

            results.push(preds);
        }

        Ok(results)
    }
}
