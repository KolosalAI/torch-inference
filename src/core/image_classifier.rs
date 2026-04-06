#![allow(dead_code)]
#[cfg(feature = "torch")]
use anyhow::Context;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "torch")]
use tch::{Device, Kind};

#[cfg(not(feature = "torch"))]
type Device = ();

#[cfg(feature = "torch")]
use crate::models::pytorch_loader::get_best_device;

#[cfg(feature = "torch")]
use crate::core::model_cache::{cache_key, CacheStats, ModelCache};
#[cfg(not(feature = "torch"))]
use crate::core::model_cache::{CacheStats, ModelCache};

#[cfg(feature = "torch")]
use crate::torch_optimization::{
    OptimizedTorchModel, TorchOptimizationConfig, TorchOptimizationConfigBuilder,
};

/// Image classification model wrapper
pub struct ImageClassifier {
    #[cfg(feature = "torch")]
    optimizer: OptimizedTorchModel,

    #[cfg(not(feature = "torch"))]
    _optimizer: (),

    labels: Vec<String>,
    input_size: (i64, i64),
    normalize_mean: Vec<f64>,
    normalize_std: Vec<f64>,
    cache: ModelCache,
    model_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub label: String,
    pub confidence: f32,
    pub class_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopKResults {
    pub predictions: Vec<ClassificationResult>,
    pub inference_time_ms: f64,
}

impl ImageClassifier {
    /// Create a new image classifier
    #[cfg(feature = "torch")]
    pub fn new(
        model_path: &Path,
        labels: Vec<String>,
        input_size: Option<(i64, i64)>,
        device: Option<Device>,
        opt_config: Option<TorchOptimizationConfig>,
    ) -> Result<Self> {
        let device = device.unwrap_or_else(|| get_best_device());
        let model_id = model_path.to_string_lossy().to_string();

        let config = opt_config.unwrap_or_else(|| {
            let is_accel = device != Device::Cpu;
            TorchOptimizationConfigBuilder::new()
                .device(device)
                .cudnn_benchmark(true)
                .warmup_iterations(3)
                .autocast(is_accel)
                .fp16(is_accel)
                .build()
        });

        log::info!("Loading image classification model from {:?}", model_path);
        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
        let mut optimizer = OptimizedTorchModel::new(model_path_str, config)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        optimizer.warmup().map_err(|e| anyhow::anyhow!("{}", e))?;

        let input_size = input_size.unwrap_or((224, 224));
        log::info!(
            "Image classifier loaded successfully with {} classes",
            labels.len()
        );

        Ok(Self {
            optimizer,
            labels,
            input_size,
            normalize_mean: vec![0.485, 0.456, 0.406],
            normalize_std: vec![0.229, 0.224, 0.225],
            cache: ModelCache::new(256),
            model_id,
        })
    }

    #[cfg(not(feature = "torch"))]
    pub fn new(
        _model_path: &Path,
        _labels: Vec<String>,
        _input_size: Option<(i64, i64)>,
        _device: Option<()>,
        _opt_config: Option<()>,
    ) -> Result<Self> {
        bail!("PyTorch feature not enabled. Compile with --features torch");
    }

    /// Preprocess image for classification
    #[cfg(feature = "torch")]
    pub fn preprocess_image(&self, image_path: &Path) -> Result<tch::Tensor> {
        use tch::vision;
        log::debug!("Preprocessing image: {:?}", image_path);

        let image = vision::image::load(image_path).context("Failed to load image")?;
        let resized = vision::image::resize(&image, self.input_size.0, self.input_size.1)?;
        let mut tensor = resized.to_kind(Kind::Float) / 255.0;

        for c in 0..3i64 {
            let mean = self.normalize_mean[c as usize];
            let std = self.normalize_std[c as usize];
            let mut channel = tensor.select(0, c);
            channel -= mean;
            channel /= std;
        }

        Ok(tensor.unsqueeze(0).to_device(self.optimizer.device()))
    }

    #[cfg(not(feature = "torch"))]
    pub fn preprocess_image(&self, _image_path: &Path) -> Result<()> {
        bail!("PyTorch feature not enabled");
    }

    /// Classify image from a file path. Reads file bytes and delegates to classify_bytes
    /// so the result cache is always used.
    #[cfg(feature = "torch")]
    pub fn classify(&self, image_path: &Path, top_k: usize) -> Result<TopKResults> {
        let image_bytes = std::fs::read(image_path)
            .with_context(|| format!("Failed to read image file: {:?}", image_path))?;
        self.classify_bytes(&image_bytes, top_k)
    }

    #[cfg(not(feature = "torch"))]
    pub fn classify(&self, _image_path: &Path, _top_k: usize) -> Result<TopKResults> {
        bail!("PyTorch feature not enabled");
    }

    /// Classify from raw image bytes. Result is cached by (model_id, bytes, top_k).
    #[cfg(feature = "torch")]
    pub fn classify_bytes(&self, image_bytes: &[u8], top_k: usize) -> Result<TopKResults> {
        let key = cache_key(&self.model_id, image_bytes, &(top_k as u64).to_le_bytes());
        self.cache.get_or_run(key, || {
            let start = std::time::Instant::now();

            // Write to temp file for image loading (tch requires a path).
            let temp_path = std::env::temp_dir().join(format!("clf_{}.jpg", uuid::Uuid::new_v4()));
            std::fs::write(&temp_path, image_bytes)?;
            let input = self.preprocess_image(&temp_path);
            let _ = std::fs::remove_file(&temp_path);
            let input = input?;

            let output = self
                .optimizer
                .infer(&input)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let probabilities = output.softmax(-1, Kind::Float);
            let (values, indices) = probabilities.topk(top_k as i64, -1, true, true);
            let values: Vec<f32> = values.try_into()?;
            let indices: Vec<i64> = indices.try_into()?;

            let predictions: Vec<ClassificationResult> = indices
                .iter()
                .enumerate()
                .map(|(i, &class_id)| {
                    let class_id = class_id as usize;
                    let label = self
                        .labels
                        .get(class_id)
                        .cloned()
                        .unwrap_or_else(|| format!("class_{}", class_id));
                    ClassificationResult {
                        label,
                        confidence: values[i],
                        class_id,
                    }
                })
                .collect();

            let inference_time_ms = start.elapsed().as_secs_f64() * 1000.0;
            if let Some(top) = predictions.first() {
                log::info!(
                    "Classification complete: {} ({:.2}%) in {:.2}ms",
                    top.label,
                    top.confidence * 100.0,
                    inference_time_ms
                );
            }
            Ok(TopKResults {
                predictions,
                inference_time_ms,
            })
        })
    }

    #[cfg(not(feature = "torch"))]
    pub fn classify_bytes(&self, _image_bytes: &[u8], _top_k: usize) -> Result<TopKResults> {
        bail!("PyTorch feature not enabled");
    }

    /// Get number of classes
    pub fn num_classes(&self) -> usize {
        self.labels.len()
    }

    /// Get class label by index
    pub fn get_label(&self, class_id: usize) -> Option<&str> {
        self.labels.get(class_id).map(|s| s.as_str())
    }

    /// Expose cache stats for observability and testing.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Construct a stub classifier for tests (no model loaded)
    #[cfg(test)]
    pub fn new_stub(labels: Vec<String>) -> Self {
        Self {
            #[cfg(not(feature = "torch"))]
            _optimizer: (),
            labels,
            input_size: (224, 224),
            normalize_mean: vec![0.485, 0.456, 0.406],
            normalize_std: vec![0.229, 0.224, 0.225],
            cache: ModelCache::new(256),
            model_id: "stub".to_string(),
        }
    }
}

/// Load ImageNet class labels
pub fn load_imagenet_labels() -> Vec<String> {
    // ImageNet 1000 classes
    // For now, return a simplified list
    // In production, load from a JSON file
    (0..1000).map(|i| format!("class_{}", i)).collect()
}

/// Common pre-trained model configurations
pub mod models {
    pub struct ModelConfig {
        pub name: String,
        pub url: String,
        pub input_size: (i64, i64),
        pub num_classes: usize,
    }

    pub fn resnet50() -> ModelConfig {
        ModelConfig {
            name: "ResNet-50".to_string(),
            url: "https://download.pytorch.org/models/resnet50-0676ba61.pth".to_string(),
            input_size: (224, 224),
            num_classes: 1000,
        }
    }

    pub fn mobilenet_v2() -> ModelConfig {
        ModelConfig {
            name: "MobileNetV2".to_string(),
            url: "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth".to_string(),
            input_size: (224, 224),
            num_classes: 1000,
        }
    }

    pub fn efficientnet_b0() -> ModelConfig {
        ModelConfig {
            name: "EfficientNet-B0".to_string(),
            url: "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth"
                .to_string(),
            input_size: (224, 224),
            num_classes: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_stats_initially_zero() {
        let clf = ImageClassifier::new_stub(vec!["cat".to_string()]);
        let stats = clf.cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_load_labels() {
        let labels = load_imagenet_labels();
        assert_eq!(labels.len(), 1000);
    }

    #[test]
    fn test_new_stub_num_classes() {
        let labels = vec!["cat".to_string(), "dog".to_string()];
        let clf = ImageClassifier::new_stub(labels.clone());
        assert_eq!(clf.num_classes(), 2);
    }

    #[test]
    fn test_new_stub_get_label() {
        let labels = vec!["apple".to_string(), "banana".to_string()];
        let clf = ImageClassifier::new_stub(labels);
        assert_eq!(clf.get_label(0), Some("apple"));
        assert_eq!(clf.get_label(1), Some("banana"));
        assert_eq!(clf.get_label(99), None);
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_new_without_torch_returns_error() {
        let path = std::path::Path::new("/nonexistent/model.pt");
        let result = ImageClassifier::new(path, vec!["cat".to_string()], None, None, None);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("PyTorch feature not enabled") || msg.contains("torch"),
            "unexpected: {}",
            msg
        );
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_preprocess_image_without_torch_returns_error() {
        let clf = ImageClassifier::new_stub(vec!["cat".to_string()]);
        let path = std::path::Path::new("/tmp/test.jpg");
        let result = clf.preprocess_image(path);
        assert!(result.is_err());
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_classify_without_torch_returns_error() {
        // Covers lines 174-176: the #[cfg(not(feature = "torch"))] classify() stub
        let clf = ImageClassifier::new_stub(vec!["cat".to_string(), "dog".to_string()]);
        let path = std::path::Path::new("/nonexistent/image.jpg");
        let result = clf.classify(path, 1);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("PyTorch feature not enabled"),
            "unexpected error message: {}",
            msg
        );
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_classify_bytes_without_torch_returns_error() {
        // Covers lines 195-197: the #[cfg(not(feature = "torch"))] classify_bytes() stub
        let clf = ImageClassifier::new_stub(vec!["cat".to_string()]);
        let fake_bytes: &[u8] = &[0xFF, 0xD8, 0xFF]; // stub ignores all input
        let result = clf.classify_bytes(fake_bytes, 1);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("PyTorch feature not enabled"),
            "unexpected error message: {}",
            msg
        );
    }

    #[test]
    fn test_models_resnet50_config() {
        let cfg = models::resnet50();
        assert_eq!(cfg.name, "ResNet-50");
        assert_eq!(cfg.input_size, (224, 224));
        assert_eq!(cfg.num_classes, 1000);
        assert!(cfg.url.contains("resnet50"));
    }

    #[test]
    fn test_models_mobilenet_v2_config() {
        let cfg = models::mobilenet_v2();
        assert_eq!(cfg.name, "MobileNetV2");
        assert_eq!(cfg.input_size, (224, 224));
        assert_eq!(cfg.num_classes, 1000);
        assert!(cfg.url.contains("mobilenet"));
    }

    #[test]
    fn test_models_efficientnet_b0_config() {
        let cfg = models::efficientnet_b0();
        assert_eq!(cfg.name, "EfficientNet-B0");
        assert_eq!(cfg.input_size, (224, 224));
        assert_eq!(cfg.num_classes, 1000);
        assert!(cfg.url.contains("efficientnet"));
    }
}
