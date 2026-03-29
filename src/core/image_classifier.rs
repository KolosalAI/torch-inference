use anyhow::{Result, Context, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "torch")]
use tch::{nn, Tensor, Device, Kind, vision};

#[cfg(not(feature = "torch"))]
type Device = ();

#[cfg(feature = "torch")]
use crate::models::pytorch_loader::get_best_device;

/// Image classification model wrapper
pub struct ImageClassifier {
    #[cfg(feature = "torch")]
    model: tch::CModule,
    #[cfg(not(feature = "torch"))]
    model: (),
    
    #[cfg(feature = "torch")]
    device: Device,
    #[cfg(not(feature = "torch"))]
    device: (),
    
    labels: Vec<String>,
    input_size: (i64, i64),
    normalize_mean: Vec<f64>,
    normalize_std: Vec<f64>,
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
    ) -> Result<Self> {
        let device = device.unwrap_or_else(|| get_best_device());
        
        log::info!("Loading image classification model from {:?}", model_path);
        let model = tch::CModule::load_on_device(model_path, device)
            .context("Failed to load image classification model")?;
        
        let input_size = input_size.unwrap_or((224, 224)); // Default ImageNet size
        
        // ImageNet normalization by default
        let normalize_mean = vec![0.485, 0.456, 0.406];
        let normalize_std = vec![0.229, 0.224, 0.225];
        
        log::info!("Image classifier loaded successfully with {} classes", labels.len());
        
        Ok(Self {
            model,
            device,
            labels,
            input_size,
            normalize_mean,
            normalize_std,
        })
    }
    
    #[cfg(not(feature = "torch"))]
    pub fn new(
        _model_path: &Path,
        labels: Vec<String>,
        _input_size: Option<(i64, i64)>,
        _device: Option<()>,
    ) -> Result<Self> {
        bail!("PyTorch feature not enabled. Compile with --features torch");
    }
    
    /// Preprocess image for classification
    #[cfg(feature = "torch")]
    pub fn preprocess_image(&self, image_path: &Path) -> Result<Tensor> {
        log::debug!("Preprocessing image: {:?}", image_path);
        
        // Load image
        let image = vision::image::load(image_path)
            .context("Failed to load image")?;
        
        // Resize to model input size
        let resized = vision::image::resize(&image, self.input_size.0, self.input_size.1)?;
        
        // Convert to float and normalize to [0, 1]
        let mut tensor = resized.to_kind(Kind::Float) / 255.0;
        
        // Normalize with ImageNet statistics
        for c in 0..3 {
            let mean = self.normalize_mean[c as usize];
            let std = self.normalize_std[c as usize];
            let mut channel = tensor.select(0, c);
            channel -= mean;
            channel /= std;
        }
        
        // Add batch dimension
        tensor = tensor.unsqueeze(0);
        
        Ok(tensor.to_device(self.device))
    }
    
    #[cfg(not(feature = "torch"))]
    pub fn preprocess_image(&self, _image_path: &Path) -> Result<()> {
        bail!("PyTorch feature not enabled");
    }
    
    /// Classify image and return top-k results
    #[cfg(feature = "torch")]
    pub fn classify(&self, image_path: &Path, top_k: usize) -> Result<TopKResults> {
        let start = std::time::Instant::now();
        
        // Preprocess image
        let input = self.preprocess_image(image_path)?;
        
        // Run inference
        let output = self.model.forward_ts(&[input])?;
        
        // Apply softmax
        let probabilities = output.softmax(-1, Kind::Float);
        
        // Get top-k predictions
        let (values, indices) = probabilities.topk(top_k as i64, -1, true, true);
        
        let values: Vec<f32> = values.try_into()?;
        let indices: Vec<i64> = indices.try_into()?;
        
        let mut predictions = Vec::new();
        for (i, &class_id) in indices.iter().enumerate() {
            let class_id = class_id as usize;
            let label = if class_id < self.labels.len() {
                self.labels[class_id].clone()
            } else {
                format!("class_{}", class_id)
            };
            
            predictions.push(ClassificationResult {
                label,
                confidence: values[i],
                class_id,
            });
        }
        
        let inference_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        log::info!(
            "Classification complete: {} ({:.2}% confidence) in {:.2}ms",
            predictions[0].label,
            predictions[0].confidence * 100.0,
            inference_time_ms
        );
        
        Ok(TopKResults {
            predictions,
            inference_time_ms,
        })
    }
    
    #[cfg(not(feature = "torch"))]
    pub fn classify(&self, _image_path: &Path, _top_k: usize) -> Result<TopKResults> {
        bail!("PyTorch feature not enabled");
    }
    
    /// Classify from raw image bytes
    #[cfg(feature = "torch")]
    pub fn classify_bytes(&self, image_bytes: &[u8], top_k: usize) -> Result<TopKResults> {
        // Save to temporary file
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join(format!("temp_image_{}.jpg", uuid::Uuid::new_v4()));
        std::fs::write(&temp_path, image_bytes)?;
        
        let result = self.classify(&temp_path, top_k);
        
        // Clean up
        let _ = std::fs::remove_file(&temp_path);
        
        result
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

    /// Construct a stub classifier for tests (no model loaded)
    #[cfg(test)]
    pub fn new_stub(labels: Vec<String>) -> Self {
        Self {
            #[cfg(not(feature = "torch"))]
            model: (),
            #[cfg(not(feature = "torch"))]
            device: (),
            labels,
            input_size: (224, 224),
            normalize_mean: vec![0.485, 0.456, 0.406],
            normalize_std: vec![0.229, 0.224, 0.225],
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
    use super::*;
    
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
            url: "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth".to_string(),
            input_size: (224, 224),
            num_classes: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let result = ImageClassifier::new(path, vec!["cat".to_string()], None, None);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("PyTorch feature not enabled") || msg.contains("torch"), "unexpected: {}", msg);
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
            msg.contains("PyTorch feature not enabled") || msg.contains("torch"),
            "unexpected error message: {}",
            msg
        );
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_classify_bytes_without_torch_returns_error() {
        // Covers lines 195-197: the #[cfg(not(feature = "torch"))] classify_bytes() stub
        let clf = ImageClassifier::new_stub(vec!["cat".to_string()]);
        let fake_bytes: &[u8] = &[0xFF, 0xD8, 0xFF]; // JPEG magic bytes (fake)
        let result = clf.classify_bytes(fake_bytes, 1);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("PyTorch feature not enabled") || msg.contains("torch"),
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
