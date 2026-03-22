// YOLO Object Detection Module
// Supports YOLOv5, YOLOv8, YOLOv10, YOLOv11, and YOLOv12 (YOLO11)

use anyhow::{Result, Context, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "torch")]
use tch::{nn, Tensor, Device, Kind, vision};

#[cfg(not(feature = "torch"))]
type Device = ();

/// YOLO Model Version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum YoloVersion {
    V5,
    V8,
    V10,
    V11,  // Also known as YOLO11
    V12,  // Latest YOLO (December 2024)
}

impl YoloVersion {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "v5" | "yolov5" => Some(Self::V5),
            "v8" | "yolov8" => Some(Self::V8),
            "v10" | "yolov10" => Some(Self::V10),
            "v11" | "yolov11" | "yolo11" => Some(Self::V11),
            "v12" | "yolov12" | "yolo12" => Some(Self::V12),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::V5 => "YOLOv5",
            Self::V8 => "YOLOv8",
            Self::V10 => "YOLOv10",
            Self::V11 => "YOLOv11",
            Self::V12 => "YOLOv12",
        }
    }
}

/// YOLO Model Size
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum YoloSize {
    Nano,    // n - fastest, least accurate
    Small,   // s
    Medium,  // m
    Large,   // l
    XLarge,  // x - slowest, most accurate
}

impl YoloSize {
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Nano => "n",
            Self::Small => "s",
            Self::Medium => "m",
            Self::Large => "l",
            Self::XLarge => "x",
        }
    }

    pub fn from_suffix(s: &str) -> Option<Self> {
        match s {
            "n" => Some(Self::Nano),
            "s" => Some(Self::Small),
            "m" => Some(Self::Medium),
            "l" => Some(Self::Large),
            "x" => Some(Self::XLarge),
            _ => None,
        }
    }
}

/// Detection result for a single object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BoundingBox {
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    pub fn center_x(&self) -> f32 {
        (self.x1 + self.x2) / 2.0
    }

    pub fn center_y(&self) -> f32 {
        (self.y1 + self.y2) / 2.0
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);

        if x2 < x1 || y2 < y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.area() + other.area() - intersection;

        intersection / union
    }
}

/// YOLO detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoloResults {
    pub detections: Vec<Detection>,
    pub inference_time_ms: f64,
    pub preprocessing_time_ms: f64,
    pub postprocessing_time_ms: f64,
    pub total_time_ms: f64,
}

/// YOLO Object Detector
pub struct YoloDetector {
    #[cfg(feature = "torch")]
    model: tch::CModule,
    #[cfg(not(feature = "torch"))]
    model: (),
    
    #[cfg(feature = "torch")]
    device: Device,
    #[cfg(not(feature = "torch"))]
    device: (),
    
    version: YoloVersion,
    size: YoloSize,
    class_names: Vec<String>,
    input_size: (i64, i64),
    conf_threshold: f32,
    iou_threshold: f32,
}

impl YoloDetector {
    /// Create a new YOLO detector
    #[cfg(feature = "torch")]
    pub fn new(
        model_path: &Path,
        version: YoloVersion,
        size: YoloSize,
        class_names: Vec<String>,
        device: Option<Device>,
    ) -> Result<Self> {
        let device = device.unwrap_or(Device::Cpu);
        
        log::info!("Loading {} ({}) model from {:?}", version.as_str(), size.suffix(), model_path);
        
        let model = tch::CModule::load_on_device(model_path, device)
            .context("Failed to load YOLO model")?;
        
        // Default input size (can be overridden)
        let input_size = match version {
            YoloVersion::V5 => (640, 640),
            YoloVersion::V8 => (640, 640),
            YoloVersion::V10 => (640, 640),
            YoloVersion::V11 => (640, 640),
            YoloVersion::V12 => (640, 640), // YOLO12 uses 640x640 by default
        };
        
        log::info!("{} detector loaded successfully with {} classes", version.as_str(), class_names.len());
        
        Ok(Self {
            model,
            device,
            version,
            size,
            class_names,
            input_size,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        })
    }
    
    #[cfg(not(feature = "torch"))]
    pub fn new(
        _model_path: &Path,
        version: YoloVersion,
        size: YoloSize,
        class_names: Vec<String>,
        _device: Option<()>,
    ) -> Result<Self> {
        bail!("PyTorch feature not enabled. Compile with --features torch");
    }

    /// Set confidence threshold
    pub fn set_conf_threshold(&mut self, threshold: f32) {
        self.conf_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set IoU threshold for NMS
    pub fn set_iou_threshold(&mut self, threshold: f32) {
        self.iou_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set custom input size
    pub fn set_input_size(&mut self, width: i64, height: i64) {
        self.input_size = (width, height);
    }

    /// Preprocess image for detection
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
        
        // Add batch dimension
        tensor = tensor.unsqueeze(0);
        
        Ok(tensor.to_device(self.device))
    }
    
    #[cfg(not(feature = "torch"))]
    pub fn preprocess_image(&self, _image_path: &Path) -> Result<()> {
        bail!("PyTorch feature not enabled");
    }

    /// Perform object detection
    #[cfg(feature = "torch")]
    pub fn detect(&self, image_path: &Path) -> Result<YoloResults> {
        let start = std::time::Instant::now();
        
        // Preprocess
        let preprocess_start = std::time::Instant::now();
        let input = self.preprocess_image(image_path)?;
        let preprocessing_time_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;
        
        // Inference
        let inference_start = std::time::Instant::now();
        let output = self.model.forward_ts(&[input])?;
        let inference_time_ms = inference_start.elapsed().as_secs_f64() * 1000.0;
        
        // Postprocess
        let postprocess_start = std::time::Instant::now();
        let detections = self.postprocess(output)?;
        let postprocessing_time_ms = postprocess_start.elapsed().as_secs_f64() * 1000.0;
        
        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        Ok(YoloResults {
            detections,
            inference_time_ms,
            preprocessing_time_ms,
            postprocessing_time_ms,
            total_time_ms,
        })
    }
    
    #[cfg(not(feature = "torch"))]
    pub fn detect(&self, _image_path: &Path) -> Result<YoloResults> {
        bail!("PyTorch feature not enabled");
    }

    /// Postprocess YOLO output
    #[cfg(feature = "torch")]
    fn postprocess(&self, output: Tensor) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();
        
        // Convert to CPU for processing
        let output = output.to_device(Device::Cpu);
        
        // Different postprocessing based on YOLO version
        match self.version {
            YoloVersion::V5 => self.postprocess_v5(output, &mut detections)?,
            YoloVersion::V8 => self.postprocess_v8(output, &mut detections)?,
            YoloVersion::V10 => self.postprocess_v10(output, &mut detections)?,
            YoloVersion::V11 => self.postprocess_v11(output, &mut detections)?,
            YoloVersion::V12 => self.postprocess_v12(output, &mut detections)?,
        }
        
        // Apply NMS
        let detections = self.non_maximum_suppression(detections);
        
        Ok(detections)
    }

    /// Postprocess YOLOv5 output
    #[cfg(feature = "torch")]
    fn postprocess_v5(&self, output: Tensor, detections: &mut Vec<Detection>) -> Result<()> {
        // YOLOv5 output shape: [batch, num_predictions, 85] (for 80 classes)
        // Each prediction: [x, y, w, h, objectness, class_0_prob, ..., class_79_prob]
        
        let output_size = output.size();
        let num_predictions = output_size[1];
        
        for i in 0..num_predictions {
            let prediction = output.get(0).get(i);
            
            let objectness: f32 = prediction.double_value(&[4]) as f32;
            
            if objectness < self.conf_threshold {
                continue;
            }
            
            // Get class probabilities
            let class_probs = prediction.narrow(0, 5, self.class_names.len() as i64);
            let (max_prob, class_id) = class_probs.max_dim(0, false);
            let max_prob: f32 = max_prob.double_value(&[]) as f32;
            let class_id: usize = class_id.int64_value(&[]) as usize;
            
            let confidence = objectness * max_prob;
            
            if confidence < self.conf_threshold {
                continue;
            }
            
            // Get bounding box
            let x: f32 = prediction.double_value(&[0]) as f32;
            let y: f32 = prediction.double_value(&[1]) as f32;
            let w: f32 = prediction.double_value(&[2]) as f32;
            let h: f32 = prediction.double_value(&[3]) as f32;
            
            let bbox = BoundingBox {
                x1: x - w / 2.0,
                y1: y - h / 2.0,
                x2: x + w / 2.0,
                y2: y + h / 2.0,
            };
            
            detections.push(Detection {
                class_id,
                class_name: self.class_names.get(class_id)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", class_id)),
                confidence,
                bbox,
            });
        }
        
        Ok(())
    }

    /// Postprocess YOLOv8 output
    #[cfg(feature = "torch")]
    fn postprocess_v8(&self, output: Tensor, detections: &mut Vec<Detection>) -> Result<()> {
        // YOLOv8 output shape: [batch, 84, 8400] (for 80 classes)
        // Transpose to [batch, 8400, 84]
        let output = output.transpose(1, 2);
        
        let output_size = output.size();
        let num_predictions = output_size[1];
        
        for i in 0..num_predictions {
            let prediction = output.get(0).get(i);
            
            // First 4 elements are bbox coordinates
            let x: f32 = prediction.double_value(&[0]) as f32;
            let y: f32 = prediction.double_value(&[1]) as f32;
            let w: f32 = prediction.double_value(&[2]) as f32;
            let h: f32 = prediction.double_value(&[3]) as f32;
            
            // Remaining elements are class probabilities
            let class_probs = prediction.narrow(0, 4, self.class_names.len() as i64);
            let (max_prob, class_id) = class_probs.max_dim(0, false);
            let confidence: f32 = max_prob.double_value(&[]) as f32;
            let class_id: usize = class_id.int64_value(&[]) as usize;
            
            if confidence < self.conf_threshold {
                continue;
            }
            
            let bbox = BoundingBox {
                x1: x - w / 2.0,
                y1: y - h / 2.0,
                x2: x + w / 2.0,
                y2: y + h / 2.0,
            };
            
            detections.push(Detection {
                class_id,
                class_name: self.class_names.get(class_id)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", class_id)),
                confidence,
                bbox,
            });
        }
        
        Ok(())
    }

    /// Postprocess YOLOv10 output
    #[cfg(feature = "torch")]
    fn postprocess_v10(&self, output: Tensor, detections: &mut Vec<Detection>) -> Result<()> {
        // YOLOv10 uses similar format to YOLOv8 but with improved architecture
        self.postprocess_v8(output, detections)
    }

    /// Postprocess YOLOv11 output
    #[cfg(feature = "torch")]
    fn postprocess_v11(&self, output: Tensor, detections: &mut Vec<Detection>) -> Result<()> {
        // YOLOv11 (YOLO11) maintains YOLOv8 output format with architecture improvements
        self.postprocess_v8(output, detections)
    }

    /// Postprocess YOLOv12 output (Latest - December 2024)
    #[cfg(feature = "torch")]
    fn postprocess_v12(&self, output: Tensor, detections: &mut Vec<Detection>) -> Result<()> {
        // YOLOv12 uses enhanced output format with better feature extraction
        // Still compatible with YOLOv8 format
        self.postprocess_v8(output, detections)
    }

    /// Non-Maximum Suppression
    fn non_maximum_suppression(&self, mut detections: Vec<Detection>) -> Vec<Detection> {
        if detections.is_empty() {
            return detections;
        }
        
        // Sort by confidence (descending)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let mut keep = Vec::new();
        
        while !detections.is_empty() {
            let current = detections.remove(0);
            keep.push(current.clone());
            
            detections.retain(|det| {
                // Keep detections of different classes
                if det.class_id != current.class_id {
                    return true;
                }
                
                // Remove detections with high IoU overlap
                det.bbox.iou(&current.bbox) < self.iou_threshold
            });
        }
        
        keep
    }

    /// Get model information
    pub fn info(&self) -> String {
        format!(
            "{} ({}) - {} classes, input: {}x{}, conf: {}, iou: {}",
            self.version.as_str(),
            self.size.suffix(),
            self.class_names.len(),
            self.input_size.0,
            self.input_size.1,
            self.conf_threshold,
            self.iou_threshold
        )
    }
}

/// Load COCO class names (80 classes)
pub fn load_coco_names() -> Vec<String> {
    vec![
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    ].iter().map(|s| s.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yolo_version_parsing() {
        assert_eq!(YoloVersion::from_str("v5"), Some(YoloVersion::V5));
        assert_eq!(YoloVersion::from_str("YOLOv8"), Some(YoloVersion::V8));
        assert_eq!(YoloVersion::from_str("yolo11"), Some(YoloVersion::V11));
        assert_eq!(YoloVersion::from_str("v12"), Some(YoloVersion::V12));
    }

    #[test]
    fn test_bbox_iou() {
        let bbox1 = BoundingBox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 };
        let bbox2 = BoundingBox { x1: 5.0, y1: 5.0, x2: 15.0, y2: 15.0 };
        
        let iou = bbox1.iou(&bbox2);
        assert!(iou > 0.0 && iou < 1.0);
    }

    #[test]
    fn test_coco_names() {
        let names = load_coco_names();
        assert_eq!(names.len(), 80);
        assert_eq!(names[0], "person");
        assert_eq!(names[79], "toothbrush");
    }
}
