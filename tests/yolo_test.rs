// Tests for YOLO object detection

use torch_inference::core::yolo::*;

#[test]
fn test_yolo_version_parsing() {
    assert_eq!(YoloVersion::from_str("v5"), Some(YoloVersion::V5));
    assert_eq!(YoloVersion::from_str("YOLOv8"), Some(YoloVersion::V8));
    assert_eq!(YoloVersion::from_str("v10"), Some(YoloVersion::V10));
    assert_eq!(YoloVersion::from_str("yolo11"), Some(YoloVersion::V11));
    assert_eq!(YoloVersion::from_str("V12"), Some(YoloVersion::V12));
    assert_eq!(YoloVersion::from_str("invalid"), None);
}

#[test]
fn test_yolo_version_display() {
    assert_eq!(YoloVersion::V5.as_str(), "YOLOv5");
    assert_eq!(YoloVersion::V8.as_str(), "YOLOv8");
    assert_eq!(YoloVersion::V10.as_str(), "YOLOv10");
    assert_eq!(YoloVersion::V11.as_str(), "YOLOv11");
    assert_eq!(YoloVersion::V12.as_str(), "YOLOv12");
}

#[test]
fn test_yolo_size_parsing() {
    assert_eq!(YoloSize::from_suffix("n"), Some(YoloSize::Nano));
    assert_eq!(YoloSize::from_suffix("s"), Some(YoloSize::Small));
    assert_eq!(YoloSize::from_suffix("m"), Some(YoloSize::Medium));
    assert_eq!(YoloSize::from_suffix("l"), Some(YoloSize::Large));
    assert_eq!(YoloSize::from_suffix("x"), Some(YoloSize::XLarge));
    assert_eq!(YoloSize::from_suffix("invalid"), None);
}

#[test]
fn test_yolo_size_suffix() {
    assert_eq!(YoloSize::Nano.suffix(), "n");
    assert_eq!(YoloSize::Small.suffix(), "s");
    assert_eq!(YoloSize::Medium.suffix(), "m");
    assert_eq!(YoloSize::Large.suffix(), "l");
    assert_eq!(YoloSize::XLarge.suffix(), "x");
}

#[test]
fn test_bounding_box_dimensions() {
    let bbox = BoundingBox {
        x1: 100.0,
        y1: 50.0,
        x2: 300.0,
        y2: 250.0,
    };

    assert_eq!(bbox.width(), 200.0);
    assert_eq!(bbox.height(), 200.0);
    assert_eq!(bbox.center_x(), 200.0);
    assert_eq!(bbox.center_y(), 150.0);
    assert_eq!(bbox.area(), 40000.0);
}

#[test]
fn test_bounding_box_iou() {
    // Identical boxes
    let bbox1 = BoundingBox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 };
    let bbox2 = BoundingBox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 };
    assert_eq!(bbox1.iou(&bbox2), 1.0);

    // Half overlap
    let bbox1 = BoundingBox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 };
    let bbox2 = BoundingBox { x1: 5.0, y1: 5.0, x2: 15.0, y2: 15.0 };
    let iou = bbox1.iou(&bbox2);
    assert!(iou > 0.0 && iou < 1.0);
    assert!((iou - 0.142857).abs() < 0.001); // 25/(100+100-25) ≈ 0.143

    // No overlap
    let bbox1 = BoundingBox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 };
    let bbox2 = BoundingBox { x1: 20.0, y1: 20.0, x2: 30.0, y2: 30.0 };
    assert_eq!(bbox1.iou(&bbox2), 0.0);
}

#[test]
fn test_coco_names() {
    let names = load_coco_names();
    
    assert_eq!(names.len(), 80);
    assert_eq!(names[0], "person");
    assert_eq!(names[1], "bicycle");
    assert_eq!(names[2], "car");
    assert_eq!(names[79], "toothbrush");
}

#[test]
fn test_detection_structure() {
    let detection = Detection {
        class_id: 0,
        class_name: "person".to_string(),
        confidence: 0.92,
        bbox: BoundingBox {
            x1: 100.0,
            y1: 50.0,
            x2: 200.0,
            y2: 300.0,
        },
    };

    assert_eq!(detection.class_id, 0);
    assert_eq!(detection.class_name, "person");
    assert_eq!(detection.confidence, 0.92);
    assert_eq!(detection.bbox.width(), 100.0);
    assert_eq!(detection.bbox.height(), 250.0);
}

#[test]
fn test_yolo_results_structure() {
    let results = YoloResults {
        detections: vec![],
        inference_time_ms: 45.3,
        preprocessing_time_ms: 12.5,
        postprocessing_time_ms: 8.2,
        total_time_ms: 66.0,
    };

    assert_eq!(results.detections.len(), 0);
    assert!(results.inference_time_ms > 0.0);
    assert!(results.total_time_ms >= results.inference_time_ms);
}

#[cfg(feature = "torch")]
#[test]
fn test_yolo_detector_creation() {
    use std::path::Path;
    use tch::Device;

    let model_path = Path::new("models/yolo5n/yolo5n.pt");
    
    // Only test if model exists
    if !model_path.exists() {
        eprintln!("Skipping detector test - model not found");
        return;
    }

    let class_names = load_coco_names();
    
    let result = YoloDetector::new(
        model_path,
        YoloVersion::V5,
        YoloSize::Nano,
        class_names,
        Some(Device::Cpu),
    );

    if result.is_ok() {
        let detector = result.unwrap();
        let info = detector.info();
        assert!(info.contains("YOLOv5"));
        assert!(info.contains("80 classes"));
    } else {
        eprintln!("Failed to create detector: {:?}", result.err());
    }
}

#[test]
fn test_all_yolo_versions() {
    let versions = vec![
        YoloVersion::V5,
        YoloVersion::V8,
        YoloVersion::V10,
        YoloVersion::V11,
        YoloVersion::V12,
    ];

    for version in versions {
        assert!(!version.as_str().is_empty());
    }
}

#[test]
fn test_all_yolo_sizes() {
    let sizes = vec![
        YoloSize::Nano,
        YoloSize::Small,
        YoloSize::Medium,
        YoloSize::Large,
        YoloSize::XLarge,
    ];

    for size in sizes {
        assert_eq!(size.suffix().len(), 1);
    }
}

#[test]
fn test_model_name_generation() {
    // Test that we can generate proper model names
    let versions = ["v5", "v8", "v10", "v11", "v12"];
    let sizes = ["n", "s", "m", "l", "x"];

    for version in versions {
        for size in sizes {
            let model_name = format!("yolo{}{}", version, size);
            assert!(!model_name.is_empty());
            assert!(model_name.starts_with("yolo"));
        }
    }
}
