// Integration tests for the complete system

use torch_inference::{cache::Cache, batch::BatchProcessor, monitor::Monitor};
use std::sync::Arc;
use std::time::Instant;
use serde_json::json;

#[tokio::test]
async fn test_end_to_end_request_flow() {
    // Setup components
    let cache = Arc::new(Cache::new(1000));
    let batch_processor = Arc::new(BatchProcessor::new(32, 100));
    let monitor = Arc::new(Monitor::new());
    
    // Simulate a request
    monitor.record_request_start();
    let request_start = Instant::now();
    
    // Check cache first
    let cache_key = "model:test_input";
    if let Some(_cached_result) = cache.get(cache_key) {
        // Cache hit - fast path
        let latency = request_start.elapsed().as_millis() as u64;
        monitor.record_request_end(latency, "/api/inference", true);
    } else {
        // Cache miss - process request
        let request = torch_inference::batch::BatchRequest {
            id: "req_1".to_string(),
            model_name: "test_model".to_string(),
            inputs: vec![json!({"data": "test"})],
            priority: 1,
            timestamp: Instant::now(),
        };
        
        batch_processor.add_request(request).await.ok();
        
        // Simulate processing
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        
        // Store in cache
        let result = json!({"output": "test_result"});
        cache.set(cache_key.to_string(), result, 60).ok();
        
        let latency = request_start.elapsed().as_millis() as u64;
        monitor.record_request_end(latency, "/api/inference", true);
    }
    
    // Verify metrics
    let metrics = monitor.get_metrics();
    assert_eq!(metrics.total_requests, 1);
    assert_eq!(metrics.total_processed, 1);
}

#[tokio::test]
async fn test_concurrent_system_load() {
    let cache = Arc::new(Cache::new(10000));
    let monitor = Arc::new(Monitor::new());
    
    let mut handles = vec![];
    
    for i in 0..100 {
        let cache_clone = Arc::clone(&cache);
        let monitor_clone = Arc::clone(&monitor);
        
        let handle = tokio::spawn(async move {
            monitor_clone.record_request_start();
            
            let key = format!("key_{}", i);
            let value = json!({"data": i});
            
            // Simulate processing
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            
            cache_clone.set(key.clone(), value, 60).ok();
            
            monitor_clone.record_request_end(10, "/api/test", true);
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    let metrics = monitor.get_metrics();
    assert_eq!(metrics.total_requests, 100);
    assert_eq!(metrics.total_processed, 100);
    assert_eq!(cache.size(), 100);
}

#[tokio::test]
async fn test_batch_processing_flow() {
    let batch_processor = BatchProcessor::new(10, 100);
    
    // Add multiple requests
    for i in 0..5 {
        let request = torch_inference::batch::BatchRequest {
            id: format!("req_{}", i),
            model_name: "test_model".to_string(),
            inputs: vec![json!({"data": i})],
            priority: i as i32,
            timestamp: Instant::now(),
        };
        
        batch_processor.add_request(request).await.ok();
    }
    
    // Process batch
    let batch = batch_processor.get_batch().await;
    assert_eq!(batch.len(), 5);
    
    // Verify all requests present
    for i in 0..5 {
        assert!(batch.iter().any(|r| r.id == format!("req_{}", i)));
    }
}

#[test]
fn test_cache_and_monitor_integration() {
    let cache = Arc::new(Cache::new(100));
    let monitor = Arc::new(Monitor::new());
    
    // Simulate cache hit scenario
    cache.set("hot_key".to_string(), json!("hot_value"), 60).ok();
    
    for _ in 0..10 {
        monitor.record_request_start();
        
        if cache.get("hot_key").is_some() {
            // Cache hit - very fast
            monitor.record_request_end(1, "/api/cache_hit", true);
        }
    }
    
    let metrics = monitor.get_metrics();
    assert_eq!(metrics.total_processed, 10);
    assert!(metrics.avg_latency_ms < 10.0); // All cache hits should be fast
}

#[test]
fn test_system_under_error_conditions() {
    let monitor = Monitor::new();
    
    // Simulate mix of success and failures
    for i in 0..50 {
        monitor.record_request_start();
        let success = i % 5 != 0; // 20% failure rate
        monitor.record_request_end(100, "/api/test", success);
    }
    
    let metrics = monitor.get_metrics();
    assert_eq!(metrics.total_requests, 50);
    assert_eq!(metrics.total_errors, 10); // 20% of 50
    
    let health = monitor.get_health_status();
    assert!(health.healthy); // Should still be healthy
}
