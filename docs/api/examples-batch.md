# POST /examples/batch - Batch Example Endpoint

**URL**: `/examples/batch`  
**Method**: `POST`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

A batch example endpoint that demonstrates batch inference functionality with predefined inputs. This endpoint showcases how to process multiple inputs simultaneously and serves as an example for batch processing workflows within the PyTorch Inference Framework.

## Request

### Request Body
None required - this endpoint uses predefined example inputs: `[1, 2, 3, 4, 5]`

### Request Examples

#### Basic Batch Example
```json
{}
```

*Note: This endpoint automatically uses a predefined batch of inputs `[1, 2, 3, 4, 5]` for demonstration purposes.*

## Response

### Success Response (200 OK)

#### Batch Example Response
```json
{
  "example": "batch_prediction",
  "input_count": 5,
  "response": {
    "success": true,
    "predictions": [
      [0.123, 0.456, 0.789],
      [0.234, 0.567, 0.890],
      [0.345, 0.678, 0.901],
      [0.456, 0.789, 0.012],
      [0.567, 0.890, 0.123]
    ],
    "model_name": "example_model",
    "inference_time_ms": 45.2,
    "timestamp": "2025-08-14T10:30:00.000Z",
    "request_id": "batch_req_12345",
    "batch_info": {
      "batch_size": 5,
      "successful_predictions": 5,
      "failed_predictions": 0,
      "processing_time": {
        "preprocessing_ms": 8.3,
        "inference_ms": 28.7,
        "postprocessing_ms": 8.2
      }
    },
    "metadata": {
      "framework_version": "1.0.0",
      "model_version": "v1.0",
      "batch_processing": true,
      "input_shape": [5, 1],
      "output_shape": [5, 3],
      "individual_inference_times": [8.2, 9.1, 8.5, 9.8, 9.6]
    }
  }
}
```

#### Batch Example with Mixed Results
```json
{
  "example": "batch_prediction",
  "input_count": 5,
  "response": {
    "success": true,
    "predictions": [
      [0.123, 0.456, 0.789],
      [0.234, 0.567, 0.890],
      null,
      [0.456, 0.789, 0.012],
      [0.567, 0.890, 0.123]
    ],
    "model_name": "example_model",
    "inference_time_ms": 42.8,
    "timestamp": "2025-08-14T10:30:00.000Z",
    "request_id": "batch_req_12346",
    "batch_info": {
      "batch_size": 5,
      "successful_predictions": 4,
      "failed_predictions": 1,
      "failed_indices": [2],
      "failure_reasons": ["Input validation failed for index 2"]
    },
    "warnings": [
      "Some predictions in the batch failed - see failed_indices for details"
    ]
  }
}
```

### Error Response (500 Internal Server Error)

```json
{
  "detail": "Batch inference failed: Model not available"
}
```

#### Response Fields

##### Root Level
| Field | Type | Description |
|-------|------|-------------|
| `example` | string | Always "batch_prediction" for this endpoint |
| `input_count` | integer | Number of inputs processed (always 5 for this example) |
| `response` | object | The batch inference response object |

##### Response Object
| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the batch inference was successful overall |
| `predictions` | array | Array of prediction results, null for failed predictions |
| `model_name` | string | Name of the model used for inference |
| `inference_time_ms` | number | Total time taken for batch inference in milliseconds |
| `timestamp` | string | ISO timestamp when inference was completed |
| `request_id` | string | Unique identifier for this batch request |
| `batch_info` | object | Information about the batch processing |
| `metadata` | object | Additional metadata about the inference |
| `warnings` | array | Any warnings about the batch processing |

##### Batch Info Object
| Field | Type | Description |
|-------|------|-------------|
| `batch_size` | integer | Number of inputs in the batch |
| `successful_predictions` | integer | Number of successful predictions |
| `failed_predictions` | integer | Number of failed predictions |
| `failed_indices` | array | Indices of failed predictions (if any) |
| `failure_reasons` | array | Reasons for prediction failures (if any) |
| `processing_time` | object | Breakdown of processing time |

##### Processing Time Object
| Field | Type | Description |
|-------|------|-------------|
| `preprocessing_ms` | number | Time spent on input preprocessing |
| `inference_ms` | number | Time spent on actual model inference |
| `postprocessing_ms` | number | Time spent on output postprocessing |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Batch prediction completed successfully |
| 500 | Internal server error |

## Examples

### Basic Batch Example

**Request:**
```bash
curl -X POST http://localhost:8000/examples/batch \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response:**
```json
{
  "example": "batch_prediction",
  "input_count": 5,
  "response": {
    "success": true,
    "predictions": [
      [0.123, 0.456, 0.789],
      [0.234, 0.567, 0.890],
      [0.345, 0.678, 0.901],
      [0.456, 0.789, 0.012],
      [0.567, 0.890, 0.123]
    ],
    "batch_info": {
      "batch_size": 5,
      "successful_predictions": 5,
      "failed_predictions": 0
    }
  }
}
```

### Python Batch Example Client

```python
import requests
import json
import time
import statistics
from typing import Dict, List, Any
import concurrent.futures
from datetime import datetime

class BatchExampleClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def run_batch_example(self) -> Dict:
        """Run a batch example prediction"""
        try:
            response = requests.post(
                f"{self.base_url}/examples/batch",
                json={},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def test_batch_connectivity(self) -> bool:
        """Test if the batch API is responsive"""
        result = self.run_batch_example()
        return "error" not in result and result.get("response", {}).get("success", False)
    
    def benchmark_batch_performance(self, iterations: int = 10) -> Dict:
        """Benchmark batch inference performance"""
        print(f"🚀 Benchmarking batch inference ({iterations} iterations)...")
        
        response_times = []
        inference_times = []
        batch_sizes = []
        throughput_rates = []  # predictions per second
        errors = 0
        
        for i in range(iterations):
            start_time = time.time()
            result = self.run_batch_example()
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to ms
            response_times.append(response_time)
            
            if "error" not in result and result.get("response", {}).get("success"):
                inference_time = result["response"].get("inference_time_ms", 0)
                input_count = result.get("input_count", 0)
                
                inference_times.append(inference_time)
                batch_sizes.append(input_count)
                
                # Calculate throughput (predictions per second)
                if inference_time > 0:
                    throughput = (input_count / inference_time) * 1000
                    throughput_rates.append(throughput)
                
                batch_info = result["response"].get("batch_info", {})
                successful = batch_info.get("successful_predictions", 0)
                failed = batch_info.get("failed_predictions", 0)
                
                print(f"  Iteration {i+1:2d}: {response_time:6.1f}ms total, {inference_time:6.1f}ms inference, {successful}/{successful+failed} successful")
            else:
                errors += 1
                print(f"  Iteration {i+1:2d}: ERROR - {result.get('error', 'Unknown error')}")
        
        if inference_times:
            benchmark_results = {
                "iterations": iterations,
                "errors": errors,
                "success_rate": (iterations - errors) / iterations * 100,
                "response_times_ms": {
                    "min": min(response_times),
                    "max": max(response_times),
                    "avg": statistics.mean(response_times),
                    "median": statistics.median(response_times),
                    "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
                },
                "inference_times_ms": {
                    "min": min(inference_times),
                    "max": max(inference_times),
                    "avg": statistics.mean(inference_times),
                    "median": statistics.median(inference_times),
                    "std_dev": statistics.stdev(inference_times) if len(inference_times) > 1 else 0
                },
                "batch_performance": {
                    "avg_batch_size": statistics.mean(batch_sizes),
                    "avg_throughput_predictions_per_sec": statistics.mean(throughput_rates) if throughput_rates else 0,
                    "max_throughput_predictions_per_sec": max(throughput_rates) if throughput_rates else 0
                }
            }
        else:
            benchmark_results = {
                "iterations": iterations,
                "errors": errors,
                "success_rate": 0,
                "error": "All requests failed"
            }
        
        return benchmark_results
    
    def analyze_batch_consistency(self, iterations: int = 20) -> Dict:
        """Analyze consistency of batch predictions"""
        print(f"🔍 Analyzing batch prediction consistency ({iterations} iterations)...")
        
        all_predictions = []
        inference_times = []
        batch_infos = []
        
        for i in range(iterations):
            result = self.run_batch_example()
            
            if "error" not in result and result.get("response", {}).get("success"):
                predictions = result["response"].get("predictions", [])
                inference_time = result["response"].get("inference_time_ms", 0)
                batch_info = result["response"].get("batch_info", {})
                
                all_predictions.append(predictions)
                inference_times.append(inference_time)
                batch_infos.append(batch_info)
                
                print(f"  Iteration {i+1:2d}: {len(predictions)} predictions, {inference_time:.1f}ms")
            else:
                print(f"  Iteration {i+1:2d}: ERROR - {result.get('error', 'Unknown error')}")
        
        if not all_predictions:
            return {"error": "No successful predictions to analyze"}
        
        # Analyze prediction consistency
        prediction_lengths = [len(preds) for preds in all_predictions]
        successful_rates = []
        
        for batch_info in batch_infos:
            total = batch_info.get("batch_size", 0)
            successful = batch_info.get("successful_predictions", 0)
            if total > 0:
                successful_rates.append(successful / total * 100)
        
        # Check if predictions are deterministic (same inputs should give same outputs)
        first_predictions = all_predictions[0] if all_predictions else []
        is_deterministic = all(preds == first_predictions for preds in all_predictions[1:])
        
        consistency_analysis = {
            "total_runs": iterations,
            "successful_runs": len(all_predictions),
            "prediction_consistency": {
                "prediction_lengths": {
                    "min": min(prediction_lengths),
                    "max": max(prediction_lengths),
                    "avg": statistics.mean(prediction_lengths),
                    "consistent": len(set(prediction_lengths)) == 1
                },
                "success_rates": {
                    "min": min(successful_rates) if successful_rates else 0,
                    "max": max(successful_rates) if successful_rates else 0,
                    "avg": statistics.mean(successful_rates) if successful_rates else 0,
                    "std_dev": statistics.stdev(successful_rates) if len(successful_rates) > 1 else 0
                },
                "deterministic": is_deterministic
            },
            "performance_consistency": {
                "inference_times_ms": {
                    "min": min(inference_times),
                    "max": max(inference_times),
                    "avg": statistics.mean(inference_times),
                    "std_dev": statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
                    "coefficient_of_variation": (statistics.stdev(inference_times) / statistics.mean(inference_times)) * 100 if len(inference_times) > 1 and statistics.mean(inference_times) > 0 else 0
                }
            }
        }
        
        return consistency_analysis
    
    def stress_test_batch(
        self,
        duration_seconds: int = 60,
        concurrent_batches: int = 2
    ) -> Dict:
        """Run a stress test on the batch example endpoint"""
        print(f"💥 Starting batch stress test for {duration_seconds}s with {concurrent_batches} concurrent batches...")
        
        def worker_batch(worker_id: int, results_list: List[Dict]):
            worker_results = []
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                batch_start = time.time()
                result = self.run_batch_example()
                batch_end = time.time()
                
                worker_results.append({
                    "worker_id": worker_id,
                    "success": "error" not in result and result.get("response", {}).get("success", False),
                    "response_time_ms": (batch_end - batch_start) * 1000,
                    "inference_time_ms": result.get("response", {}).get("inference_time_ms", 0) if "error" not in result else 0,
                    "batch_size": result.get("input_count", 0) if "error" not in result else 0,
                    "successful_predictions": result.get("response", {}).get("batch_info", {}).get("successful_predictions", 0) if "error" not in result else 0,
                    "error": result.get("error")
                })
            
            results_list.extend(worker_results)
        
        # Run concurrent batch requests
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_batches) as executor:
            futures = []
            for i in range(concurrent_batches):
                future = executor.submit(worker_batch, i, all_results)
                futures.append(future)
            
            concurrent.futures.wait(futures)
        
        # Analyze results
        successful = [r for r in all_results if r["success"]]
        failed = [r for r in all_results if not r["success"]]
        
        stress_results = {
            "duration_seconds": duration_seconds,
            "concurrent_batches": concurrent_batches,
            "total_batch_requests": len(all_results),
            "successful_batches": len(successful),
            "failed_batches": len(failed),
            "batches_per_second": len(all_results) / duration_seconds,
            "success_rate": len(successful) / len(all_results) * 100 if all_results else 0
        }
        
        if successful:
            response_times = [r["response_time_ms"] for r in successful]
            inference_times = [r["inference_time_ms"] for r in successful if r["inference_time_ms"]]
            total_predictions = sum(r["successful_predictions"] for r in successful)
            total_inference_time = sum(r["inference_time_ms"] for r in successful if r["inference_time_ms"])
            
            stress_results["performance"] = {
                "avg_response_time_ms": statistics.mean(response_times),
                "min_response_time_ms": min(response_times),
                "max_response_time_ms": max(response_times),
                "avg_inference_time_ms": statistics.mean(inference_times) if inference_times else 0,
                "total_predictions_processed": total_predictions,
                "predictions_per_second": (total_predictions / total_inference_time * 1000) if total_inference_time > 0 else 0
            }
            
            # Analyze per-worker performance
            worker_stats = {}
            for result in successful:
                worker_id = result["worker_id"]
                if worker_id not in worker_stats:
                    worker_stats[worker_id] = {
                        "batches": 0,
                        "total_predictions": 0,
                        "total_time": 0
                    }
                
                worker_stats[worker_id]["batches"] += 1
                worker_stats[worker_id]["total_predictions"] += result["successful_predictions"]
                worker_stats[worker_id]["total_time"] += result["response_time_ms"]
            
            stress_results["worker_analysis"] = worker_stats
        
        if failed:
            error_types = {}
            for result in failed:
                error = result.get("error", "Unknown error")
                error_types[error] = error_types.get(error, 0) + 1
            
            stress_results["error_analysis"] = error_types
        
        return stress_results
    
    def compare_batch_vs_individual(self, iterations: int = 10) -> Dict:
        """Compare batch processing vs individual predictions"""
        print(f"⚖️  Comparing batch vs individual processing ({iterations} iterations)...")
        
        # Test batch processing
        print("  Testing batch processing...")
        batch_times = []
        batch_throughputs = []
        
        for i in range(iterations):
            start_time = time.time()
            result = self.run_batch_example()
            end_time = time.time()
            
            if "error" not in result and result.get("response", {}).get("success"):
                batch_time = (end_time - start_time) * 1000
                input_count = result.get("input_count", 0)
                inference_time = result["response"].get("inference_time_ms", 0)
                
                batch_times.append(batch_time)
                if inference_time > 0:
                    throughput = (input_count / inference_time) * 1000
                    batch_throughputs.append(throughput)
                
                print(f"    Batch {i+1:2d}: {batch_time:.1f}ms for {input_count} predictions")
        
        # Simulate individual processing (using simple endpoint)
        print("  Simulating individual processing...")
        individual_times = []
        individual_total_times = []
        
        for i in range(iterations):
            total_start = time.time()
            individual_batch_times = []
            
            # Simulate processing 5 individual requests (same as batch size)
            for j in range(5):
                try:
                    individual_start = time.time()
                    response = requests.post(
                        f"{self.base_url}/examples/simple",
                        json={"input": j + 1},
                        timeout=30
                    )
                    individual_end = time.time()
                    
                    if response.status_code == 200:
                        individual_time = (individual_end - individual_start) * 1000
                        individual_batch_times.append(individual_time)
                except:
                    individual_batch_times.append(float('inf'))  # Mark as failed
            
            total_end = time.time()
            total_time = (total_end - total_start) * 1000
            
            if all(t != float('inf') for t in individual_batch_times):
                individual_times.extend(individual_batch_times)
                individual_total_times.append(total_time)
                print(f"    Individual {i+1:2d}: {total_time:.1f}ms for 5 predictions")
        
        # Calculate comparison metrics
        comparison = {
            "iterations": iterations,
            "batch_performance": {
                "avg_time_ms": statistics.mean(batch_times) if batch_times else float('inf'),
                "avg_throughput_predictions_per_sec": statistics.mean(batch_throughputs) if batch_throughputs else 0,
                "successful_runs": len(batch_times)
            },
            "individual_performance": {
                "avg_total_time_ms": statistics.mean(individual_total_times) if individual_total_times else float('inf'),
                "avg_individual_time_ms": statistics.mean(individual_times) if individual_times else float('inf'),
                "successful_runs": len(individual_total_times)
            }
        }
        
        # Calculate efficiency gains
        if batch_times and individual_total_times:
            avg_batch_time = statistics.mean(batch_times)
            avg_individual_total = statistics.mean(individual_total_times)
            
            comparison["efficiency"] = {
                "batch_speedup": avg_individual_total / avg_batch_time if avg_batch_time > 0 else 0,
                "time_savings_ms": avg_individual_total - avg_batch_time,
                "time_savings_percentage": ((avg_individual_total - avg_batch_time) / avg_individual_total * 100) if avg_individual_total > 0 else 0
            }
        
        return comparison
    
    def print_benchmark_results(self, results: Dict, title: str = "Benchmark Results"):
        """Print formatted benchmark results"""
        print(f"\n{'='*80}")
        print(f"{title.upper()}")
        print(f"{'='*80}")
        
        if "error" in results:
            print(f"❌ Benchmark failed: {results['error']}")
            return
        
        print(f"Iterations: {results.get('iterations', 'N/A')}")
        print(f"Errors: {results.get('errors', 'N/A')}")
        print(f"Success Rate: {results.get('success_rate', 0):.1f}%")
        
        if "response_times_ms" in results:
            rt = results["response_times_ms"]
            print(f"\nResponse Times:")
            print(f"  Min: {rt.get('min', 0):.1f}ms")
            print(f"  Max: {rt.get('max', 0):.1f}ms")
            print(f"  Avg: {rt.get('avg', 0):.1f}ms")
            print(f"  Median: {rt.get('median', 0):.1f}ms")
            print(f"  Std Dev: {rt.get('std_dev', 0):.1f}ms")
        
        if "inference_times_ms" in results:
            it = results["inference_times_ms"]
            print(f"\nInference Times:")
            print(f"  Min: {it.get('min', 0):.1f}ms")
            print(f"  Max: {it.get('max', 0):.1f}ms")
            print(f"  Avg: {it.get('avg', 0):.1f}ms")
            print(f"  Median: {it.get('median', 0):.1f}ms")
            print(f"  Std Dev: {it.get('std_dev', 0):.1f}ms")
        
        if "batch_performance" in results:
            bp = results["batch_performance"]
            print(f"\nBatch Performance:")
            print(f"  Avg Batch Size: {bp.get('avg_batch_size', 0):.1f}")
            print(f"  Avg Throughput: {bp.get('avg_throughput_predictions_per_sec', 0):.1f} predictions/sec")
            print(f"  Max Throughput: {bp.get('max_throughput_predictions_per_sec', 0):.1f} predictions/sec")
        
        print(f"{'='*80}")

# Usage Examples
client = BatchExampleClient()

# Test connectivity
print("🔗 Testing batch API connectivity...")
if client.test_batch_connectivity():
    print("✅ Batch API is responsive")
else:
    print("❌ Batch API is not responsive")

# Run basic batch example
print("\n🚀 Running basic batch example...")
result = client.run_batch_example()
print(f"Result: {json.dumps(result, indent=2)}")

# Benchmark performance
print("\n📊 Running batch performance benchmark...")
benchmark = client.benchmark_batch_performance(iterations=10)
client.print_benchmark_results(benchmark, "Batch Performance Benchmark")

# Analyze consistency
print("\n🔍 Analyzing batch prediction consistency...")
consistency = client.analyze_batch_consistency(iterations=5)
print(f"Consistency Analysis: {json.dumps(consistency, indent=2)}")

# Compare batch vs individual processing
print("\n⚖️  Comparing batch vs individual processing...")
comparison = client.compare_batch_vs_individual(iterations=3)
if "efficiency" in comparison:
    efficiency = comparison["efficiency"]
    print(f"Batch is {efficiency['batch_speedup']:.1f}x faster than individual processing")
    print(f"Time savings: {efficiency['time_savings_ms']:.1f}ms ({efficiency['time_savings_percentage']:.1f}%)")

# Stress test (uncomment to run)
# print("\n💥 Running batch stress test...")
# stress_results = client.stress_test_batch(duration_seconds=30, concurrent_batches=2)
# print(f"Stress test: {stress_results['successful_batches']}/{stress_results['total_batch_requests']} batches successful")
# print(f"Batches per second: {stress_results['batches_per_second']:.1f}")
```

### JavaScript Batch Testing Utility

```javascript
class BatchExampleTester {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }
  
  async runBatchExample() {
    try {
      const response = await fetch(`${this.baseUrl}/examples/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
      });
      
      return await response.json();
    } catch (error) {
      return { error: error.message };
    }
  }
  
  async testBatchConnectivity() {
    const result = await this.runBatchExample();
    return !result.error && result.response?.success;
  }
  
  async benchmarkBatchPerformance(iterations = 10) {
    console.log(`🚀 Benchmarking batch performance (${iterations} iterations)...`);
    
    const results = [];
    const startOverall = performance.now();
    
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      const result = await this.runBatchExample();
      const end = performance.now();
      
      const responseTime = end - start;
      const success = !result.error && result.response?.success;
      
      results.push({
        iteration: i + 1,
        success,
        responseTime,
        inferenceTime: result.response?.inference_time_ms || 0,
        batchSize: result.input_count || 0,
        successfulPredictions: result.response?.batch_info?.successful_predictions || 0,
        failedPredictions: result.response?.batch_info?.failed_predictions || 0,
        error: result.error
      });
      
      console.log(`  Iteration ${i + 1}: ${responseTime.toFixed(1)}ms (${success ? '✅' : '❌'})`);
    }
    
    const endOverall = performance.now();
    const totalTime = endOverall - startOverall;
    
    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);
    
    return {
      iterations,
      successful: successful.length,
      failed: failed.length,
      successRate: (successful.length / iterations) * 100,
      totalTime,
      avgResponseTime: successful.reduce((sum, r) => sum + r.responseTime, 0) / successful.length,
      avgInferenceTime: successful.reduce((sum, r) => sum + r.inferenceTime, 0) / successful.length,
      avgBatchSize: successful.reduce((sum, r) => sum + r.batchSize, 0) / successful.length,
      totalPredictions: successful.reduce((sum, r) => sum + r.successfulPredictions, 0),
      results
    };
  }
  
  async analyzeBatchConsistency(iterations = 10) {
    console.log(`🔍 Analyzing batch consistency (${iterations} iterations)...`);
    
    const results = [];
    const allPredictions = [];
    
    for (let i = 0; i < iterations; i++) {
      const result = await this.runBatchExample();
      
      if (!result.error && result.response?.success) {
        const predictions = result.response.predictions || [];
        const batchInfo = result.response.batch_info || {};
        
        results.push({
          iteration: i + 1,
          predictions,
          inferenceTime: result.response.inference_time_ms,
          successfulPredictions: batchInfo.successful_predictions,
          failedPredictions: batchInfo.failed_predictions,
          batchSize: result.input_count
        });
        
        allPredictions.push(predictions);
        console.log(`  Iteration ${i + 1}: ${predictions.length} predictions, ${result.response.inference_time_ms}ms`);
      } else {
        console.log(`  Iteration ${i + 1}: ERROR - ${result.error}`);
      }
    }
    
    if (results.length === 0) {
      return { error: 'No successful batches to analyze' };
    }
    
    // Check consistency
    const predictionLengths = results.map(r => r.predictions.length);
    const inferenceTimes = results.map(r => r.inferenceTime);
    const successRates = results.map(r => r.successfulPredictions / r.batchSize * 100);
    
    // Check if predictions are deterministic
    const firstPredictions = JSON.stringify(allPredictions[0]);
    const isDeterministic = allPredictions.every(preds => JSON.stringify(preds) === firstPredictions);
    
    return {
      totalRuns: iterations,
      successfulRuns: results.length,
      consistency: {
        predictionLengthConsistent: predictionLengths.every(len => len === predictionLengths[0]),
        avgPredictionLength: predictionLengths.reduce((sum, len) => sum + len, 0) / predictionLengths.length,
        deterministic: isDeterministic,
        avgSuccessRate: successRates.reduce((sum, rate) => sum + rate, 0) / successRates.length,
        avgInferenceTime: inferenceTimes.reduce((sum, time) => sum + time, 0) / inferenceTimes.length,
        inferenceTimeVariation: Math.max(...inferenceTimes) - Math.min(...inferenceTimes)
      },
      results
    };
  }
  
  renderBatchTestInterface(containerId) {
    const container = document.getElementById(containerId);
    
    container.innerHTML = `
      <div class="batch-example-tester">
        <h2>Batch Example Tester</h2>
        
        <div class="test-section">
          <h3>Quick Batch Test</h3>
          <button onclick="batchTester.runQuickBatchTest()" class="btn-primary">
            🚀 Run Batch Example
          </button>
          <div id="quick-batch-result" class="result-display"></div>
        </div>
        
        <div class="test-section">
          <h3>Performance Tests</h3>
          <div class="button-group">
            <button onclick="batchTester.runBatchConnectivityTest()" class="btn-secondary">
              🔗 Connectivity Test
            </button>
            <button onclick="batchTester.runBatchBenchmark()" class="btn-secondary">
              📊 Benchmark (10x)
            </button>
            <button onclick="batchTester.runConsistencyTest()" class="btn-secondary">
              🔍 Consistency Test
            </button>
          </div>
        </div>
        
        <div id="batch-test-results" class="results-section"></div>
      </div>
    `;
  }
  
  async runQuickBatchTest() {
    const resultDiv = document.getElementById('quick-batch-result');
    resultDiv.innerHTML = '<div class="loading">Running batch test...</div>';
    
    const result = await this.runBatchExample();
    
    if (result.error) {
      resultDiv.innerHTML = `<div class="error">❌ Error: ${result.error}</div>`;
    } else if (result.response?.success) {
      const batchInfo = result.response.batch_info || {};
      const predictions = result.response.predictions || [];
      
      resultDiv.innerHTML = `
        <div class="success">
          ✅ Batch Success!
          <br>Input Count: ${result.input_count}
          <br>Inference Time: ${result.response.inference_time_ms}ms
          <br>Successful Predictions: ${batchInfo.successful_predictions || 0}
          <br>Failed Predictions: ${batchInfo.failed_predictions || 0}
          <br>Predictions: <code>${JSON.stringify(predictions.slice(0, 2))}${predictions.length > 2 ? '...' : ''}</code>
        </div>
      `;
    } else {
      resultDiv.innerHTML = '<div class="error">❌ Batch test failed</div>';
    }
  }
  
  async runBatchConnectivityTest() {
    const resultsDiv = document.getElementById('batch-test-results');
    resultsDiv.innerHTML = '<div class="loading">Testing batch connectivity...</div>';
    
    const isConnected = await this.testBatchConnectivity();
    
    resultsDiv.innerHTML = `
      <div class="test-result">
        <h4>Batch Connectivity Test</h4>
        <div class="${isConnected ? 'success' : 'error'}">
          ${isConnected ? '✅ Batch API is responsive and working correctly' : '❌ Batch API is not responding or not working'}
        </div>
      </div>
    `;
  }
  
  async runBatchBenchmark() {
    const resultsDiv = document.getElementById('batch-test-results');
    resultsDiv.innerHTML = '<div class="loading">Running batch benchmark (10 iterations)...</div>';
    
    const benchmark = await this.benchmarkBatchPerformance(10);
    
    resultsDiv.innerHTML = `
      <div class="test-result">
        <h4>Batch Performance Benchmark</h4>
        <div class="benchmark-stats">
          <div class="stat">Success Rate: ${benchmark.successRate.toFixed(1)}%</div>
          <div class="stat">Successful Batches: ${benchmark.successful}/${benchmark.iterations}</div>
          <div class="stat">Avg Response: ${benchmark.avgResponseTime.toFixed(1)}ms</div>
          <div class="stat">Avg Inference: ${benchmark.avgInferenceTime.toFixed(1)}ms</div>
          <div class="stat">Avg Batch Size: ${benchmark.avgBatchSize.toFixed(0)}</div>
          <div class="stat">Total Predictions: ${benchmark.totalPredictions}</div>
        </div>
        <div class="benchmark-chart">
          ${benchmark.results.map(r => `
            <div class="benchmark-bar ${r.success ? 'success' : 'error'}" 
                 style="height: ${(r.responseTime / Math.max(...benchmark.results.map(x => x.responseTime))) * 100}%"
                 title="Iteration ${r.iteration}: ${r.responseTime.toFixed(1)}ms, ${r.successfulPredictions} predictions">
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }
  
  async runConsistencyTest() {
    const resultsDiv = document.getElementById('batch-test-results');
    resultsDiv.innerHTML = '<div class="loading">Running consistency test...</div>';
    
    const consistency = await this.analyzeBatchConsistency(5);
    
    if (consistency.error) {
      resultsDiv.innerHTML = `
        <div class="test-result">
          <h4>Batch Consistency Test</h4>
          <div class="error">❌ Error: ${consistency.error}</div>
        </div>
      `;
      return;
    }
    
    const cons = consistency.consistency;
    
    resultsDiv.innerHTML = `
      <div class="test-result">
        <h4>Batch Consistency Analysis</h4>
        <div class="consistency-stats">
          <div class="stat">Successful Runs: ${consistency.successfulRuns}/${consistency.totalRuns}</div>
          <div class="stat">Length Consistent: ${cons.predictionLengthConsistent ? '✅ Yes' : '❌ No'}</div>
          <div class="stat">Deterministic: ${cons.deterministic ? '✅ Yes' : '❌ No'}</div>
          <div class="stat">Avg Success Rate: ${cons.avgSuccessRate.toFixed(1)}%</div>
          <div class="stat">Avg Inference Time: ${cons.avgInferenceTime.toFixed(1)}ms</div>
          <div class="stat">Time Variation: ${cons.inferenceTimeVariation.toFixed(1)}ms</div>
        </div>
      </div>
    `;
  }
}

// Global instance
const batchTester = new BatchExampleTester();

// Initialize interface
// batchTester.renderBatchTestInterface('batch-example-container');
```

## Use Cases

### Batch Processing Demonstration
- Show batch inference capabilities
- Demonstrate efficient processing of multiple inputs
- Compare batch vs individual processing performance

### Performance Testing
- Benchmark batch inference speed
- Test system throughput under batch loads
- Validate batch processing consistency

### Integration Testing
- Verify batch endpoint functionality
- Test batch processing error handling
- Validate batch response format

### Development Examples
- Provide working batch processing example
- Demonstrate proper batch request/response handling
- Show batch processing best practices

## Related Endpoints

- [Batch Predict](./batch-predict.md) - Full batch prediction endpoint
- [Simple Example](./examples-simple.md) - Single prediction example
- [Predict](./predict.md) - Individual prediction endpoint
- [Health](./health.md) - System health check

## Best Practices

1. **Batch Size Optimization**: Test different batch sizes for optimal performance
2. **Error Handling**: Implement proper handling for partial batch failures
3. **Throughput Monitoring**: Track predictions per second metrics
4. **Resource Management**: Monitor memory usage with large batches
5. **Consistency Validation**: Verify batch processing produces consistent results
