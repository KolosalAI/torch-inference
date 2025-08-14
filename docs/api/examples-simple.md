# POST /examples/simple - Simple Example Endpoint

**URL**: `/examples/simple`  
**Method**: `POST`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

A simple example endpoint for testing basic inference functionality. This endpoint demonstrates the simplest way to use the PyTorch Inference Framework and serves as a quick test for system connectivity and basic model operation.

## Request

### Request Body

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input` | any | No | 42 | Input value for the example prediction |

### Request Examples

#### Basic Example
```json
{
  "input": 42
}
```

#### String Input Example
```json
{
  "input": "test string"
}
```

#### Array Input Example
```json
{
  "input": [1, 2, 3, 4, 5]
}
```

#### Object Input Example
```json
{
  "input": {
    "data": [1.5, 2.3, 0.8],
    "metadata": {
      "type": "numerical"
    }
  }
}
```

## Response

### Success Response (200 OK)

#### Simple Example Response
```json
{
  "example": "simple_prediction",
  "input": 42,
  "response": {
    "success": true,
    "prediction": [0.123, 0.456, 0.789],
    "model_name": "example_model",
    "inference_time_ms": 15.7,
    "timestamp": "2025-08-14T10:30:00.000Z",
    "request_id": "req_12345",
    "metadata": {
      "framework_version": "1.0.0",
      "model_version": "v1.0",
      "input_shape": [1],
      "output_shape": [3],
      "processing_time": {
        "preprocessing_ms": 2.1,
        "inference_ms": 10.4,
        "postprocessing_ms": 3.2
      }
    }
  }
}
```

#### String Input Response
```json
{
  "example": "simple_prediction",
  "input": "test string",
  "response": {
    "success": true,
    "prediction": "processed_test_string",
    "model_name": "example_model",
    "inference_time_ms": 8.2,
    "timestamp": "2025-08-14T10:30:00.000Z",
    "request_id": "req_12346"
  }
}
```

#### Array Input Response
```json
{
  "example": "simple_prediction",
  "input": [1, 2, 3, 4, 5],
  "response": {
    "success": true,
    "prediction": [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
      [0.7, 0.8, 0.9],
      [0.2, 0.4, 0.6],
      [0.8, 0.1, 0.9]
    ],
    "model_name": "example_model",
    "inference_time_ms": 25.1,
    "timestamp": "2025-08-14T10:30:00.000Z",
    "request_id": "req_12347",
    "metadata": {
      "input_length": 5,
      "output_length": 5
    }
  }
}
```

### Error Response (500 Internal Server Error)

```json
{
  "detail": "Model inference failed: Invalid input format"
}
```

#### Response Fields

##### Root Level
| Field | Type | Description |
|-------|------|-------------|
| `example` | string | Always "simple_prediction" for this endpoint |
| `input` | any | The input value that was processed |
| `response` | object | The inference response object |

##### Response Object
| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the inference was successful |
| `prediction` | any | The model prediction result |
| `model_name` | string | Name of the model used for inference |
| `inference_time_ms` | number | Total time taken for inference in milliseconds |
| `timestamp` | string | ISO timestamp when inference was completed |
| `request_id` | string | Unique identifier for this request |
| `metadata` | object | Additional metadata about the inference |

##### Metadata Object
| Field | Type | Description |
|-------|------|-------------|
| `framework_version` | string | Version of the inference framework |
| `model_version` | string | Version of the model used |
| `input_shape` | array | Shape of the processed input |
| `output_shape` | array | Shape of the prediction output |
| `processing_time` | object | Breakdown of processing time |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Prediction completed successfully |
| 400 | Bad Request - Invalid input format |
| 500 | Internal server error |

## Examples

### Basic Simple Example

**Request:**
```bash
curl -X POST http://localhost:8000/examples/simple \
  -H "Content-Type: application/json" \
  -d '{"input": 42}'
```

**Response:**
```json
{
  "example": "simple_prediction",
  "input": 42,
  "response": {
    "success": true,
    "prediction": [0.123, 0.456, 0.789],
    "model_name": "example_model",
    "inference_time_ms": 15.7
  }
}
```

### Custom Input Example

**Request:**
```bash
curl -X POST http://localhost:8000/examples/simple \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello World"}'
```

### Array Input Example

**Request:**
```bash
curl -X POST http://localhost:8000/examples/simple \
  -H "Content-Type: application/json" \
  -d '{"input": [1, 2, 3, 4, 5]}'
```

### Python Example Client

```python
import requests
import json
import time
from typing import Any, Dict, List, Union

class SimpleExampleClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def run_simple_example(self, input_value: Any = None) -> Dict:
        """Run a simple example prediction"""
        payload = {}
        if input_value is not None:
            payload["input"] = input_value
        
        try:
            response = requests.post(
                f"{self.base_url}/examples/simple",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def test_connectivity(self) -> bool:
        """Test if the API is responsive"""
        result = self.run_simple_example()
        return "error" not in result and result.get("response", {}).get("success", False)
    
    def benchmark_simple_inference(
        self, 
        input_value: Any = None, 
        iterations: int = 10
    ) -> Dict:
        """Benchmark simple inference performance"""
        print(f"🚀 Benchmarking simple inference ({iterations} iterations)...")
        
        response_times = []
        inference_times = []
        errors = 0
        
        for i in range(iterations):
            start_time = time.time()
            result = self.run_simple_example(input_value)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to ms
            response_times.append(response_time)
            
            if "error" not in result and result.get("response", {}).get("success"):
                inference_time = result["response"].get("inference_time_ms", 0)
                inference_times.append(inference_time)
                print(f"  Iteration {i+1:2d}: {response_time:6.1f}ms total, {inference_time:6.1f}ms inference")
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
                    "avg": sum(response_times) / len(response_times),
                    "total": sum(response_times)
                },
                "inference_times_ms": {
                    "min": min(inference_times),
                    "max": max(inference_times),
                    "avg": sum(inference_times) / len(inference_times),
                    "total": sum(inference_times)
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
    
    def test_different_inputs(self) -> Dict:
        """Test the endpoint with different input types"""
        test_cases = [
            {"name": "Default (no input)", "input": None},
            {"name": "Integer", "input": 42},
            {"name": "Float", "input": 3.14159},
            {"name": "String", "input": "Hello, World!"},
            {"name": "List of integers", "input": [1, 2, 3, 4, 5]},
            {"name": "List of floats", "input": [1.1, 2.2, 3.3]},
            {"name": "Dictionary", "input": {"key": "value", "number": 123}},
            {"name": "Nested structure", "input": {
                "data": [1, 2, 3],
                "metadata": {"type": "test", "version": 1}
            }},
            {"name": "Boolean", "input": True},
            {"name": "Empty list", "input": []},
            {"name": "Empty dict", "input": {}}
        ]
        
        results = []
        
        print("🧪 Testing different input types...")
        
        for test_case in test_cases:
            print(f"  Testing: {test_case['name']}...")
            
            start_time = time.time()
            result = self.run_simple_example(test_case['input'])
            end_time = time.time()
            
            test_result = {
                "name": test_case["name"],
                "input": test_case["input"],
                "success": "error" not in result and result.get("response", {}).get("success", False),
                "response_time_ms": (end_time - start_time) * 1000,
                "inference_time_ms": None,
                "error": result.get("error")
            }
            
            if test_result["success"]:
                test_result["inference_time_ms"] = result["response"].get("inference_time_ms")
                test_result["prediction_type"] = type(result["response"]["prediction"]).__name__
                print(f"    ✅ Success ({test_result['response_time_ms']:.1f}ms)")
            else:
                print(f"    ❌ Failed: {test_result['error']}")
            
            results.append(test_result)
        
        # Summary
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        summary = {
            "total_tests": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) * 100,
            "results": results
        }
        
        if successful:
            response_times = [r["response_time_ms"] for r in successful]
            inference_times = [r["inference_time_ms"] for r in successful if r["inference_time_ms"]]
            
            summary["performance"] = {
                "avg_response_time_ms": sum(response_times) / len(response_times),
                "avg_inference_time_ms": sum(inference_times) / len(inference_times) if inference_times else 0
            }
        
        return summary
    
    def stress_test(self, duration_seconds: int = 60, concurrent_requests: int = 1) -> Dict:
        """Run a stress test on the simple example endpoint"""
        import threading
        import queue
        
        print(f"💥 Starting stress test for {duration_seconds}s with {concurrent_requests} concurrent requests...")
        
        results_queue = queue.Queue()
        stop_event = threading.Event()
        
        def worker():
            local_results = []
            while not stop_event.is_set():
                start_time = time.time()
                result = self.run_simple_example(42)
                end_time = time.time()
                
                local_results.append({
                    "success": "error" not in result and result.get("response", {}).get("success", False),
                    "response_time_ms": (end_time - start_time) * 1000,
                    "inference_time_ms": result.get("response", {}).get("inference_time_ms", 0) if "error" not in result else 0,
                    "error": result.get("error")
                })
                
                if stop_event.is_set():
                    break
            
            results_queue.put(local_results)
        
        # Start worker threads
        threads = []
        for i in range(concurrent_requests):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Run for specified duration
        time.sleep(duration_seconds)
        stop_event.set()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect all results
        all_results = []
        while not results_queue.empty():
            thread_results = results_queue.get()
            all_results.extend(thread_results)
        
        # Analyze results
        successful = [r for r in all_results if r["success"]]
        failed = [r for r in all_results if not r["success"]]
        
        stress_results = {
            "duration_seconds": duration_seconds,
            "concurrent_requests": concurrent_requests,
            "total_requests": len(all_results),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "requests_per_second": len(all_results) / duration_seconds,
            "success_rate": len(successful) / len(all_results) * 100 if all_results else 0
        }
        
        if successful:
            response_times = [r["response_time_ms"] for r in successful]
            inference_times = [r["inference_time_ms"] for r in successful if r["inference_time_ms"]]
            
            stress_results["performance"] = {
                "avg_response_time_ms": sum(response_times) / len(response_times),
                "min_response_time_ms": min(response_times),
                "max_response_time_ms": max(response_times),
                "avg_inference_time_ms": sum(inference_times) / len(inference_times) if inference_times else 0
            }
        
        if failed:
            error_types = {}
            for result in failed:
                error = result.get("error", "Unknown error")
                error_types[error] = error_types.get(error, 0) + 1
            
            stress_results["error_analysis"] = error_types
        
        return stress_results
    
    def print_benchmark_results(self, results: Dict):
        """Print formatted benchmark results"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS")
        print(f"{'='*60}")
        
        if "error" in results:
            print(f"❌ Benchmark failed: {results['error']}")
            return
        
        print(f"Iterations: {results['iterations']}")
        print(f"Errors: {results['errors']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        
        if "response_times_ms" in results:
            rt = results["response_times_ms"]
            print(f"\nResponse Times:")
            print(f"  Min: {rt['min']:.1f}ms")
            print(f"  Max: {rt['max']:.1f}ms")
            print(f"  Avg: {rt['avg']:.1f}ms")
            print(f"  Total: {rt['total']:.1f}ms")
        
        if "inference_times_ms" in results:
            it = results["inference_times_ms"]
            print(f"\nInference Times:")
            print(f"  Min: {it['min']:.1f}ms")
            print(f"  Max: {it['max']:.1f}ms")
            print(f"  Avg: {it['avg']:.1f}ms")
            print(f"  Total: {it['total']:.1f}ms")
        
        print(f"{'='*60}")

# Usage Examples
client = SimpleExampleClient()

# Test connectivity
print("🔗 Testing API connectivity...")
if client.test_connectivity():
    print("✅ API is responsive")
else:
    print("❌ API is not responsive")

# Run basic example
print("\n🚀 Running basic example...")
result = client.run_simple_example(42)
print(f"Result: {json.dumps(result, indent=2)}")

# Benchmark performance
print("\n📊 Running performance benchmark...")
benchmark = client.benchmark_simple_inference(input_value=42, iterations=10)
client.print_benchmark_results(benchmark)

# Test different input types
print("\n🧪 Testing different input types...")
input_test_results = client.test_different_inputs()
print(f"Input type tests: {input_test_results['successful']}/{input_test_results['total_tests']} successful")
print(f"Success rate: {input_test_results['success_rate']:.1f}%")

# Stress test (uncomment to run)
# print("\n💥 Running stress test...")
# stress_results = client.stress_test(duration_seconds=30, concurrent_requests=2)
# print(f"Stress test: {stress_results['successful_requests']}/{stress_results['total_requests']} successful")
# print(f"Requests per second: {stress_results['requests_per_second']:.1f}")
```

### JavaScript Testing Utility

```javascript
class SimpleExampleTester {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }
  
  async runSimpleExample(inputValue = null) {
    const payload = {};
    if (inputValue !== null) {
      payload.input = inputValue;
    }
    
    try {
      const response = await fetch(`${this.baseUrl}/examples/simple`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      return await response.json();
    } catch (error) {
      return { error: error.message };
    }
  }
  
  async testConnectivity() {
    const result = await this.runSimpleExample();
    return !result.error && result.response?.success;
  }
  
  async benchmarkPerformance(iterations = 10) {
    console.log(`🚀 Benchmarking performance (${iterations} iterations)...`);
    
    const results = [];
    const startOverall = performance.now();
    
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      const result = await this.runSimpleExample(42);
      const end = performance.now();
      
      const responseTime = end - start;
      const success = !result.error && result.response?.success;
      
      results.push({
        iteration: i + 1,
        success,
        responseTime,
        inferenceTime: result.response?.inference_time_ms || 0,
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
      results
    };
  }
  
  async testDifferentInputs() {
    const testCases = [
      { name: 'Default (no input)', input: null },
      { name: 'Integer', input: 42 },
      { name: 'Float', input: 3.14159 },
      { name: 'String', input: 'Hello, World!' },
      { name: 'Array', input: [1, 2, 3, 4, 5] },
      { name: 'Object', input: { key: 'value', number: 123 } },
      { name: 'Boolean', input: true },
      { name: 'Empty array', input: [] },
      { name: 'Empty object', input: {} }
    ];
    
    console.log('🧪 Testing different input types...');
    const results = [];
    
    for (const testCase of testCases) {
      console.log(`  Testing: ${testCase.name}...`);
      
      const start = performance.now();
      const result = await this.runSimpleExample(testCase.input);
      const end = performance.now();
      
      const success = !result.error && result.response?.success;
      const responseTime = end - start;
      
      results.push({
        name: testCase.name,
        input: testCase.input,
        success,
        responseTime,
        inferenceTime: result.response?.inference_time_ms || 0,
        error: result.error
      });
      
      console.log(`    ${success ? '✅' : '❌'} ${success ? 'Success' : 'Failed'} (${responseTime.toFixed(1)}ms)`);
    }
    
    const successful = results.filter(r => r.success);
    
    return {
      totalTests: results.length,
      successful: successful.length,
      failed: results.length - successful.length,
      successRate: (successful.length / results.length) * 100,
      avgResponseTime: successful.reduce((sum, r) => sum + r.responseTime, 0) / successful.length,
      results
    };
  }
  
  renderTestInterface(containerId) {
    const container = document.getElementById(containerId);
    
    container.innerHTML = `
      <div class="simple-example-tester">
        <h2>Simple Example Tester</h2>
        
        <div class="test-section">
          <h3>Quick Test</h3>
          <div class="input-group">
            <input type="text" id="test-input" placeholder="Enter test value (leave empty for default)" />
            <button onclick="tester.runQuickTest()" class="btn-primary">Test</button>
          </div>
          <div id="quick-result" class="result-display"></div>
        </div>
        
        <div class="test-section">
          <h3>Performance Tests</h3>
          <div class="button-group">
            <button onclick="tester.runConnectivityTest()" class="btn-secondary">
              🔗 Connectivity Test
            </button>
            <button onclick="tester.runBenchmark()" class="btn-secondary">
              📊 Benchmark (10x)
            </button>
            <button onclick="tester.runInputTests()" class="btn-secondary">
              🧪 Input Type Tests
            </button>
          </div>
        </div>
        
        <div id="test-results" class="results-section"></div>
      </div>
    `;
  }
  
  async runQuickTest() {
    const input = document.getElementById('test-input').value.trim();
    const resultDiv = document.getElementById('quick-result');
    
    let inputValue = null;
    if (input) {
      try {
        // Try to parse as JSON first
        inputValue = JSON.parse(input);
      } catch {
        // If not JSON, treat as string
        inputValue = input;
      }
    }
    
    resultDiv.innerHTML = '<div class="loading">Running test...</div>';
    
    const result = await this.runSimpleExample(inputValue);
    
    if (result.error) {
      resultDiv.innerHTML = `<div class="error">❌ Error: ${result.error}</div>`;
    } else if (result.response?.success) {
      resultDiv.innerHTML = `
        <div class="success">
          ✅ Success! 
          <br>Inference Time: ${result.response.inference_time_ms}ms
          <br>Prediction: <code>${JSON.stringify(result.response.prediction)}</code>
        </div>
      `;
    } else {
      resultDiv.innerHTML = '<div class="error">❌ Test failed</div>';
    }
  }
  
  async runConnectivityTest() {
    const resultsDiv = document.getElementById('test-results');
    resultsDiv.innerHTML = '<div class="loading">Testing connectivity...</div>';
    
    const isConnected = await this.testConnectivity();
    
    resultsDiv.innerHTML = `
      <div class="test-result">
        <h4>Connectivity Test</h4>
        <div class="${isConnected ? 'success' : 'error'}">
          ${isConnected ? '✅ API is responsive and working correctly' : '❌ API is not responding or not working'}
        </div>
      </div>
    `;
  }
  
  async runBenchmark() {
    const resultsDiv = document.getElementById('test-results');
    resultsDiv.innerHTML = '<div class="loading">Running benchmark (10 iterations)...</div>';
    
    const benchmark = await this.benchmarkPerformance(10);
    
    resultsDiv.innerHTML = `
      <div class="test-result">
        <h4>Performance Benchmark</h4>
        <div class="benchmark-stats">
          <div class="stat">Success Rate: ${benchmark.successRate.toFixed(1)}%</div>
          <div class="stat">Successful: ${benchmark.successful}/${benchmark.iterations}</div>
          <div class="stat">Avg Response: ${benchmark.avgResponseTime.toFixed(1)}ms</div>
          <div class="stat">Avg Inference: ${benchmark.avgInferenceTime.toFixed(1)}ms</div>
          <div class="stat">Total Time: ${benchmark.totalTime.toFixed(0)}ms</div>
        </div>
        <div class="benchmark-chart">
          ${benchmark.results.map(r => `
            <div class="benchmark-bar ${r.success ? 'success' : 'error'}" 
                 style="height: ${(r.responseTime / Math.max(...benchmark.results.map(x => x.responseTime))) * 100}%"
                 title="Iteration ${r.iteration}: ${r.responseTime.toFixed(1)}ms">
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }
  
  async runInputTests() {
    const resultsDiv = document.getElementById('test-results');
    resultsDiv.innerHTML = '<div class="loading">Testing different input types...</div>';
    
    const inputTests = await this.testDifferentInputs();
    
    const resultItems = inputTests.results.map(test => `
      <div class="input-test-item ${test.success ? 'success' : 'error'}">
        <div class="test-name">${test.success ? '✅' : '❌'} ${test.name}</div>
        <div class="test-details">
          ${test.success ? `${test.responseTime.toFixed(1)}ms` : `Error: ${test.error}`}
        </div>
      </div>
    `).join('');
    
    resultsDiv.innerHTML = `
      <div class="test-result">
        <h4>Input Type Tests</h4>
        <div class="test-summary">
          Success Rate: ${inputTests.successRate.toFixed(1)}% 
          (${inputTests.successful}/${inputTests.totalTests})
        </div>
        <div class="input-test-results">
          ${resultItems}
        </div>
      </div>
    `;
  }
}

// Global instance
const tester = new SimpleExampleTester();

// Initialize interface
// tester.renderTestInterface('simple-example-container');
```

## Use Cases

### Development Testing
- Quick API connectivity verification
- Basic functionality testing
- Input format validation
- Performance baseline establishment

### Integration Testing
- System health checks
- End-to-end workflow validation
- Load testing preparation
- Error handling verification

### Demonstration
- Simple API usage examples
- Client library testing
- Framework capabilities showcase
- Educational purposes

## Related Endpoints

- [Predict](./predict.md) - Full prediction endpoint
- [Batch Example](./examples-batch.md) - Batch processing example
- [Health](./health.md) - System health check
- [Stats](./stats.md) - System statistics

## Best Practices

1. **Start Simple**: Use this endpoint for initial API testing
2. **Input Validation**: Test various input types to understand accepted formats
3. **Performance Baseline**: Use for establishing performance expectations
4. **Error Handling**: Test error scenarios for robust client development
5. **Documentation**: Use responses to understand API behavior and format
