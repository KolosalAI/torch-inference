# POST /predict/batch - Batch Prediction

**URL**: `/predict/batch`  
**Method**: `POST`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Performs batch model inference on multiple input samples simultaneously. This endpoint is optimized for processing multiple inputs efficiently in a single request.

## Request

### URL Parameters
None

### Query Parameters
None

### Request Body

```json
{
  "inputs": ["any", "any", "..."],
  "priority": 0,
  "timeout": 60.0
}
```

#### Request Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `inputs` | array | Yes | - | Array of input data for batch inference |
| `priority` | integer | No | 0 | Request priority (higher = processed first) |
| `timeout` | float | No | null | Request timeout in seconds |

#### Input Format Examples

**Mixed Input Types:**
```json
{
  "inputs": [
    42,
    [1.0, 2.0, 3.0],
    "text_input",
    [4.0, 5.0, 6.0, 7.0, 8.0]
  ]
}
```

**Numeric Arrays:**
```json
{
  "inputs": [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
  ]
}
```

## Response

### Success Response (200 OK)

```json
{
  "success": true,
  "results": [0.7234, -0.4567, 1.2345, 0.8901],
  "error": null,
  "processing_time": 0.0234,
  "batch_size": 4
}
```

### Error Response (200 OK with error)

```json
{
  "success": false,
  "results": [],
  "error": "Batch processing failed: invalid input format",
  "processing_time": null,
  "batch_size": 0
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the batch prediction was successful |
| `results` | array | Array of prediction results (one per input) |
| `error` | string\|null | Error message if batch prediction failed |
| `processing_time` | float\|null | Total processing time in seconds |
| `batch_size` | integer | Number of inputs processed |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success (check `success` field for operation result) |
| 422 | Validation error (invalid request format) |
| 503 | Service unavailable (inference engine not ready) |

## Examples

### Basic Batch Prediction

**Request:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [42, [1.0, 2.0, 3.0], "test", 100]
  }'
```

**Response:**
```json
{
  "success": true,
  "results": [0.7234, -0.4567, 1.2345, 0.8901],
  "error": null,
  "processing_time": 0.0234,
  "batch_size": 4
}
```

### High Priority Batch with Timeout

**Request:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      [1.0, 2.0, 3.0, 4.0, 5.0],
      [6.0, 7.0, 8.0, 9.0, 10.0],
      [11.0, 12.0, 13.0, 14.0, 15.0]
    ],
    "priority": 10,
    "timeout": 30.0
  }'
```

### Python Example

```python
import requests

# Basic batch prediction
inputs = [42, [1.0, 2.0, 3.0], "sample_text", 100]

response = requests.post('http://localhost:8000/predict/batch', 
                        json={'inputs': inputs})
data = response.json()

if data['success']:
    print(f"Batch size: {data['batch_size']}")
    print(f"Processing time: {data['processing_time']:.3f}s")
    print(f"Results: {data['results']}")
    
    # Calculate throughput
    throughput = data['batch_size'] / data['processing_time']
    print(f"Throughput: {throughput:.1f} predictions/second")
else:
    print(f"Error: {data['error']}")

# High priority batch
large_batch = [[i] * 10 for i in range(50)]  # 50 inputs

response = requests.post('http://localhost:8000/predict/batch', 
                        json={
                            'inputs': large_batch,
                            'priority': 5,
                            'timeout': 60.0
                        })
```

### JavaScript Example

```javascript
// Batch prediction with mixed input types
const inputs = [
  42,
  [1.0, 2.0, 3.0, 4.0, 5.0],
  "text_sample",
  [10.0, 20.0, 30.0]
];

fetch('http://localhost:8000/predict/batch', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    inputs: inputs,
    priority: 0,
    timeout: 30.0
  })
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    console.log('Batch size:', data.batch_size);
    console.log('Processing time:', data.processing_time);
    console.log('Results:', data.results);
    
    // Calculate per-input average time
    const avgTime = data.processing_time / data.batch_size;
    console.log('Average time per input:', avgTime);
  } else {
    console.error('Batch error:', data.error);
  }
});
```

## Error Handling

### Service Unavailable (503)
```json
{
  "detail": "Inference engine not available"
}
```

### Validation Error (422)
```json
{
  "detail": [
    {
      "loc": ["body", "inputs"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Processing Error (200 with success: false)
```json
{
  "success": false,
  "results": [],
  "error": "Batch processing failed: timeout exceeded",
  "processing_time": null,
  "batch_size": 0
}
```

## Performance Characteristics

### Batch Size Impact

| Batch Size | Typical Processing Time | Throughput |
|------------|------------------------|------------|
| 1 | 0.015s | 67 pred/s |
| 10 | 0.025s | 400 pred/s |
| 50 | 0.080s | 625 pred/s |
| 100 | 0.150s | 667 pred/s |

### Optimization Tips

1. **Optimal Batch Size**: 20-100 inputs for best throughput
2. **Input Consistency**: Similar input types process faster
3. **Memory Usage**: Larger batches use more memory
4. **Timeout Settings**: Set based on expected batch size
5. **Priority Usage**: Use higher priority for time-sensitive batches

## Batch Processing Behavior

- **Sequential Processing**: Each input is processed individually but efficiently batched
- **Error Handling**: One failed input doesn't stop the entire batch
- **Memory Management**: Automatic memory cleanup after batch completion
- **Device Utilization**: Optimal GPU/CPU resource usage for batch operations

## Related Endpoints

- [Single Prediction](./predict.md) - Process individual inputs
- [Batch Example](./example-batch.md) - Example batch usage
- [Statistics](./stats.md) - Monitor batch performance
- [Health Check](./health.md) - Verify batch processing capability

## Best Practices

1. **Batch Size**: Use 20-100 items for optimal performance
2. **Input Validation**: Ensure all inputs are valid before sending
3. **Timeout Management**: Set timeouts based on batch size and model complexity
4. **Error Recovery**: Handle partial failures gracefully
5. **Rate Limiting**: Don't overwhelm the server with too many concurrent batch requests
