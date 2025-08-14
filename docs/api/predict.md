# POST /predict - Single Prediction

**URL**: `/predict`  
**Method**: `POST`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Performs single model inference on provided input data. The endpoint accepts various input formats and returns predictions with processing metadata.

## Request

### URL Parameters
None

### Query Parameters
None

### Request Body

```json
{
  "inputs": "any",
  "priority": 0,
  "timeout": 30.0
}
```

#### Request Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `inputs` | any | Yes | - | Input data for inference (flexible format) |
| `priority` | integer | No | 0 | Request priority (higher = processed first) |
| `timeout` | float | No | null | Request timeout in seconds |

#### Input Format Examples

The `inputs` field accepts various data types:

**Numeric Input:**
```json
{"inputs": 42}
```

**Array Input:**
```json
{"inputs": [1.0, 2.0, 3.0, 4.0, 5.0]}
```

**String Input:**
```json
{"inputs": "sample_text"}
```

## Response

### Success Response (200 OK)

```json
{
  "success": true,
  "result": 0.7234,
  "error": null,
  "processing_time": 0.0156,
  "model_info": {
    "model": "example",
    "device": "cpu"
  }
}
```

### Error Response (200 OK with error)

```json
{
  "success": false,
  "result": null,
  "error": "Error message describing what went wrong",
  "processing_time": null,
  "model_info": null
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the prediction was successful |
| `result` | any | Prediction result (format depends on model) |
| `error` | string\|null | Error message if prediction failed |
| `processing_time` | float\|null | Processing time in seconds |
| `model_info` | object\|null | Information about the model used |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success (check `success` field for operation result) |
| 422 | Validation error (invalid request format) |
| 503 | Service unavailable (inference engine not ready) |

## Examples

### Basic Numeric Prediction

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": 42}'
```

**Response:**
```json
{
  "success": true,
  "result": 0.7234,
  "error": null,
  "processing_time": 0.0156,
  "model_info": {
    "model": "example",
    "device": "cpu"
  }
}
```

### Array Input with Priority

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [1.0, 2.0, 3.0, 4.0, 5.0],
    "priority": 10,
    "timeout": 5.0
  }'
```

### Python Example

```python
import requests

# Simple numeric input
response = requests.post('http://localhost:8000/predict', 
                        json={'inputs': 42})
data = response.json()

if data['success']:
    print(f"Prediction: {data['result']}")
    print(f"Processing time: {data['processing_time']:.3f}s")
else:
    print(f"Error: {data['error']}")

# Array input with priority
response = requests.post('http://localhost:8000/predict', 
                        json={
                            'inputs': [1.0, 2.0, 3.0],
                            'priority': 5,
                            'timeout': 10.0
                        })
```

### JavaScript Example

```javascript
// Simple prediction
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    inputs: 42,
    priority: 0
  })
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    console.log('Prediction:', data.result);
    console.log('Processing time:', data.processing_time);
  } else {
    console.error('Error:', data.error);
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
  "result": null,
  "error": "Model inference failed: invalid input format",
  "processing_time": null,
  "model_info": null
}
```

## Performance Considerations

- **Input Size**: Larger inputs take more processing time
- **Priority**: Higher priority requests are processed first
- **Timeout**: Set appropriate timeouts for your use case
- **Device**: GPU processing (when available) is typically faster than CPU

## Model-Specific Behavior

The example model accepts various input formats:
- **Numbers**: Padded to 10 features
- **Arrays**: Truncated or padded to 10 features  
- **Strings**: Converted to numeric representation
- **Other**: Fallback to tensor conversion

## Related Endpoints

- [Batch Prediction](./batch-predict.md) - Process multiple inputs at once
- [Health Check](./health.md) - Verify model readiness
- [Statistics](./stats.md) - Monitor prediction performance
- [Simple Example](./example-simple.md) - Example usage
