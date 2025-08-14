# GET / - Root Endpoint

**URL**: `/`  
**Method**: `GET`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Returns basic API information, available endpoints, and current status. This is the primary entry point for discovering API capabilities.

## Request

### URL Parameters
None

### Query Parameters
None

### Request Body
None (GET request)

## Response

### Success Response (200 OK)

```json
{
  "message": "PyTorch Inference Framework API",
  "version": "1.0.0",
  "status": "running",
  "timestamp": "2025-08-14T10:30:00.000000",
  "environment": "development",
  "endpoints": {
    "inference": "/predict",
    "batch_inference": "/predict/batch",
    "health": "/health",
    "stats": "/stats",
    "models": "/models",
    "config": "/config",
    "model_downloads": {
      "download": "/models/download",
      "available": "/models/available",
      "cache_info": "/models/cache/info",
      "remove": "/models/download/{model_name}"
    }
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | API welcome message |
| `version` | string | API version |
| `status` | string | Current API status |
| `timestamp` | string | Current server timestamp (ISO format) |
| `environment` | string | Current environment (development/production) |
| `endpoints` | object | Available API endpoints organized by category |

## Examples

### cURL Request
```bash
curl -X GET http://localhost:8000/
```

### Python Request
```python
import requests

response = requests.get('http://localhost:8000/')
data = response.json()

print(f"API Status: {data['status']}")
print(f"Version: {data['version']}")
print(f"Available endpoints: {list(data['endpoints'].keys())}")
```

### JavaScript Request
```javascript
fetch('http://localhost:8000/')
  .then(response => response.json())
  .then(data => {
    console.log('API Status:', data.status);
    console.log('Version:', data.version);
    console.log('Available endpoints:', Object.keys(data.endpoints));
  });
```

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - API information returned |
| 500 | Internal server error |

## Error Handling

### Server Error (500)
```json
{
  "detail": "Internal server error"
}
```

## Notes

- This endpoint is always available and requires no authentication
- Use this endpoint to verify API availability and discover other endpoints
- The `endpoints` object provides a map of all available API functionality
- Response includes server timestamp for time synchronization purposes
- Environment field helps distinguish between development and production deployments

## Related Endpoints

- [Health Check](./health.md) - Check API health status
- [Configuration](./config.md) - Get detailed configuration information
- [Statistics](./stats.md) - Get performance statistics
