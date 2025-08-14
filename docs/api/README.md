# 🚀 API Documentation

Welcome to the PyTorch Inference Framework API documentation. This directory contains detailed documentation for all available REST API endpoints.

## 📋 Available Endpoints

### Core Inference
- [**GET /** - Root Endpoint](./root.md) - API information and status
- [**POST /predict** - Single Prediction](./predict.md) - Perform single model inference
- [**POST /predict/batch** - Batch Prediction](./batch-predict.md) - Perform batch model inference

### Monitoring & Health
- [**GET /health** - Health Check](./health.md) - Check API and engine health status
- [**GET /stats** - Statistics](./stats.md) - Get engine performance statistics
- [**GET /config** - Configuration](./config.md) - Get current configuration information

### Model Management
- [**GET /models** - List Models](./models.md) - List registered models
- [**POST /models/download** - Download Model](./model-download.md) - Download models from various sources
- [**GET /models/available** - Available Downloads](./model-available.md) - List downloadable models
- [**GET /models/download/{model_name}/info** - Model Info](./model-info.md) - Get download information
- [**DELETE /models/download/{model_name}** - Remove Model](./model-remove.md) - Remove model from cache
- [**GET /models/cache/info** - Model Cache Info](./models-cache-info.md) - Get model cache information

### Cache Management
- [**GET /cache-info** - Cache Information](./cache-info.md) - Get detailed cache information and statistics
- [**POST /cache-clear** - Clear Cache](./cache-clear.md) - Clear cache data selectively

### Examples & Testing
- [**POST /examples/simple** - Simple Example](./examples-simple.md) - Simple prediction example
- [**POST /examples/batch** - Batch Example](./examples-batch.md) - Batch prediction example

## 🔧 Quick Start

### Base URL
```
http://localhost:8000
```

### Authentication
Currently, the API does not require authentication for development purposes. In production, consider implementing proper authentication mechanisms.

### Content Type
All API endpoints expect and return `application/json` content type unless otherwise specified.

### Response Format
All responses follow a consistent structure:

```json
{
  "success": true,
  "result": "...",
  "error": null,
  "processing_time": 0.123
}
```

## 📊 Interactive Documentation

The API provides automatically generated interactive documentation:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

## 🚀 Getting Started

1. **Start the API Server**:
   ```bash
   uv run python main.py
   ```

2. **Check Health Status**:
   ```bash
   curl http://localhost:8000/health
   ```

3. **Make Your First Prediction**:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"inputs": 42}'
   ```

## 📝 Request/Response Examples

Each endpoint documentation includes:
- **Purpose**: What the endpoint does
- **HTTP Method**: GET, POST, DELETE, etc.
- **URL**: Full endpoint path with parameters
- **Request Format**: Expected request body structure
- **Response Format**: Response body structure
- **Status Codes**: Possible HTTP status codes
- **Examples**: Real request/response examples
- **Error Handling**: Common error scenarios

## 🔍 Error Handling

The API uses standard HTTP status codes:

- **200 OK**: Successful operation
- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Server-side error
- **503 Service Unavailable**: Service temporarily unavailable

Error responses include detailed error messages:

```json
{
  "success": false,
  "result": null,
  "error": "Detailed error message",
  "processing_time": null
}
```

## 📚 Additional Resources

- [Installation Guide](../installation.md)
- [Configuration Guide](../configuration.md)
- [Examples & Tutorials](../examples.md)
- [Testing Guide](../testing.md)

---

For questions or support, please refer to the main project documentation or create an issue in the project repository.
