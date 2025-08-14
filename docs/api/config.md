# GET /config - Configuration

**URL**: `/config`  
**Method**: `GET`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Returns current configuration information for the API server and inference engine. This endpoint provides detailed settings for debugging, monitoring, and system introspection.

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
  "configuration": {
    "environment": "development",
    "config_sources": ["config.yaml", "environment_variables", "defaults"],
    "last_loaded": "2025-08-14T10:30:00.000Z"
  },
  "inference_config": {
    "device_type": "cpu",
    "batch_size": 4,
    "use_fp16": false,
    "enable_profiling": false
  },
  "server_config": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": true,
    "log_level": "INFO"
  }
}
```

#### Response Fields

##### Configuration Object
| Field | Type | Description |
|-------|------|-------------|
| `environment` | string | Current environment (development/production) |
| `config_sources` | array | Sources used to load configuration |
| `last_loaded` | string | When configuration was last loaded (ISO format) |

##### Inference Config Object
| Field | Type | Description |
|-------|------|-------------|
| `device_type` | string | Inference device (cpu/cuda/auto) |
| `batch_size` | integer | Default batch size for processing |
| `use_fp16` | boolean | Whether to use 16-bit floating point precision |
| `enable_profiling` | boolean | Whether performance profiling is enabled |

##### Server Config Object
| Field | Type | Description |
|-------|------|-------------|
| `host` | string | Server bind address |
| `port` | integer | Server port number |
| `reload` | boolean | Whether auto-reload is enabled |
| `log_level` | string | Logging level (DEBUG/INFO/WARNING/ERROR) |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Configuration returned |
| 500 | Internal server error |

## Examples

### Basic Configuration Request

**Request:**
```bash
curl -X GET http://localhost:8000/config
```

**Response:**
```json
{
  "configuration": {
    "environment": "development",
    "config_sources": ["config.yaml", "environment_variables", "defaults"],
    "last_loaded": "2025-08-14T10:30:00.000Z"
  },
  "inference_config": {
    "device_type": "cpu",
    "batch_size": 4,
    "use_fp16": false,
    "enable_profiling": false
  },
  "server_config": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": true,
    "log_level": "INFO"
  }
}
```

### Python Configuration Inspector

```python
import requests
import json
from datetime import datetime

class ConfigurationInspector:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_config(self):
        try:
            response = requests.get(f"{self.base_url}/config")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching configuration: {e}")
            return None
    
    def print_config_summary(self, config_data):
        print("="*60)
        print("PYTORCH INFERENCE API - CONFIGURATION SUMMARY")
        print("="*60)
        
        # General Configuration
        general = config_data['configuration']
        print(f"Environment: {general['environment'].upper()}")
        print(f"Config Sources: {', '.join(general['config_sources'])}")
        print(f"Last Loaded: {general['last_loaded']}")
        
        # Inference Configuration
        inference = config_data['inference_config']
        print(f"\nINFERENCE SETTINGS:")
        print(f"  Device Type: {inference['device_type'].upper()}")
        print(f"  Batch Size: {inference['batch_size']}")
        print(f"  FP16 Precision: {'✅ Enabled' if inference['use_fp16'] else '❌ Disabled'}")
        print(f"  Profiling: {'✅ Enabled' if inference['enable_profiling'] else '❌ Disabled'}")
        
        # Server Configuration
        server = config_data['server_config']
        print(f"\nSERVER SETTINGS:")
        print(f"  Bind Address: {server['host']}:{server['port']}")
        print(f"  Auto-Reload: {'✅ Enabled' if server['reload'] else '❌ Disabled'}")
        print(f"  Log Level: {server['log_level']}")
        
        print("="*60)
    
    def validate_production_config(self, config_data):
        """Check if configuration is suitable for production"""
        warnings = []
        recommendations = []
        
        # Environment check
        if config_data['configuration']['environment'] != 'production':
            warnings.append("⚠️ Not running in production environment")
        
        # Security checks
        server = config_data['server_config']
        if server['host'] == '0.0.0.0':
            recommendations.append("🔒 Consider binding to specific interface in production")
        
        if server['reload']:
            warnings.append("⚠️ Auto-reload should be disabled in production")
        
        # Performance checks
        inference = config_data['inference_config']
        if not inference['use_fp16'] and 'cuda' in inference['device_type'].lower():
            recommendations.append("⚡ Consider enabling FP16 for GPU inference")
        
        if inference['batch_size'] < 4:
            recommendations.append("📊 Consider larger batch size for better throughput")
        
        # Logging checks
        if server['log_level'] == 'DEBUG':
            warnings.append("⚠️ Debug logging may impact performance in production")
        
        return warnings, recommendations
    
    def export_config(self, config_data, filename="api_config.json"):
        """Export configuration to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"✅ Configuration exported to {filename}")
        except Exception as e:
            print(f"❌ Export failed: {e}")

# Usage
inspector = ConfigurationInspector()
config = inspector.get_config()

if config:
    inspector.print_config_summary(config)
    
    # Production validation
    warnings, recommendations = inspector.validate_production_config(config)
    
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
    
    if recommendations:
        print("\nRECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  {rec}")
    
    # Export configuration
    inspector.export_config(config)
```

### JavaScript Configuration Display

```javascript
class ConfigurationViewer {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }
  
  async fetchConfig() {
    try {
      const response = await fetch(`${this.baseUrl}/config`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching configuration:', error);
      return null;
    }
  }
  
  displayConfig(configData) {
    const container = document.getElementById('config-container');
    
    const html = `
      <div class="config-section">
        <h3>General Configuration</h3>
        <div class="config-item">
          <label>Environment:</label>
          <span class="config-value environment-${configData.configuration.environment}">
            ${configData.configuration.environment.toUpperCase()}
          </span>
        </div>
        <div class="config-item">
          <label>Config Sources:</label>
          <span class="config-value">${configData.configuration.config_sources.join(', ')}</span>
        </div>
      </div>
      
      <div class="config-section">
        <h3>Inference Configuration</h3>
        <div class="config-item">
          <label>Device Type:</label>
          <span class="config-value">${configData.inference_config.device_type.toUpperCase()}</span>
        </div>
        <div class="config-item">
          <label>Batch Size:</label>
          <span class="config-value">${configData.inference_config.batch_size}</span>
        </div>
        <div class="config-item">
          <label>FP16 Precision:</label>
          <span class="config-value ${configData.inference_config.use_fp16 ? 'enabled' : 'disabled'}">
            ${configData.inference_config.use_fp16 ? '✅ Enabled' : '❌ Disabled'}
          </span>
        </div>
        <div class="config-item">
          <label>Profiling:</label>
          <span class="config-value ${configData.inference_config.enable_profiling ? 'enabled' : 'disabled'}">
            ${configData.inference_config.enable_profiling ? '✅ Enabled' : '❌ Disabled'}
          </span>
        </div>
      </div>
      
      <div class="config-section">
        <h3>Server Configuration</h3>
        <div class="config-item">
          <label>Address:</label>
          <span class="config-value">${configData.server_config.host}:${configData.server_config.port}</span>
        </div>
        <div class="config-item">
          <label>Auto-Reload:</label>
          <span class="config-value ${configData.server_config.reload ? 'enabled' : 'disabled'}">
            ${configData.server_config.reload ? '✅ Enabled' : '❌ Disabled'}
          </span>
        </div>
        <div class="config-item">
          <label>Log Level:</label>
          <span class="config-value log-level-${configData.server_config.log_level.toLowerCase()}">
            ${configData.server_config.log_level}
          </span>
        </div>
      </div>
    `;
    
    container.innerHTML = html;
  }
  
  async loadAndDisplay() {
    const config = await this.fetchConfig();
    if (config) {
      this.displayConfig(config);
      this.performConfigChecks(config);
    }
  }
  
  performConfigChecks(configData) {
    const checksContainer = document.getElementById('config-checks');
    const checks = [];
    
    // Environment check
    if (configData.configuration.environment === 'production') {
      checks.push({ type: 'success', message: '✅ Running in production environment' });
    } else {
      checks.push({ type: 'warning', message: '⚠️ Not running in production environment' });
    }
    
    // Performance checks
    if (configData.inference_config.use_fp16) {
      checks.push({ type: 'success', message: '⚡ FP16 precision enabled for performance' });
    }
    
    if (configData.inference_config.batch_size >= 4) {
      checks.push({ type: 'success', message: '📊 Good batch size for throughput' });
    } else {
      checks.push({ type: 'info', message: '💡 Consider larger batch size for better throughput' });
    }
    
    // Security checks
    if (configData.server_config.host === '0.0.0.0' && 
        configData.configuration.environment === 'production') {
      checks.push({ type: 'warning', message: '🔒 Consider binding to specific interface in production' });
    }
    
    const checksHtml = checks.map(check => 
      `<div class="config-check ${check.type}">${check.message}</div>`
    ).join('');
    
    checksContainer.innerHTML = `<h3>Configuration Analysis</h3>${checksHtml}`;
  }
}

// Usage
const viewer = new ConfigurationViewer();
viewer.loadAndDisplay();
```

### Environment-Specific Configurations

#### Development Environment
```json
{
  "configuration": {
    "environment": "development"
  },
  "inference_config": {
    "device_type": "cpu",
    "batch_size": 2,
    "use_fp16": false,
    "enable_profiling": true
  },
  "server_config": {
    "host": "127.0.0.1",
    "port": 8000,
    "reload": true,
    "log_level": "DEBUG"
  }
}
```

#### Production Environment
```json
{
  "configuration": {
    "environment": "production"
  },
  "inference_config": {
    "device_type": "cuda",
    "batch_size": 16,
    "use_fp16": true,
    "enable_profiling": false
  },
  "server_config": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": false,
    "log_level": "INFO"
  }
}
```

## Configuration Sources

The configuration is loaded from multiple sources in order of priority:

1. **Environment Variables** (highest priority)
2. **Configuration File** (`config.yaml`)
3. **Default Values** (lowest priority)

### Environment Variable Mapping

| Environment Variable | Configuration Path | Default |
|---------------------|-------------------|---------|
| `ENVIRONMENT` | `environment` | `development` |
| `DEVICE` | `inference_config.device_type` | `auto` |
| `BATCH_SIZE` | `inference_config.batch_size` | `4` |
| `USE_FP16` | `inference_config.use_fp16` | `false` |
| `ENABLE_PROFILING` | `inference_config.enable_profiling` | `false` |
| `HOST` | `server_config.host` | `0.0.0.0` |
| `PORT` | `server_config.port` | `8000` |
| `LOG_LEVEL` | `server_config.log_level` | `INFO` |

## Error Handling

### Internal Server Error (500)
```json
{
  "detail": "Failed to retrieve configuration"
}
```

## Configuration Validation

The API validates configuration values at startup and provides warnings for:

- **Invalid device types**: Non-existent CUDA devices
- **Memory constraints**: Batch sizes too large for available memory
- **Port conflicts**: Attempting to bind to occupied ports
- **Permission issues**: Insufficient rights for specified bind address

## Related Endpoints

- [Health Check](./health.md) - Current system health
- [Statistics](./stats.md) - Performance metrics affected by configuration
- [Root](./root.md) - Basic API information

## Configuration Best Practices

1. **Environment Separation**: Use different configs for dev/prod
2. **Security**: Don't expose sensitive configuration in public environments
3. **Performance**: Tune batch_size and fp16 settings based on workload
4. **Monitoring**: Enable profiling only when needed (performance impact)
5. **Logging**: Use appropriate log levels for environment
