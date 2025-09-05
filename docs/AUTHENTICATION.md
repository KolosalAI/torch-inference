# Authentication System Setup Guide

This guide explains how to use the JWT-based authentication system integrated into the PyTorch Inference Framework.

## Overview

The authentication system provides:
- JWT-based authentication with access and refresh tokens
- API key authentication for programmatic access
- File-based user storage (no database required)
- Role-based access control
- Rate limiting and security features

## Quick Start

### 1. Server Setup

The authentication system is automatically initialized when the server starts. Check the logs for:

```
INITIALIZING AUTHENTICATION SYSTEM
✓ Authentication system initialized successfully
```

### 2. Default Admin User

A default admin user is created automatically:
- **Username:** `admin`
- **Password:** `admin123`
- **Role:** `admin`

⚠️ **IMPORTANT:** Change this password immediately after first login!

### 3. Available Endpoints

#### Authentication Endpoints

- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user  
- `POST /auth/refresh` - Refresh access token
- `POST /auth/logout` - Logout user
- `GET /auth/profile` - Get user profile
- `PUT /auth/password` - Change password

#### API Key Management

- `POST /auth/generate-key` - Generate new API key
- `GET /auth/api-keys` - List user's API keys
- `DELETE /auth/api-keys/{key_id}` - Revoke API key

#### Admin Endpoints (Admin Only)

- `GET /auth/users` - List all users
- `DELETE /auth/users/{username}` - Delete user
- `GET /auth/stats` - Get authentication statistics
- `POST /auth/cleanup` - Cleanup expired tokens/keys

## Usage Examples

### 1. User Registration

```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "myuser",
    "email": "user@example.com", 
    "password": "MySecurePassword123!",
    "full_name": "My User"
  }'
```

### 2. User Login

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "myuser",
    "password": "MySecurePassword123!"
  }'
```

Response:
```json
{
  "success": true,
  "message": "Login successful",
  "token": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 1800
  },
  "user": {
    "id": "user_abc123",
    "username": "myuser",
    "email": "user@example.com",
    "roles": ["user"]
  }
}
```

### 3. Using JWT Token for API Requests

```bash
# Using Authorization header
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [1, 2, 3, 4, 5]
  }'
```

### 4. Generate API Key

```bash
curl -X POST "http://localhost:8000/auth/generate-key" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My API Key",
    "scopes": ["read", "write"],
    "expires_in_days": 30
  }'
```

### 5. Using API Key for Requests

```bash
# Using X-API-Key header
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: sk_YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [1, 2, 3, 4, 5]
  }'

# Or using Authorization header
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer sk_YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [1, 2, 3, 4, 5]
  }'
```

## Protected Endpoints

The following endpoints require authentication when enabled:

### Required Authentication
- `GET /stats` - Engine statistics
- `POST /auth/generate-key` - Generate API key
- `GET /auth/profile` - User profile
- `PUT /auth/password` - Change password

### Optional Authentication
- `POST /predict` - General prediction
- `POST /predict/batch` - Batch prediction  
- `POST /{model_name}/predict` - Model-specific prediction
- `GET /models` - List models

### Public Endpoints (No Authentication)
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `POST /auth/refresh` - Token refresh

## Configuration

Authentication is configured in `config/testing.yaml`:

```yaml
auth:
  jwt_secret_key: "your-super-secret-jwt-key-for-testing-change-in-production"
  jwt_algorithm: "HS256"
  access_token_expire_minutes: 30
  refresh_token_expire_days: 7
  api_key_length: 32
  user_store_file: "./data/users.json"
  session_store_file: "./data/sessions.json"

security:
  enable_api_keys: true
  enable_auth: true
  rate_limit_per_minute: 100
  cors_origins: ["http://localhost:3000"]
  protected_endpoints: ["/inference", "/models", "/predict", "/stats"]
  public_endpoints: ["/health", "/auth", "/"]
```

## Password Requirements

Passwords must meet the following requirements:
- At least 8 characters long
- At least one lowercase letter
- At least one uppercase letter  
- At least one digit
- At least one special character (!@#$%^&*()-_+=)
- No common patterns (123, abc, password, etc.)

## Data Storage

User data is stored in simple JSON files:

- `./data/users.json` - User accounts and profiles
- `./data/sessions.json` - API keys and session data

These files are automatically created and managed by the system.

## Security Features

- **Password Hashing:** Uses bcrypt for secure password storage
- **JWT Tokens:** Stateless authentication with configurable expiry
- **API Keys:** Alternative authentication method for automation
- **Rate Limiting:** Configurable request rate limits per user
- **Account Locking:** Automatic lockout after failed login attempts
- **Token Blacklisting:** Immediate token revocation capability

## Troubleshooting

### Authentication System Not Loading

Check the server logs for authentication initialization messages. Common issues:

1. **Missing dependencies:** Install required packages:
   ```bash
   pip install python-jose[cryptography] passlib[bcrypt] python-multipart
   ```

2. **Configuration file not found:** Ensure `config/testing.yaml` exists

3. **Permissions:** Ensure the application can write to the `./data/` directory

### Can't Access Protected Endpoints

1. **Check token validity:** Tokens expire after 30 minutes by default
2. **Verify token format:** Should be in `Authorization: Bearer TOKEN` header
3. **Check user permissions:** Some endpoints require specific roles

### Default Admin Password

If you've lost the admin password, delete the `./data/users.json` file and restart the server to recreate the default admin user.

## Security Best Practices

1. **Change default passwords** immediately after setup
2. **Use strong JWT secret keys** in production  
3. **Configure CORS origins** appropriately
4. **Enable HTTPS** in production
5. **Regularly rotate API keys**
6. **Monitor authentication logs** for suspicious activity
7. **Keep dependencies updated**

## API Documentation

For complete API documentation with interactive examples, visit:
http://localhost:8000/docs

The authentication endpoints are documented in the "Authentication" section.
