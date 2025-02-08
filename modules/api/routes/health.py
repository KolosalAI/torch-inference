#!/usr/bin/env python3
"""
routes/health.py - Health Check Endpoints for the PyTorch Inference Engine API

This module defines endpoints that are used to verify the health and status of the
inference service. The health-check endpoint returns structured data (uptime, model
loading status, and version) which can be used by monitoring systems (such as Prometheus,
Grafana, or Kubernetes probes) to assess the application's overall health.
"""

import time
import logging
from fastapi import APIRouter, Request
from api.schemas import HealthStatus

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    # Configure basic logging if no handlers are present.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# -----------------------------------------------------------------------------
# API Router Setup
# -----------------------------------------------------------------------------
router = APIRouter()


@router.get("/", response_model=HealthStatus, tags=["Health"])
async def health_check(request: Request) -> HealthStatus:
    """
    Health Check Endpoint

    This endpoint returns the current health status of the API service. It computes the uptime,
    retrieves the model loading status, and provides the API version. The application's state
    (start_time and models_loaded flag) is accessed via the Request object's app state, which should
    be initialized during application startup.

    Args:
        request (Request): The incoming HTTP request. It provides access to the application's state.

    Returns:
        HealthStatus: A Pydantic model instance containing the following fields:
            - status (str): Overall health status of the API (e.g., "healthy").
            - uptime (str): Uptime of the API in seconds, represented as a string.
            - models_loaded (bool): Indicator whether the required models have been loaded.
            - version (str): The current API version.
    """
    # Retrieve the application state. The main application should set these attributes during startup.
    app_state = request.app.state
    start_time = getattr(app_state, "start_time", time.time())
    models_loaded = getattr(app_state, "models_loaded", False)
    
    # Calculate uptime in seconds.
    current_time = time.time()
    uptime_seconds = current_time - start_time
    uptime_str = f"{int(uptime_seconds)} seconds"

    # Construct the health status response.
    health_status = HealthStatus(
        status="healthy",
        uptime=uptime_str,
        models_loaded=models_loaded,
        version="1.0.0"
    )

    # Log the health check status for monitoring and debugging purposes.
    logger.info("Health check performed: %s", health_status.json())
    return health_status
