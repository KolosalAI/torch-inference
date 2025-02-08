#!/usr/bin/env python3
"""
schemas.py - Production-level Pydantic models for the PyTorch Inference Engine API.

This module defines the request and response schemas used in the API endpoints.
It ensures that all data inputs are validated and outputs are formatted consistently.
Additionally, a test harness is provided at the bottom to allow for quick verification
of the schemas using sample data, logging the results instead of printing.
"""

import logging
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------

class InferenceRequest(BaseModel):
    """
    Schema for an inference request payload.

    Attributes:
        model_name (str): Identifier or name of the model to be used for inference.
        input_data (List[float]): Input tensor data for inference.
        precision (Optional[str]): Precision mode for inference (e.g., "FP32", "FP16", "INT8").
            Defaults to "FP32".
        device (Optional[str]): Target device for inference (e.g., "cpu", "gpu", "auto").
            Defaults to "auto" for automatic device selection.
    """
    model_name: str = Field(..., description="Name or identifier of the model to use")
    input_data: List[float] = Field(..., description="Input tensor data for inference")
    precision: Optional[str] = Field("FP32", description="Precision mode: FP32, FP16, INT8")
    device: Optional[str] = Field("auto", description="Device to use (e.g., CPU, GPU, auto-detect)")


class InferenceResponse(BaseModel):
    """
    Schema for an inference response payload.

    Attributes:
        predictions (List[float]): Output predictions from the inference.
        latency (float): Total time taken for the inference in milliseconds.
        status (str): Status message indicating the result of the inference, defaulting to "success".
    """
    predictions: List[float] = Field(..., description="Output predictions from the model")
    latency: float = Field(..., description="Inference latency in milliseconds")
    status: str = Field("success", description="Status message, typically 'success'")


class HealthStatus(BaseModel):
    """
    Schema representing the health status of the API service.

    Attributes:
        status (str): Overall health status of the API.
        uptime (str): Uptime of the API in a human-readable format.
        models_loaded (bool): Indicator whether the required models are loaded.
        version (str): The current API version.
    """
    status: str = Field(..., description="Overall health status of the API")
    uptime: str = Field(..., description="Uptime of the API")
    models_loaded: bool = Field(..., description="Indicator whether models are loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """
    Schema for an error response payload.

    Attributes:
        detail (str): Detailed error message for debugging or user feedback.
    """
    detail: str = Field(..., description="Detailed error message")


# -----------------------------------------------------------------------------
# Test Harness for Schema Validation (using logging)
# -----------------------------------------------------------------------------

def _test_models() -> None:
    """
    Run basic tests on the defined schemas to verify that they validate and serialize data correctly.
    All outputs are logged using the logging module.
    """
    logger.info("Running basic tests for schemas...")

    # Test InferenceRequest
    try:
        req = InferenceRequest(
            model_name="resnet50",
            input_data=[0.1, 0.2, 0.3],
            precision="FP16",
            device="gpu"
        )
        logger.info("InferenceRequest instance created successfully:")
        logger.info(req.json(indent=4))
    except ValidationError as ve:
        logger.error("Validation error in InferenceRequest: %s", ve)

    logger.info("-" * 40)

    # Test InferenceResponse
    try:
        res = InferenceResponse(
            predictions=[0.2, 0.4, 0.6],
            latency=50.0,
            status="success"
        )
        logger.info("InferenceResponse instance created successfully:")
        logger.info(res.json(indent=4))
    except ValidationError as ve:
        logger.error("Validation error in InferenceResponse: %s", ve)

    logger.info("-" * 40)

    # Test HealthStatus
    try:
        health = HealthStatus(
            status="healthy",
            uptime="3600 seconds",
            models_loaded=True,
            version="1.0.0"
        )
        logger.info("HealthStatus instance created successfully:")
        logger.info(health.json(indent=4))
    except ValidationError as ve:
        logger.error("Validation error in HealthStatus: %s", ve)

    logger.info("-" * 40)

    # Test ErrorResponse
    try:
        err = ErrorResponse(
            detail="An error occurred during processing."
        )
        logger.info("ErrorResponse instance created successfully:")
        logger.info(err.json(indent=4))
    except ValidationError as ve:
        logger.error("Validation error in ErrorResponse: %s", ve)

    logger.info("All tests completed successfully.")


# -----------------------------------------------------------------------------
# Main Execution for Testing Schemas
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    _test_models()
