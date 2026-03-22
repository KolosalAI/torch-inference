"""
Inference endpoints for model predictions and processing.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from pydantic import BaseModel, Field

from ...services.inference import InferenceService
from ...models.api.inference import InferenceRequest, InferenceResponse, BatchInferenceRequest, BatchInferenceResponse
from ...models.api.base import SuccessResponse, ErrorResponse
from ...core.exceptions import ValidationError, ModelNotFoundError, InternalServerError
from ..dependencies import get_inference_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["Inference"])


class InferenceStats(BaseModel):
    """Inference statistics."""
    requests_processed: int
    average_latency: Optional[float] = None
    total_processing_time: Optional[float] = None
    successful_predictions: int = 0
    failed_predictions: int = 0
    last_prediction_time: Optional[datetime] = None


class PerformanceReport(BaseModel):
    """Performance report for inference operations."""
    timestamp: datetime
    uptime: Optional[float] = None
    throughput: Optional[float] = None
    latency_stats: Optional[Dict[str, float]] = None
    memory_usage: Optional[Dict[str, Any]] = None
    error_rate: Optional[float] = None


@router.post("/predict",
            response_model=InferenceResponse,
            summary="Single Prediction",
            description="Perform inference on a single input")
async def predict(
    request: InferenceRequest,
    background_tasks: BackgroundTasks,
    inference_service: InferenceService = Depends(get_inference_service)
) -> InferenceResponse:
    """
    Perform inference on a single input.
    
    Args:
        request: Inference request with input data and configuration
        background_tasks: Background tasks for async operations
        inference_service: Inference service dependency
    
    Returns:
        InferenceResponse: Prediction results with metadata
    
    Raises:
        HTTPException: If prediction fails or model not found
    """
    try:
        logger.info(f"[PREDICT] Processing prediction request for model: {request.model_name}")
        logger.debug(f"[PREDICT] Input type: {type(request.inputs).__name__}")
        
        # Perform prediction
        result = await inference_service.predict(
            inputs=request.inputs,
            model_name=request.model_name,
            parameters=request.parameters,
            return_metadata=request.return_metadata
        )
        
        logger.info(f"[PREDICT] Prediction completed successfully for model: {request.model_name}")
        
        # Add background logging task
        background_tasks.add_task(
            _log_prediction_metrics,
            model_name=request.model_name,
            success=True,
            processing_time=result.get("processing_time", 0)
        )
        
        return InferenceResponse(**result)
        
    except ModelNotFoundError as e:
        logger.warning(f"[PREDICT] Model not found: {e}")
        background_tasks.add_task(_log_prediction_metrics, model_name=request.model_name, success=False)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model_name}' not found"
        )
    
    except ValidationError as e:
        logger.warning(f"[PREDICT] Validation error: {e}")
        background_tasks.add_task(_log_prediction_metrics, model_name=request.model_name, success=False)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {e}"
        )
    
    except Exception as e:
        logger.error(f"[PREDICT] Prediction failed: {e}")
        background_tasks.add_task(_log_prediction_metrics, model_name=request.model_name, success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}"
        )


@router.post("/predict/batch",
            response_model=BatchInferenceResponse,
            summary="Batch Predictions",
            description="Perform inference on multiple inputs in a single request")
async def predict_batch(
    request: BatchInferenceRequest,
    background_tasks: BackgroundTasks,
    inference_service: InferenceService = Depends(get_inference_service)
) -> BatchInferenceResponse:
    """
    Perform batch inference on multiple inputs.
    
    Args:
        request: Batch inference request with multiple inputs
        background_tasks: Background tasks for async operations
        inference_service: Inference service dependency
    
    Returns:
        BatchInferenceResponse: Batch prediction results
    
    Raises:
        HTTPException: If batch prediction fails
    """
    try:
        logger.info(f"[BATCH_PREDICT] Processing batch prediction for model: {request.model_name}")
        logger.debug(f"[BATCH_PREDICT] Batch size: {len(request.inputs_list)}")
        
        if not request.inputs_list:
            raise ValidationError("Batch inputs list cannot be empty")
        
        if len(request.inputs_list) > 100:  # Reasonable batch size limit
            raise ValidationError("Batch size too large (max: 100)")
        
        # Perform batch prediction
        results = await inference_service.predict_batch(
            inputs_list=request.inputs_list,
            model_name=request.model_name,
            parameters=request.parameters,
            return_metadata=request.return_metadata
        )
        
        logger.info(f"[BATCH_PREDICT] Batch prediction completed for model: {request.model_name}")
        
        # Add background logging task
        background_tasks.add_task(
            _log_batch_prediction_metrics,
            model_name=request.model_name,
            batch_size=len(request.inputs_list),
            success=True,
            processing_time=results.get("total_processing_time", 0)
        )
        
        return BatchInferenceResponse(**results)
        
    except ValidationError as e:
        logger.warning(f"[BATCH_PREDICT] Validation error: {e}")
        background_tasks.add_task(_log_batch_prediction_metrics, model_name=request.model_name, success=False)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"[BATCH_PREDICT] Batch prediction failed: {e}")
        background_tasks.add_task(_log_batch_prediction_metrics, model_name=request.model_name, success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {e}"
        )


@router.get("/stats",
           response_model=InferenceStats,
           summary="Inference Statistics",
           description="Get inference performance statistics")
async def get_inference_stats(
    inference_service: InferenceService = Depends(get_inference_service)
) -> InferenceStats:
    """
    Get current inference statistics.
    
    Args:
        inference_service: Inference service dependency
    
    Returns:
        InferenceStats: Current inference statistics
    """
    try:
        logger.debug("[STATS] Getting inference statistics...")
        
        stats = await inference_service.get_stats()
        
        return InferenceStats(
            requests_processed=stats.get("requests_processed", 0),
            average_latency=stats.get("average_latency"),
            total_processing_time=stats.get("total_processing_time"),
            successful_predictions=stats.get("successful_predictions", 0),
            failed_predictions=stats.get("failed_predictions", 0),
            last_prediction_time=stats.get("last_prediction_time")
        )
        
    except Exception as e:
        logger.error(f"[STATS] Failed to get inference statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {e}"
        )


@router.get("/performance",
           response_model=PerformanceReport,
           summary="Performance Report",
           description="Get detailed performance report for inference operations")
async def get_performance_report(
    include_memory: bool = Query(False, description="Include memory usage information"),
    inference_service: InferenceService = Depends(get_inference_service)
) -> PerformanceReport:
    """
    Get detailed performance report.
    
    Args:
        include_memory: Whether to include memory usage information
        inference_service: Inference service dependency
    
    Returns:
        PerformanceReport: Detailed performance report
    """
    try:
        logger.debug("[PERFORMANCE] Getting performance report...")
        
        report = await inference_service.get_performance_report(include_memory=include_memory)
        
        return PerformanceReport(
            timestamp=datetime.now(),
            uptime=report.get("uptime"),
            throughput=report.get("throughput"),
            latency_stats=report.get("latency_stats"),
            memory_usage=report.get("memory_usage") if include_memory else None,
            error_rate=report.get("error_rate")
        )
        
    except Exception as e:
        logger.error(f"[PERFORMANCE] Failed to get performance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance report: {e}"
        )


@router.post("/warmup",
            response_model=SuccessResponse,
            summary="Warmup Model",
            description="Warmup a model to improve first prediction performance")
async def warmup_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    iterations: int = Query(3, ge=1, le=10, description="Number of warmup iterations"),
    inference_service: InferenceService = Depends(get_inference_service)
) -> SuccessResponse:
    """
    Warmup a model to improve performance of subsequent predictions.
    
    Args:
        model_name: Name of the model to warmup
        iterations: Number of warmup iterations
        background_tasks: Background tasks for async operations
        inference_service: Inference service dependency
    
    Returns:
        SuccessResponse: Warmup completion status
    """
    try:
        logger.info(f"[WARMUP] Starting warmup for model: {model_name} ({iterations} iterations)")
        
        # Add warmup as background task
        background_tasks.add_task(
            _warmup_model_background,
            inference_service=inference_service,
            model_name=model_name,
            iterations=iterations
        )
        
        return SuccessResponse(
            success=True,
            message=f"Warmup started for model '{model_name}' with {iterations} iterations",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"[WARMUP] Failed to start warmup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start warmup: {e}"
        )


# Background task functions

async def _log_prediction_metrics(model_name: str, success: bool, processing_time: float = 0):
    """Log prediction metrics in background."""
    try:
        status = "success" if success else "failed"
        logger.info(f"[METRICS] Prediction {status} - Model: {model_name}, Time: {processing_time:.4f}s")
    except Exception as e:
        logger.error(f"[METRICS] Failed to log prediction metrics: {e}")


async def _log_batch_prediction_metrics(model_name: str, success: bool, batch_size: int = 0, processing_time: float = 0):
    """Log batch prediction metrics in background."""
    try:
        status = "success" if success else "failed"
        logger.info(f"[METRICS] Batch prediction {status} - Model: {model_name}, Size: {batch_size}, Time: {processing_time:.4f}s")
    except Exception as e:
        logger.error(f"[METRICS] Failed to log batch prediction metrics: {e}")


async def _warmup_model_background(inference_service: InferenceService, model_name: str, iterations: int):
    """Perform model warmup in background."""
    try:
        logger.info(f"[WARMUP] Starting background warmup for model: {model_name}")
        
        await inference_service.warmup_model(model_name, iterations)
        
        logger.info(f"[WARMUP] Completed background warmup for model: {model_name}")
        
    except Exception as e:
        logger.error(f"[WARMUP] Background warmup failed for model {model_name}: {e}")
