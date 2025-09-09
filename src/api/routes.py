"""
Core API routes for PyTorch Inference Framework.

This module provides the main API endpoints with enhanced error handling,
standardized responses, and comprehensive validation.
"""

import time
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from ..core.config import get_config
from ..core.exceptions import TorchInferenceError, ValidationError, PredictionError
from .schemas import (
    PredictionRequest, PredictionResponse, TTSRequest, TTSResponse,
    STTRequest, STTResponse, HealthResponse, SystemInfoResponse,
    ModelListResponse, ModelInfo, create_success_response, 
    create_error_response, create_health_response, validate_request_model
)
from .dependencies import get_inference_engine, get_model_manager, get_memory_manager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1",
    tags=["Core API"],
    responses={
        404: {"description": "Not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)


@router.post("/predict", response_model=PredictionResponse, summary="Universal Prediction Endpoint")
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    inference_engine = Depends(get_inference_engine),
    model_manager = Depends(get_model_manager),
    memory_manager = Depends(get_memory_manager)
) -> PredictionResponse:
    """
    Universal prediction endpoint for all model types.
    
    This endpoint handles:
    - Single and batch predictions
    - Multiple model types (TTS, STT, classification, etc.)
    - Automatic optimization and caching
    - Comprehensive error handling
    - Performance monitoring
    """
    start_time = time.time()
    
    try:
        logger.info(f"Prediction request received - Model: {request.model_name}, "
                   f"Request ID: {request.request_id}")
        
        # Validate model availability
        if not model_manager.has_model(request.model_name):
            available_models = model_manager.list_models()
            raise PredictionError(
                details=f"Model '{request.model_name}' not available",
                context={
                    "requested_model": request.model_name,
                    "available_models": available_models
                },
                suggestions=[
                    f"Use one of the available models: {', '.join(available_models[:3])}",
                    "Check model name spelling",
                    "Ensure model is loaded and ready"
                ]
            )
        
        # Get model info for optimization
        model_info = model_manager.get_model_info(request.model_name)
        
        # Determine if this is a batch request
        is_batch = isinstance(request.inputs, list) and len(request.inputs) > 1
        
        # Use memory management context
        with memory_manager.inference_context(model_name=request.model_name):
            if is_batch and request.enable_batching:
                # Batch processing
                result = await inference_engine.predict_batch(
                    inputs_list=request.inputs,
                    priority=request.priority,
                    timeout=request.timeout
                )
                batch_info = {
                    "batch_size": len(request.inputs),
                    "processing_mode": "batch"
                }
            else:
                # Single prediction
                result = await inference_engine.predict(
                    inputs=request.inputs,
                    priority=request.priority,
                    timeout=request.timeout,
                    metadata=request.metadata
                )
                batch_info = {
                    "batch_size": 1 if not is_batch else len(request.inputs),
                    "processing_mode": "single" if not is_batch else "individual"
                }
        
        processing_time = time.time() - start_time
        
        # Log successful prediction
        logger.info(f"Prediction completed - Request ID: {request.request_id}, "
                   f"Time: {processing_time:.3f}s")
        
        # Schedule background cleanup if needed
        background_tasks.add_task(_cleanup_after_prediction, request.model_name)
        
        return create_success_response(
            request_id=request.request_id,
            result=result,
            processing_time=processing_time,
            model_info=model_info,
            batch_info=batch_info,
            metadata={
                "optimization_applied": model_info.get("optimization_enabled", False),
                "device": str(inference_engine.device),
                "memory_cached": memory_manager.get_cached_model(request.model_name) is not None
            }
        )
        
    except TorchInferenceError as e:
        processing_time = time.time() - start_time
        logger.error(f"Prediction failed - Request ID: {request.request_id}, "
                    f"Error: {e.message}")
        
        return create_error_response(
            request_id=request.request_id,
            error=e,
            processing_time=processing_time
        )
    
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Unexpected prediction error - Request ID: {request.request_id}, "
                    f"Error: {str(e)}")
        
        # Convert to standardized error
        inference_error = PredictionError(
            details=f"Unexpected error: {str(e)}",
            cause=e,
            context={"request_id": request.request_id}
        )
        
        return create_error_response(
            request_id=request.request_id,
            error=inference_error,
            processing_time=processing_time
        )


@router.post("/tts/synthesize", response_model=TTSResponse, summary="Text-to-Speech Synthesis")
async def synthesize_speech(
    request: TTSRequest,
    background_tasks: BackgroundTasks,
    inference_engine = Depends(get_inference_engine),
    model_manager = Depends(get_model_manager)
) -> TTSResponse:
    """
    Text-to-Speech synthesis endpoint.
    
    Provides high-quality speech synthesis with:
    - Multiple voice options
    - Emotion and style control
    - Multiple output formats
    - Real-time and batch processing
    """
    start_time = time.time()
    
    try:
        logger.info(f"TTS request received - Model: {request.model_name}, "
                   f"Text length: {len(request.text)}, Request ID: {request.request_id}")
        
        # Validate TTS model
        if not model_manager.is_tts_model(request.model_name):
            tts_models = model_manager.get_tts_models()
            raise PredictionError(
                details=f"'{request.model_name}' is not a TTS model",
                context={
                    "requested_model": request.model_name,
                    "available_tts_models": tts_models
                },
                suggestions=[
                    f"Use a TTS model: {', '.join(tts_models[:3])}",
                    "Check the model type in your request"
                ]
            )
        
        # Prepare TTS-specific inputs
        tts_inputs = {
            "text": request.text,
            "voice": request.voice,
            "language": request.language,
            "speed": request.speed,
            "pitch": request.pitch,
            "volume": request.volume,
            "emotion": request.emotion,
            "output_format": request.output_format.value,
            "sample_rate": request.sample_rate,
            "enable_noise_reduction": request.enable_noise_reduction,
            "enable_enhancement": request.enable_enhancement
        }
        
        # Perform synthesis
        result = await inference_engine.predict(
            inputs=tts_inputs,
            priority=request.priority,
            timeout=request.timeout or 60.0,  # Default 60s timeout for TTS
            metadata={**request.metadata, "task_type": "text-to-speech"}
        )
        
        processing_time = time.time() - start_time
        
        # Extract audio information
        audio_data = result.get("audio_data")
        audio_info = result.get("audio_info", {})
        
        logger.info(f"TTS synthesis completed - Request ID: {request.request_id}, "
                   f"Duration: {audio_info.get('duration', 0):.2f}s, "
                   f"Processing time: {processing_time:.3f}s")
        
        return TTSResponse(
            request_id=request.request_id,
            status="success",
            audio_data=audio_data,
            audio_format=request.output_format,
            duration=audio_info.get("duration"),
            sample_rate=audio_info.get("sample_rate", request.sample_rate),
            file_size=audio_info.get("file_size"),
            processing_time=processing_time,
            model_info={
                "model_name": request.model_name,
                "voice_used": request.voice or "default",
                "language": request.language
            },
            synthesis_info={
                "text_length": len(request.text),
                "estimated_duration": len(request.text) * 0.05,  # Rough estimate
                "optimizations_applied": audio_info.get("optimizations", [])
            }
        )
        
    except TorchInferenceError as e:
        processing_time = time.time() - start_time
        logger.error(f"TTS synthesis failed - Request ID: {request.request_id}, "
                    f"Error: {e.message}")
        
        return TTSResponse(
            request_id=request.request_id,
            status="error",
            processing_time=processing_time,
            error=e.to_dict()
        )
    
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"TTS synthesis unexpected error - Request ID: {request.request_id}, "
                    f"Error: {str(e)}")
        
        return TTSResponse(
            request_id=request.request_id,
            status="error",
            processing_time=processing_time,
            error={
                "error_code": "TTS_SYNTHESIS_ERROR",
                "message": str(e),
                "type": type(e).__name__
            }
        )


@router.post("/stt/transcribe", response_model=STTResponse, summary="Speech-to-Text Transcription")
async def transcribe_speech(
    request: STTRequest,
    background_tasks: BackgroundTasks,
    inference_engine = Depends(get_inference_engine),
    model_manager = Depends(get_model_manager)
) -> STTResponse:
    """
    Speech-to-Text transcription endpoint.
    
    Provides accurate speech transcription with:
    - Multiple language support
    - Timestamp and confidence information
    - Speaker diarization
    - Real-time processing
    """
    start_time = time.time()
    
    try:
        logger.info(f"STT request received - Model: {request.model_name}, "
                   f"Language: {request.language}, Request ID: {request.request_id}")
        
        # Validate STT model
        if not model_manager.is_stt_model(request.model_name):
            stt_models = model_manager.get_stt_models()
            raise PredictionError(
                details=f"'{request.model_name}' is not an STT model",
                context={
                    "requested_model": request.model_name,
                    "available_stt_models": stt_models
                },
                suggestions=[
                    f"Use an STT model: {', '.join(stt_models[:3])}",
                    "Check the model type in your request"
                ]
            )
        
        # Note: Audio data would typically be uploaded separately
        # For this example, we'll assume it's passed in metadata
        audio_data = request.metadata.get("audio_data")
        if not audio_data:
            raise ValidationError(
                field="audio_data",
                value=None,
                expected="base64 encoded audio or file reference",
                suggestions=["Upload audio file or provide audio data in metadata"]
            )
        
        # Prepare STT-specific inputs
        stt_inputs = {
            "audio_data": audio_data,
            "language": request.language,
            "enable_timestamps": request.enable_timestamps,
            "enable_word_confidence": request.enable_word_confidence,
            "enable_speaker_diarization": request.enable_speaker_diarization,
            "beam_size": request.beam_size,
            "temperature": request.temperature,
            "suppress_blank": request.suppress_blank,
            "suppress_tokens": request.suppress_tokens,
            "initial_prompt": request.initial_prompt,
            "enable_vad": request.enable_vad,
            "enable_noise_reduction": request.enable_noise_reduction,
            "normalize_audio": request.normalize_audio
        }
        
        # Perform transcription
        result = await inference_engine.predict(
            inputs=stt_inputs,
            priority=request.priority,
            timeout=request.timeout or 120.0,  # Default 120s timeout for STT
            metadata={**request.metadata, "task_type": "speech-to-text"}
        )
        
        processing_time = time.time() - start_time
        
        # Extract transcription information
        text = result.get("text", "")
        segments = result.get("segments", [])
        audio_info = result.get("audio_info", {})
        transcription_info = result.get("transcription_info", {})
        
        logger.info(f"STT transcription completed - Request ID: {request.request_id}, "
                   f"Text length: {len(text)}, Processing time: {processing_time:.3f}s")
        
        return STTResponse(
            request_id=request.request_id,
            status="success",
            text=text,
            segments=segments,
            language=transcription_info.get("detected_language", request.language),
            language_confidence=transcription_info.get("language_confidence"),
            duration=audio_info.get("duration"),
            sample_rate=audio_info.get("sample_rate"),
            overall_confidence=transcription_info.get("overall_confidence"),
            word_count=len(text.split()) if text else 0,
            processing_time=processing_time,
            model_info={
                "model_name": request.model_name,
                "language_used": transcription_info.get("detected_language", request.language)
            },
            processing_info={
                "audio_duration": audio_info.get("duration", 0),
                "processing_speed": (
                    audio_info.get("duration", 0) / processing_time
                    if processing_time > 0 else 0
                ),
                "optimizations_applied": transcription_info.get("optimizations", [])
            }
        )
        
    except TorchInferenceError as e:
        processing_time = time.time() - start_time
        logger.error(f"STT transcription failed - Request ID: {request.request_id}, "
                    f"Error: {e.message}")
        
        return STTResponse(
            request_id=request.request_id,
            status="error",
            processing_time=processing_time,
            error=e.to_dict()
        )
    
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"STT transcription unexpected error - Request ID: {request.request_id}, "
                    f"Error: {str(e)}")
        
        return STTResponse(
            request_id=request.request_id,
            status="error",
            processing_time=processing_time,
            error={
                "error_code": "STT_TRANSCRIPTION_ERROR",
                "message": str(e),
                "type": type(e).__name__
            }
        )


@router.get("/health", response_model=HealthResponse, summary="Comprehensive Health Check")
async def health_check(
    inference_engine = Depends(get_inference_engine),
    model_manager = Depends(get_model_manager),
    memory_manager = Depends(get_memory_manager)
) -> HealthResponse:
    """
    Comprehensive health check endpoint.
    
    Checks:
    - Inference engine status
    - Model availability
    - Memory usage
    - GPU/device status
    - System resources
    """
    try:
        logger.debug("Health check requested")
        
        checks = {}
        
        # Inference engine health
        if inference_engine:
            engine_health = await inference_engine.health_check()
            checks["inference_engine"] = engine_health["checks"]
            checks["inference_engine"]["overall"] = engine_health["healthy"]
        else:
            checks["inference_engine"] = {
                "overall": False,
                "error": "Inference engine not available"
            }
        
        # Model manager health
        if model_manager:
            checks["models"] = {
                "overall": True,
                "total_models": len(model_manager.list_models()),
                "loaded_models": len(model_manager.get_loaded_models()),
                "tts_models": len(model_manager.get_tts_models()),
                "stt_models": len(model_manager.get_stt_models())
            }
        else:
            checks["models"] = {
                "overall": False,
                "error": "Model manager not available"
            }
        
        # Memory health
        if memory_manager:
            memory_stats = memory_manager.get_memory_stats()
            checks["memory"] = {
                "overall": memory_stats["cache"]["cached_models"] < memory_manager.max_cached_models,
                "cached_models": memory_stats["cache"]["cached_models"],
                "cache_hit_rate": memory_stats["cache"]["hit_rate"],
                "memory_usage": memory_stats.get("system", {}),
                "gpu_memory": memory_stats.get("gpu", {})
            }
        else:
            checks["memory"] = {
                "overall": False,
                "error": "Memory manager not available"
            }
        
        # System health
        checks["system"] = {
            "overall": True,
            "timestamp": time.time(),
            "config_loaded": True
        }
        
        # Calculate overall health
        overall_healthy = all(
            check.get("overall", False) for check in checks.values()
        )
        
        # Add uptime if available
        config = get_config()
        uptime = getattr(config, '_start_time', None)
        if uptime:
            uptime = time.time() - uptime
        
        logger.debug(f"Health check completed - Overall healthy: {overall_healthy}")
        
        return create_health_response(
            checks=checks,
            healthy=overall_healthy,
            uptime=uptime,
            version="1.0.0",
            environment=config.environment
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        
        return create_health_response(
            checks={"error": {"overall": False, "message": str(e)}},
            healthy=False
        )


@router.get("/models", response_model=ModelListResponse, summary="List Available Models")
async def list_models(
    category: Optional[str] = None,
    model_type: Optional[str] = None,
    loaded_only: bool = False,
    model_manager = Depends(get_model_manager)
) -> ModelListResponse:
    """
    List all available models with filtering options.
    
    Query parameters:
    - category: Filter by model category
    - model_type: Filter by model type (tts, stt, etc.)
    - loaded_only: Show only currently loaded models
    """
    try:
        logger.debug(f"Models list requested - Category: {category}, "
                    f"Type: {model_type}, Loaded only: {loaded_only}")
        
        # Get all models
        all_models = model_manager.list_models()
        filtered_models = []
        
        for model_name in all_models:
            model_info = model_manager.get_model_info(model_name)
            
            # Apply filters
            if category and model_info.get("category") != category:
                continue
            if model_type and model_info.get("model_type") != model_type:
                continue
            if loaded_only and not model_manager.is_model_loaded(model_name):
                continue
            
            # Create ModelInfo object
            model_detail = ModelInfo(
                name=model_name,
                display_name=model_info.get("display_name", model_name),
                description=model_info.get("description"),
                model_type=model_info.get("model_type", "custom"),
                version=model_info.get("version"),
                supported_languages=model_info.get("supported_languages", []),
                supported_formats=model_info.get("supported_formats", []),
                max_input_length=model_info.get("max_input_length"),
                status=model_info.get("status", "available"),
                loaded=model_manager.is_model_loaded(model_name),
                load_time=model_info.get("load_time"),
                memory_usage_mb=model_info.get("memory_usage_mb"),
                total_requests=model_info.get("total_requests", 0),
                last_used=model_info.get("last_used"),
                average_processing_time=model_info.get("average_processing_time"),
                optimization_enabled=model_info.get("optimization_enabled", False),
                optimization_info=model_info.get("optimization_info", {})
            )
            
            filtered_models.append(model_detail)
        
        # Group by categories
        categories = {}
        for model in filtered_models:
            model_type_key = model.model_type.value if hasattr(model.model_type, 'value') else str(model.model_type)
            if model_type_key not in categories:
                categories[model_type_key] = []
            categories[model_type_key].append(model.name)
        
        # Calculate statistics
        statistics = {
            "total_models": len(filtered_models),
            "loaded_models": sum(1 for m in filtered_models if m.loaded),
            "categories": len(categories),
            "tts_models": len([m for m in filtered_models if m.model_type == "text-to-speech"]),
            "stt_models": len([m for m in filtered_models if m.model_type == "speech-to-text"]),
            "total_requests": sum(m.total_requests for m in filtered_models),
            "average_load_time": (
                sum(m.load_time for m in filtered_models if m.load_time) / 
                len([m for m in filtered_models if m.load_time])
            ) if any(m.load_time for m in filtered_models) else None
        }
        
        logger.debug(f"Models list completed - Found {len(filtered_models)} models")
        
        return ModelListResponse(
            request_id=f"models_list_{int(time.time())}",
            status="success",
            models=filtered_models,
            total_models=len(filtered_models),
            categories=categories,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"Models list failed: {str(e)}")
        
        return ModelListResponse(
            request_id=f"models_list_error_{int(time.time())}",
            status="error",
            models=[],
            total_models=0,
            error={
                "error_code": "MODEL_LIST_ERROR",
                "message": str(e),
                "type": type(e).__name__
            }
        )


@router.get("/system/info", response_model=SystemInfoResponse, summary="System Information")
async def get_system_info(
    include_performance: bool = True,
    include_config: bool = False,
    inference_engine = Depends(get_inference_engine),
    memory_manager = Depends(get_memory_manager)
) -> SystemInfoResponse:
    """
    Get comprehensive system information.
    
    Query parameters:
    - include_performance: Include performance metrics
    - include_config: Include configuration details
    """
    try:
        logger.debug("System info requested")
        
        import psutil
        import platform
        import torch
        
        # System information
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "uptime": time.time() - psutil.boot_time()
        }
        
        # Hardware information
        hardware_info = {
            "cpu": {
                "model": platform.processor(),
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "used_percent": psutil.virtual_memory().percent
            }
        }
        
        # GPU information
        if torch.cuda.is_available():
            gpu_info = {}
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info[f"gpu_{i}"] = {
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multi_processor_count
                }
            hardware_info["gpu"] = gpu_info
        
        # Software information
        software_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        }
        
        # Performance metrics
        performance_info = {}
        if include_performance:
            if inference_engine:
                performance_info["inference_engine"] = inference_engine.get_performance_report()
            
            if memory_manager:
                performance_info["memory"] = memory_manager.get_memory_stats()
        
        # Configuration
        config_info = {}
        if include_config:
            config = get_config()
            config_info = {
                "environment": config.environment,
                "debug": config.debug,
                "server": {
                    "host": config.server.host,
                    "port": config.server.port,
                    "log_level": config.server.log_level.value
                },
                "inference": {
                    "device_type": config.inference.device.device_type.value,
                    "batch_size": config.inference.batch.batch_size,
                    "optimization_enabled": config.inference.optimization.enabled
                }
            }
        
        logger.debug("System info completed")
        
        return SystemInfoResponse(
            request_id=f"system_info_{int(time.time())}",
            status="success",
            system=system_info,
            hardware=hardware_info,
            software=software_info,
            performance=performance_info,
            configuration=config_info
        )
        
    except Exception as e:
        logger.error(f"System info failed: {str(e)}")
        
        return SystemInfoResponse(
            request_id=f"system_info_error_{int(time.time())}",
            status="error",
            system={},
            hardware={},
            software={},
            performance={},
            configuration={},
            error={
                "error_code": "SYSTEM_INFO_ERROR",
                "message": str(e),
                "type": type(e).__name__
            }
        )


# Helper functions

async def _cleanup_after_prediction(model_name: str) -> None:
    """Background cleanup task after prediction."""
    try:
        # Light cleanup - can be extended as needed
        memory_manager = get_memory_manager()
        if memory_manager:
            # Update model access statistics
            cached_model = memory_manager.get_cached_model(model_name)
            if cached_model:
                logger.debug(f"Model '{model_name}' accessed from cache")
    except Exception as e:
        logger.debug(f"Background cleanup failed: {e}")


# Error handlers for this router

@router.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc.message}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": exc.to_dict(),
            "timestamp": time.time()
        }
    )


@router.exception_handler(TorchInferenceError)
async def torch_inference_error_handler(request: Request, exc: TorchInferenceError):
    """Handle torch inference errors."""
    logger.error(f"Torch inference error: {exc.message}")
    
    status_code = 500
    if "not found" in exc.message.lower():
        status_code = 404
    elif "validation" in exc.message.lower():
        status_code = 422
    elif "timeout" in exc.message.lower():
        status_code = 408
    elif "resource" in exc.message.lower():
        status_code = 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.to_dict(),
            "timestamp": time.time()
        }
    )
