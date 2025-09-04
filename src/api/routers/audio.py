"""
Audio processing endpoints for TTS, STT, and audio manipulation.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import io
import tempfile
import os

from ...services.audio import AudioService
from ...models.api.audio import TTSRequest, TTSResponse, STTRequest, STTResponse, AudioProcessRequest
from ...models.api.base import SuccessResponse, ErrorResponse
from ...core.exceptions import ValidationError, ProcessingError, InternalServerError
from ..dependencies import get_audio_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["Audio Processing"])


class AudioInfo(BaseModel):
    """Audio file information."""
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None
    size: Optional[int] = None
    bitrate: Optional[int] = None


class AudioProcessingStatus(BaseModel):
    """Audio processing status."""
    task_id: str
    status: str  # processing, completed, failed
    progress: Optional[float] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


@router.post("/tts",
            response_model=TTSResponse,
            summary="Text-to-Speech",
            description="Convert text to speech audio")
async def text_to_speech(
    request: TTSRequest,
    background_tasks: BackgroundTasks,
    audio_service: AudioService = Depends(get_audio_service)
) -> TTSResponse:
    """
    Convert text to speech.
    
    Args:
        request: TTS request with text and voice configuration
        background_tasks: Background tasks for async operations
        audio_service: Audio service dependency
    
    Returns:
        TTSResponse: Generated audio data and metadata
    
    Raises:
        HTTPException: If TTS processing fails
    """
    try:
        logger.info(f"[TTS] Processing TTS request - Voice: {request.voice_id}, Length: {len(request.text)} chars")
        
        if len(request.text) > 5000:  # Reasonable text length limit
            raise ValidationError("Text too long (max: 5000 characters)")
        
        # Process TTS
        result = await audio_service.text_to_speech(
            text=request.text,
            voice_id=request.voice_id,
            speed=request.speed,
            pitch=request.pitch,
            volume=request.volume,
            output_format=request.output_format
        )
        
        logger.info(f"[TTS] TTS processing completed successfully")
        
        # Add background logging
        background_tasks.add_task(
            _log_tts_metrics,
            text_length=len(request.text),
            voice_id=request.voice_id,
            success=True,
            processing_time=result.get("processing_time", 0)
        )
        
        return TTSResponse(**result)
        
    except ValidationError as e:
        logger.warning(f"[TTS] Validation error: {e}")
        background_tasks.add_task(_log_tts_metrics, text_length=len(request.text), success=False)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except ProcessingError as e:
        logger.error(f"[TTS] Processing error: {e}")
        background_tasks.add_task(_log_tts_metrics, text_length=len(request.text), success=False)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"TTS processing failed: {e}"
        )
    
    except Exception as e:
        logger.error(f"[TTS] Unexpected error: {e}")
        background_tasks.add_task(_log_tts_metrics, text_length=len(request.text), success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS processing failed: {e}"
        )


@router.post("/stt",
            response_model=STTResponse,
            summary="Speech-to-Text",
            description="Convert speech audio to text")
async def speech_to_text(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Language code for transcription"),
    model_name: Optional[str] = Form("whisper-base", description="STT model to use"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    audio_service: AudioService = Depends(get_audio_service)
) -> STTResponse:
    """
    Convert speech to text.
    
    Args:
        audio_file: Audio file to transcribe
        language: Target language for transcription
        model_name: STT model to use
        background_tasks: Background tasks for async operations
        audio_service: Audio service dependency
    
    Returns:
        STTResponse: Transcribed text and metadata
    
    Raises:
        HTTPException: If STT processing fails
    """
    try:
        logger.info(f"[STT] Processing STT request - File: {audio_file.filename}, Model: {model_name}")
        
        # Validate file size (50MB limit)
        if audio_file.size and audio_file.size > 50 * 1024 * 1024:
            raise ValidationError("Audio file too large (max: 50MB)")
        
        # Read audio data
        audio_data = await audio_file.read()
        
        if not audio_data:
            raise ValidationError("Empty audio file")
        
        # Process STT
        result = await audio_service.speech_to_text(
            audio_data=audio_data,
            filename=audio_file.filename,
            language=language,
            model_name=model_name
        )
        
        logger.info(f"[STT] STT processing completed successfully")
        
        # Add background logging
        background_tasks.add_task(
            _log_stt_metrics,
            file_size=len(audio_data),
            filename=audio_file.filename,
            model_name=model_name,
            success=True,
            processing_time=result.get("processing_time", 0)
        )
        
        return STTResponse(**result)
        
    except ValidationError as e:
        logger.warning(f"[STT] Validation error: {e}")
        background_tasks.add_task(_log_stt_metrics, filename=audio_file.filename, success=False)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except ProcessingError as e:
        logger.error(f"[STT] Processing error: {e}")
        background_tasks.add_task(_log_stt_metrics, filename=audio_file.filename, success=False)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"STT processing failed: {e}"
        )
    
    except Exception as e:
        logger.error(f"[STT] Unexpected error: {e}")
        background_tasks.add_task(_log_stt_metrics, filename=audio_file.filename, success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"STT processing failed: {e}"
        )


@router.post("/process",
            response_model=SuccessResponse,
            summary="Process Audio",
            description="Apply various audio processing operations")
async def process_audio(
    audio_file: UploadFile = File(..., description="Audio file to process"),
    operations: str = Form(..., description="JSON string of processing operations"),
    output_format: str = Form("wav", description="Output audio format"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    audio_service: AudioService = Depends(get_audio_service)
) -> SuccessResponse:
    """
    Apply audio processing operations.
    
    Args:
        audio_file: Audio file to process
        operations: JSON string describing processing operations
        output_format: Output audio format
        background_tasks: Background tasks for async operations
        audio_service: Audio service dependency
    
    Returns:
        SuccessResponse: Processing status with task ID
    """
    try:
        logger.info(f"[AUDIO_PROCESS] Processing audio file: {audio_file.filename}")
        
        # Validate file size
        if audio_file.size and audio_file.size > 100 * 1024 * 1024:
            raise ValidationError("Audio file too large (max: 100MB)")
        
        # Parse operations
        import json
        try:
            operations_dict = json.loads(operations)
        except json.JSONDecodeError:
            raise ValidationError("Invalid operations JSON format")
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # Start processing in background
        task_id = f"audio_process_{int(datetime.now().timestamp())}"
        
        background_tasks.add_task(
            _process_audio_background,
            audio_service=audio_service,
            task_id=task_id,
            audio_data=audio_data,
            filename=audio_file.filename,
            operations=operations_dict,
            output_format=output_format
        )
        
        return SuccessResponse(
            success=True,
            message=f"Audio processing started with task ID: {task_id}",
            data={"task_id": task_id},
            timestamp=datetime.now()
        )
        
    except ValidationError as e:
        logger.warning(f"[AUDIO_PROCESS] Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"[AUDIO_PROCESS] Failed to start processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start audio processing: {e}"
        )


@router.get("/info",
           response_model=AudioInfo,
           summary="Audio File Info",
           description="Get information about an audio file")
async def get_audio_info(
    audio_file: UploadFile = File(..., description="Audio file to analyze"),
    audio_service: AudioService = Depends(get_audio_service)
) -> AudioInfo:
    """
    Get audio file information.
    
    Args:
        audio_file: Audio file to analyze
        audio_service: Audio service dependency
    
    Returns:
        AudioInfo: Audio file metadata
    """
    try:
        logger.debug(f"[AUDIO_INFO] Analyzing audio file: {audio_file.filename}")
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # Get audio info
        info = await audio_service.get_audio_info(audio_data, audio_file.filename)
        
        return AudioInfo(**info)
        
    except Exception as e:
        logger.error(f"[AUDIO_INFO] Failed to get audio info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze audio file: {e}"
        )


@router.get("/voices",
           response_model=List[Dict[str, Any]],
           summary="List Available Voices",
           description="Get list of available TTS voices")
async def list_voices(
    language: Optional[str] = Query(None, description="Filter by language"),
    audio_service: AudioService = Depends(get_audio_service)
) -> List[Dict[str, Any]]:
    """
    Get list of available TTS voices.
    
    Args:
        language: Filter voices by language
        audio_service: Audio service dependency
    
    Returns:
        List[Dict[str, Any]]: Available voices with metadata
    """
    try:
        logger.debug("[VOICES] Getting available voices...")
        
        voices = await audio_service.list_voices(language_filter=language)
        
        return voices
        
    except Exception as e:
        logger.error(f"[VOICES] Failed to list voices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list voices: {e}"
        )


@router.get("/processing/{task_id}",
           response_model=AudioProcessingStatus,
           summary="Get Processing Status",
           description="Get status of an audio processing task")
async def get_processing_status(
    task_id: str,
    audio_service: AudioService = Depends(get_audio_service)
) -> AudioProcessingStatus:
    """
    Get audio processing task status.
    
    Args:
        task_id: Audio processing task ID
        audio_service: Audio service dependency
    
    Returns:
        AudioProcessingStatus: Current task status
    """
    try:
        logger.debug(f"[PROCESSING_STATUS] Getting status for task: {task_id}")
        
        status_info = await audio_service.get_processing_status(task_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processing task '{task_id}' not found"
            )
        
        return AudioProcessingStatus(**status_info)
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"[PROCESSING_STATUS] Failed to get status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing status: {e}"
        )


# Background task functions

async def _log_tts_metrics(text_length: int, success: bool, voice_id: str = None, processing_time: float = 0):
    """Log TTS metrics in background."""
    try:
        status = "success" if success else "failed"
        logger.info(f"[METRICS] TTS {status} - Length: {text_length}, Voice: {voice_id}, Time: {processing_time:.4f}s")
    except Exception as e:
        logger.error(f"[METRICS] Failed to log TTS metrics: {e}")


async def _log_stt_metrics(success: bool, filename: str = None, file_size: int = 0, model_name: str = None, processing_time: float = 0):
    """Log STT metrics in background."""
    try:
        status = "success" if success else "failed"
        logger.info(f"[METRICS] STT {status} - File: {filename}, Size: {file_size}, Model: {model_name}, Time: {processing_time:.4f}s")
    except Exception as e:
        logger.error(f"[METRICS] Failed to log STT metrics: {e}")


async def _process_audio_background(
    audio_service: AudioService,
    task_id: str,
    audio_data: bytes,
    filename: str,
    operations: Dict[str, Any],
    output_format: str
):
    """Process audio in background."""
    try:
        logger.info(f"[AUDIO_PROCESS] Starting background processing for task: {task_id}")
        
        result = await audio_service.process_audio(
            audio_data=audio_data,
            filename=filename,
            operations=operations,
            output_format=output_format,
            task_id=task_id
        )
        
        logger.info(f"[AUDIO_PROCESS] Background processing completed for task: {task_id}")
        
    except Exception as e:
        logger.error(f"[AUDIO_PROCESS] Background processing failed for task {task_id}: {e}")
        # Update task status with error
        try:
            await audio_service.update_processing_status(task_id, "failed", error_message=str(e))
        except Exception as update_error:
            logger.error(f"[AUDIO_PROCESS] Failed to update task status: {update_error}")
