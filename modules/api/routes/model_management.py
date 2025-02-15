from fastapi import APIRouter, HTTPException
from core.model_manager import ModelManager
from modules.utils.downloader import DownloadHandler
from schemas import ModelDownloadRequest, ModelStatusResponse

router = APIRouter()
model_manager = ModelManager()

@router.post("/models/download", response_model=ModelStatusResponse)
async def download_model(request: ModelDownloadRequest):
    """Endpoint to initiate model downloads"""
    return model_manager.download_model(
        source=request.source,
        model_id=request.model_id,
        version=request.version,
        force=request.force
    )

@router.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status(model_id: str):
    """Check download progress and model availability"""
    return model_manager.get_status(model_id)

@router.post("/models/activate")
async def activate_model(model_id: str):
    """Switch active model at runtime"""
    success = model_manager.activate_model(model_id)
    return {"status": "success" if success else "failed"}