import os
import logging
import torch
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from pathlib import Path

# Import your custom modules (adjust the import paths as necessary)
from ..core.engine import InferenceEngine
from ..core.preprocessor import MultiTaskPreprocessor
from ..core.postprocessor import (
    ClassificationPostprocessor,
    DetectionPostprocessor,
    SegmentationPostprocessor
)
from ..utils.config import load_config
from ..utils.logger import setup_logging

# Set up logging
logger = setup_logging()

app = FastAPI(
    title="Model Inference API",
    description="An API for performing inference on various tasks (classification, detection, segmentation).",
    version="1.0.0"
)

# Add CORS middleware to allow requests from any origin (customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for engine, preprocessor, and postprocessors.
engine: InferenceEngine = None
preprocessor: MultiTaskPreprocessor = None
postprocessors: dict = {}

MODEL_DIR = Path("./models/model_store")
MODEL_URLS = {
    "default": "https://example.com/path/to/model.pt",  # Replace with a real URL.
}

class InferenceRequest(BaseModel):
    """
    Request model for inference.
    - **image**: A list representing the image in HWC format.
    - **model_name**: Name of the model to use.
    - **task_type**: Type of task: 'classification', 'detection', or 'segmentation'.
    """
    image: list
    model_name: str = "default"
    task_type: str = "classification"

@app.on_event("startup")
async def startup_event():
    """
    Startup event: Load configuration, ensure the model exists (download if necessary),
    initialize the InferenceEngine, preprocessor, and postprocessors.
    """
    global engine, preprocessor, postprocessors

    config = load_config()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{config.MODEL_NAME}.pt"

    if not model_path.exists():
        logger.info(f"Model '{config.MODEL_NAME}' not found, downloading...")
        # Offload model download to a thread to avoid blocking the event loop.
        await run_in_threadpool(download_model, config.MODEL_NAME, model_path)

    try:
        # Offload engine initialization to a thread if it is blocking.
        engine = await run_in_threadpool(InferenceEngine, str(model_path), config)
        preprocessor = MultiTaskPreprocessor(config)
        postprocessors["classification"] = ClassificationPostprocessor(config)
        postprocessors["detection"] = DetectionPostprocessor(config)
        postprocessors["segmentation"] = SegmentationPostprocessor(config)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.exception("Initialization failed")
        raise RuntimeError(f"Initialization failed: {str(e)}")

def download_model(model_name: str, save_path: Path):
    """
    Download the model from a given URL.
    """
    url = MODEL_URLS.get(model_name)
    if not url:
        raise ValueError(f"No download URL available for model '{model_name}'")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f"Downloaded model: {save_path}")
    except Exception as e:
        logger.exception(f"Failed to download model '{model_name}'")
        raise RuntimeError(f"Failed to download model '{model_name}': {str(e)}")

@app.post("/predict", tags=["Inference"], summary="Perform model inference")
async def predict(request: InferenceRequest, background_tasks: BackgroundTasks):
    """
    Perform inference on an input image.
    - **image**: Image data in HWC list format.
    - **model_name**: Model to use.
    - **task_type**: Inference task type ('classification', 'detection', or 'segmentation').
    
    Returns a JSON object with the inference result.
    """
    try:
        model_path = MODEL_DIR / f"{request.model_name}.pt"
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found. Please check available models."
            )
        # Convert the input image (assumed HWC list) to a NumPy array.
        image_np = np.array(request.image, dtype=np.uint8)
        # Use the preprocessor __call__ method (instead of a custom .process)
        tensor = await run_in_threadpool(preprocessor, image_np)
        # Offload inference to a thread if engine.infer is blocking.
        output = await run_in_threadpool(engine.infer, tensor)
        postprocessor = postprocessors.get(request.task_type)
        if not postprocessor:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported task type: {request.task_type}"
            )
        # Process the model outputs.
        result = await run_in_threadpool(postprocessor, output)
        return {"success": True, "result": result}
    except Exception as e:
        logger.exception(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["Health"], summary="Health check endpoint")
async def health_check():
    """
    Health check endpoint. Returns the service status,
    GPU availability, and whether the model is loaded.
    """
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": engine is not None
    }

@app.get("/models", tags=["Models"], summary="List available models")
async def list_available_models():
    """
    List the names of available models in the model store.
    """
    models = [model.stem for model in MODEL_DIR.glob("*.pt")]
    return {"available_models": models}
