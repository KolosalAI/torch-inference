import os
import logging
import torch
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from ..core.engine import InferenceEngine
from ..core.preprocessor import ImagePreprocessor
from ..core.postprocessor import ClassificationPostprocessor
from ..utils.config import load_config
from ..utils.logger import setup_logging

# Setup logging
logger = setup_logging()

app = FastAPI()
engine = None
preprocessor = None
postprocessor = None

MODEL_DIR = Path("./models/model_store")
MODEL_URLS = {
    "default": "https://example.com/path/to/model.pt",  # Replace with real URL
}

class InferenceRequest(BaseModel):
    image: list  # Expecting HWC format list
    model_name: str = "default"  # Allows selecting different models

@app.on_event("startup")
async def startup_event():
    global engine, preprocessor, postprocessor
    config = load_config()

    # Ensure model directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Download model if not present
    model_path = MODEL_DIR / f"{config.MODEL_NAME}.pt"
    if not model_path.exists():
        logger.info(f"Model '{config.MODEL_NAME}' not found, downloading...")
        download_model(config.MODEL_NAME, model_path)

    try:
        engine = InferenceEngine(str(model_path), config)
        preprocessor = ImagePreprocessor(config)
        postprocessor = ClassificationPostprocessor(config)
        logger.info("Model loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Initialization failed: {str(e)}")

def download_model(model_name: str, save_path: Path):
    """
    Downloads the model file if it doesn't exist.
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
        raise RuntimeError(f"Failed to download model '{model_name}': {str(e)}")

@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        # Validate model selection
        model_path = MODEL_DIR / f"{request.model_name}.pt"
        if not model_path.exists():
            return {"error": f"Model '{request.model_name}' not found. Please check available models."}

        # Convert list to numpy array
        image_np = np.array(request.image, dtype=np.uint8)

        # Preprocess
        tensor = preprocessor.process(image_np)

        # Inference
        output = engine.infer(tensor)

        # Postprocess
        result = postprocessor.process(output)

        return {"success": True, "result": result}
    
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check API to verify server and model status.
    """
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": engine is not None
    }

@app.get("/models")
async def list_available_models():
    """
    Returns a list of available models in the model store.
    """
    models = [model.stem for model in MODEL_DIR.glob("*.pt")]
    return {"available_models": models}

