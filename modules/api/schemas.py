from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class InferenceRequest(BaseModel):
    image: List[List[List[int]]] = Field(
        ...,
        description="3D list representing an image in HWC format. Example: [[[R, G, B], ...], ...]"
    )
    model_name: str = Field(
        "default",
        description="Name of the model to use for inference."
    )
    task_type: str = Field(
        "classification",
        description="Task type: one of 'classification', 'detection', or 'segmentation'."
    )

    class Config:
        schema_extra = {
            "example": {
                "image": [
                    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    [[255, 255, 0], [0, 255, 255], [255, 0, 255]],
                    [[0, 0, 0], [128, 128, 128], [255, 255, 255]]
                ],
                "model_name": "default",
                "task_type": "classification"
            }
        }

class InferenceResponse(BaseModel):
    success: bool = Field(
        ...,
        description="Indicates if the inference was successful."
    )
    result: Optional[Dict[str, Any]] = Field(
        None,
        description="The output of the model inference. Structure depends on the task type."
    )
    error: Optional[str] = Field(
        None,
        description="Error message if inference failed."
    )

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "result": {"label": "cat", "confidence": 0.98},
                "error": None
            }
        }

class HealthCheckResponse(BaseModel):
    status: str = Field(
        "healthy",
        description="Server health status."
    )
    gpu_available: bool = Field(
        ...,
        description="Indicates whether GPU is available for inference."
    )
    model_loaded: bool = Field(
        ...,
        description="Indicates whether the model is loaded and ready for inference."
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "gpu_available": True,
                "model_loaded": True
            }
        }

class ModelListResponse(BaseModel):
    available_models: List[str] = Field(
        ...,
        description="List of available model names in the model store."
    )

    class Config:
        schema_extra = {
            "example": {
                "available_models": ["default", "model_v2", "experimental_model"]
            }
        }
