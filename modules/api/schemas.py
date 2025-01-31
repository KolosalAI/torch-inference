from pydantic import BaseModel, Field
from typing import List, Optional

class InferenceRequest(BaseModel):
    image: List[List[List[int]]] = Field(..., description="3D list representing an image in HWC format")
    model_name: str = Field("default", description="Name of the model to use for inference")
    task_type: str = Field("classification", description="Task type: classification, detection, segmentation")

class InferenceResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the inference was successful")
    result: Optional[dict] = Field(None, description="The output of the model inference")
    error: Optional[str] = Field(None, description="Error message if inference failed")

class HealthCheckResponse(BaseModel):
    status: str = Field("healthy", description="Server health status")
    gpu_available: bool = Field(..., description="Whether GPU is available for inference")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")

class ModelListResponse(BaseModel):
    available_models: List[str] = Field(..., description="List of available models in the model store")
