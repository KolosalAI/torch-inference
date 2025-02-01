import yaml
import torch
from typing import Tuple
from pydantic import BaseModel

class Config(BaseModel):
    # Device and precision settings
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PRECISION: str = "fp16" if torch.cuda.is_available() else "fp32"
    
    # Model and inference parameters
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    MODEL_PATH: str = "./modules/models/model_store/default.pt"
    TRT_FP16: bool = True
    MODEL_NAME: str = "default"

    class Config:
        extra = "allow"  # Allow extra keys from the YAML file

def load_config(path: str = "config.yaml") -> Config:
    """
    Load configuration from a YAML file and return a Config object.
    If certain keys are missing, defaults are applied.
    """
    with open(path, "r") as f:
        config_data = yaml.safe_load(f) or {}
    return Config(**config_data)
