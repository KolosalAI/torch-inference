import yaml
import torch
from typing import Dict, Any

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    # Set default values
    config.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    config.setdefault("precision", "fp16" if config["device"] == "cuda" else "fp32")
    
    return config