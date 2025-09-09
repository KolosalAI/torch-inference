"""
Model Configuration Manager for PyTorch Inference Framework

This module provides functionality to load and manage model configurations
from models.json file, determining which models are available and how they
should be loaded and configured.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelCategory(Enum):
    """Model categories."""
    DEMO = "demo"
    NLP = "nlp"
    COMPUTER_VISION = "computer-vision"
    AUDIO = "audio"
    CUSTOM = "custom"


class ModelSource(Enum):
    """Model sources."""
    BUILTIN = "builtin"
    HUGGINGFACE = "huggingface"
    TORCHVISION = "torchvision"
    PYTORCH_HUB = "pytorch_hub"
    CUSTOM = "custom"
    URL = "url"


@dataclass
class HardwareRequirements:
    """Hardware requirements for a model."""
    min_memory_mb: int
    recommended_memory_mb: int
    gpu_required: bool
    min_gpu_memory_mb: int


@dataclass
class InferenceConfig:
    """Inference configuration for a model."""
    batch_size: int
    max_batch_size: int
    timeout_seconds: int
    warmup_iterations: int


@dataclass
class ModelMetadata:
    """Model metadata."""
    parameters: int
    size_mb: float
    architecture: str
    framework: str
    version: str
    languages: Optional[List[str]] = None
    accuracy: Optional[float] = None
    input_size: Optional[List[int]] = None
    num_classes: Optional[int] = None
    sample_rate: Optional[int] = None


@dataclass
class TTSFeatures:
    """Text-to-Speech specific features."""
    supports_tts: bool
    quality: str
    speed: str
    supports_voice_cloning: bool
    supports_emotions: bool
    max_text_length: int
    supports_music: Optional[bool] = False
    supports_sound_effects: Optional[bool] = False


@dataclass
class STTFeatures:
    """Speech-to-Text specific features."""
    supports_stt: bool
    quality: str
    speed: str
    supports_timestamps: bool
    supports_language_detection: bool
    max_audio_length_seconds: int


@dataclass
class ModelConfig:
    """Complete model configuration."""
    name: str
    display_name: str
    description: str
    source: ModelSource
    model_id: Optional[str]
    model_type: str
    task: str
    category: ModelCategory
    enabled: bool
    auto_load: bool
    priority: int
    hardware_requirements: HardwareRequirements
    inference_config: InferenceConfig
    metadata: ModelMetadata
    tts_features: Optional[TTSFeatures] = None
    stt_features: Optional[STTFeatures] = None


@dataclass
class ModelGroup:
    """Model group configuration."""
    name: str
    description: str
    models: List[str]
    default_model: str
    enabled: bool


@dataclass
class DeploymentProfile:
    """Deployment profile configuration."""
    name: str
    description: str
    auto_load_models: List[str]
    preload_models: List[str]
    max_models_in_memory: int
    prefer_cpu: bool
    enable_model_caching: bool
    enable_gpu_optimization: Optional[bool] = False


@dataclass
class HardwareProfile:
    """Hardware profile configuration."""
    name: str
    description: str
    allowed_models: List[str]
    blocked_models: List[str]
    max_memory_usage_mb: int
    max_gpu_memory_mb: Optional[int] = None
    optimization_level: str = "medium"


class ModelConfigManager:
    """Manager for model configurations from models.json."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize the model configuration manager.
        
        Args:
            config_file: Path to models.json file. If None, uses default location.
        """
        if config_file is None:
            # Try multiple possible locations
            possible_paths = [
                Path("data/models.json"),
                Path("models.json"),
                Path(__file__).parent.parent.parent / "data" / "models.json"
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_file = path
                    break
            
            if config_file is None:
                logger.warning("No models.json file found, using empty configuration")
                self._config = {}
                return
        
        self.config_file = Path(config_file)
        self._config = self._load_config()
        self._models = self._parse_models()
        self._model_groups = self._parse_model_groups()
        self._deployment_profiles = self._parse_deployment_profiles()
        self._hardware_profiles = self._parse_hardware_profiles()
        
        logger.info(f"Loaded model configuration from {self.config_file}")
        logger.info(f"Available models: {len(self._models)}")
        logger.info(f"Model groups: {len(self._model_groups)}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load models.json: {e}")
            return {}
    
    def _parse_models(self) -> Dict[str, ModelConfig]:
        """Parse model configurations."""
        models = {}
        
        available_models = self._config.get("available_models", {})
        for model_name, model_data in available_models.items():
            try:
                # Parse hardware requirements
                hw_req_data = model_data.get("hardware_requirements", {})
                hardware_requirements = HardwareRequirements(
                    min_memory_mb=hw_req_data.get("min_memory_mb", 256),
                    recommended_memory_mb=hw_req_data.get("recommended_memory_mb", 512),
                    gpu_required=hw_req_data.get("gpu_required", False),
                    min_gpu_memory_mb=hw_req_data.get("min_gpu_memory_mb", 0)
                )
                
                # Parse inference config
                inf_config_data = model_data.get("inference_config", {})
                inference_config = InferenceConfig(
                    batch_size=inf_config_data.get("batch_size", 1),
                    max_batch_size=inf_config_data.get("max_batch_size", 8),
                    timeout_seconds=inf_config_data.get("timeout_seconds", 30),
                    warmup_iterations=inf_config_data.get("warmup_iterations", 3)
                )
                
                # Parse metadata
                metadata_data = model_data.get("metadata", {})
                metadata = ModelMetadata(
                    parameters=metadata_data.get("parameters", 0),
                    size_mb=metadata_data.get("size_mb", 0.0),
                    architecture=metadata_data.get("architecture", "Unknown"),
                    framework=metadata_data.get("framework", "pytorch"),
                    version=metadata_data.get("version", "1.0.0"),
                    languages=metadata_data.get("languages"),
                    accuracy=metadata_data.get("accuracy"),
                    input_size=metadata_data.get("input_size"),
                    num_classes=metadata_data.get("num_classes"),
                    sample_rate=metadata_data.get("sample_rate")
                )
                
                # Parse TTS features if present
                tts_features = None
                if "tts_features" in model_data:
                    tts_data = model_data["tts_features"]
                    tts_features = TTSFeatures(
                        supports_tts=tts_data.get("supports_tts", False),
                        quality=tts_data.get("quality", "medium"),
                        speed=tts_data.get("speed", "medium"),
                        supports_voice_cloning=tts_data.get("supports_voice_cloning", False),
                        supports_emotions=tts_data.get("supports_emotions", False),
                        max_text_length=tts_data.get("max_text_length", 1000),
                        supports_music=tts_data.get("supports_music", False),
                        supports_sound_effects=tts_data.get("supports_sound_effects", False)
                    )
                
                # Parse STT features if present
                stt_features = None
                if "stt_features" in model_data:
                    stt_data = model_data["stt_features"]
                    stt_features = STTFeatures(
                        supports_stt=stt_data.get("supports_stt", False),
                        quality=stt_data.get("quality", "medium"),
                        speed=stt_data.get("speed", "medium"),
                        supports_timestamps=stt_data.get("supports_timestamps", False),
                        supports_language_detection=stt_data.get("supports_language_detection", False),
                        max_audio_length_seconds=stt_data.get("max_audio_length_seconds", 3600)
                    )
                
                # Create model config
                model_config = ModelConfig(
                    name=model_data.get("name", model_name),
                    display_name=model_data.get("display_name", model_name),
                    description=model_data.get("description", ""),
                    source=ModelSource(model_data.get("source", "custom")),
                    model_id=model_data.get("model_id"),
                    model_type=model_data.get("model_type", "unknown"),
                    task=model_data.get("task", "unknown"),
                    category=ModelCategory(model_data.get("category", "custom")),
                    enabled=model_data.get("enabled", True),
                    auto_load=model_data.get("auto_load", False),
                    priority=model_data.get("priority", 5),
                    hardware_requirements=hardware_requirements,
                    inference_config=inference_config,
                    metadata=metadata,
                    tts_features=tts_features,
                    stt_features=stt_features
                )
                
                models[model_name] = model_config
                
            except Exception as e:
                logger.error(f"Failed to parse model config for {model_name}: {e}")
        
        return models
    
    def _parse_model_groups(self) -> Dict[str, ModelGroup]:
        """Parse model group configurations."""
        groups = {}
        
        model_groups_data = self._config.get("model_groups", {})
        for group_name, group_data in model_groups_data.items():
            try:
                group = ModelGroup(
                    name=group_data.get("name", group_name),
                    description=group_data.get("description", ""),
                    models=group_data.get("models", []),
                    default_model=group_data.get("default_model", ""),
                    enabled=group_data.get("enabled", True)
                )
                groups[group_name] = group
            except Exception as e:
                logger.error(f"Failed to parse model group {group_name}: {e}")
        
        return groups
    
    def _parse_deployment_profiles(self) -> Dict[str, DeploymentProfile]:
        """Parse deployment profile configurations."""
        profiles = {}
        
        deployment_profiles_data = self._config.get("deployment_profiles", {})
        for profile_name, profile_data in deployment_profiles_data.items():
            try:
                profile = DeploymentProfile(
                    name=profile_data.get("name", profile_name),
                    description=profile_data.get("description", ""),
                    auto_load_models=profile_data.get("auto_load_models", []),
                    preload_models=profile_data.get("preload_models", []),
                    max_models_in_memory=profile_data.get("max_models_in_memory", 5),
                    prefer_cpu=profile_data.get("prefer_cpu", False),
                    enable_model_caching=profile_data.get("enable_model_caching", True),
                    enable_gpu_optimization=profile_data.get("enable_gpu_optimization", False)
                )
                profiles[profile_name] = profile
            except Exception as e:
                logger.error(f"Failed to parse deployment profile {profile_name}: {e}")
        
        return profiles
    
    def _parse_hardware_profiles(self) -> Dict[str, HardwareProfile]:
        """Parse hardware profile configurations."""
        profiles = {}
        
        hardware_profiles_data = self._config.get("hardware_profiles", {})
        for profile_name, profile_data in hardware_profiles_data.items():
            try:
                profile = HardwareProfile(
                    name=profile_data.get("name", profile_name),
                    description=profile_data.get("description", ""),
                    allowed_models=profile_data.get("allowed_models", []),
                    blocked_models=profile_data.get("blocked_models", []),
                    max_memory_usage_mb=profile_data.get("max_memory_usage_mb", 8192),
                    max_gpu_memory_mb=profile_data.get("max_gpu_memory_mb"),
                    optimization_level=profile_data.get("optimization_level", "medium")
                )
                profiles[profile_name] = profile
            except Exception as e:
                logger.error(f"Failed to parse hardware profile {profile_name}: {e}")
        
        return profiles
    
    def get_available_models(self, enabled_only: bool = True) -> List[str]:
        """Get list of available model names."""
        if enabled_only:
            return [name for name, config in self._models.items() if config.enabled]
        return list(self._models.keys())
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self._models.get(model_name)
    
    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a model is enabled."""
        config = self.get_model_config(model_name)
        return config.enabled if config else False
    
    def get_auto_load_models(self) -> List[str]:
        """Get list of models that should be auto-loaded."""
        return [name for name, config in self._models.items() 
                if config.enabled and config.auto_load]
    
    def get_models_by_category(self, category: ModelCategory) -> List[str]:
        """Get models by category."""
        return [name for name, config in self._models.items()
                if config.enabled and config.category == category]
    
    def get_models_by_task(self, task: str) -> List[str]:
        """Get models by task."""
        return [name for name, config in self._models.items()
                if config.enabled and config.task == task]
    
    def get_tts_models(self) -> List[str]:
        """Get TTS-enabled models."""
        return [name for name, config in self._models.items()
                if config.enabled and config.tts_features and config.tts_features.supports_tts]
    
    def get_stt_models(self) -> List[str]:
        """Get STT-enabled models."""
        return [name for name, config in self._models.items()
                if config.enabled and config.stt_features and config.stt_features.supports_stt]
    
    def get_model_group(self, group_name: str) -> Optional[ModelGroup]:
        """Get model group configuration."""
        return self._model_groups.get(group_name)
    
    def get_model_groups(self) -> Dict[str, ModelGroup]:
        """Get all model groups."""
        return self._model_groups.copy()
    
    def get_deployment_profile(self, profile_name: str) -> Optional[DeploymentProfile]:
        """Get deployment profile configuration."""
        return self._deployment_profiles.get(profile_name)
    
    def get_hardware_profile(self, profile_name: str) -> Optional[HardwareProfile]:
        """Get hardware profile configuration."""
        return self._hardware_profiles.get(profile_name)
    
    def get_compatible_models(self, hardware_profile: str = "cpu_only") -> List[str]:
        """Get models compatible with a hardware profile."""
        profile = self.get_hardware_profile(hardware_profile)
        if not profile:
            return self.get_available_models()
        
        available = set(self.get_available_models())
        allowed = set(profile.allowed_models) if profile.allowed_models else available
        blocked = set(profile.blocked_models) if profile.blocked_models else set()
        
        return list(allowed - blocked)
    
    def validate_model_requirements(self, model_name: str, 
                                  available_memory_mb: int,
                                  available_gpu_memory_mb: int = 0) -> bool:
        """Validate if system meets model requirements."""
        config = self.get_model_config(model_name)
        if not config:
            return False
        
        hw_req = config.hardware_requirements
        
        # Check memory requirements
        if available_memory_mb < hw_req.min_memory_mb:
            return False
        
        # Check GPU requirements
        if hw_req.gpu_required and available_gpu_memory_mb < hw_req.min_gpu_memory_mb:
            return False
        
        return True
    
    def get_model_priority(self, model_name: str) -> int:
        """Get model priority (lower number = higher priority)."""
        config = self.get_model_config(model_name)
        return config.priority if config else 999
    
    def get_models_by_priority(self) -> List[str]:
        """Get models sorted by priority (highest priority first)."""
        return sorted(self.get_available_models(), 
                     key=lambda name: self.get_model_priority(name))


# Global instance
_model_config_manager: Optional[ModelConfigManager] = None


def get_model_config_manager() -> ModelConfigManager:
    """Get the global model configuration manager."""
    global _model_config_manager
    if _model_config_manager is None:
        _model_config_manager = ModelConfigManager()
    return _model_config_manager


def is_model_available(model_name: str) -> bool:
    """Check if a model is available and enabled."""
    manager = get_model_config_manager()
    return manager.is_model_enabled(model_name)


def get_available_models() -> List[str]:
    """Get list of available models."""
    manager = get_model_config_manager()
    return manager.get_available_models()


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get model information as a dictionary."""
    manager = get_model_config_manager()
    config = manager.get_model_config(model_name)
    
    if not config:
        return None
    
    return {
        "name": config.name,
        "display_name": config.display_name,
        "description": config.description,
        "source": config.source.value,
        "model_id": config.model_id,
        "task": config.task,
        "category": config.category.value,
        "enabled": config.enabled,
        "auto_load": config.auto_load,
        "priority": config.priority,
        "hardware_requirements": {
            "min_memory_mb": config.hardware_requirements.min_memory_mb,
            "recommended_memory_mb": config.hardware_requirements.recommended_memory_mb,
            "gpu_required": config.hardware_requirements.gpu_required,
            "min_gpu_memory_mb": config.hardware_requirements.min_gpu_memory_mb
        },
        "metadata": {
            "parameters": config.metadata.parameters,
            "size_mb": config.metadata.size_mb,
            "architecture": config.metadata.architecture,
            "framework": config.metadata.framework,
            "version": config.metadata.version
        },
        "tts_enabled": config.tts_features is not None and config.tts_features.supports_tts,
        "stt_enabled": config.stt_features is not None and config.stt_features.supports_stt
    }
