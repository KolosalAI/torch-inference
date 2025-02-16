# core/model_manager.py
import os
import json
import threading
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from shutil import move, rmtree
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.segmentation_model import SegmentationModel
from utils.downloader import DownloadHandler
from utils.checksum import validate_checksum
from models.model_sources import get_downloader_for_source

log = logging.getLogger(__name__)

@dataclass
class ModelStatus:
    model_id: str
    status: str = "pending"  # pending, downloading, validating, active, error
    progress: float = 0.0
    error: Optional[str] = None
    download_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.active_model: Optional[SegmentationModel] = None
        self.model_registry_path = Path("models/model_registry.json")
        self.model_store = Path("models/model_store")
        self.download_queue: Dict[str, ModelStatus] = {}
        self.lock = threading.Lock()
        self._load_registry()
        
        # Ensure directories exist
        self.model_store.mkdir(parents=True, exist_ok=True)

    def _load_registry(self):
        """Load or initialize the model registry"""
        try:
            if self.model_registry_path.exists():
                with open(self.model_registry_path) as f:
                    self.registry = json.load(f)
            else:
                self.registry = {}
        except Exception as e:
            log.error(f"Failed to load model registry: {str(e)}")
            self.registry = {}

    def _save_registry(self):
        """Persist model registry to disk"""
        try:
            with open(self.model_registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save model registry: {str(e)}")

    def download_model(self, source: str, model_id: str, version: Optional[str] = None, force: bool = False) -> ModelStatus:
        """Initiate model download from specified source"""
        with self.lock:
            # Check existing models
            if model_id in self.registry and not force:
                return ModelStatus(
                    model_id=model_id,
                    status="exists",
                    metadata=self.registry[model_id]
                )

            # Create status tracker
            status = ModelStatus(model_id=model_id)
            self.download_queue[model_id] = status

            # Start download thread
            thread = threading.Thread(
                target=self._download_thread,
                args=(source, model_id, version, force, status),
                daemon=True
            )
            thread.start()

            return status

    def _download_thread(self, source: str, model_id: str, version: str, force: bool, status: ModelStatus):
        """Background thread handling download process"""
        try:
            downloader = get_downloader_for_source(source)(self.config)
            temp_dir = self.model_store / "temp" / model_id
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Update status
            status.status = "downloading"
            status.download_path = temp_dir

            # Download files
            download_result = downloader.download(
                model_id=model_id,
                version=version,
                output_dir=temp_dir,
                progress_callback=lambda p: setattr(status, 'progress', p),
                force=force
            )

            if not download_result:
                raise RuntimeError(f"Download failed for {model_id}")

            # Validate downloaded files
            status.status = "validating"
            if not self._validate_model(temp_dir):
                raise RuntimeError("Model validation failed")

            # Move to final location
            final_path = self.model_store / model_id
            if final_path.exists():
                rmtree(final_path)
            move(temp_dir, final_path)

            # Update registry
            self.registry[model_id] = {
                "source": source,
                "version": version or "latest",
                "path": str(final_path),
                "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checksum": self._calculate_checksum(final_path)
            }
            self._save_registry()

            # Update status
            status.status = "completed"
            status.progress = 100
            status.end_time = time.time()

        except Exception as e:
            status.status = "error"
            status.error = str(e)
            log.exception(f"Model download failed: {str(e)}")
            if temp_dir.exists():
                rmtree(temp_dir)
        finally:
            status.end_time = time.time()

    def _validate_model(self, model_path: Path) -> bool:
        """Validate downloaded model files"""
        # Check for required files
        required_files = ["model_weights.pth", "config.json"]
        for f in required_files:
            if not (model_path / f).exists():
                raise ValueError(f"Missing required file: {f}")

        # Validate checksum if available
        if self.config.model_validation.checksum_verification:
            checksum_file = model_path / "checksum.sha256"
            if checksum_file.exists():
                if not validate_checksum(model_path, checksum_file):
                    raise ValueError("Checksum validation failed")

        return True

    def _calculate_checksum(self, model_path: Path) -> str:
        """Calculate checksum for model files"""
        # Implementation depends on your checksum utility
        return "sha256:..."  # Replace with actual checksum

    def get_model_status(self, model_id: str) -> Optional[ModelStatus]:
        """Get current status of a model"""
        with self.lock:
            return self.download_queue.get(model_id)

    def list_available_models(self) -> Dict[str, Any]:
        """List all models in registry and download queue"""
        with self.lock:
            return {
                "registry": self.registry,
                "active": self.active_model.model_id if self.active_model else None,
                "downloads": {k: v.__dict__ for k, v in self.download_queue.items()}
            }

    def activate_model(self, model_id: str) -> bool:
        """Switch active model implementation"""
        with self.lock:
            if model_id not in self.registry:
                log.error(f"Model {model_id} not in registry")
                return False

            model_path = Path(self.registry[model_id]["path"])
            
            try:
                new_model = SegmentationModel.load(
                    model_path=model_path,
                    device=self.config.device,
                    use_tensorrt=self.config.use_tensorrt
                )
                
                # Warmup model
                new_model.warmup(self.config.warmup_iterations)
                
                # Switch models
                old_model = self.active_model
                self.active_model = new_model
                
                # Cleanup old model
                if old_model:
                    old_model.unload()
                
                log.info(f"Activated model: {model_id}")
                return True
            except Exception as e:
                log.error(f"Model activation failed: {str(e)}")
                return False

    def cleanup_downloads(self):
        """Clean up incomplete downloads"""
        with self.lock:
            temp_dir = self.model_store / "temp"
            if temp_dir.exists():
                rmtree(temp_dir)
                temp_dir.mkdir()

if __name__ == "__main__":
    # Example usage
    from utils.config import EngineConfig
    
    config = EngineConfig()
    manager = ModelManager(config)
    
    # Start download
    status = manager.download_model(
        source="huggingface",
        model_id="bert-base-uncased"
    )
    
    # Monitor progress
    while status.status not in ["completed", "error"]:
        print(f"Download progress: {status.progress}%")
        time.sleep(1)
    
    # Activate model
    if manager.activate_model("bert-base-uncased"):
        print("Model activated successfully")