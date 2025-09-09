"""
Inference service wrapper for backward compatibility.

This module provides a wrapper around the enhanced inference engine
to maintain API compatibility with existing code while leveraging
the new architecture and capabilities.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import asyncio
from datetime import datetime

from ..core.exceptions import InferenceError, ModelNotFoundError, ValidationError

logger = logging.getLogger(__name__)


class InferenceServiceWrapper:
    """
    Wrapper around the enhanced inference engine to provide backward compatibility
    with the existing service interface while leveraging new capabilities.
    """
    
    def __init__(self, inference_engine, model_manager):
        """
        Initialize the inference service wrapper.
        
        Args:
            inference_engine: Enhanced inference engine instance
            model_manager: Model manager instance
        """
        self.inference_engine = inference_engine
        self.model_manager = model_manager
        self._initialized = False
        
        logger.debug("InferenceServiceWrapper initialized")
    
    async def initialize(self):
        """Initialize the service wrapper."""
        if not self._initialized:
            # Ensure inference engine is started
            if hasattr(self.inference_engine, 'is_running') and not self.inference_engine.is_running():
                await self.inference_engine.start()
            
            self._initialized = True
            logger.debug("InferenceServiceWrapper initialization completed")
    
    async def predict(self, input_data: Any, model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform inference using the enhanced engine.
        
        Args:
            input_data: Input data for inference
            model_name: Optional model name to use
            **kwargs: Additional inference parameters
        
        Returns:
            Dict containing prediction results
        """
        try:
            await self.initialize()
            
            # Convert input data to appropriate format
            if isinstance(input_data, str):
                input_tensor = input_data  # Let the engine handle text
            elif hasattr(input_data, 'numpy'):
                input_tensor = input_data
            elif isinstance(input_data, (list, tuple)):
                input_tensor = input_data
            else:
                input_tensor = input_data
            
            # Use the enhanced inference engine
            result = await self.inference_engine.predict(
                input_data=input_tensor,
                model_name=model_name,
                **kwargs
            )
            
            # Ensure result is in expected format
            if not isinstance(result, dict):
                result = {"prediction": result}
            
            # Add metadata for compatibility
            result.update({
                "timestamp": datetime.utcnow().isoformat(),
                "model_name": model_name or "default",
                "service": "inference_service_wrapper"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise InferenceError(
                details=f"Prediction failed: {e}",
                context={"model_name": model_name, "input_type": type(input_data).__name__},
                cause=e
            )
    
    async def predict_batch(self, inputs: List[Any], model_name: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform batch inference.
        
        Args:
            inputs: List of input data
            model_name: Optional model name to use
            **kwargs: Additional inference parameters
        
        Returns:
            List of prediction results
        """
        try:
            await self.initialize()
            
            # Check if engine supports batch processing
            if hasattr(self.inference_engine, 'predict_batch'):
                results = await self.inference_engine.predict_batch(
                    inputs=inputs,
                    model_name=model_name,
                    **kwargs
                )
                
                # Ensure results are in expected format
                if not isinstance(results, list):
                    results = [results]
                
                # Add metadata to each result
                timestamp = datetime.utcnow().isoformat()
                for i, result in enumerate(results):
                    if not isinstance(result, dict):
                        results[i] = {"prediction": result}
                    results[i].update({
                        "timestamp": timestamp,
                        "model_name": model_name or "default",
                        "batch_index": i,
                        "service": "inference_service_wrapper"
                    })
                
                return results
            else:
                # Fallback to individual predictions
                results = []
                for i, input_data in enumerate(inputs):
                    try:
                        result = await self.predict(input_data, model_name, **kwargs)
                        result["batch_index"] = i
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Batch prediction failed for item {i}: {e}")
                        results.append({
                            "error": str(e),
                            "batch_index": i,
                            "timestamp": datetime.utcnow().isoformat(),
                            "service": "inference_service_wrapper"
                        })
                
                return results
        
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise InferenceError(
                details=f"Batch prediction failed: {e}",
                context={"model_name": model_name, "batch_size": len(inputs)},
                cause=e
            )
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            return self.model_manager.list_models()
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        try:
            return self.model_manager.get_loaded_models()
        except Exception as e:
            logger.error(f"Failed to get loaded models: {e}")
            return []
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded."""
        try:
            return self.model_manager.is_model_loaded(model_name)
        except Exception as e:
            logger.error(f"Failed to check model status: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        try:
            return self.model_manager.get_model_info(model_name)
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                "name": model_name,
                "status": "unknown",
                "error": str(e)
            }
    
    async def load_model(self, model_name: str, **kwargs) -> bool:
        """Load a model (if supported by engine)."""
        try:
            if hasattr(self.inference_engine, 'load_model'):
                await self.inference_engine.load_model(model_name, **kwargs)
                return True
            else:
                logger.warning(f"Model loading not supported by current engine")
                return False
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model (if supported by engine)."""
        try:
            if hasattr(self.inference_engine, 'unload_model'):
                await self.inference_engine.unload_model(model_name)
                return True
            else:
                logger.warning(f"Model unloading not supported by current engine")
                return False
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        try:
            if hasattr(self.inference_engine, 'get_stats'):
                stats = self.inference_engine.get_stats()
            else:
                stats = {}
            
            # Add model manager stats
            stats.update({
                "available_models": len(self.get_available_models()),
                "loaded_models": len(self.get_loaded_models()),
                "service_type": "inference_service_wrapper"
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get inference stats: {e}")
            return {"error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the service."""
        try:
            health = {
                "healthy": True,
                "service": "inference_service_wrapper",
                "initialized": self._initialized,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check inference engine health
            if hasattr(self.inference_engine, 'get_health_status'):
                engine_health = self.inference_engine.get_health_status()
                health.update({
                    "engine_healthy": engine_health.get("healthy", True),
                    "engine_status": engine_health
                })
                if not engine_health.get("healthy", True):
                    health["healthy"] = False
            
            # Check model manager health
            if hasattr(self.model_manager, 'get_health_status'):
                manager_health = self.model_manager.get_health_status()
                health.update({
                    "model_manager_healthy": manager_health.get("healthy", True),
                    "model_manager_status": manager_health
                })
                if not manager_health.get("healthy", True):
                    health["healthy"] = False
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "service": "inference_service_wrapper",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def shutdown(self):
        """Shutdown the service."""
        try:
            if hasattr(self.inference_engine, 'stop'):
                await self.inference_engine.stop()
            
            if hasattr(self.model_manager, 'cleanup_all'):
                self.model_manager.cleanup_all()
            
            self._initialized = False
            logger.info("InferenceServiceWrapper shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
    
    # Legacy compatibility methods
    
    def predict_sync(self, input_data: Any, model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Synchronous prediction for backward compatibility."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.predict(input_data, model_name, **kwargs))
        except RuntimeError:
            # No event loop running
            return asyncio.run(self.predict(input_data, model_name, **kwargs))
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status (legacy method)."""
        return {
            "status": "running" if self._initialized else "not_initialized",
            "available_models": self.get_available_models(),
            "loaded_models": self.get_loaded_models(),
            "health": self.get_health_status()
        }
    
    def cleanup(self):
        """Cleanup method for backward compatibility."""
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.shutdown())
        except RuntimeError:
            asyncio.run(self.shutdown())


# Convenience function for creating the service
def create_inference_service(inference_engine, model_manager) -> InferenceServiceWrapper:
    """Create an inference service wrapper instance."""
    return InferenceServiceWrapper(inference_engine, model_manager)
