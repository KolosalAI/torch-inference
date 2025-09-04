"""
Core inference service handling prediction requests.
"""

import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Union

from ..core.exceptions import InferenceError, ModelNotFoundError, ServiceUnavailableError
# Import models directly from specific files to avoid circular imports
from ..models.api.inference import InferenceRequest, InferenceResponse, BatchInferenceRequest, BatchInferenceResponse

logger = logging.getLogger(__name__)


class InferenceService:
    """Service for handling inference requests."""
    
    def __init__(self, model_manager=None, inference_engine=None, autoscaler=None):
        self.model_manager = model_manager
        self.inference_engine = inference_engine
        self.autoscaler = autoscaler
        self.logger = logger
    
    async def predict(
        self, 
        model_name: str, 
        request: InferenceRequest
    ) -> InferenceResponse:
        """
        Perform prediction using specified model.
        
        Args:
            model_name: Name of the model to use
            request: Inference request with inputs and options
            
        Returns:
            InferenceResponse with results or error
        """
        start_time = time.perf_counter()
        
        try:
            # Validate model exists
            if not self._is_model_available(model_name):
                if model_name != "example":
                    self.logger.warning(f"Model '{model_name}' not found, using 'example'")
                    model_name = "example"
                    
                if not self._is_model_available(model_name):
                    raise ModelNotFoundError(f"Model '{model_name}' not available")
            
            # Determine processing strategy
            is_batch_input = isinstance(request.inputs, list) and len(request.inputs) > 1
            input_count = len(request.inputs) if isinstance(request.inputs, list) else 1
            
            self.logger.debug(
                f"Processing prediction - Model: {model_name}, "
                f"Type: {'batch' if is_batch_input else 'single'}, Count: {input_count}"
            )
            
            # Perform inference
            result = await self._execute_prediction(model_name, request, is_batch_input)
            
            processing_time = time.perf_counter() - start_time
            
            self.logger.debug(
                f"Prediction completed - Model: {model_name}, "
                f"Time: {processing_time*1000:.1f}ms"
            )
            
            return InferenceResponse(
                success=True,
                result=result,
                processing_time=processing_time,
                model_info={
                    "model": model_name,
                    "device": self._get_device_info(),
                    "input_type": "batch" if is_batch_input else "single",
                    "input_count": input_count,
                    "processing_path": "optimized"
                },
                batch_info={
                    "inflight_batching_enabled": request.enable_batching,
                    "processed_as_batch": is_batch_input,
                    "concurrent_optimization": is_batch_input and input_count > 8
                }
            )
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Prediction timeout for model {model_name}")
            return InferenceResponse(
                success=False,
                error="Request timed out",
                model_info={"model": model_name},
                processing_time=time.perf_counter() - start_time
            )
        except Exception as e:
            self.logger.error(f"Prediction failed for model {model_name}: {e}")
            return InferenceResponse(
                success=False,
                error=str(e),
                model_info={"model": model_name},
                processing_time=time.perf_counter() - start_time
            )
    
    async def predict_batch(
        self, 
        model_name: str, 
        inputs_list: List[Any],
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Perform batch prediction.
        
        Args:
            model_name: Name of the model to use
            inputs_list: List of inputs to process
            priority: Request priority
            timeout: Request timeout
            
        Returns:
            List of prediction results
        """
        if not self.inference_engine and not self.autoscaler:
            raise ServiceUnavailableError("No inference services available")
        
        try:
            if self.autoscaler:
                # Process through autoscaler
                tasks = [
                    self.autoscaler.predict(model_name, input_item, priority=priority, timeout=timeout)
                    for input_item in inputs_list
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions in results
                final_results = []
                for result in results:
                    if isinstance(result, Exception):
                        final_results.append({"error": str(result)})
                    else:
                        final_results.append(result)
                return final_results
            else:
                # Use inference engine batch processing
                return await self.inference_engine.predict_batch(
                    inputs_list=inputs_list,
                    priority=priority,
                    timeout=timeout or 1.0
                )
                
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise InferenceError(f"Batch prediction failed: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of inference services."""
        if not self.inference_engine:
            return {
                "healthy": False,
                "checks": {"inference_engine": False},
                "timestamp": time.time()
            }
        
        try:
            health_status = await self.inference_engine.health_check()
            return health_status
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "checks": {"error": str(e)},
                "timestamp": time.time()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        if not self.inference_engine:
            raise ServiceUnavailableError("Inference engine not available")
        
        try:
            stats = self.inference_engine.get_stats()
            performance_report = self.inference_engine.get_performance_report()
            return {
                "stats": stats,
                "performance_report": performance_report
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            raise InferenceError(f"Failed to get stats: {e}")
    
    def _is_model_available(self, model_name: str) -> bool:
        """Check if model is available."""
        if not self.model_manager:
            return model_name == "example"  # Fallback for basic service
        return model_name in self.model_manager.list_models()
    
    async def _execute_prediction(
        self, 
        model_name: str, 
        request: InferenceRequest, 
        is_batch_input: bool
    ) -> Any:
        """Execute the actual prediction."""
        if is_batch_input and len(request.inputs) <= 8:  # Small batches
            self.logger.debug(f"Processing small batch ({len(request.inputs)} items)")
            return await self.predict_batch(
                model_name, request.inputs, request.priority, request.timeout
            )
        elif is_batch_input:  # Large batches - process concurrently
            self.logger.debug(f"Processing large batch ({len(request.inputs)} items) concurrently")
            chunk_size = 4
            chunks = [request.inputs[i:i + chunk_size] for i in range(0, len(request.inputs), chunk_size)]
            
            # Process chunks concurrently
            chunk_tasks = [
                self.predict_batch(model_name, chunk, request.priority, request.timeout)
                for chunk in chunks
            ]
            
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            # Flatten results
            result = []
            for chunk_result in chunk_results:
                if isinstance(chunk_result, Exception):
                    result.extend([{"error": str(chunk_result)}] * len(chunks[0]))
                else:
                    result.extend(chunk_result)
            return result
        else:  # Single input - fastest path
            self.logger.debug("Processing single input (fastest path)")
            
            if self.autoscaler:
                return await self.autoscaler.predict(
                    model_name, request.inputs, 
                    priority=request.priority, timeout=request.timeout
                )
            else:
                return await self.inference_engine.predict(
                    inputs=request.inputs,
                    priority=request.priority,
                    timeout=request.timeout or 1.0
                )
    
    def _get_device_info(self) -> str:
        """Get device information."""
        if self.inference_engine:
            return str(self.inference_engine.device)
        return "autoscaler" if self.autoscaler else "unknown"
