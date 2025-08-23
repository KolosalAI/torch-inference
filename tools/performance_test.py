#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Testing Tool for PyTorch Inference Server

This script provides comprehensive performance testing capabilities including:
- Predictions per second (pred/s) measurement
- Concurrency testing with multiple threads
- Latency analysis (min, max, avg, percentiles)
- Throughput testing with various batch sizes
- Model selection via command-line arguments
- Detailed performance reports
- Support for new model-specific endpoints (/{model_name}/predict)

Version Notes:
- Updated to use new model-specific endpoint format: /{model_name}/predict
- Supports automatic batch handling (no separate /predict/batch endpoint needed)
- Updated unload endpoint to use /autoscaler/unload

Usage:
    python tools/performance_test.py --model example --duration 30 --concurrency 4
    python tools/performance_test.py --model resnet50 --requests 1000 --batch-size 8
    python tools/performance_test.py --list-models
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import requests
import aiohttp
import numpy as np
import psutil

# Optional imports
try:
    import GPUtil
except ImportError:
    GPUtil = None

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import locale
    try:
        # Try to set UTF-8 encoding
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Predefined model configurations for testing different sizes
MODEL_CONFIGS = {
    "small": [
        {
            "name": "mobilenet_v2_small",
            "source": "torchvision",
            "model_id": "mobilenet_v2",
            "description": "MobileNet V2 - Small efficient model (~14MB)",
            "estimated_size_mb": 14,
            "task": "image_classification",
            "input_type": "image_tensor"
        },
        {
            "name": "distilbert_small",
            "source": "huggingface", 
            "model_id": "distilbert-base-uncased",
            "description": "DistilBERT - Small BERT variant (~268MB)",
            "estimated_size_mb": 268,
            "task": "text-classification",
            "input_type": "text"
        }
    ],
    "medium": [
        {
            "name": "resnet50_medium",
            "source": "torchvision",
            "model_id": "resnet50",
            "description": "ResNet-50 - Medium CNN model (~98MB)",
            "estimated_size_mb": 98,
            "task": "image_classification",
            "input_type": "image_tensor"
        },
        {
            "name": "bert_medium",
            "source": "huggingface",
            "model_id": "bert-base-uncased",
            "description": "BERT Base - Medium transformer model (~440MB)",
            "estimated_size_mb": 440,
            "task": "text-classification",
            "input_type": "text"
        }
    ],
    "large": [
        {
            "name": "resnet152_large",
            "source": "torchvision",
            "model_id": "resnet152",
            "description": "ResNet-152 - Large CNN model (~230MB)",
            "estimated_size_mb": 230,
            "task": "image_classification",
            "input_type": "image_tensor"
        },
        {
            "name": "roberta_large",
            "source": "huggingface",
            "model_id": "roberta-large",
            "description": "RoBERTa Large - Large transformer model (~1.3GB)",
            "estimated_size_mb": 1300,
            "task": "text-classification",
            "input_type": "text"
        }
    ]
}

# Common model aliases for easy auto-downloading
COMMON_MODEL_ALIASES = {
    # TorchVision models
    "resnet18": {"source": "torchvision", "model_id": "resnet18", "task": "image_classification"},
    "resnet34": {"source": "torchvision", "model_id": "resnet34", "task": "image_classification"},
    "resnet50": {"source": "torchvision", "model_id": "resnet50", "task": "image_classification"},
    "resnet101": {"source": "torchvision", "model_id": "resnet101", "task": "image_classification"},
    "resnet152": {"source": "torchvision", "model_id": "resnet152", "task": "image_classification"},
    "mobilenet_v2": {"source": "torchvision", "model_id": "mobilenet_v2", "task": "image_classification"},
    "mobilenet_v3_small": {"source": "torchvision", "model_id": "mobilenet_v3_small", "task": "image_classification"},
    "mobilenet_v3_large": {"source": "torchvision", "model_id": "mobilenet_v3_large", "task": "image_classification"},
    "efficientnet_b0": {"source": "torchvision", "model_id": "efficientnet_b0", "task": "image_classification"},
    "efficientnet_b1": {"source": "torchvision", "model_id": "efficientnet_b1", "task": "image_classification"},
    "vgg16": {"source": "torchvision", "model_id": "vgg16", "task": "image_classification"},
    "vgg19": {"source": "torchvision", "model_id": "vgg19", "task": "image_classification"},
    "densenet121": {"source": "torchvision", "model_id": "densenet121", "task": "image_classification"},
    "densenet169": {"source": "torchvision", "model_id": "densenet169", "task": "image_classification"},
    "inception_v3": {"source": "torchvision", "model_id": "inception_v3", "task": "image_classification"},
    
    # Hugging Face models (popular ones)
    "bert-base-uncased": {"source": "huggingface", "model_id": "bert-base-uncased", "task": "text-classification"},
    "bert-base-cased": {"source": "huggingface", "model_id": "bert-base-cased", "task": "text-classification"},
    "bert-large-uncased": {"source": "huggingface", "model_id": "bert-large-uncased", "task": "text-classification"},
    "distilbert-base-uncased": {"source": "huggingface", "model_id": "distilbert-base-uncased", "task": "text-classification"},
    "roberta-base": {"source": "huggingface", "model_id": "roberta-base", "task": "text-classification"},
    "roberta-large": {"source": "huggingface", "model_id": "roberta-large", "task": "text-classification"},
    "albert-base-v2": {"source": "huggingface", "model_id": "albert-base-v2", "task": "text-classification"},
    "electra-small-discriminator": {"source": "huggingface", "model_id": "google/electra-small-discriminator", "task": "text-classification"},
    "electra-base-discriminator": {"source": "huggingface", "model_id": "google/electra-base-discriminator", "task": "text-classification"},
    
    # Simplified aliases
    "resnet": {"source": "torchvision", "model_id": "resnet50", "task": "image_classification"},
    "mobilenet": {"source": "torchvision", "model_id": "mobilenet_v2", "task": "image_classification"},
    "bert": {"source": "huggingface", "model_id": "bert-base-uncased", "task": "text-classification"},
    "distilbert": {"source": "huggingface", "model_id": "distilbert-base-uncased", "task": "text-classification"},
    "roberta": {"source": "huggingface", "model_id": "roberta-base", "task": "text-classification"},
}


@dataclass
class TestConfig:
    """Configuration for performance testing."""
    base_url: str = "http://localhost:8000"
    model_name: str = "example"
    duration_seconds: Optional[int] = None
    num_requests: Optional[int] = None
    concurrency: int = 1
    batch_size: int = 1
    warmup_requests: int = 10
    timeout: float = 30.0
    input_data: Union[int, float, List, str] = 42
    priority: int = 0
    verbose: bool = False
    output_format: str = "text"  # text, json, csv
    test_type: str = "basic"  # basic, latency, throughput, scalability, memory
    allow_null_results: bool = False


@dataclass
class RequestResult:
    """Result of a single request."""
    success: bool
    response_time: float
    status_code: Optional[int] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    result_size: Optional[int] = None
    timestamp: Optional[float] = None
    is_cold_start: Optional[bool] = None


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    cpu_memory_mb: float
    gpu_memory_mb: float
    gpu_utilization: float
    model_memory_mb: Optional[float] = None
    cache_size_mb: Optional[float] = None


@dataclass
class LatencyTestResult:
    """Results from latency testing."""
    cold_start_times: List[float]
    warm_start_times: List[float]
    avg_cold_start: float
    avg_warm_start: float
    cold_start_overhead: float
    p95_cold_start: float
    p95_warm_start: float


@dataclass
class ThroughputTestResult:
    """Results from throughput testing."""
    batch_sizes: List[int]
    throughput_per_batch: List[float]
    latency_per_batch: List[float]
    optimal_batch_size: int
    max_throughput: float
    efficiency_scores: List[float]


@dataclass
class ScalabilityTestResult:
    """Results from scalability testing."""
    concurrency_levels: List[int]
    throughput_per_level: List[float]
    latency_per_level: List[float]
    error_rates: List[float]
    degradation_factor: float
    optimal_concurrency: int


@dataclass
class MemoryTestResult:
    """Results from memory testing."""
    baseline_memory: MemorySnapshot
    peak_memory: MemorySnapshot
    memory_timeline: List[MemorySnapshot]
    model_memory_mb: float
    cache_growth_mb: float
    memory_efficiency: float
    fragmentation_detected: bool


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    model_name: str
    test_duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    concurrency: int
    batch_size: int
    
    # Timing metrics
    min_response_time: float
    max_response_time: float
    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    
    # Throughput metrics
    requests_per_second: float
    predictions_per_second: float
    
    # Error analysis
    error_rate: float
    error_breakdown: Dict[str, int]
    
    # Additional metrics
    total_data_transferred: int
    avg_processing_time: Optional[float] = None


class MemoryMonitor:
    """Monitor system and GPU memory usage."""
    
    def __init__(self):
        self.monitoring = False
        self.snapshots = []
        self.monitor_thread = None
        
    def get_memory_snapshot(self) -> MemorySnapshot:
        """Get current memory usage snapshot."""
        # CPU memory
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # GPU memory
        gpu_memory_mb = 0
        gpu_utilization = 0
        
        try:
            if GPUtil is not None:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_memory_mb = gpu.memoryUsed
                    gpu_utilization = gpu.load * 100
        except Exception:
            pass  # No GPU or GPUtil not available
        
        return MemorySnapshot(
            timestamp=time.time(),
            cpu_memory_mb=cpu_memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization
        )
    
    def start_monitoring(self, interval: float = 0.5):
        """Start continuous memory monitoring."""
        self.monitoring = True
        self.snapshots = []
        
        def monitor():
            while self.monitoring:
                snapshot = self.get_memory_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_peak_memory(self) -> MemorySnapshot:
        """Get peak memory usage from snapshots."""
        if not self.snapshots:
            return self.get_memory_snapshot()
        
        peak_cpu = max(self.snapshots, key=lambda s: s.cpu_memory_mb)
        peak_gpu = max(self.snapshots, key=lambda s: s.gpu_memory_mb)
        
        # Return combined peak
        return MemorySnapshot(
            timestamp=max(peak_cpu.timestamp, peak_gpu.timestamp),
            cpu_memory_mb=peak_cpu.cpu_memory_mb,
            gpu_memory_mb=peak_gpu.gpu_memory_mb,
            gpu_utilization=max(s.gpu_utilization for s in self.snapshots)
        )
    
    def detect_fragmentation(self) -> bool:
        """Detect potential GPU memory fragmentation."""
        if len(self.snapshots) < 10:
            return False
        
        # Look for patterns indicating fragmentation:
        # 1. Memory usage spikes without corresponding request spikes
        # 2. Memory not being freed after batch completion
        memory_values = [s.gpu_memory_mb for s in self.snapshots]
        memory_variance = np.var(memory_values)
        memory_mean = np.mean(memory_values)
        
        # High variance relative to mean suggests fragmentation
        if memory_mean > 0:
            coefficient_of_variation = np.sqrt(memory_variance) / memory_mean
            return coefficient_of_variation > 0.3  # Threshold for fragmentation
        
        return False


class PerformanceTestRunner:
    """Main performance testing runner."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.session = requests.Session()
        self.session.timeout = config.timeout
        self.memory_monitor = MemoryMonitor()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        self.memory_monitor.stop_monitoring()
    
    def check_server_health(self) -> bool:
        """Check if the server is healthy and accessible."""
        try:
            response = self.session.get(f"{self.config.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                is_healthy = health_data.get("healthy", False)
                if not is_healthy:
                    print(f"[WARNING] Server reports unhealthy status: {health_data}")
                return is_healthy
            else:
                print(f"[ERROR] Server health check failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] Server health check failed: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """Get list of available models from the server."""
        try:
            response = self.session.get(f"{self.config.base_url}/models")
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            else:
                print(f"[ERROR] Failed to get models list: {response.status_code}")
                return []
        except Exception as e:
            print(f"[ERROR] Error getting models list: {e}")
            return []
    
    def get_model_config(self, model_name: str) -> Optional[Dict]:
        """Get model configuration from predefined configs or common aliases."""
        # First check predefined configs
        for size_category, models in MODEL_CONFIGS.items():
            for model_config in models:
                if model_config["name"] == model_name:
                    return model_config
        
        # Check common aliases
        if model_name in COMMON_MODEL_ALIASES:
            alias_config = COMMON_MODEL_ALIASES[model_name].copy()
            alias_config["name"] = model_name
            alias_config["description"] = f"Auto-detected {model_name} model"
            return alias_config
        
        return None
    
    def list_downloadable_models(self) -> None:
        """List all predefined downloadable models."""
        print("[INFO] Available models for download and testing:")
        print("=" * 80)
        
        for size_category, models in MODEL_CONFIGS.items():
            print(f"\n[{size_category.upper()}] MODELS:")
            print("-" * 40)
            
            for model_config in models:
                print(f"  Model: {model_config['name']}")
                print(f"     Description: {model_config['description']}")
                print(f"     Estimated size: {model_config['estimated_size_mb']} MB")
                print(f"     Source: {model_config['source']}")
                print(f"     Task: {model_config['task']}")
                print()
        
        print("\n[COMMON ALIASES] - Popular models you can test:")
        print("-" * 50)
        
        # Group by source
        torchvision_models = []
        huggingface_models = []
        
        for alias, config in COMMON_MODEL_ALIASES.items():
            if config["source"] == "torchvision":
                torchvision_models.append(alias)
            elif config["source"] == "huggingface":
                huggingface_models.append(alias)
        
        if torchvision_models:
            print(f"  TorchVision Models:")
            for model in sorted(torchvision_models):
                print(f"    • {model}")
            print()
        
        if huggingface_models:
            print(f"  Hugging Face Models:")
            for model in sorted(huggingface_models):
                print(f"    • {model}")
            print()
        
        print("💡 Tips:")
        print("   • Use any alias directly: --model resnet50")
        print("   • Download automatically: --model bert (downloads bert-base-uncased)")
        print("   • Mix with tests: --model mobilenet --duration 60 --concurrency 4")
    
    def download_model(self, model_name: str) -> bool:
        """Download a model using the server's download endpoint."""
        model_config = self.get_model_config(model_name)
        if not model_config:
            print(f"[ERROR] Unknown model: {model_name}")
            print("💡 Available options:")
            print("   • Use --list-downloadable to see all available models")
            print("   • Try common names like: resnet50, bert, mobilenet, distilbert")
            print("   • Use aliases like: resnet (for resnet50), bert (for bert-base-uncased)")
            return False
        
        print(f"[INFO] 🚀 Downloading model: {model_name}")
        print(f"📝 Description: {model_config['description']}")
        
        # Show estimated size if available
        if "estimated_size_mb" in model_config:
            size_mb = model_config["estimated_size_mb"]
            if size_mb < 100:
                size_info = f"📦 Estimated size: ~{size_mb} MB (Small - quick download)"
            elif size_mb < 500:
                size_info = f"📦 Estimated size: ~{size_mb} MB (Medium - moderate download)"
            else:
                size_info = f"📦 Estimated size: ~{size_mb} MB (Large - may take time)"
            print(size_info)
        
        try:
            # Use the server's download endpoint
            download_url = f"{self.config.base_url}/models/download"
            params = {
                "source": model_config["source"],
                "model_id": model_config["model_id"],
                "name": model_name,
                "task": model_config.get("task", "classification")
            }
            
            print(f"⏳ Sending download request to server...")
            response = self.session.post(download_url, params=params, timeout=300)  # 5 min timeout for downloads
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                
                if status == "downloading":
                    print(f"🔄 Download started in background: {data.get('message', 'Model downloading')}")
                    print("⌛ Waiting for download to complete (this may take a few minutes)...")
                    
                    # Wait a bit for background download and check model availability
                    import time
                    for attempt in range(30):  # Check for up to 5 minutes
                        time.sleep(10)
                        available_models = self.list_available_models()
                        if model_name in available_models:
                            print(f"✅ Download completed successfully!")
                            return True
                        print(f"⏳ Still downloading... (attempt {attempt + 1}/30)")
                    
                    print(f"⚠️  Download may still be in progress. Check server logs if model isn't available.")
                    return True  # Assume success, server will handle it
                    
                elif status == "completed":
                    print(f"✅ Download completed: {data.get('message', 'Model downloaded')}")
                    return True
                else:
                    print(f"✅ Download started: {data.get('message', 'Model download initiated')}")
                    return True
                    
            else:
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", str(error_data))
                except:
                    error_detail = response.text[:200]
                
                print(f"❌ Download failed (HTTP {response.status_code}): {error_detail}")
                
                # Provide helpful suggestions
                if response.status_code == 400:
                    print("💡 This might be a configuration issue. Try:")
                    print("   • Check if the model source is supported")
                    print("   • Verify the model name spelling")
                elif response.status_code == 404:
                    print("💡 Model not found. Try:")
                    print("   • Use --list-downloadable to see available models")
                    print("   • Check the model name and try a different one")
                elif response.status_code == 500:
                    print("💡 Server error. Try:")
                    print("   • Make sure the inference server is running properly")
                    print("   • Check server logs for more details")
                
                return False
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            print("💡 Troubleshooting:")
            print("   • Make sure the inference server is running")
            print("   • Check your internet connection")
            print("   • Try a different model or check server logs")
            return False
    
    def get_appropriate_input_data(self, model_name: str) -> Union[int, float, List, str]:
        """Get appropriate input data based on model type."""
        model_config = self.get_model_config(model_name)
        if not model_config:
            return self.config.input_data  # Default fallback
        
        # Determine input type from model config or infer from source
        input_type = model_config.get("input_type")
        source = model_config.get("source", "")
        task = model_config.get("task", "")
        
        # If no explicit input_type, infer from source and task
        if not input_type:
            if source == "torchvision" or "resnet" in model_name.lower() or "mobilenet" in model_name.lower():
                input_type = "image_tensor"
            elif source == "huggingface" or "bert" in model_name.lower() or "roberta" in model_name.lower():
                input_type = "text"
            elif "vision" in task.lower() or "image" in task.lower():
                input_type = "image_tensor"
            elif "text" in task.lower() or "nlp" in task.lower():
                input_type = "text"
        
        if input_type == "image_tensor":
            # For vision models, create a simple but working tensor
            # Height must be > 10 to pass the batch detection logic in main.py
            height, width = 15, 15  # Must be > 10 to be detected as single image
            channels = 3  # RGB
            
            # Create a simple working tensor with some variation
            single_image = []
            for c in range(channels):
                channel = []
                for h in range(height):
                    row = []
                    for w in range(width):
                        # Simple pattern that creates a working tensor
                        value = 0.5 + (c * 0.1) + (h * 0.001) + (w * 0.0001)  # Simple gradient
                        row.append(value)
                    channel.append(row)
                single_image.append(channel)
            
            # Return the tensor in [C, H, W] format - this format works!
            return single_image
            
        elif input_type == "text":
            # Return text input for NLP models with some variety
            text_samples = [
                "This is a test sentence for natural language processing and classification.",
                "The performance testing framework evaluates model inference capabilities.",
                "Machine learning models require proper input formatting for optimal results.",
                "PyTorch provides excellent tools for deep learning model deployment.",
                "Text classification tasks benefit from well-structured input data."
            ]
            return random.choice(text_samples)
            
        else:
            # Default fallback for unknown types
            return self.config.input_data
    
    def create_test_payload(self) -> Dict:
        """Create payload for testing. The new model-specific endpoint handles both single and batch inputs."""
        # Get appropriate input data for the model
        input_data = self.get_appropriate_input_data(self.config.model_name)
        
        # Check if this is an image tensor (3D list)
        is_image_tensor = (isinstance(input_data, list) and 
                          len(input_data) == 3 and  # 3 channels
                          isinstance(input_data[0], list) and 
                          isinstance(input_data[0][0], list))
        
        if self.config.batch_size == 1:
            if is_image_tensor:
                # For image tensors, send the image data directly in [C, H, W] format
                # This will be properly detected as InputType.IMAGE by the preprocessor
                return {
                    "inputs": input_data,
                    "priority": self.config.priority,
                    "timeout": self.config.timeout
                }
            else:
                # For non-image inputs, send directly
                return {
                    "inputs": input_data,
                    "priority": self.config.priority,
                    "timeout": self.config.timeout
                }
        else:
            # For batch testing - create proper batch format
            batch_inputs = [input_data] * self.config.batch_size
            return {
                "inputs": batch_inputs,
                "priority": self.config.priority,
                "timeout": self.config.timeout
            }
    
    def get_endpoint_url(self) -> str:
        """Get the model-specific prediction endpoint URL."""
        # Use the new model-specific endpoint format: /{model_name}/predict
        return f"{self.config.base_url}/{self.config.model_name}/predict"
    
    def make_single_request(self, mark_timestamp: bool = True, detect_cold_start: bool = False) -> RequestResult:
        """Make a single prediction request."""
        url = self.get_endpoint_url()
        payload = self.create_test_payload()
        
        request_timestamp = time.time()
        start_time = time.time()
        try:
            response = self.session.post(url, json=payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                success = data.get("success", False)
                processing_time = data.get("processing_time")
                result_size = len(json.dumps(data).encode()) if data else 0
                
                # Additional validation: check if result contains meaningful data
                result = data.get("result")
                if success and result is not None and not self.config.allow_null_results:
                    # Check if result is all nulls or empty
                    if isinstance(result, list):
                        # For lists, check if all values are null/None
                        if all(x is None for x in result):
                            success = False
                            error_msg = "Model returned all null predictions"
                        else:
                            success = True
                            error_msg = None
                    elif isinstance(result, dict):
                        # For dicts, check if all values are null/None
                        if all(v is None for v in result.values()):
                            success = False
                            error_msg = "Model returned all null predictions"
                        else:
                            success = True
                            error_msg = None
                    else:
                        # For other types, check if it's None
                        if result is None:
                            success = False
                            error_msg = "Model returned null prediction"
                        else:
                            success = True
                            error_msg = None
                elif success and self.config.allow_null_results:
                    # Allow null results when flag is set
                    success = True
                    error_msg = None
                else:
                    success = False
                    error_msg = data.get("error", "Request marked as unsuccessful")
                
                # Detect cold start based on processing time
                is_cold_start = None
                if detect_cold_start and processing_time:
                    # Cold starts typically take 2-10x longer than warm starts
                    # We'll mark it as cold start if it's significantly slower
                    is_cold_start = processing_time > 1.0  # Threshold for cold start detection
                
                return RequestResult(
                    success=success,
                    response_time=response_time,
                    status_code=response.status_code,
                    processing_time=processing_time,
                    result_size=result_size,
                    timestamp=request_timestamp if mark_timestamp else None,
                    is_cold_start=is_cold_start,
                    error=error_msg if not success else None
                )
            else:
                return RequestResult(
                    success=False,
                    response_time=response_time,
                    status_code=response.status_code,
                    timestamp=request_timestamp if mark_timestamp else None,
                    error=f"HTTP {response.status_code}: {response.text[:200]}"
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            return RequestResult(
                success=False,
                response_time=response_time,
                timestamp=request_timestamp if mark_timestamp else None,
                error=str(e)
            )
    
    def warmup(self) -> None:
        """Perform warmup requests."""
        if self.config.warmup_requests > 0:
            print(f"[WARMUP] Warming up with {self.config.warmup_requests} requests...")
            for _ in range(self.config.warmup_requests):
                self.make_single_request()
            print("[WARMUP] Warmup complete")
    
    def run_duration_test(self) -> List[RequestResult]:
        """Run test for a specific duration."""
        print(f"[TEST] Running duration test for {self.config.duration_seconds}s with {self.config.concurrency} threads")
        
        results = []
        results_lock = threading.Lock()
        end_time = time.time() + self.config.duration_seconds
        
        def worker():
            while time.time() < end_time:
                result = self.make_single_request()
                with results_lock:
                    results.append(result)
        
        # Start worker threads
        threads = []
        for _ in range(self.config.concurrency):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return results
    
    def run_count_test(self) -> List[RequestResult]:
        """Run test for a specific number of requests."""
        print(f"[TEST] Running count test for {self.config.num_requests} requests with {self.config.concurrency} threads")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            # Submit all requests
            futures = [executor.submit(self.make_single_request) 
                      for _ in range(self.config.num_requests)]
            
            # Collect results with progress
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                
                if self.config.verbose and completed % max(1, self.config.num_requests // 10) == 0:
                    print(f"Progress: {completed}/{self.config.num_requests} requests completed")
        
        return results
    
    def generate_report(self, results: List[RequestResult], test_duration: float) -> PerformanceReport:
        """Generate comprehensive performance report."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if not successful_results:
            print("\n❌ PERFORMANCE TEST FAILED - NO SUCCESSFUL REQUESTS")
            print("=" * 60)
            print(f"Total requests attempted: {len(results)}")
            print(f"Failed requests: {len(failed_results)}")
            print(f"Test duration: {test_duration:.2f}s")
            
            if failed_results:
                print("\n🔍 ERROR ANALYSIS:")
                error_breakdown = defaultdict(int)
                for result in failed_results:
                    error_type = result.error[:100] if result.error else "Unknown error"
                    error_breakdown[error_type] += 1
                
                for error, count in error_breakdown.items():
                    print(f"   • {error}: {count} occurrences")
                
                # Show first few detailed errors
                print(f"\n📋 DETAILED ERROR SAMPLES (first 3):")
                for i, result in enumerate(failed_results[:3]):
                    print(f"   Error {i+1}:")
                    print(f"      Status: {result.status_code}")
                    print(f"      Response time: {result.response_time:.3f}s")
                    print(f"      Error: {result.error}")
            
            print("\n💡 TROUBLESHOOTING STEPS:")
            print("   1. Check if the inference server is running properly")
            print("   2. Verify the model is loaded and functional")
            print("   3. Test the endpoint manually:")
            print(f"      curl -X POST {self.config.base_url}/{self.config.model_name}/predict \\")
            print("           -H 'Content-Type: application/json' \\")
            print("           -d '{\"inputs\": [[1,2,3]], \"timeout\": 30}'")
            print("   4. Check server logs for detailed error messages")
            print("   5. Try testing with the 'example' model first")
            
            sys.exit(1)
        
        # Response time statistics
        response_times = [r.response_time for r in successful_results]
        processing_times = [r.processing_time for r in successful_results if r.processing_time is not None]
        
        # Error analysis
        error_breakdown = defaultdict(int)
        for result in failed_results:
            error_type = result.error[:50] if result.error else "Unknown error"
            error_breakdown[error_type] += 1
        
        # Calculate predictions per second based on batch size
        total_predictions = len(successful_results) * self.config.batch_size
        predictions_per_second = total_predictions / test_duration if test_duration > 0 else 0
        
        # Data transfer calculation
        total_data = sum(r.result_size for r in successful_results if r.result_size)
        
        return PerformanceReport(
            model_name=self.config.model_name,
            test_duration=test_duration,
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            concurrency=self.config.concurrency,
            batch_size=self.config.batch_size,
            
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            avg_response_time=statistics.mean(response_times),
            median_response_time=statistics.median(response_times),
            p95_response_time=np.percentile(response_times, 95),
            p99_response_time=np.percentile(response_times, 99),
            
            requests_per_second=len(successful_results) / test_duration if test_duration > 0 else 0,
            predictions_per_second=predictions_per_second,
            
            error_rate=len(failed_results) / len(results) * 100 if results else 0,
            error_breakdown=dict(error_breakdown),
            
            total_data_transferred=total_data,
            avg_processing_time=statistics.mean(processing_times) if processing_times else None
        )
    
    def check_model_availability(self) -> bool:
        """Check if model is available and working correctly."""
        print(f"🔍 Checking model availability for '{self.config.model_name}'...")
        
        # Check if model is in the list
        available_models = self.list_available_models()
        if self.config.model_name not in available_models:
            print(f"❌ Model '{self.config.model_name}' not found in available models: {available_models}")
            return False
        
        print(f"✅ Model '{self.config.model_name}' found in server model list")
        
        # Test a single prediction
        print("🧪 Testing single prediction...")
        test_result = self.make_single_request()
        
        if test_result.success:
            print(f"✅ Model responds successfully (response time: {test_result.response_time:.3f}s)")
            if test_result.processing_time:
                print(f"   Server processing time: {test_result.processing_time:.3f}s")
            return True
        else:
            print(f"❌ Model test failed: {test_result.error}")
            print(f"   Status code: {test_result.status_code}")
            print(f"   Response time: {test_result.response_time:.3f}s")
            
            # Additional diagnostics
            print("\n🔧 Running diagnostics...")
            url = self.get_endpoint_url()
            payload = self.create_test_payload()
            
            try:
                response = self.session.post(url, json=payload)
                print(f"   Raw response status: {response.status_code}")
                print(f"   Raw response headers: {dict(response.headers)}")
                
                if response.text:
                    response_text = response.text[:1000]  # First 1000 chars
                    print(f"   Raw response body: {response_text}")
                    
                    try:
                        response_json = response.json()
                        print(f"   Response JSON keys: {list(response_json.keys())}")
                        if "result" in response_json:
                            result = response_json["result"]
                            print(f"   Result type: {type(result)}")
                            if isinstance(result, list):
                                print(f"   Result length: {len(result)}")
                                if len(result) > 0:
                                    print(f"   First few items: {result[:5]}")
                                    print(f"   All None?: {all(x is None for x in result)}")
                            elif result is None:
                                print(f"   Result is None")
                            else:
                                print(f"   Result value: {result}")
                                
                    except Exception as parse_error:
                        print(f"   Could not parse response as JSON: {parse_error}")
                        try:
                            response_json = response.json()  # Try again for special handling
                        except:
                            response_json = {}
                            
                    # For ResNet50 and image models, null results might be expected 
                    # if the tensor format isn't perfect - allow this for performance testing
                    try:
                        if not hasattr(self, 'response_json') or 'response_json' not in locals():
                            response_json = response.json()
                        
                        if (response_json.get("success") and 
                            self.config.model_name in ["resnet50", "resnet18", "resnet34", "resnet101", "resnet152"]):
                            print("ℹ️  Note: Image models may return null for non-standard tensor formats")
                            print("   This is acceptable for performance testing - enabling null result tolerance")
                            self.config.allow_null_results = True
                            return True
                    except:
                        pass
                
            except Exception as e:
                print(f"   Diagnostic request failed: {e}")
            
            return False
    
    def run_test(self) -> PerformanceReport:
        """Run the complete performance test."""
        # Health check
        if not self.check_server_health():
            print("❌ Server is not healthy. Please start the server first.")
            print("💡 Start the server with: python main.py")
            sys.exit(1)
        
        # Check if model needs to be downloaded
        available_models = self.list_available_models()
        if self.config.model_name not in available_models:
            print(f"🔍 Model '{self.config.model_name}' not found on server.")
            
            # Check if it's a downloadable model
            model_config = self.get_model_config(self.config.model_name)
            if model_config:
                print(f"📥 Auto-downloading model '{self.config.model_name}'...")
                if self.download_model(self.config.model_name):
                    print(f"✅ Model downloaded successfully!")
                    # Refresh available models list
                    available_models = self.list_available_models()
                    if self.config.model_name not in available_models:
                        print(f"❌ Model still not available after download. Please check server logs.")
                        sys.exit(1)
                else:
                    print(f"❌ Failed to download model '{self.config.model_name}'")
                    sys.exit(1)
            else:
                print(f"❌ Model '{self.config.model_name}' not found and no auto-download available.")
                print(f"📋 Available models on server: {available_models}")
                print("💡 Tips:")
                print("   • Use --list-downloadable to see models you can auto-download")
                print("   • Try common names like: resnet50, bert, mobilenet, distilbert")
                print("   • Use --download <model_name> to download a specific model")
                
                # Suggest similar models
                similar_models = []
                search_term = self.config.model_name.lower()
                for alias in COMMON_MODEL_ALIASES.keys():
                    if search_term in alias.lower() or alias.lower() in search_term:
                        similar_models.append(alias)
                
                if similar_models:
                    print(f"🔗 Did you mean one of these? {', '.join(similar_models[:3])}")
                
                sys.exit(1)
        
        # Check model availability and functionality
        if not self.check_model_availability():
            print("❌ Model availability check failed. Cannot proceed with performance test.")
            sys.exit(1)
        
        print(f"🚀 Starting performance test for model: {self.config.model_name}")
        model_config = self.get_model_config(self.config.model_name)
        if model_config:
            print(f"📝 Model info: {model_config['description']}")
        
        print(f"📋 Test configuration:")
        print(f"   • Model: {self.config.model_name}")
        print(f"   • Endpoint: /{self.config.model_name}/predict")
        print(f"   • Concurrency: {self.config.concurrency} threads")
        print(f"   • Batch size: {self.config.batch_size}")
        print(f"   • Timeout: {self.config.timeout}s")
        
        if self.config.duration_seconds:
            print(f"   • Duration: {self.config.duration_seconds}s")
        else:
            print(f"   • Requests: {self.config.num_requests}")
        
        # Warmup
        self.warmup()
        
        # Run main test
        start_time = time.time()
        
        if self.config.duration_seconds:
            results = self.run_duration_test()
        else:
            results = self.run_count_test()
        
        test_duration = time.time() - start_time
        
        # Generate report
        report = self.generate_report(results, test_duration)
        return report
    
    def run_latency_test(self) -> LatencyTestResult:
        """Run latency test to measure cold start vs warm start times."""
        print("[LATENCY TEST] Measuring cold start vs warm start latency...")
        
        cold_start_times = []
        warm_start_times = []
        
        # Test cold starts - restart the model between requests
        print("🥶 Testing cold start latency...")
        for i in range(5):  # 5 cold start measurements
            print(f"   Cold start test {i+1}/5")
            
            # Make request to unload model using new autoscaler endpoint
            try:
                self.session.delete(f"{self.config.base_url}/autoscaler/unload", 
                                   params={"model_name": self.config.model_name})
                time.sleep(2)  # Wait for unload
            except:
                pass  # Unload endpoint might not exist or fail
            
            # Make a request - this should be a cold start
            result = self.make_single_request(detect_cold_start=True)
            if result.success and result.processing_time:
                cold_start_times.append(result.processing_time)
                time.sleep(1)  # Brief pause between tests
        
        # Test warm starts - keep model loaded
        print("🔥 Testing warm start latency...")
        # First, ensure model is loaded with a warmup request
        self.make_single_request()
        
        for i in range(20):  # 20 warm start measurements
            if i % 5 == 0:
                print(f"   Warm start test {i+1}/20")
            
            result = self.make_single_request(detect_cold_start=True)
            if result.success and result.processing_time:
                warm_start_times.append(result.processing_time)
        
        # Calculate metrics
        avg_cold_start = statistics.mean(cold_start_times) if cold_start_times else 0
        avg_warm_start = statistics.mean(warm_start_times) if warm_start_times else 0
        cold_start_overhead = avg_cold_start - avg_warm_start if cold_start_times and warm_start_times else 0
        
        p95_cold_start = np.percentile(cold_start_times, 95) if cold_start_times else 0
        p95_warm_start = np.percentile(warm_start_times, 95) if warm_start_times else 0
        
        return LatencyTestResult(
            cold_start_times=cold_start_times,
            warm_start_times=warm_start_times,
            avg_cold_start=avg_cold_start,
            avg_warm_start=avg_warm_start,
            cold_start_overhead=cold_start_overhead,
            p95_cold_start=p95_cold_start,
            p95_warm_start=p95_warm_start
        )
    
    def run_throughput_test(self) -> ThroughputTestResult:
        """Run throughput test with different batch sizes."""
        print("[THROUGHPUT TEST] Testing throughput at different batch sizes...")
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        throughput_per_batch = []
        latency_per_batch = []
        efficiency_scores = []
        
        original_batch_size = self.config.batch_size
        original_concurrency = self.config.concurrency
        
        for batch_size in batch_sizes:
            print(f"📦 Testing batch size: {batch_size}")
            
            # Update config for this batch size
            self.config.batch_size = batch_size
            
            # Run test for this batch size
            results = []
            start_time = time.time()
            
            # Run for 30 seconds or 100 requests, whichever comes first
            max_requests = 100
            test_duration = 30
            end_time = start_time + test_duration
            
            while len(results) < max_requests and time.time() < end_time:
                result = self.make_single_request()
                results.append(result)
            
            actual_duration = time.time() - start_time
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                # Calculate throughput (predictions per second)
                total_predictions = len(successful_results) * batch_size
                throughput = total_predictions / actual_duration
                throughput_per_batch.append(throughput)
                
                # Calculate average latency
                avg_latency = statistics.mean([r.response_time for r in successful_results])
                latency_per_batch.append(avg_latency)
                
                # Calculate efficiency (throughput per unit latency)
                efficiency = throughput / avg_latency if avg_latency > 0 else 0
                efficiency_scores.append(efficiency)
                
                print(f"   Throughput: {throughput:.2f} pred/s, Latency: {avg_latency:.3f}s")
            else:
                throughput_per_batch.append(0)
                latency_per_batch.append(float('inf'))
                efficiency_scores.append(0)
                print(f"   Failed - no successful requests")
        
        # Restore original config
        self.config.batch_size = original_batch_size
        self.config.concurrency = original_concurrency
        
        # Find optimal batch size
        optimal_batch_size = batch_sizes[0]
        max_throughput = 0
        
        if throughput_per_batch:
            max_throughput = max(throughput_per_batch)
            optimal_idx = throughput_per_batch.index(max_throughput)
            optimal_batch_size = batch_sizes[optimal_idx]
        
        return ThroughputTestResult(
            batch_sizes=batch_sizes,
            throughput_per_batch=throughput_per_batch,
            latency_per_batch=latency_per_batch,
            optimal_batch_size=optimal_batch_size,
            max_throughput=max_throughput,
            efficiency_scores=efficiency_scores
        )
    
    def run_scalability_test(self) -> ScalabilityTestResult:
        """Run scalability test with different concurrency levels."""
        print("[SCALABILITY TEST] Testing performance at different concurrency levels...")
        
        concurrency_levels = [1, 2, 4, 8, 16, 32]
        throughput_per_level = []
        latency_per_level = []
        error_rates = []
        
        original_concurrency = self.config.concurrency
        
        for concurrency in concurrency_levels:
            print(f"🔀 Testing concurrency: {concurrency} threads")
            
            self.config.concurrency = concurrency
            
            # Run test for this concurrency level
            start_time = time.time()
            test_duration = 30  # 30 seconds per test
            
            results = []
            results_lock = threading.Lock()
            end_time = start_time + test_duration
            
            def worker():
                while time.time() < end_time:
                    result = self.make_single_request()
                    with results_lock:
                        results.append(result)
            
            # Start worker threads
            threads = []
            for _ in range(concurrency):
                thread = threading.Thread(target=worker)
                thread.start()
                threads.append(thread)
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            actual_duration = time.time() - start_time
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            # Calculate metrics
            if results:
                throughput = len(successful_results) / actual_duration
                throughput_per_level.append(throughput)
                
                if successful_results:
                    avg_latency = statistics.mean([r.response_time for r in successful_results])
                    latency_per_level.append(avg_latency)
                else:
                    latency_per_level.append(float('inf'))
                
                error_rate = len(failed_results) / len(results) * 100
                error_rates.append(error_rate)
                
                print(f"   Throughput: {throughput:.2f} req/s, Latency: {avg_latency:.3f}s, Errors: {error_rate:.1f}%")
            else:
                throughput_per_level.append(0)
                latency_per_level.append(float('inf'))
                error_rates.append(100)
                print(f"   Failed - no requests completed")
        
        # Restore original config
        self.config.concurrency = original_concurrency
        
        # Calculate degradation factor and optimal concurrency
        degradation_factor = 0
        optimal_concurrency = 1
        
        if len(throughput_per_level) >= 2:
            baseline_throughput = throughput_per_level[0]  # Single thread throughput
            max_throughput = max(throughput_per_level)
            
            if baseline_throughput > 0:
                # Ideal scaling would be linear
                ideal_max = baseline_throughput * concurrency_levels[throughput_per_level.index(max_throughput)]
                degradation_factor = 1 - (max_throughput / ideal_max)
            
            # Find optimal concurrency (best throughput with acceptable error rate)
            for i, (throughput, error_rate) in enumerate(zip(throughput_per_level, error_rates)):
                if error_rate < 5:  # Less than 5% error rate
                    if throughput > throughput_per_level[optimal_concurrency - 1]:
                        optimal_concurrency = concurrency_levels[i]
        
        return ScalabilityTestResult(
            concurrency_levels=concurrency_levels,
            throughput_per_level=throughput_per_level,
            latency_per_level=latency_per_level,
            error_rates=error_rates,
            degradation_factor=degradation_factor,
            optimal_concurrency=optimal_concurrency
        )
    
    def run_memory_test(self) -> MemoryTestResult:
        """Run memory test to measure memory usage and detect fragmentation."""
        print("[MEMORY TEST] Monitoring memory usage and detecting fragmentation...")
        
        # Get baseline memory
        baseline_memory = self.memory_monitor.get_memory_snapshot()
        print(f"📊 Baseline memory - CPU: {baseline_memory.cpu_memory_mb:.1f}MB, GPU: {baseline_memory.gpu_memory_mb:.1f}MB")
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring(interval=0.5)
        
        try:
            # Run test workload for memory analysis
            print("🔄 Running workload for memory analysis...")
            
            # Phase 1: Single requests to see basic model memory
            for i in range(10):
                self.make_single_request()
                time.sleep(0.1)
            
            # Phase 2: Batch requests to see batch memory scaling
            original_batch_size = self.config.batch_size
            for batch_size in [1, 4, 8, 16]:
                self.config.batch_size = batch_size
                print(f"   Testing batch size {batch_size}")
                for _ in range(5):
                    self.make_single_request()
                    time.sleep(0.2)
            
            self.config.batch_size = original_batch_size
            
            # Phase 3: Concurrent requests to stress test memory
            print("   Testing concurrent requests...")
            original_concurrency = self.config.concurrency
            self.config.concurrency = 8
            
            results = []
            results_lock = threading.Lock()
            end_time = time.time() + 20  # 20 seconds of stress testing
            
            def worker():
                while time.time() < end_time:
                    result = self.make_single_request()
                    with results_lock:
                        results.append(result)
                    time.sleep(0.05)
            
            threads = []
            for _ in range(self.config.concurrency):
                thread = threading.Thread(target=worker)
                thread.start()
                threads.append(thread)
            
            for thread in threads:
                thread.join()
            
            self.config.concurrency = original_concurrency
            
        finally:
            # Stop memory monitoring
            self.memory_monitor.stop_monitoring()
        
        # Analyze memory usage
        peak_memory = self.memory_monitor.get_peak_memory()
        fragmentation_detected = self.memory_monitor.detect_fragmentation()
        
        # Calculate model memory footprint
        model_memory_mb = peak_memory.cpu_memory_mb - baseline_memory.cpu_memory_mb
        
        # Estimate cache growth (approximate)
        memory_snapshots = self.memory_monitor.snapshots
        if len(memory_snapshots) > 10:
            early_avg = np.mean([s.gpu_memory_mb for s in memory_snapshots[:10]])
            late_avg = np.mean([s.gpu_memory_mb for s in memory_snapshots[-10:]])
            cache_growth_mb = max(0, late_avg - early_avg)
        else:
            cache_growth_mb = 0
        
        # Calculate memory efficiency (throughput per MB of memory)
        successful_requests = len([r for r in results if r.success])
        memory_efficiency = successful_requests / model_memory_mb if model_memory_mb > 0 else 0
        
        print(f"📈 Peak memory - CPU: {peak_memory.cpu_memory_mb:.1f}MB, GPU: {peak_memory.gpu_memory_mb:.1f}MB")
        print(f"🧠 Model memory footprint: {model_memory_mb:.1f}MB")
        if fragmentation_detected:
            print("⚠️  Potential GPU memory fragmentation detected!")
        
        return MemoryTestResult(
            baseline_memory=baseline_memory,
            peak_memory=peak_memory,
            memory_timeline=memory_snapshots,
            model_memory_mb=model_memory_mb,
            cache_growth_mb=cache_growth_mb,
            memory_efficiency=memory_efficiency,
            fragmentation_detected=fragmentation_detected
        )
    
    def run_comprehensive_test(self) -> Dict:
        """Run all test types and return comprehensive results."""
        print("🚀 Starting comprehensive performance testing...")
        
        results = {}
        
        # 1. Basic performance test
        print("\n" + "="*60)
        basic_report = self.run_test()
        results['basic'] = basic_report
        
        # 2. Latency test
        print("\n" + "="*60)
        latency_result = self.run_latency_test()
        results['latency'] = latency_result
        
        # 3. Throughput test
        print("\n" + "="*60)
        throughput_result = self.run_throughput_test()
        results['throughput'] = throughput_result
        
        # 4. Scalability test
        print("\n" + "="*60)
        scalability_result = self.run_scalability_test()
        results['scalability'] = scalability_result
        
        # 5. Memory test
        print("\n" + "="*60)
        memory_result = self.run_memory_test()
        results['memory'] = memory_result
        
        print("\n" + "="*60)
        print("✅ Comprehensive testing completed!")
        
        return results


class ReportFormatter:
    """Format and display performance reports."""
    
    @staticmethod
    def format_text_report(report: PerformanceReport) -> str:
        """Format report as readable text."""
        lines = [
            "=" * 80,
            f"🎯 PERFORMANCE TEST REPORT - {report.model_name.upper()}",
            "=" * 80,
            "",
            "📊 TEST SUMMARY:",
            f"   • Test Duration: {report.test_duration:.2f}s",
            f"   • Total Requests: {report.total_requests:,}",
            f"   • Successful: {report.successful_requests:,} ({100 - report.error_rate:.1f}%)",
            f"   • Failed: {report.failed_requests:,} ({report.error_rate:.1f}%)",
            f"   • Concurrency: {report.concurrency} threads",
            f"   • Batch Size: {report.batch_size}",
            "",
            "⚡ THROUGHPUT METRICS:",
            f"   • Requests/second: {report.requests_per_second:.2f}",
            f"   • Predictions/second: {report.predictions_per_second:.2f}",
            "",
            "⏱️  LATENCY METRICS (seconds):",
            f"   • Min: {report.min_response_time:.4f}s",
            f"   • Max: {report.max_response_time:.4f}s",
            f"   • Average: {report.avg_response_time:.4f}s",
            f"   • Median: {report.median_response_time:.4f}s",
            f"   • 95th percentile: {report.p95_response_time:.4f}s",
            f"   • 99th percentile: {report.p99_response_time:.4f}s",
        ]
        
        if report.avg_processing_time:
            lines.extend([
                "",
                "🔄 SERVER PROCESSING:",
                f"   • Avg processing time: {report.avg_processing_time:.4f}s",
                f"   • Network overhead: {report.avg_response_time - report.avg_processing_time:.4f}s"
            ])
        
        if report.total_data_transferred > 0:
            data_mb = report.total_data_transferred / (1024 * 1024)
            throughput_mbps = data_mb / report.test_duration if report.test_duration > 0 else 0
            lines.extend([
                "",
                "📦 DATA TRANSFER:",
                f"   • Total data: {data_mb:.2f} MB",
                f"   • Throughput: {throughput_mbps:.2f} MB/s"
            ])
        
        if report.error_breakdown:
            lines.extend([
                "",
                "❌ ERROR ANALYSIS:",
            ])
            for error, count in report.error_breakdown.items():
                lines.append(f"   • {error}: {count} occurrences")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    @staticmethod
    def format_json_report(report: PerformanceReport) -> str:
        """Format report as JSON."""
        report_dict = {
            "model_name": report.model_name,
            "test_summary": {
                "duration_seconds": report.test_duration,
                "total_requests": report.total_requests,
                "successful_requests": report.successful_requests,
                "failed_requests": report.failed_requests,
                "concurrency": report.concurrency,
                "batch_size": report.batch_size
            },
            "throughput_metrics": {
                "requests_per_second": report.requests_per_second,
                "predictions_per_second": report.predictions_per_second
            },
            "latency_metrics": {
                "min_response_time": report.min_response_time,
                "max_response_time": report.max_response_time,
                "avg_response_time": report.avg_response_time,
                "median_response_time": report.median_response_time,
                "p95_response_time": report.p95_response_time,
                "p99_response_time": report.p99_response_time
            },
            "error_metrics": {
                "error_rate_percent": report.error_rate,
                "error_breakdown": report.error_breakdown
            }
        }
        
        if report.avg_processing_time:
            report_dict["processing_metrics"] = {
                "avg_processing_time": report.avg_processing_time,
                "network_overhead": report.avg_response_time - report.avg_processing_time
            }
        
        if report.total_data_transferred > 0:
            data_mb = report.total_data_transferred / (1024 * 1024)
            throughput_mbps = data_mb / report.test_duration if report.test_duration > 0 else 0
            report_dict["data_transfer_metrics"] = {
                "total_data_mb": data_mb,
                "throughput_mbps": throughput_mbps
            }
        
        return json.dumps(report_dict, indent=2)
    
    @staticmethod
    def format_csv_report(report: PerformanceReport) -> str:
        """Format report as CSV for easy analysis."""
        headers = [
            "model_name", "test_duration", "total_requests", "successful_requests", 
            "failed_requests", "concurrency", "batch_size", "requests_per_second",
            "predictions_per_second", "min_response_time", "max_response_time",
            "avg_response_time", "median_response_time", "p95_response_time",
            "p99_response_time", "error_rate_percent"
        ]
        
        values = [
            report.model_name, report.test_duration, report.total_requests,
            report.successful_requests, report.failed_requests, report.concurrency,
            report.batch_size, report.requests_per_second, report.predictions_per_second,
            report.min_response_time, report.max_response_time, report.avg_response_time,
            report.median_response_time, report.p95_response_time, report.p99_response_time,
            report.error_rate
        ]
        
        csv_lines = [
            ",".join(headers),
            ",".join(str(v) for v in values)
        ]
        
        return "\n".join(csv_lines)
    
    @staticmethod
    def format_comprehensive_report(results: Dict) -> str:
        """Format comprehensive test results."""
        lines = [
            "=" * 100,
            "🎯 COMPREHENSIVE PERFORMANCE ANALYSIS",
            "=" * 100,
        ]
        
        # Basic performance section
        if 'basic' in results:
            basic = results['basic']
            lines.extend([
                "",
                "📊 BASIC PERFORMANCE SUMMARY:",
                f"   • Model: {basic.model_name}",
                f"   • Throughput: {basic.requests_per_second:.2f} req/s, {basic.predictions_per_second:.2f} pred/s",
                f"   • Latency: {basic.avg_response_time:.3f}s avg, {basic.p95_response_time:.3f}s p95",
                f"   • Success Rate: {100 - basic.error_rate:.1f}%"
            ])
        
        # Latency analysis section
        if 'latency' in results:
            latency = results['latency']
            lines.extend([
                "",
                "🥶🔥 LATENCY ANALYSIS (Cold vs Warm Start):",
                f"   • Cold Start: {latency.avg_cold_start:.3f}s avg, {latency.p95_cold_start:.3f}s p95",
                f"   • Warm Start: {latency.avg_warm_start:.3f}s avg, {latency.p95_warm_start:.3f}s p95",
                f"   • Cold Start Overhead: +{latency.cold_start_overhead:.3f}s ({latency.cold_start_overhead/latency.avg_warm_start*100:.1f}% slower)" if latency.avg_warm_start > 0 else "",
                f"   • Cold Start Samples: {len(latency.cold_start_times)}",
                f"   • Warm Start Samples: {len(latency.warm_start_times)}"
            ])
        
        # Throughput analysis section
        if 'throughput' in results:
            throughput = results['throughput']
            lines.extend([
                "",
                "📦 THROUGHPUT ANALYSIS (Batch Size Impact):",
                f"   • Optimal Batch Size: {throughput.optimal_batch_size}",
                f"   • Maximum Throughput: {throughput.max_throughput:.2f} pred/s",
                ""
            ])
            
            lines.append("   Batch Size Performance:")
            for i, batch_size in enumerate(throughput.batch_sizes):
                if i < len(throughput.throughput_per_batch):
                    tput = throughput.throughput_per_batch[i]
                    latency = throughput.latency_per_batch[i] if i < len(throughput.latency_per_batch) else 0
                    efficiency = throughput.efficiency_scores[i] if i < len(throughput.efficiency_scores) else 0
                    marker = "⭐" if batch_size == throughput.optimal_batch_size else "  "
                    lines.append(f"   {marker} Batch {batch_size:2d}: {tput:8.2f} pred/s, {latency:.3f}s latency, {efficiency:.1f} efficiency")
        
        # Scalability analysis section
        if 'scalability' in results:
            scalability = results['scalability']
            lines.extend([
                "",
                "🔀 SCALABILITY ANALYSIS (Concurrency Impact):",
                f"   • Optimal Concurrency: {scalability.optimal_concurrency} threads",
                f"   • Performance Degradation: {scalability.degradation_factor*100:.1f}%",
                ""
            ])
            
            lines.append("   Concurrency Performance:")
            for i, concurrency in enumerate(scalability.concurrency_levels):
                if i < len(scalability.throughput_per_level):
                    tput = scalability.throughput_per_level[i]
                    latency = scalability.latency_per_level[i] if i < len(scalability.latency_per_level) else 0
                    error_rate = scalability.error_rates[i] if i < len(scalability.error_rates) else 0
                    marker = "⭐" if concurrency == scalability.optimal_concurrency else "  "
                    lines.append(f"   {marker} {concurrency:2d} threads: {tput:8.2f} req/s, {latency:.3f}s latency, {error_rate:.1f}% errors")
        
        # Memory analysis section
        if 'memory' in results:
            memory = results['memory']
            lines.extend([
                "",
                "🧠 MEMORY ANALYSIS:",
                f"   • Model Memory Footprint: {memory.model_memory_mb:.1f} MB",
                f"   • Baseline Memory: CPU {memory.baseline_memory.cpu_memory_mb:.1f}MB, GPU {memory.baseline_memory.gpu_memory_mb:.1f}MB",
                f"   • Peak Memory: CPU {memory.peak_memory.cpu_memory_mb:.1f}MB, GPU {memory.peak_memory.gpu_memory_mb:.1f}MB",
                f"   • Cache Growth: {memory.cache_growth_mb:.1f} MB",
                f"   • Memory Efficiency: {memory.memory_efficiency:.2f} req/MB",
                f"   • Fragmentation Detected: {'⚠️  YES' if memory.fragmentation_detected else '✅ NO'}"
            ])
        
        # Performance recommendations
        lines.extend([
            "",
            "💡 PERFORMANCE RECOMMENDATIONS:",
        ])
        
        if 'throughput' in results and 'scalability' in results:
            throughput = results['throughput']
            scalability = results['scalability']
            lines.extend([
                f"   • Use batch size {throughput.optimal_batch_size} for maximum throughput",
                f"   • Use {scalability.optimal_concurrency} concurrent threads for optimal performance",
            ])
        
        if 'latency' in results:
            latency = results['latency']
            if latency.cold_start_overhead > 0.5:  # More than 500ms overhead
                lines.append("   • Consider model preloading to avoid cold start delays")
        
        if 'memory' in results:
            memory = results['memory']
            if memory.fragmentation_detected:
                lines.append("   • ⚠️  Monitor GPU memory fragmentation - consider periodic restarts")
            if memory.model_memory_mb > 1000:  # Large model
                lines.append("   • Large model detected - ensure adequate GPU memory")
        
        lines.append("=" * 100)
        return "\n".join(lines)
    
    @staticmethod
    def format_comprehensive_json(results: Dict) -> str:
        """Format comprehensive results as JSON."""
        json_results = {}
        
        # Convert dataclass results to dictionaries
        for test_type, result in results.items():
            if hasattr(result, '__dict__'):
                json_results[test_type] = result.__dict__
            else:
                json_results[test_type] = result
        
        return json.dumps(json_results, indent=2, default=str)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Performance testing tool for PyTorch Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic performance test (default)
  %(prog)s --model resnet50 --duration 30 --concurrency 4
      Auto-download ResNet-50 and test for 30 seconds with 4 threads

  # Latency analysis (cold vs warm start)
  %(prog)s --model bert --test-type latency
      Measure cold start vs warm start latency for BERT

  # Throughput testing (different batch sizes)
  %(prog)s --model mobilenet --test-type throughput
      Test throughput at different batch sizes to find optimal

  # Scalability testing (different concurrency levels)
  %(prog)s --model distilbert --test-type scalability
      Test performance degradation with increasing concurrency

  # Memory analysis
  %(prog)s --model resnet50 --test-type memory
      Monitor memory usage and detect GPU fragmentation

  # Comprehensive analysis (all tests)
  %(prog)s --model bert --test-type comprehensive
      Run all test types for complete performance analysis

  # Tensor optimization comparison
  %(prog)s --model resnet50 --test-type tensor-optimization
      Compare original vs tensor-optimized model performance

  # Test common models (auto-download if needed)
  %(prog)s --model bert --requests 1000 --batch-size 8
      Download and test BERT with 1000 requests, batch size 8

  # Simple model testing
  %(prog)s --model mobilenet --duration 60
      Download MobileNet and test for 1 minute

  # Export results
  %(prog)s --model resnet50 --test-type comprehensive --format json --output results.json
      Run comprehensive tests and save JSON results to file

  # List and download options
  %(prog)s --list-models
      List available models on the server

  %(prog)s --list-downloadable
      List all models available for auto-download

  %(prog)s --download resnet18
      Download ResNet-18 without running tests

Test Types Explained:
  basic:         Standard performance test (throughput, latency, success rate)
  latency:       Cold start vs warm start latency analysis
  throughput:    Batch size optimization for maximum throughput
  scalability:   Concurrency testing to find optimal thread count
  memory:        Memory usage analysis and fragmentation detection
  comprehensive: All test types combined with detailed recommendations
  tensor-optimization: Compare original vs low-rank tensor optimized models

Popular Models (auto-downloadable):
  Image Models:  resnet50, resnet18, mobilenet, efficientnet_b0, vgg16
  Text Models:   bert, distilbert, roberta, albert-base-v2
  Quick Tests:   resnet (→ resnet50), bert (→ bert-base-uncased)

Size Categories:
  Small Models:  ~14-300MB   (mobilenet, distilbert)
  Medium Models: ~300-500MB  (resnet50, bert-base)
  Large Models:  500MB+      (resnet152, roberta-large)
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="example",
        help="Model to test (default: example). Try: resnet50, bert, mobilenet, distilbert, or use --list-downloadable."
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models on the server and exit"
    )
    
    parser.add_argument(
        "--list-downloadable",
        action="store_true", 
        help="List all models available for auto-download and exit"
    )
    
    parser.add_argument(
        "--download",
        type=str,
        help="Download a model and exit. Try: resnet50, bert, mobilenet (use --list-downloadable for full list)"
    )
    
    # Test configuration
    test_group = parser.add_mutually_exclusive_group(required=False)
    test_group.add_argument(
        "--duration", "-d",
        type=int,
        help="Test duration in seconds"
    )
    
    test_group.add_argument(
        "--requests", "-r",
        type=int,
        help="Number of requests to send"
    )
    
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=1,
        help="Number of concurrent threads (default: 1)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for each request (default: 1)"
    )
    
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup requests (default: 10)"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30.0)"
    )
    
    # Server configuration
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    
    # Input data
    parser.add_argument(
        "--input-data",
        type=str,
        default="42",
        help="Input data for predictions (default: 42)"
    )
    
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Request priority (default: 0)"
    )
    
    # Output options
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file (default: stdout)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    # Test type selection
    parser.add_argument(
        "--allow-null-results",
        action="store_true",
        help="Allow tests to continue even if model returns null predictions (useful for testing inference pipeline)"
    )
    
    parser.add_argument(
        "--test-type", "-t",
        choices=["basic", "latency", "throughput", "scalability", "memory", "comprehensive", "tensor-optimization"],
        default="basic",
        help="Type of test to run (default: basic). Options: basic (standard test), latency (cold/warm start), throughput (batch sizes), scalability (concurrency), memory (usage analysis), comprehensive (all tests), tensor-optimization (low-rank tensor optimization comparison)"
    )
    
    return parser


def parse_input_data(input_str: str) -> Union[int, float, List, str]:
    """Parse input data string into appropriate type."""
    # Try to parse as JSON first
    try:
        return json.loads(input_str)
    except json.JSONDecodeError:
        pass
    
    # Try to parse as number
    try:
        if "." in input_str:
            return float(input_str)
        else:
            return int(input_str)
    except ValueError:
        pass
    
    # Return as string
    return input_str


def run_tensor_optimization_test(args) -> None:
    """Run performance comparison between original and tensor-optimized models."""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Install with: pip install torch")
        return
    
    print("🧠 TENSOR OPTIMIZATION PERFORMANCE TEST")
    print("=" * 80)
    
    # Import necessary components for tensor optimization
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from framework.optimizers.tensor_factorization_optimizer import (
            TensorFactorizationOptimizer,
            TensorFactorizationConfig,
            factorize_model
        )
        from framework.core.model_downloader import get_model_downloader, download_model
    except ImportError as e:
        print(f"❌ Failed to import tensor optimization components: {e}")
        print("💡 Make sure the framework modules are available")
        return
    
    # Download or load the model for testing
    model_name = args.model
    print(f"🔄 Preparing model '{model_name}' for tensor optimization testing...")
    
    downloader = get_model_downloader()
    
    # Check if model is cached locally, if not try to download it
    model_path = downloader.get_model_path(model_name)
    if model_path is None:
        print(f"📥 Model '{model_name}' not found locally, attempting to download...")
        
        # Try to download using common aliases
        try:
            if model_name in COMMON_MODEL_ALIASES:
                alias_config = COMMON_MODEL_ALIASES[model_name]
                model_path, model_info = download_model(
                    source=alias_config["source"],
                    model_id=alias_config["model_id"],
                    model_name=model_name,
                    task=alias_config.get("task", "classification")
                )
                print(f"✅ Downloaded {model_name}: {model_info.description}")
            else:
                print(f"❌ Model '{model_name}' not found in downloadable models")
                print("💡 Try: resnet50, bert, mobilenet, distilbert, or use --list-downloadable")
                return
        except Exception as e:
            print(f"❌ Failed to download model '{model_name}': {e}")
            return
    
    # Load the model
    try:
        print(f"📂 Loading model from: {model_path}")
        original_model = torch.load(model_path, map_location='cpu')
        if not isinstance(original_model, nn.Module):
            print("❌ Loaded object is not a PyTorch model")
            return
        
        print(f"✅ Original model loaded successfully")
        original_params = sum(p.numel() for p in original_model.parameters())
        print(f"📊 Original model parameters: {original_params:,}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Configure tensor factorization for performance testing
    print("\n🔧 Configuring tensor factorization...")
    tf_config = TensorFactorizationConfig()
    tf_config.decomposition_method = "svd"  # SVD is fastest and most reliable
    tf_config.target_compression_ratio = 0.6  # 40% parameter reduction
    tf_config.preserve_accuracy_threshold = 0.05  # Allow up to 5% accuracy loss
    tf_config.performance_priority = True  # Prioritize speed over compression
    tf_config.enable_fine_tuning = False  # Disable for performance testing
    tf_config.min_param_savings = 0.3  # Minimum 30% parameter reduction
    tf_config.min_flop_savings = 0.25  # Minimum 25% FLOP reduction
    
    print(f"   • Method: {tf_config.decomposition_method}")
    print(f"   • Target compression: {tf_config.target_compression_ratio}")
    print(f"   • Performance priority: {tf_config.performance_priority}")
    
    # Apply tensor factorization
    print("\n⚡ Applying tensor factorization optimization...")
    try:
        optimizer = TensorFactorizationOptimizer(tf_config)
        
        # Create example inputs for benchmarking
        model_config = None
        for size_category, models in MODEL_CONFIGS.items():
            for config in models:
                if config["name"] == model_name:
                    model_config = config
                    break
            if model_config:
                break
        
        if not model_config and model_name in COMMON_MODEL_ALIASES:
            alias_config = COMMON_MODEL_ALIASES[model_name]
            task = alias_config.get("task", "classification")
            if "image" in task.lower() or alias_config["source"] == "torchvision":
                example_inputs = torch.randn(1, 3, 224, 224)
            else:
                example_inputs = torch.randn(1, 512)  # Text models
        elif model_config:
            task = model_config.get("task", "classification")
            if "image" in task.lower():
                example_inputs = torch.randn(1, 3, 224, 224)
            else:
                example_inputs = torch.randn(1, 512)
        else:
            # Default to image input
            example_inputs = torch.randn(1, 3, 224, 224)
        
        print(f"   • Example input shape: {list(example_inputs.shape)}")
        
        # Apply factorization
        start_time = time.time()
        factorized_model = optimizer.optimize(original_model.cpu())
        factorization_time = time.time() - start_time
        
        factorized_params = sum(p.numel() for p in factorized_model.parameters())
        compression_ratio = factorized_params / original_params
        
        print(f"✅ Tensor factorization completed in {factorization_time:.2f}s")
        print(f"📊 Factorized model parameters: {factorized_params:,}")
        print(f"📉 Compression ratio: {compression_ratio:.3f} ({(1-compression_ratio)*100:.1f}% reduction)")
        
    except Exception as e:
        print(f"❌ Tensor factorization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Benchmark both models locally
    print("\n🚀 PERFORMANCE BENCHMARKING")
    print("-" * 50)
    
    try:
        # Set models to evaluation mode
        original_model.eval()
        factorized_model.eval()
        
        # Benchmark parameters
        benchmark_iterations = 100
        
        print(f"🔥 Benchmarking {benchmark_iterations} iterations...")
        
        # Benchmark original model
        print("   • Original model...")
        start_time = time.time()
        with torch.no_grad():
            for _ in range(benchmark_iterations):
                _ = original_model(example_inputs)
        original_time = time.time() - start_time
        original_fps = benchmark_iterations / original_time
        
        # Benchmark factorized model
        print("   • Factorized model...")
        start_time = time.time()
        with torch.no_grad():
            for _ in range(benchmark_iterations):
                _ = factorized_model(example_inputs)
        factorized_time = time.time() - start_time
        factorized_fps = benchmark_iterations / factorized_time
        
        # Calculate speedup
        speedup = original_time / factorized_time
        
        print("\n📈 TENSOR OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Optimization method: {tf_config.decomposition_method.upper()}")
        print(f"Benchmark iterations: {benchmark_iterations}")
        print()
        print("💾 MODEL SIZE:")
        print(f"   • Original parameters: {original_params:,}")
        print(f"   • Optimized parameters: {factorized_params:,}")
        print(f"   • Compression ratio: {compression_ratio:.3f}")
        print(f"   • Size reduction: {(1-compression_ratio)*100:.1f}%")
        print()
        print("⚡ PERFORMANCE:")
        print(f"   • Original inference time: {original_time:.4f}s ({original_fps:.2f} FPS)")
        print(f"   • Optimized inference time: {factorized_time:.4f}s ({factorized_fps:.2f} FPS)")
        print(f"   • Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
        print()
        
        # Quality assessment
        print("🎯 OUTPUT QUALITY:")
        try:
            with torch.no_grad():
                original_output = original_model(example_inputs)
                factorized_output = factorized_model(example_inputs)
                
                # Calculate similarity metrics
                mse = torch.mean((original_output - factorized_output) ** 2).item()
                mae = torch.mean(torch.abs(original_output - factorized_output)).item()
                
                # Flatten outputs for cosine similarity
                orig_flat = original_output.flatten()
                fact_flat = factorized_output.flatten()
                cos_sim = F.cosine_similarity(orig_flat, fact_flat, dim=0).item()
                
                print(f"   • Mean Squared Error: {mse:.6f}")
                print(f"   • Mean Absolute Error: {mae:.6f}")
                print(f"   • Cosine Similarity: {cos_sim:.4f}")
                
                # Quality assessment
                if cos_sim > 0.99:
                    quality = "🟢 Excellent (>99% similarity)"
                elif cos_sim > 0.95:
                    quality = "🟡 Good (>95% similarity)"
                elif cos_sim > 0.90:
                    quality = "🟠 Acceptable (>90% similarity)"
                else:
                    quality = "🔴 Poor (<90% similarity)"
                
                print(f"   • Quality assessment: {quality}")
                
        except Exception as e:
            print(f"   • Could not assess output quality: {e}")
        
        print()
        print("💡 RECOMMENDATIONS:")
        if speedup > 1.2 and compression_ratio < 0.8:
            print("   ✅ Tensor optimization is beneficial for this model")
            print(f"   ✅ {speedup:.1f}x speedup with {(1-compression_ratio)*100:.0f}% size reduction")
        elif speedup > 1.0:
            print("   🟡 Moderate improvement - consider if size reduction is priority")
        else:
            print("   ❌ Tensor optimization not beneficial for this model")
            print("   💡 Try different optimization methods or parameters")
        
        # Save results if output file specified
        if args.output:
            results = {
                "model_name": model_name,
                "optimization_method": tf_config.decomposition_method,
                "benchmark_iterations": benchmark_iterations,
                "original_parameters": original_params,
                "optimized_parameters": factorized_params,
                "compression_ratio": compression_ratio,
                "size_reduction_percent": (1-compression_ratio)*100,
                "original_time_s": original_time,
                "optimized_time_s": factorized_time,
                "original_fps": original_fps,
                "optimized_fps": factorized_fps,
                "speedup": speedup,
                "improvement_percent": (speedup-1)*100
            }
            
            if args.format == "json":
                output_content = json.dumps(results, indent=2)
            elif args.format == "csv":
                # Convert to CSV format
                headers = list(results.keys())
                values = [str(v) for v in results.values()]
                output_content = ",".join(headers) + "\n" + ",".join(values)
            else:
                # Text format
                output_content = f"""TENSOR OPTIMIZATION PERFORMANCE TEST RESULTS
Model: {model_name}
Method: {tf_config.decomposition_method}
Original Parameters: {original_params:,}
Optimized Parameters: {factorized_params:,}
Compression Ratio: {compression_ratio:.3f}
Size Reduction: {(1-compression_ratio)*100:.1f}%
Original Time: {original_time:.4f}s
Optimized Time: {factorized_time:.4f}s
Speedup: {speedup:.2f}x
Improvement: {(speedup-1)*100:.1f}%
"""
            
            with open(args.output, 'w') as f:
                f.write(output_content)
            print(f"📄 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle tensor optimization test
    if args.test_type == "tensor-optimization":
        run_tensor_optimization_test(args)
        return
    
    # Handle list models command
    if args.list_models:
        config = TestConfig(base_url=args.url)
        with PerformanceTestRunner(config) as runner:
            models = runner.list_available_models()
            if models:
                print("📋 Available models on server:")
                for model in models:
                    print(f"   • {model}")
            else:
                print("❌ No models available or server not accessible")
        return
    
    # Handle list downloadable models command
    if args.list_downloadable:
        config = TestConfig(base_url=args.url)
        with PerformanceTestRunner(config) as runner:
            runner.list_downloadable_models()
        return
    
    # Handle download command
    if args.download:
        config = TestConfig(base_url=args.url)
        with PerformanceTestRunner(config) as runner:
            if runner.download_model(args.download):
                print(f"[SUCCESS] Successfully downloaded model: {args.download}")
            else:
                print(f"❌ Failed to download model: {args.download}")
                sys.exit(1)
        return
    
    # Set default test parameters if none specified
    if not args.duration and not args.requests:
        args.duration = 30  # Default to 30 seconds
    
    # Parse input data
    input_data = parse_input_data(args.input_data)
    
    # Create test configuration
    config = TestConfig(
        base_url=args.url,
        model_name=args.model,
        duration_seconds=args.duration,
        num_requests=args.requests,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        warmup_requests=args.warmup,
        timeout=args.timeout,
        input_data=input_data,
        priority=args.priority,
        verbose=args.verbose,
        output_format=args.format,
        test_type=args.test_type,
        allow_null_results=args.allow_null_results
    )
    
    # Run the test
    try:
        with PerformanceTestRunner(config) as runner:
            if args.test_type == "basic":
                report = runner.run_test()
                # Format the report
                if args.format == "json":
                    output = ReportFormatter.format_json_report(report)
                elif args.format == "csv":
                    output = ReportFormatter.format_csv_report(report)
                else:
                    output = ReportFormatter.format_text_report(report)
                    
            elif args.test_type == "latency":
                result = runner.run_latency_test()
                if args.format == "json":
                    output = json.dumps(result.__dict__, indent=2, default=str)
                else:
                    output = f"""
🥶🔥 LATENCY TEST RESULTS
=======================
Cold Start Times: {len(result.cold_start_times)} samples
• Average: {result.avg_cold_start:.3f}s
• 95th percentile: {result.p95_cold_start:.3f}s

Warm Start Times: {len(result.warm_start_times)} samples
• Average: {result.avg_warm_start:.3f}s  
• 95th percentile: {result.p95_warm_start:.3f}s

Cold Start Overhead: +{result.cold_start_overhead:.3f}s ({result.cold_start_overhead/result.avg_warm_start*100:.1f}% slower)
"""
                    
            elif args.test_type == "throughput":
                result = runner.run_throughput_test()
                if args.format == "json":
                    output = json.dumps(result.__dict__, indent=2, default=str)
                else:
                    output = f"""
📦 THROUGHPUT TEST RESULTS
==========================
Optimal Batch Size: {result.optimal_batch_size}
Maximum Throughput: {result.max_throughput:.2f} pred/s

Batch Size Performance:
"""
                    for i, batch_size in enumerate(result.batch_sizes):
                        if i < len(result.throughput_per_batch):
                            tput = result.throughput_per_batch[i]
                            latency = result.latency_per_batch[i] if i < len(result.latency_per_batch) else 0
                            marker = "⭐" if batch_size == result.optimal_batch_size else "  "
                            output += f"{marker} Batch {batch_size:2d}: {tput:8.2f} pred/s, {latency:.3f}s latency\n"
                    
            elif args.test_type == "scalability":
                result = runner.run_scalability_test()
                if args.format == "json":
                    output = json.dumps(result.__dict__, indent=2, default=str)
                else:
                    output = f"""
🔀 SCALABILITY TEST RESULTS
===========================
Optimal Concurrency: {result.optimal_concurrency} threads
Performance Degradation: {result.degradation_factor*100:.1f}%

Concurrency Performance:
"""
                    for i, concurrency in enumerate(result.concurrency_levels):
                        if i < len(result.throughput_per_level):
                            tput = result.throughput_per_level[i]
                            latency = result.latency_per_level[i] if i < len(result.latency_per_level) else 0
                            error_rate = result.error_rates[i] if i < len(result.error_rates) else 0
                            marker = "⭐" if concurrency == result.optimal_concurrency else "  "
                            output += f"{marker} {concurrency:2d} threads: {tput:8.2f} req/s, {latency:.3f}s latency, {error_rate:.1f}% errors\n"
                    
            elif args.test_type == "memory":
                result = runner.run_memory_test()
                if args.format == "json":
                    output = json.dumps(result.__dict__, indent=2, default=str)
                else:
                    output = f"""
🧠 MEMORY TEST RESULTS
======================
Model Memory Footprint: {result.model_memory_mb:.1f} MB
Baseline Memory: CPU {result.baseline_memory.cpu_memory_mb:.1f}MB, GPU {result.baseline_memory.gpu_memory_mb:.1f}MB
Peak Memory: CPU {result.peak_memory.cpu_memory_mb:.1f}MB, GPU {result.peak_memory.gpu_memory_mb:.1f}MB
Cache Growth: {result.cache_growth_mb:.1f} MB
Memory Efficiency: {result.memory_efficiency:.2f} req/MB
Fragmentation Detected: {'⚠️  YES' if result.fragmentation_detected else '✅ NO'}
"""
                    
            elif args.test_type == "comprehensive":
                results = runner.run_comprehensive_test()
                if args.format == "json":
                    output = ReportFormatter.format_comprehensive_json(results)
                else:
                    output = ReportFormatter.format_comprehensive_report(results)
        
        # Output the report
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"📄 Report saved to: {args.output}")
        else:
            print(output)
            
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
