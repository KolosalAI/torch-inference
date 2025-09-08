"""
GPU-aware demo model for image benchmarking.

This module provides a demo image generation function that properly utilizes GPU
for computation instead of just sleeping.
"""

import time
import logging
from typing import Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


def gpu_demo_image_model(
    prompt: str,
    width: int = 512,
    height: int = 512,
    num_images: int = 1,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    **kwargs
) -> Dict[str, Any]:
    """
    GPU-aware demo image generation function.
    
    This function actually utilizes GPU memory and computation instead of just
    sleeping, providing realistic GPU usage patterns for benchmarking.
    """
    start_time = time.perf_counter()
    
    # Determine device to use
    device = _get_optimal_device()
    
    try:
        # Perform actual GPU computation if available
        if device.type in ['cuda', 'mps'] and TORCH_AVAILABLE:
            _perform_gpu_image_generation(device, width, height, num_inference_steps, guidance_scale)
        else:
            # CPU fallback with realistic timing
            cpu_time = _calculate_cpu_processing_time(width, height, num_inference_steps)
            time.sleep(cpu_time)
        
        # Generate dummy image data
        images = []
        for i in range(num_images):
            image_size = width * height * 3  # RGB
            dummy_data = b'demo_image_data' * (image_size // 15)
            images.append(dummy_data)
        
        total_time = time.perf_counter() - start_time
        
        # Return result in expected format
        return {
            'images': images,
            'width': width,
            'height': height,
            'num_images': num_images,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'file_size_bytes': len(images[0]) if images else 0,
            'image_format': 'PNG',
            'processing_time': total_time,
            'device': str(device),
            'gpu_computation_used': device.type in ['cuda', 'mps'] and TORCH_AVAILABLE
        }
        
    except Exception as e:
        logger.warning(f"GPU computation failed, falling back to simple timing: {e}")
        
        # Fallback to simple timing
        fallback_time = _calculate_fallback_time(width, height, num_inference_steps)
        time.sleep(fallback_time)
        
        images = []
        for i in range(num_images):
            image_size = width * height * 3
            dummy_data = b'fallback_image_data' * (image_size // 18)
            images.append(dummy_data)
        
        return {
            'images': images,
            'width': width,
            'height': height,
            'num_images': num_images,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'file_size_bytes': len(images[0]) if images else 0,
            'image_format': 'PNG',
            'processing_time': time.perf_counter() - start_time,
            'device': str(device),
            'error': str(e),
            'gpu_computation_used': False
        }


def _get_optimal_device():
    """Get the optimal device for computation - prefer GPU for maximum performance."""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available - using CPU fallback")
        return 'cpu'
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS device")
        return device
    else:
        logger.warning("No GPU available - falling back to CPU (performance will be degraded)")
        return torch.device('cpu')


def _perform_gpu_image_generation(device, width: int, height: int, 
                                num_inference_steps: int, guidance_scale: float):
    """Perform actual GPU computation to simulate image generation."""
    try:
        with torch.no_grad():
            # Use appropriate dtype
            dtype = torch.float16 if device.type == 'cuda' else torch.float32
            
            # Simulate diffusion model latent space (compressed representation)
            latent_height, latent_width = height // 8, width // 8
            latents = torch.randn(1, 4, latent_height, latent_width, device=device, dtype=dtype)
            
            # Simulate text embeddings for conditioning
            text_embeddings = torch.randn(1, 77, 768, device=device, dtype=dtype)
            
            # Simulate iterative denoising process
            for step in range(min(num_inference_steps, 25)):  # Limit for demo
                # Simulate U-Net noise prediction
                noise_pred = _simulate_unet_step(latents, step, text_embeddings, device, dtype)
                
                # Simulate scheduler step (DDPM/DDIM)
                alpha = 1.0 - (step / num_inference_steps) * 0.99
                beta = 0.02 * (1 - alpha)
                latents = latents - beta * noise_pred
                
                # Simulate some additional processing
                if step % 5 == 0:  # Every 5 steps, do some extra computation
                    temp_processing = torch.nn.functional.conv2d(
                        latents,
                        torch.randn(4, 4, 3, 3, device=device, dtype=dtype),
                        padding=1
                    )
                    del temp_processing
                
                # Synchronize for realistic timing
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            # Simulate VAE decoder (latent to image space)
            final_image = _simulate_vae_decoder(latents, device, dtype, width, height)
            
            # Cleanup
            del latents, text_embeddings, final_image
            
            # Clear cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
    except Exception as e:
        logger.debug(f"GPU computation error (non-critical): {e}")
        # Fallback to simple delay
        time.sleep(0.01 * num_inference_steps)


def _simulate_unet_step(latents, timestep: int, text_embeddings, device, dtype):
    """Simulate a single U-Net denoising step."""
    batch_size, channels, height, width = latents.shape
    
    # Simulate U-Net encoder
    x = latents
    for i in range(3):  # 3 down-sampling blocks
        # Convolution
        x = torch.nn.functional.conv2d(
            x,
            torch.randn(channels * 2, channels, 3, 3, device=device, dtype=dtype),
            padding=1
        )
        x = torch.nn.functional.gelu(x)
        
        # Attention (simplified)
        if i == 1:  # Middle layer gets attention
            attn_weights = torch.randn(batch_size, height * width, 77, device=device, dtype=dtype)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, text_embeddings)
            # Reshape and add
            attn_reshaped = attn_output.mean(dim=-1).view(batch_size, 1, height, width)
            x = x + attn_reshaped.expand_as(x)
        
        # Down-sample
        if i < 2:
            x = torch.nn.functional.avg_pool2d(x, 2)
            height, width = height // 2, width // 2
        
        channels *= 2
    
    # Simulate U-Net decoder (simplified)
    for i in range(3):
        if i > 0:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        
        x = torch.nn.functional.conv2d(
            x,
            torch.randn(channels // 2, channels, 3, 3, device=device, dtype=dtype),
            padding=1
        )
        x = torch.nn.functional.gelu(x)
        channels //= 2
    
    # Final prediction layer
    noise_pred = torch.nn.functional.conv2d(
        x,
        torch.randn(latents.size(1), x.size(1), 3, 3, device=device, dtype=dtype),
        padding=1
    )
    
    return noise_pred


def _simulate_vae_decoder(latents, device, dtype, target_width: int, target_height: int):
    """Simulate VAE decoder from latents to image."""
    x = latents
    
    # Up-sample through decoder layers
    for i in range(3):  # 3 up-sampling layers (8x total)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        
        out_channels = 3 if i == 2 else x.size(1) // 2
        x = torch.nn.functional.conv2d(
            x,
            torch.randn(out_channels, x.size(1), 3, 3, device=device, dtype=dtype),
            padding=1
        )
        
        if i < 2:
            x = torch.nn.functional.gelu(x)
        else:
            x = torch.sigmoid(x)  # Final RGB output
    
    # Ensure correct output size
    if x.shape[-2:] != (target_height, target_width):
        x = torch.nn.functional.interpolate(
            x, size=(target_height, target_width), mode='bilinear', align_corners=False
        )
    
    return x


def _calculate_cpu_processing_time(width: int, height: int, num_inference_steps: int) -> float:
    """Calculate realistic CPU processing time."""
    base_time = 0.1  # Base time for CPU
    resolution_factor = (width * height) / (512 * 512)
    steps_factor = num_inference_steps / 50
    return base_time * resolution_factor * steps_factor


def _calculate_fallback_time(width: int, height: int, num_inference_steps: int) -> float:
    """Calculate fallback processing time."""
    base_time = 0.05
    complexity = (width * height * num_inference_steps) / (512 * 512 * 50)
    return base_time * complexity


# Additional utility functions for integration

def create_gpu_aware_image_function(**default_params):
    """Create a GPU-aware image function with default parameters."""
    def image_function(prompt: str, **kwargs):
        # Merge default params with provided kwargs
        merged_params = {**default_params, **kwargs}
        return gpu_demo_image_model(prompt, **merged_params)
    
    return image_function


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {
        'torch_available': TORCH_AVAILABLE,
        'cuda_available': False,
        'mps_available': False,
        'device_count': 0,
        'device_names': [],
        'recommended_device': 'cpu'
    }
    
    if TORCH_AVAILABLE:
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['device_count'] = torch.cuda.device_count()
            info['device_names'] = [torch.cuda.get_device_name(i) for i in range(info['device_count'])]
            info['recommended_device'] = 'cuda:0'
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['mps_available'] = True
            if not info['cuda_available']:
                info['recommended_device'] = 'mps'
    
    return info
