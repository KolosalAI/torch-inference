#!/usr/bin/env python3
"""Comprehensive PyTorch vs ONNX benchmark"""
import torch
import torchvision.models as models
import onnxruntime as ort
import numpy as np
import time
import os

torch.backends.cudnn.benchmark = True
os.makedirs('onnx_models', exist_ok=True)

print('='*70)
print('COMPREHENSIVE BENCHMARK: PyTorch vs ONNX (Batch=1)')
print('='*70)
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'ONNX Runtime: {ort.__version__}')
print('='*70)

def benchmark(fn, warmup=30, iters=100):
    for _ in range(warmup): 
        fn()
    if hasattr(torch.cuda, 'synchronize'): 
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters): 
        fn()
    if hasattr(torch.cuda, 'synchronize'): 
        torch.cuda.synchronize()
    return iters / (time.perf_counter() - start)

results = []
for name, model_fn in [
    ('ResNet-18', models.resnet18),
    ('ResNet-50', models.resnet50),
    ('MobileNetV3-L', models.mobilenet_v3_large),
    ('EfficientNet-B0', models.efficientnet_b0),
    ('ViT-B/16', models.vit_b_16),
]:
    print(f'\n{name}:')
    
    # PyTorch models
    m32 = model_fn(weights='DEFAULT').cuda().eval()
    m16 = model_fn(weights='DEFAULT').cuda().half().eval()
    x32 = torch.randn(1, 3, 224, 224).cuda()
    x16 = x32.half()
    
    with torch.no_grad():
        fps32 = benchmark(lambda: m32(x32))
        fps16 = benchmark(lambda: m16(x16))
    
    # Export ONNX
    safe_name = name.lower().replace("/","_").replace("-","_")
    onnx_path = f'onnx_models/{safe_name}.onnx'
    torch.onnx.export(m32, x32, onnx_path, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input':{0:'b'},'output':{0:'b'}}, opset_version=17)
    
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_opts, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    xnp = np.random.randn(1,3,224,224).astype(np.float32)
    fps_onnx = benchmark(lambda: sess.run(None, {'input': xnp}))
    
    # Find best
    all_fps = [('PyTorch FP32', fps32), ('PyTorch FP16', fps16), ('ONNX CUDA', fps_onnx)]
    best = max(all_fps, key=lambda x: x[1])
    
    print(f'  PyTorch FP32: {fps32:7.1f} FPS')
    print(f'  PyTorch FP16: {fps16:7.1f} FPS ({fps16/fps32:.2f}x)')
    print(f'  ONNX CUDA:    {fps_onnx:7.1f} FPS ({fps_onnx/fps32:.2f}x)')
    print(f'  >>> Best: {best[0]} ({best[1]:.1f} FPS)')
    
    results.append((name, fps32, fps16, fps_onnx, best[0]))
    del m32, m16, sess
    torch.cuda.empty_cache()

print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print(f'{"Model":<18} {"FP32":<10} {"FP16":<10} {"ONNX":<10} {"Best":<15}')
print('-'*70)
for name, fp32, fp16, onnx, best in results:
    print(f'{name:<18} {fp32:<10.1f} {fp16:<10.1f} {onnx:<10.1f} {best:<15}')
