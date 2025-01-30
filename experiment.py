#!/usr/bin/env python3

import torch
import torch_tensorrt
import torchvision.models as models
import time

class InferenceTimer:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.total_time = 0.0
        self.batch_count = 0
        
    def record(self, elapsed_time, batch_size):
        self.total_time += elapsed_time
        self.batch_count += batch_size
        
    def get_stats(self):
        avg_time_per_batch = self.total_time / self.batch_count if self.batch_count > 0 else 0
        samples_per_second = self.batch_count / self.total_time if self.total_time > 0 else 0
        return avg_time_per_batch, samples_per_second

def main():
    # --------------------------------------------------------------------------
    # 1. SETUP
    # --------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained ResNet-18, half precision, evaluation mode, and move to device
    model = models.resnet18(pretrained=True).half().eval().to(device)
    original_model = model  # Keep reference to original model

    # --------------------------------------------------------------------------
    # 2. CONFIGURATION
    # --------------------------------------------------------------------------
    compile_config = {
        "enabled_precisions": {torch.half},
        "debug": True,
        "workspace_size": 20 << 30,  # 20 GB
        "min_block_size": 90,        # Forces single TRT engine
        "torch_executed_ops": {},
    }

    # --------------------------------------------------------------------------
    # 3. PYTORCH NATIVE INFERENCE BASELINE
    # --------------------------------------------------------------------------
    print("\n=== PyTorch Native Inference ===")
    
    pytorch_timer = InferenceTimer()
    
    def run_pytorch_inference(batch_size):
        inputs = torch.randn((batch_size, 3, 224, 224), dtype=torch.half, device=device)
        
        # Warmup
        for _ in range(3):
            _ = original_model(inputs)
        
        # Timed runs
        torch.cuda.synchronize()
        start_time = time.time()
        _ = original_model(inputs)
        torch.cuda.synchronize()
        
        elapsed = (time.time() - start_time) * 1000  # ms
        pytorch_timer.record(elapsed, batch_size)
        
        print(f"PyTorch BS={batch_size}: {elapsed:.2f}ms | "
              f"{elapsed/batch_size:.2f}ms/sample | "
              f"{1000*batch_size/elapsed:.1f} samples/s")

    print("Running PyTorch native inferences:")
    for bs in [1, 8]:
        run_pytorch_inference(bs)

    # --------------------------------------------------------------------------
    # 4. TENSORRT STATIC SHAPES COMPILATION (torch_compile)
    # --------------------------------------------------------------------------
    print("\n=== TensorRT Static Shapes Compilation (ir='torch_compile') ===")
    
    static_timer = InferenceTimer()
    inputs_bs1 = torch.randn((1, 3, 224, 224), dtype=torch.half, device=device)
    
    with torch.no_grad():
        optimized_model = torch_tensorrt.compile(
            original_model,
            ir="torch_compile",
            inputs=[inputs_bs1],
            **compile_config
        )

        def run_static_inference(batch_size):
            inputs = torch.randn((batch_size, 3, 224, 224), dtype=torch.half, device=device)
            
            # Warmup
            for _ in range(3):
                _ = optimized_model(inputs)
            
            # Timed runs
            torch.cuda.synchronize()
            start_time = time.time()
            _ = optimized_model(inputs)
            torch.cuda.synchronize()
            
            elapsed = (time.time() - start_time) * 1000  # ms
            static_timer.record(elapsed, batch_size)
            
            speedup = pytorch_timer.total_time / static_timer.total_time
            print(f"TRT Static BS={batch_size}: {elapsed:.2f}ms | "
                  f"{elapsed/batch_size:.2f}ms/sample | "
                  f"{1000*batch_size/elapsed:.1f} samples/s | "
                  f"{speedup:.1f}x speedup")

        print("Running static shape inferences:")
        for bs in [1, 8]:
            run_static_inference(bs)

    # --------------------------------------------------------------------------
    # 5. TENSORRT DYNAMIC SHAPES COMPILATION (dynamo)
    # --------------------------------------------------------------------------
    print("\n=== TensorRT Dynamic Shapes Compilation (ir='dynamo') ===")

    dynamic_timer = InferenceTimer()
    with torch.no_grad():
        trt_model = torch_tensorrt.compile(
            original_model,
            ir="dynamo",
            inputs=[torch_tensorrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(8, 3, 224, 224),
                max_shape=(16, 3, 224, 224),
                dtype=torch.half,
            )],
            **compile_config
        )

        def run_dynamic_inference(batch_size):
            inputs = torch.randn((batch_size, 3, 224, 224), dtype=torch.half, device=device)
            
            # Warmup
            for _ in range(3):
                _ = trt_model(inputs)
            
            # Timed runs
            torch.cuda.synchronize()
            start_time = time.time()
            _ = trt_model(inputs)
            torch.cuda.synchronize()
            
            elapsed = (time.time() - start_time) * 1000  # ms
            dynamic_timer.record(elapsed, batch_size)
            
            speedup = pytorch_timer.total_time / dynamic_timer.total_time
            print(f"TRT Dynamic BS={batch_size}: {elapsed:.2f}ms | "
                  f"{elapsed/batch_size:.2f}ms/sample | "
                  f"{1000*batch_size/elapsed:.1f} samples/s | "
                  f"{speedup:.1f}x speedup")

        print("Running dynamic shape inferences:")
        for bs in [1, 4, 12, 16]:
            run_dynamic_inference(bs)

    # --------------------------------------------------------------------------
    # 6. FINAL STATISTICS
    # --------------------------------------------------------------------------
    print("\n=== Final Statistics ===")
    
    pytorch_avg, pytorch_sps = pytorch_timer.get_stats()
    static_avg, static_sps = static_timer.get_stats()
    dynamic_avg, dynamic_sps = dynamic_timer.get_stats()
    
    print(f"[PyTorch] Average: {pytorch_avg:.2f}ms/batch | {pytorch_sps:.1f} samples/s")
    print(f"[TRT Static] Average: {static_avg:.2f}ms/batch | {static_sps:.1f} samples/s | "
          f"Speedup: {pytorch_avg/static_avg:.1f}x")
    print(f"[TRT Dynamic] Average: {dynamic_avg:.2f}ms/batch | {dynamic_sps:.1f} samples/s | "
          f"Speedup: {pytorch_avg/dynamic_avg:.1f}x")

if __name__ == "__main__":
    main()