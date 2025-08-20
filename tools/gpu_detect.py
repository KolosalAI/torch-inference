"""
GPU Detection CLI Tool

Command-line interface for the GPU detection system.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from framework.core.gpu_detection import GPUDetector, print_gpu_report
from framework.core.gpu_manager import GPUManager, print_gpu_configuration_report


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="GPU Detection and Configuration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Basic GPU detection report
  %(prog)s --detailed                # Detailed report with recommendations
  %(prog)s --json                    # Output in JSON format
  %(prog)s --benchmark               # Include performance benchmarks
  %(prog)s --config-only             # Show only configuration recommendations
        """
    )
    
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed report with configuration recommendations"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Include performance benchmarks (slower but more detailed)"
    )
    
    parser.add_argument(
        "--config-only", "-c",
        action="store_true",
        help="Show only configuration recommendations"
    )
    
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Disable performance benchmarks (faster)"
    )
    
    parser.add_argument(
        "--force-refresh", "-f",
        action="store_true",
        help="Force refresh of cached detection results"
    )
    
    args = parser.parse_args()
    
    # Determine benchmark setting
    enable_benchmarks = True
    if args.no_benchmark:
        enable_benchmarks = False
    elif args.benchmark:
        enable_benchmarks = True
    
    try:
        if args.json:
            # JSON output
            manager = GPUManager()
            gpus, device_config = manager.detect_and_configure(force_refresh=args.force_refresh)
            
            output = {
                "gpus": [],
                "device_config": {
                    "device_type": device_config.device_type.value,
                    "device_id": device_config.device_id,
                    "use_fp16": device_config.use_fp16,
                    "use_int8": device_config.use_int8,
                    "use_tensorrt": device_config.use_tensorrt,
                    "use_torch_compile": device_config.use_torch_compile,
                    "compile_mode": device_config.compile_mode
                },
                "recommendations": {
                    "memory": manager.get_memory_recommendations(),
                    "optimization": manager.get_optimization_recommendations()
                }
            }
            
            # Convert GPUs to dict format
            for gpu in gpus:
                gpu_dict = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "vendor": gpu.vendor.value,
                    "architecture": gpu.architecture.value,
                    "device_id": gpu.device_id,
                    "memory_mb": gpu.memory.total_mb,
                    "available_memory_mb": gpu.memory.available_mb,
                    "pytorch_support": gpu.pytorch_support,
                    "suitable_for_inference": gpu.is_suitable_for_inference(),
                    "recommended_precisions": gpu.get_recommended_precision(),
                    "supported_accelerators": [acc.value for acc in gpu.supported_accelerators]
                }
                
                if gpu.compute_capability:
                    gpu_dict["compute_capability"] = {
                        "major": gpu.compute_capability.major,
                        "minor": gpu.compute_capability.minor,
                        "version": gpu.compute_capability.version,
                        "supports_fp16": gpu.compute_capability.supports_fp16,
                        "supports_int8": gpu.compute_capability.supports_int8,
                        "supports_tensor_cores": gpu.compute_capability.supports_tensor_cores,
                        "supports_tf32": gpu.compute_capability.supports_tf32
                    }
                
                if gpu.benchmark_results:
                    gpu_dict["benchmark_results"] = gpu.benchmark_results
                
                output["gpus"].append(gpu_dict)
            
            print(json.dumps(output, indent=2, default=str))
            
        elif args.config_only:
            # Configuration recommendations only
            manager = GPUManager()
            manager.detect_and_configure(force_refresh=args.force_refresh)
            
            memory_rec = manager.get_memory_recommendations()
            optimization_rec = manager.get_optimization_recommendations()
            
            print("GPU CONFIGURATION RECOMMENDATIONS")
            print("=" * 50)
            
            print(f"\nMemory Recommendations:")
            for rec in memory_rec.get("recommendations", []):
                print(f"  - {rec}")
            
            print(f"\nOptimization Recommendations:")
            for rec in optimization_rec.get("recommendations", []):
                print(f"  - {rec}")
                
        elif args.detailed:
            # Detailed report with recommendations
            print_gpu_configuration_report()
            
        else:
            # Basic detection report
            print_gpu_report(enable_benchmarks=enable_benchmarks)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
