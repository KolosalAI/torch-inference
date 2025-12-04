"""
Image Model Benchmark Reporter - Generate comprehensive reports and visualizations.

Provides functionality to generate CSV reports, comparison tables, and plots
for image model benchmark results analysis and presentation.
"""

import csv
import io
import statistics
from typing import List, Dict, Optional, Any
import logging

from .image_metrics import ImageMetrics
from .image_benchmark import ImageBenchmarkResult

logger = logging.getLogger(__name__)


class ImageBenchmarkReporter:
    """
    Generate comprehensive reports and visualizations for image model benchmark results.
    
    Supports CSV generation, comparison tables, and performance plots
    following recommended image generation benchmarking metrics.
    """
    
    def __init__(self):
        """Initialize the reporter."""
        pass
    
    def generate_detailed_csv_report(
        self,
        results: Dict[int, ImageBenchmarkResult],
        test_prompts: Optional[List[str]] = None
    ) -> str:
        """
        Generate a detailed CSV report with individual request/iteration data.
        
        Args:
            results: Dictionary mapping concurrency levels to benchmark results
            test_prompts: Optional list of test prompts used in benchmark
            
        Returns:
            CSV content as string with individual request metrics
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header for detailed data
        writer.writerow([
            "Concurrency_Level",
            "Request_ID",
            "Iteration",
            "Prompt_Input",
            "Prompt_Length_Chars",
            "Prompt_Length_Tokens",
            "Negative_Prompt_Length_Chars",
            "Start_Time_Sec",
            "First_Image_Time_Sec",
            "End_Time_Sec",
            "Wall_Time_Sec",
            "TTFI_Sec",
            "Width",
            "Height",
            "Num_Images",
            "Total_Pixels",
            "Num_Inference_Steps",
            "Guidance_Scale",
            "Seed",
            "Images_Per_Sec",
            "Pixels_Per_Sec",
            "Steps_Per_Sec",
            "Memory_Peak_MB",
            "GPU_Memory_MB",
            "File_Size_Bytes",
            "Image_Format",
            "Success",
            "Error_Message"
        ])
        
        # Extract test prompts lookup if available
        test_prompts_lookup = {}
        if test_prompts:
            for i, prompt in enumerate(test_prompts):
                test_prompts_lookup[i] = prompt
        
        # Data rows for each request
        for concurrency in sorted(results.keys()):
            result = results[concurrency]
            request_metrics = result.request_metrics
            
            for idx, request in enumerate(request_metrics):
                # Extract iteration number from request_id (assumes format "req_N")
                iteration = idx
                if hasattr(request, 'request_id') and request.request_id:
                    try:
                        if request.request_id.startswith('req_'):
                            iteration = int(request.request_id.split('_')[1])
                    except (ValueError, IndexError):
                        pass
                
                # Get prompt input (cycle through test prompts based on iteration)
                prompt_input = ""
                if test_prompts:
                    prompt_idx = iteration % len(test_prompts)
                    prompt_input = test_prompts[prompt_idx]
                
                # Calculate derived metrics
                wall_time = request.wall_time if request.wall_time is not None else 0.0
                ttfi = request.ttfi if request.ttfi is not None else 0.0
                images_per_sec = request.images_per_second if request.images_per_second is not None else 0.0
                pixels_per_sec = request.pixels_per_second if request.pixels_per_second is not None else 0.0
                steps_per_sec = request.steps_per_second if request.steps_per_second is not None else 0.0
                
                # Success status
                success = "True" if request.error is None else "False"
                error_msg = request.error if request.error else ""
                
                writer.writerow([
                    concurrency,
                    request.request_id,
                    iteration,
                    prompt_input,
                    request.prompt_len_chars,
                    request.prompt_len_tokens,
                    request.negative_prompt_len_chars,
                    f"{request.t_start:.6f}",
                    f"{request.t_first_image:.6f}" if request.t_first_image is not None else "",
                    f"{request.t_end:.6f}" if request.t_end is not None else "",
                    f"{wall_time:.6f}",
                    f"{ttfi:.6f}",
                    request.width,
                    request.height,
                    request.num_images,
                    request.total_pixels,
                    request.num_inference_steps,
                    f"{request.guidance_scale:.2f}",
                    request.seed if request.seed is not None else "",
                    f"{images_per_sec:.6f}",
                    f"{pixels_per_sec:.1f}",
                    f"{steps_per_sec:.2f}",
                    f"{request.memory_peak_mb:.2f}",
                    f"{request.gpu_memory_mb:.2f}",
                    request.file_size_bytes,
                    request.image_format,
                    success,
                    error_msg
                ])
        
        return output.getvalue()
    
    def generate_summary_csv_report(
        self,
        results: Dict[int, ImageBenchmarkResult],
        test_prompts: Optional[List[str]] = None
    ) -> str:
        """
        Generate a summary CSV report with aggregated statistics only.
        
        Args:
            results: Dictionary mapping concurrency levels to benchmark results
            test_prompts: Optional list of test prompts used in benchmark
            
        Returns:
            CSV content as string with summary statistics
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Concurrency",
            "IPS",
            "PPS", 
            "SPS",
            "RPS",
            "CPS",
            "Tokens_Per_Sec",
            "TTFI_p50_ms",
            "TTFI_p95_ms",
            "TTFI_p99_ms",
            "TTFI_Mean_ms",
            "Total_Requests",
            "Success_Rate_%",
            "Failed_Requests",
            "Total_Wall_Time_s",
            "Total_Images",
            "Total_Pixels",
            "Total_Inference_Steps",
            "Avg_Prompt_Length_chars",
            "Avg_Images_Per_Request",
            "Avg_Image_Width",
            "Avg_Image_Height",
            "Avg_Inference_Steps",
            "Avg_Guidance_Scale",
            "Avg_Memory_Peak_MB",
            "Max_Memory_Peak_MB",
            "Avg_GPU_Memory_MB",
            "Max_GPU_Memory_MB"
        ])
        
        # Data rows
        for concurrency in sorted(results.keys()):
            result = results[concurrency]
            metrics = result.metrics
            
            writer.writerow([
                concurrency,
                f"{metrics.ips:.4f}",
                f"{metrics.pps:.1f}",
                f"{metrics.sps:.2f}",
                f"{metrics.rps:.2f}",
                f"{metrics.cps:.1f}",
                f"{metrics.tokens_per_sec:.1f}",
                f"{metrics.ttfi_p50 * 1000:.1f}",
                f"{metrics.ttfi_p95 * 1000:.1f}",
                f"{metrics.ttfi_p99 * 1000:.1f}",
                f"{metrics.ttfi_mean * 1000:.1f}",
                metrics.total_requests,
                f"{metrics.success_rate:.1f}",
                metrics.failed_requests,
                f"{metrics.total_wall_time:.2f}",
                metrics.total_images,
                metrics.total_pixels,
                metrics.total_inference_steps,
                f"{metrics.avg_prompt_length:.0f}",
                f"{metrics.avg_images_per_request:.1f}",
                f"{metrics.avg_image_width:.0f}",
                f"{metrics.avg_image_height:.0f}",
                f"{metrics.avg_inference_steps:.1f}",
                f"{metrics.avg_guidance_scale:.2f}",
                f"{metrics.avg_memory_peak_mb:.2f}",
                f"{metrics.max_memory_peak_mb:.2f}",
                f"{metrics.avg_gpu_memory_mb:.2f}",
                f"{metrics.max_gpu_memory_mb:.2f}"
            ])
        
        # Add summary section
        writer.writerow([])
        writer.writerow(["Summary Statistics"])
        writer.writerow(["Metric", "Min", "Max", "Range", "Best_Concurrency"])
        
        if results:
            ips_values = [(c, r.metrics.ips) for c, r in results.items()]
            pps_values = [(c, r.metrics.pps) for c, r in results.items()]
            rps_values = [(c, r.metrics.rps) for c, r in results.items()]
            ttfi_values = [(c, r.metrics.ttfi_p95) for c, r in results.items()]
            memory_values = [(c, r.metrics.avg_memory_peak_mb) for c, r in results.items()]
            
            best_ips = max(ips_values, key=lambda x: x[1])
            best_pps = max(pps_values, key=lambda x: x[1])
            best_rps = max(rps_values, key=lambda x: x[1])
            best_ttfi = min(ttfi_values, key=lambda x: x[1])
            lowest_memory = min(memory_values, key=lambda x: x[1])
            
            ips_min, ips_max = min(v[1] for v in ips_values), max(v[1] for v in ips_values)
            pps_min, pps_max = min(v[1] for v in pps_values), max(v[1] for v in pps_values)
            rps_min, rps_max = min(v[1] for v in rps_values), max(v[1] for v in rps_values)
            ttfi_min, ttfi_max = min(v[1] for v in ttfi_values), max(v[1] for v in ttfi_values)
            mem_min, mem_max = min(v[1] for v in memory_values), max(v[1] for v in memory_values)
            
            writer.writerow(["IPS", f"{ips_min:.4f}", f"{ips_max:.4f}", 
                           f"{ips_max-ips_min:.4f}", f"{best_ips[0]}"])
            writer.writerow(["PPS", f"{pps_min:.1f}", f"{pps_max:.1f}", 
                           f"{pps_max-pps_min:.1f}", f"{best_pps[0]}"])
            writer.writerow(["RPS", f"{rps_min:.2f}", f"{rps_max:.2f}", 
                           f"{rps_max-rps_min:.2f}", f"{best_rps[0]}"])
            writer.writerow(["TTFI_p95_ms", f"{ttfi_min*1000:.1f}", f"{ttfi_max*1000:.1f}", 
                           f"{(ttfi_max-ttfi_min)*1000:.1f}", f"{best_ttfi[0]}"])
            writer.writerow(["Memory_MB", f"{mem_min:.2f}", f"{mem_max:.2f}", 
                           f"{mem_max-mem_min:.2f}", f"{lowest_memory[0]}"])
        
        # Add test configuration
        if test_prompts:
            writer.writerow([])
            writer.writerow(["Test Configuration"])
            writer.writerow(["Test_Prompts_Count", len(test_prompts)])
            prompt_lengths = [len(prompt) for prompt in test_prompts]
            writer.writerow(["Min_Prompt_Length", min(prompt_lengths)])
            writer.writerow(["Max_Prompt_Length", max(prompt_lengths)])
            writer.writerow(["Avg_Prompt_Length", statistics.mean(prompt_lengths)])
            writer.writerow(["Median_Prompt_Length", statistics.median(prompt_lengths)])
        
        return output.getvalue()
    
    def generate_comparison_csv(
        self,
        benchmark_results: Dict[str, Dict[int, ImageBenchmarkResult]]
    ) -> str:
        """
        Generate a comparison CSV for multiple image benchmark runs.
        
        Args:
            benchmark_results: Dictionary mapping benchmark names to results
            
        Returns:
            Comparison CSV content as string
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Determine all concurrency levels
        all_concurrencies = set()
        for results in benchmark_results.values():
            all_concurrencies.update(results.keys())
        concurrencies = sorted(all_concurrencies)
        
        # Header
        header = ["Benchmark"]
        for concurrency in concurrencies:
            header.extend([
                f"IPS_C{concurrency}",
                f"PPS_C{concurrency}",
                f"RPS_C{concurrency}",
                f"TTFI_p95_ms_C{concurrency}",
                f"Memory_MB_C{concurrency}"
            ])
        writer.writerow(header)
        
        # Data rows
        for benchmark_name, results in benchmark_results.items():
            row = [benchmark_name]
            for concurrency in concurrencies:
                if concurrency in results:
                    metrics = results[concurrency].metrics
                    row.extend([
                        f"{metrics.ips:.4f}",
                        f"{metrics.pps:.1f}",
                        f"{metrics.rps:.2f}",
                        f"{metrics.ttfi_p95 * 1000:.1f}",
                        f"{metrics.avg_memory_peak_mb:.2f}"
                    ])
                else:
                    row.extend(["N/A", "N/A", "N/A", "N/A", "N/A"])
            writer.writerow(row)
        
        # Add best performance summary
        writer.writerow([])
        writer.writerow(["Best Performance Summary"])
        writer.writerow(["Metric", "Concurrency", "Best_Benchmark", "Best_Value"])
        
        for concurrency in concurrencies:
            # Find best IPS
            ips_values = []
            for name, results in benchmark_results.items():
                if concurrency in results:
                    ips_values.append((name, results[concurrency].metrics.ips))
            
            if ips_values:
                best_ips = max(ips_values, key=lambda x: x[1])
                writer.writerow([f"IPS_C{concurrency}", concurrency, best_ips[0], f"{best_ips[1]:.4f}"])
        
        return output.getvalue()
    
    def generate_detailed_comparison_csv(
        self,
        benchmark_results: Dict[str, Dict[int, ImageBenchmarkResult]]
    ) -> str:
        """
        Generate a detailed comparison CSV with individual request data from multiple benchmarks.
        
        Args:
            benchmark_results: Dictionary mapping benchmark names to results
            
        Returns:
            Detailed comparison CSV content as string
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header for detailed comparison
        writer.writerow([
            "Benchmark_Name",
            "Concurrency_Level",
            "Request_ID",
            "Iteration",
            "Prompt_Length_Chars",
            "Start_Time_Sec",
            "First_Image_Time_Sec",
            "End_Time_Sec",
            "Wall_Time_Sec",
            "TTFI_Sec",
            "Width",
            "Height",
            "Num_Images",
            "Total_Pixels",
            "Num_Inference_Steps",
            "Images_Per_Sec",
            "Pixels_Per_Sec",
            "Memory_Peak_MB",
            "GPU_Memory_MB",
            "Success",
            "Error_Message"
        ])
        
        # Data rows for each benchmark and request
        for benchmark_name, results in benchmark_results.items():
            for concurrency in sorted(results.keys()):
                result = results[concurrency]
                request_metrics = result.request_metrics
                
                for idx, request in enumerate(request_metrics):
                    # Extract iteration number from request_id (assumes format "req_N")
                    iteration = idx
                    if hasattr(request, 'request_id') and request.request_id:
                        try:
                            if request.request_id.startswith('req_'):
                                iteration = int(request.request_id.split('_')[1])
                        except (ValueError, IndexError):
                            pass
                    
                    # Calculate derived metrics
                    wall_time = request.wall_time if request.wall_time is not None else 0.0
                    ttfi = request.ttfi if request.ttfi is not None else 0.0
                    images_per_sec = request.images_per_second if request.images_per_second is not None else 0.0
                    pixels_per_sec = request.pixels_per_second if request.pixels_per_second is not None else 0.0
                    
                    # Success status
                    success = "True" if request.error is None else "False"
                    error_msg = request.error if request.error else ""
                    
                    writer.writerow([
                        benchmark_name,
                        concurrency,
                        request.request_id,
                        iteration,
                        request.prompt_len_chars,
                        f"{request.t_start:.6f}",
                        f"{request.t_first_image:.6f}" if request.t_first_image is not None else "",
                        f"{request.t_end:.6f}" if request.t_end is not None else "",
                        f"{wall_time:.6f}",
                        f"{ttfi:.6f}",
                        request.width,
                        request.height,
                        request.num_images,
                        request.total_pixels,
                        request.num_inference_steps,
                        f"{images_per_sec:.6f}",
                        f"{pixels_per_sec:.1f}",
                        f"{request.memory_peak_mb:.2f}",
                        f"{request.gpu_memory_mb:.2f}",
                        success,
                        error_msg
                    ])
        
        return output.getvalue()
    
    def plot_throughput_curve(
        self,
        results: Dict[int, ImageBenchmarkResult],
        output_path: str,
        title: str = "Image Model Throughput vs Concurrency"
    ) -> None:
        """
        Generate throughput curve plot (IPS vs concurrency).
        
        Args:
            results: Benchmark results
            output_path: Path to save the plot
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.style as style
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")
            return
        
        concurrencies = sorted(results.keys())
        ips_values = [results[c].metrics.ips for c in concurrencies]
        pps_values = [results[c].metrics.pps for c in concurrencies]
        rps_values = [results[c].metrics.rps for c in concurrencies]
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # IPS plot
        ax1.plot(concurrencies, ips_values, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Concurrency Level')
        ax1.set_ylabel('IPS (Images Per Second)')
        ax1.set_title('IPS vs Concurrency')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(concurrencies)
        
        # Add value labels
        for i, (c, ips) in enumerate(zip(concurrencies, ips_values)):
            ax1.annotate(f'{ips:.3f}', (c, ips), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        # PPS plot
        ax2.plot(concurrencies, pps_values, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Concurrency Level')
        ax2.set_ylabel('PPS (Pixels Per Second)')
        ax2.set_title('PPS vs Concurrency')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(concurrencies)
        
        # RPS plot
        ax3.plot(concurrencies, rps_values, 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Concurrency Level')
        ax3.set_ylabel('RPS (Requests Per Second)')
        ax3.set_title('RPS vs Concurrency')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(concurrencies)
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_latency_metrics(
        self,
        results: Dict[int, ImageBenchmarkResult],
        output_path: str,
        title: str = "Image Model Latency Metrics"
    ) -> None:
        """
        Generate latency metrics plot (TTFI percentiles).
        
        Args:
            results: Benchmark results
            output_path: Path to save the plot
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")
            return
        
        concurrencies = sorted(results.keys())
        ttfi_p50 = [results[c].metrics.ttfi_p50 * 1000 for c in concurrencies]  # Convert to ms
        ttfi_p95 = [results[c].metrics.ttfi_p95 * 1000 for c in concurrencies]
        ttfi_p99 = [results[c].metrics.ttfi_p99 * 1000 for c in concurrencies]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.array(concurrencies)
        width = 0.25
        
        ax.bar(x - width, ttfi_p50, width, label='TTFI p50', alpha=0.8)
        ax.bar(x, ttfi_p95, width, label='TTFI p95', alpha=0.8)
        ax.bar(x + width, ttfi_p99, width, label='TTFI p99', alpha=0.8)
        
        ax.set_xlabel('Concurrency Level')
        ax.set_ylabel('Time to First Image (ms)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary_table(
        self,
        results: Dict[int, ImageBenchmarkResult],
        benchmark_name: str = "Image Model Benchmark"
    ) -> None:
        """
        Print a formatted summary table to console.
        
        Args:
            results: Benchmark results
            benchmark_name: Name of the benchmark
        """
        print(f"\n{benchmark_name} Results Summary")
        print("=" * 80)
        print(f"{'Conc':<4} {'IPS':<8} {'PPS':<10} {'RPS':<8} {'TTFI p95':<10} {'Memory':<8} {'Success':<8}")
        print("-" * 80)
        
        for concurrency in sorted(results.keys()):
            metrics = results[concurrency].metrics
            print(f"{concurrency:<4} "
                  f"{metrics.ips:<8.3f} "
                  f"{metrics.pps:<10.0f} "
                  f"{metrics.rps:<8.1f} "
                  f"{metrics.ttfi_p95*1000:<10.1f} "
                  f"{metrics.avg_memory_peak_mb:<8.1f} "
                  f"{metrics.success_rate:<8.1f}%")
        
        print("=" * 80)
        
        # Find best performing concurrency for each metric
        if results:
            best_ips = max(results.items(), key=lambda x: x[1].metrics.ips)
            best_pps = max(results.items(), key=lambda x: x[1].metrics.pps)
            best_rps = max(results.items(), key=lambda x: x[1].metrics.rps)
            best_ttfi = min(results.items(), key=lambda x: x[1].metrics.ttfi_p95)
            best_memory = min(results.items(), key=lambda x: x[1].metrics.avg_memory_peak_mb)
            
            print("Best Performance:")
            print(f"  Highest IPS: {best_ips[1].metrics.ips:.3f} at concurrency {best_ips[0]}")
            print(f"  Highest PPS: {best_pps[1].metrics.pps:.0f} at concurrency {best_pps[0]}")
            print(f"  Highest RPS: {best_rps[1].metrics.rps:.1f} at concurrency {best_rps[0]}")
            print(f"  Lowest TTFI p95: {best_ttfi[1].metrics.ttfi_p95*1000:.1f}ms at concurrency {best_ttfi[0]}")
            print(f"  Lowest Memory: {best_memory[1].metrics.avg_memory_peak_mb:.1f}MB at concurrency {best_memory[0]}")
            print()
