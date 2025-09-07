"""
TTS Benchmark Reporter - Generate comprehensive reports and visualizations.

Provides functionality to generate CSV reports, comparison tables, and plots
for TTS benchmark results analysis and presentation.
"""

import csv
import io
import statistics
from typing import List, Dict, Optional, Any
import logging

from .metrics import TTSMetrics
from .tts_benchmark import TTSBenchmarkResult

logger = logging.getLogger(__name__)


class TTSBenchmarkReporter:
    """
    Generate comprehensive reports and visualizations for TTS benchmark results.
    
    Supports CSV generation, comparison tables, and performance plots
    following the recommended TTS benchmarking metrics.
    """
    
    def __init__(self):
        """Initialize the reporter."""
        pass
    
    def generate_csv_report(
        self,
        results: Dict[int, TTSBenchmarkResult],
        test_texts: Optional[List[str]] = None
    ) -> str:
        """
        Generate a CSV report with comprehensive TTS benchmark metrics.
        
        Args:
            results: Dictionary mapping concurrency levels to benchmark results
            test_texts: Optional list of test texts used in benchmark
            
        Returns:
            CSV content as string
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Concurrency",
            "ASPS",
            "RTF_Mean", 
            "RTF_Median",
            "RPS",
            "CPS",
            "Tokens_Per_Sec",
            "TTFA_p50_ms",
            "TTFA_p95_ms",
            "TTFA_p99_ms",
            "TTFA_Mean_ms",
            "Total_Requests",
            "Success_Rate_%",
            "Failed_Requests",
            "Total_Wall_Time_s",
            "Total_Audio_Duration_s",
            "Avg_Text_Length_chars",
            "Avg_Audio_Duration_s",
            "Sample_Rate",
            "Bit_Depth"
        ])
        
        # Data rows
        for concurrency in sorted(results.keys()):
            result = results[concurrency]
            metrics = result.metrics
            
            writer.writerow([
                concurrency,
                f"{metrics.asps:.4f}",
                f"{metrics.rtf_mean:.4f}",
                f"{metrics.rtf_median:.4f}",
                f"{metrics.rps:.2f}",
                f"{metrics.cps:.1f}",
                f"{metrics.tokens_per_sec:.1f}",
                f"{metrics.ttfa_p50 * 1000:.1f}",
                f"{metrics.ttfa_p95 * 1000:.1f}",
                f"{metrics.ttfa_p99 * 1000:.1f}",
                f"{metrics.ttfa_mean * 1000:.1f}",
                metrics.total_requests,
                f"{metrics.success_rate:.1f}",
                metrics.failed_requests,
                f"{metrics.total_wall_time:.2f}",
                f"{metrics.total_audio_duration:.2f}",
                f"{metrics.avg_text_length:.0f}",
                f"{metrics.avg_audio_duration:.2f}",
                metrics.sample_rate,
                metrics.bit_depth
            ])
        
        # Add summary section
        writer.writerow([])
        writer.writerow(["Summary Statistics"])
        writer.writerow(["Metric", "Min", "Max", "Range", "Best_Concurrency"])
        
        if results:
            asps_values = [(c, r.metrics.asps) for c, r in results.items()]
            rtf_values = [(c, r.metrics.rtf_median) for c, r in results.items()]
            rps_values = [(c, r.metrics.rps) for c, r in results.items()]
            ttfa_values = [(c, r.metrics.ttfa_p95) for c, r in results.items()]
            
            best_asps = max(asps_values, key=lambda x: x[1])
            best_rtf = min(rtf_values, key=lambda x: x[1])
            best_rps = max(rps_values, key=lambda x: x[1])
            best_ttfa = min(ttfa_values, key=lambda x: x[1])
            
            asps_min, asps_max = min(v[1] for v in asps_values), max(v[1] for v in asps_values)
            rtf_min, rtf_max = min(v[1] for v in rtf_values), max(v[1] for v in rtf_values)
            rps_min, rps_max = min(v[1] for v in rps_values), max(v[1] for v in rps_values)
            ttfa_min, ttfa_max = min(v[1] for v in ttfa_values), max(v[1] for v in ttfa_values)
            
            writer.writerow(["ASPS", f"{asps_min:.4f}", f"{asps_max:.4f}", 
                           f"{asps_max-asps_min:.4f}", f"{best_asps[0]}"])
            writer.writerow(["RTF_Median", f"{rtf_min:.4f}", f"{rtf_max:.4f}", 
                           f"{rtf_max-rtf_min:.4f}", f"{best_rtf[0]}"])
            writer.writerow(["RPS", f"{rps_min:.2f}", f"{rps_max:.2f}", 
                           f"{rps_max-rps_min:.2f}", f"{best_rps[0]}"])
            writer.writerow(["TTFA_p95_ms", f"{ttfa_min*1000:.1f}", f"{ttfa_max*1000:.1f}", 
                           f"{(ttfa_max-ttfa_min)*1000:.1f}", f"{best_ttfa[0]}"])
        
        # Add test configuration
        if test_texts:
            writer.writerow([])
            writer.writerow(["Test Configuration"])
            writer.writerow(["Test_Texts_Count", len(test_texts)])
            text_lengths = [len(text) for text in test_texts]
            writer.writerow(["Min_Text_Length", min(text_lengths)])
            writer.writerow(["Max_Text_Length", max(text_lengths)])
            writer.writerow(["Avg_Text_Length", statistics.mean(text_lengths)])
            writer.writerow(["Median_Text_Length", statistics.median(text_lengths)])
        
        return output.getvalue()
    
    def generate_comparison_csv(
        self,
        benchmark_results: Dict[str, Dict[int, TTSBenchmarkResult]]
    ) -> str:
        """
        Generate a comparison CSV for multiple benchmark runs.
        
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
                f"ASPS_C{concurrency}",
                f"RTF_C{concurrency}",
                f"RPS_C{concurrency}",
                f"TTFA_p95_ms_C{concurrency}"
            ])
        writer.writerow(header)
        
        # Data rows
        for benchmark_name, results in benchmark_results.items():
            row = [benchmark_name]
            for concurrency in concurrencies:
                if concurrency in results:
                    metrics = results[concurrency].metrics
                    row.extend([
                        f"{metrics.asps:.4f}",
                        f"{metrics.rtf_median:.4f}",
                        f"{metrics.rps:.2f}",
                        f"{metrics.ttfa_p95 * 1000:.1f}"
                    ])
                else:
                    row.extend(["N/A", "N/A", "N/A", "N/A"])
            writer.writerow(row)
        
        # Add best performance summary
        writer.writerow([])
        writer.writerow(["Best Performance Summary"])
        writer.writerow(["Metric", "Concurrency", "Best_Benchmark", "Best_Value"])
        
        for concurrency in concurrencies:
            # Find best ASPS
            asps_values = []
            for name, results in benchmark_results.items():
                if concurrency in results:
                    asps_values.append((name, results[concurrency].metrics.asps))
            
            if asps_values:
                best_asps = max(asps_values, key=lambda x: x[1])
                writer.writerow([f"ASPS_C{concurrency}", concurrency, best_asps[0], f"{best_asps[1]:.4f}"])
        
        return output.getvalue()
    
    def plot_throughput_curve(
        self,
        results: Dict[int, TTSBenchmarkResult],
        output_path: str,
        title: str = "TTS Throughput vs Concurrency"
    ) -> None:
        """
        Generate throughput curve plot (ASPS vs concurrency).
        
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
        asps_values = [results[c].metrics.asps for c in concurrencies]
        rtf_values = [results[c].metrics.rtf_median for c in concurrencies]
        rps_values = [results[c].metrics.rps for c in concurrencies]
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # ASPS plot
        ax1.plot(concurrencies, asps_values, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Concurrency Level')
        ax1.set_ylabel('ASPS (Audio Seconds Per Second)')
        ax1.set_title('ASPS vs Concurrency')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(concurrencies)
        
        # Add value labels
        for i, (c, asps) in enumerate(zip(concurrencies, asps_values)):
            ax1.annotate(f'{asps:.3f}', (c, asps), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        # RTF plot
        ax2.plot(concurrencies, rtf_values, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Concurrency Level')
        ax2.set_ylabel('RTF (Real Time Factor)')
        ax2.set_title('RTF vs Concurrency')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(concurrencies)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Real-time')
        ax2.legend()
        
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
        results: Dict[int, TTSBenchmarkResult],
        output_path: str,
        title: str = "TTS Latency Metrics"
    ) -> None:
        """
        Generate latency metrics plot (TTFA percentiles).
        
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
        ttfa_p50 = [results[c].metrics.ttfa_p50 * 1000 for c in concurrencies]  # Convert to ms
        ttfa_p95 = [results[c].metrics.ttfa_p95 * 1000 for c in concurrencies]
        ttfa_p99 = [results[c].metrics.ttfa_p99 * 1000 for c in concurrencies]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.array(concurrencies)
        width = 0.25
        
        ax.bar(x - width, ttfa_p50, width, label='TTFA p50', alpha=0.8)
        ax.bar(x, ttfa_p95, width, label='TTFA p95', alpha=0.8)
        ax.bar(x + width, ttfa_p99, width, label='TTFA p99', alpha=0.8)
        
        ax.set_xlabel('Concurrency Level')
        ax.set_ylabel('Time to First Audio (ms)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_benchmark_comparison(
        self,
        benchmark_results: Dict[str, Dict[int, TTSBenchmarkResult]],
        output_path: str,
        metric: str = "asps"
    ) -> None:
        """
        Generate comparison plot across multiple benchmarks.
        
        Args:
            benchmark_results: Dictionary mapping benchmark names to results
            output_path: Path to save the plot
            metric: Metric to compare ('asps', 'rtf', 'rps', 'ttfa_p95')
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")
            return
        
        # Determine all concurrency levels
        all_concurrencies = set()
        for results in benchmark_results.values():
            all_concurrencies.update(results.keys())
        concurrencies = sorted(all_concurrencies)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for benchmark_name, results in benchmark_results.items():
            values = []
            for concurrency in concurrencies:
                if concurrency in results:
                    metrics = results[concurrency].metrics
                    if metric == "asps":
                        values.append(metrics.asps)
                    elif metric == "rtf":
                        values.append(metrics.rtf_median)
                    elif metric == "rps":
                        values.append(metrics.rps)
                    elif metric == "ttfa_p95":
                        values.append(metrics.ttfa_p95 * 1000)  # Convert to ms
                    else:
                        values.append(0)
                else:
                    values.append(None)
            
            # Plot line (skip None values)
            valid_x = [c for c, v in zip(concurrencies, values) if v is not None]
            valid_y = [v for v in values if v is not None]
            
            if valid_x:
                ax.plot(valid_x, valid_y, 'o-', linewidth=2, markersize=6, label=benchmark_name)
        
        ax.set_xlabel('Concurrency Level')
        
        ylabel_map = {
            "asps": "ASPS (Audio Seconds Per Second)",
            "rtf": "RTF (Real Time Factor)",
            "rps": "RPS (Requests Per Second)",
            "ttfa_p95": "TTFA p95 (ms)"
        }
        ax.set_ylabel(ylabel_map.get(metric, metric.upper()))
        
        title_map = {
            "asps": "ASPS Comparison Across Benchmarks",
            "rtf": "RTF Comparison Across Benchmarks", 
            "rps": "RPS Comparison Across Benchmarks",
            "ttfa_p95": "TTFA p95 Comparison Across Benchmarks"
        }
        ax.set_title(title_map.get(metric, f"{metric.upper()} Comparison"))
        
        ax.set_xticks(concurrencies)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for RTF = 1.0
        if metric == "rtf":
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Real-time')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary_table(
        self,
        results: Dict[int, TTSBenchmarkResult],
        benchmark_name: str = "TTS Benchmark"
    ) -> None:
        """
        Print a formatted summary table to console.
        
        Args:
            results: Benchmark results
            benchmark_name: Name of the benchmark
        """
        print(f"\n{benchmark_name} Results Summary")
        print("=" * 80)
        print(f"{'Conc':<4} {'ASPS':<8} {'RTF':<8} {'RPS':<8} {'CPS':<8} {'TTFA p95':<10} {'Success':<8}")
        print("-" * 80)
        
        for concurrency in sorted(results.keys()):
            metrics = results[concurrency].metrics
            print(f"{concurrency:<4} "
                  f"{metrics.asps:<8.3f} "
                  f"{metrics.rtf_median:<8.3f} "
                  f"{metrics.rps:<8.1f} "
                  f"{metrics.cps:<8.0f} "
                  f"{metrics.ttfa_p95*1000:<10.1f} "
                  f"{metrics.success_rate:<8.1f}%")
        
        print("=" * 80)
        
        # Find best performing concurrency for each metric
        if results:
            best_asps = max(results.items(), key=lambda x: x[1].metrics.asps)
            best_rtf = min(results.items(), key=lambda x: x[1].metrics.rtf_median)
            best_rps = max(results.items(), key=lambda x: x[1].metrics.rps)
            best_ttfa = min(results.items(), key=lambda x: x[1].metrics.ttfa_p95)
            
            print("Best Performance:")
            print(f"  Highest ASPS: {best_asps[1].metrics.asps:.3f} at concurrency {best_asps[0]}")
            print(f"  Lowest RTF: {best_rtf[1].metrics.rtf_median:.3f} at concurrency {best_rtf[0]}")
            print(f"  Highest RPS: {best_rps[1].metrics.rps:.1f} at concurrency {best_rps[0]}")
            print(f"  Lowest TTFA p95: {best_ttfa[1].metrics.ttfa_p95*1000:.1f}ms at concurrency {best_ttfa[0]}")
            print()
