#!/usr/bin/env python3
"""
Performance Benchmark Script for Latency Optimization

This script benchmarks the optimized inference pipeline to validate 
the latency improvements and ensure we meet the <600ms target.
"""

import asyncio
import time
import statistics
import requests
import json
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


class LatencyBenchmark:
    """
    Comprehensive latency benchmark for the inference system.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def benchmark_single_request(self, data: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
        """Benchmark a single request and return metrics."""
        start_time = time.perf_counter()
        
        try:
            response = self.session.post(
                f"{self.base_url}/example/predict",
                json=data,
                timeout=timeout
            )
            
            end_time = time.perf_counter()
            latency = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "latency_seconds": latency,
                    "latency_ms": latency * 1000,
                    "processing_time": result.get("processing_time", 0),
                    "status_code": response.status_code,
                    "response_size": len(response.content)
                }
            else:
                return {
                    "success": False,
                    "latency_seconds": latency,
                    "latency_ms": latency * 1000,
                    "status_code": response.status_code,
                    "error": response.text
                }
                
        except Exception as e:
            end_time = time.perf_counter()
            latency = end_time - start_time
            return {
                "success": False,
                "latency_seconds": latency,
                "latency_ms": latency * 1000,
                "error": str(e)
            }
    
    def run_sequential_benchmark(self, num_requests: int = 100, 
                                input_data: Any = None) -> Dict[str, Any]:
        """Run sequential requests to measure individual latency."""
        print(f"🔄 Running sequential benchmark with {num_requests} requests...")
        
        if input_data is None:
            input_data = [1.0, 2.0, 3.0, 4.0, 5.0]  # Simple numeric input
        
        request_data = {
            "inputs": input_data,
            "priority": 0,
            "timeout": 5.0,
            "enable_batching": True
        }
        
        results = []
        for i in range(num_requests):
            if i % 20 == 0:
                print(f"  Progress: {i}/{num_requests}")
            
            result = self.benchmark_single_request(request_data)
            results.append(result)
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.01)
        
        return self._analyze_results(results, "Sequential")
    
    def run_concurrent_benchmark(self, num_requests: int = 100, 
                                concurrency: int = 10,
                                input_data: Any = None) -> Dict[str, Any]:
        """Run concurrent requests to measure throughput and latency under load."""
        print(f"🚀 Running concurrent benchmark with {num_requests} requests, "
              f"concurrency: {concurrency}...")
        
        if input_data is None:
            input_data = [1.0, 2.0, 3.0, 4.0, 5.0]  # Simple numeric input
        
        request_data = {
            "inputs": input_data,
            "priority": 0,
            "timeout": 5.0,
            "enable_batching": True
        }
        
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all requests
            futures = [
                executor.submit(self.benchmark_single_request, request_data)
                for _ in range(num_requests)
            ]
            
            # Collect results
            for i, future in enumerate(as_completed(futures)):
                if i % 20 == 0:
                    print(f"  Completed: {i}/{num_requests}")
                
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "success": False,
                        "latency_seconds": 0,
                        "latency_ms": 0,
                        "error": str(e)
                    })
        
        return self._analyze_results(results, "Concurrent")
    
    def run_batch_benchmark(self, batch_sizes: List[int] = None,
                           iterations: int = 20) -> Dict[str, Any]:
        """Benchmark different batch sizes to find optimal configuration."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        
        print(f"📊 Running batch size benchmark with sizes: {batch_sizes}")
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Create batch input
            batch_input = [[i + j for j in range(5)] for i in range(batch_size)]
            
            request_data = {
                "inputs": batch_input,
                "priority": 0,
                "timeout": 10.0,
                "enable_batching": True
            }
            
            results = []
            for i in range(iterations):
                result = self.benchmark_single_request(request_data, timeout=10.0)
                results.append(result)
                time.sleep(0.05)  # Small delay between tests
            
            analysis = self._analyze_results(results, f"Batch-{batch_size}")
            batch_results[batch_size] = analysis
        
        return batch_results
    
    def _analyze_results(self, results: List[Dict[str, Any]], 
                        test_name: str) -> Dict[str, Any]:
        """Analyze benchmark results and compute statistics."""
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        if not successful_results:
            return {
                "test_name": test_name,
                "total_requests": len(results),
                "successful_requests": 0,
                "failed_requests": len(failed_results),
                "success_rate": 0.0,
                "error": "All requests failed"
            }
        
        latencies_ms = [r["latency_ms"] for r in successful_results]
        processing_times = [r.get("processing_time", 0) for r in successful_results if r.get("processing_time")]
        
        analysis = {
            "test_name": test_name,
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate": len(successful_results) / len(results) * 100,
            
            # Latency statistics (end-to-end)
            "latency_stats_ms": {
                "min": min(latencies_ms),
                "max": max(latencies_ms),
                "mean": statistics.mean(latencies_ms),
                "median": statistics.median(latencies_ms),
                "p95": self._percentile(latencies_ms, 95),
                "p99": self._percentile(latencies_ms, 99),
                "std": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0
            },
            
            # Throughput
            "throughput": {
                "requests_per_second": len(successful_results) / (max(latencies_ms) / 1000) if latencies_ms else 0,
                "avg_requests_per_second": 1000 / statistics.mean(latencies_ms) if latencies_ms else 0
            }
        }
        
        # Processing time statistics (if available)
        if processing_times:
            processing_times_ms = [t * 1000 for t in processing_times]
            analysis["processing_time_stats_ms"] = {
                "min": min(processing_times_ms),
                "max": max(processing_times_ms),
                "mean": statistics.mean(processing_times_ms),
                "median": statistics.median(processing_times_ms),
                "p95": self._percentile(processing_times_ms, 95),
                "p99": self._percentile(processing_times_ms, 99)
            }
        
        # Performance assessment
        mean_latency = analysis["latency_stats_ms"]["mean"]
        analysis["performance_assessment"] = self._assess_performance(mean_latency)
        
        return analysis
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile / 100 * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _assess_performance(self, mean_latency_ms: float) -> Dict[str, Any]:
        """Assess performance against targets."""
        target_latency = 600  # ms
        
        if mean_latency_ms <= target_latency:
            status = "✅ EXCELLENT"
            improvement_needed = 0
        elif mean_latency_ms <= target_latency * 1.2:
            status = "🟡 GOOD"
            improvement_needed = mean_latency_ms - target_latency
        elif mean_latency_ms <= target_latency * 2:
            status = "🟠 NEEDS IMPROVEMENT"
            improvement_needed = mean_latency_ms - target_latency
        else:
            status = "❌ POOR"
            improvement_needed = mean_latency_ms - target_latency
        
        return {
            "status": status,
            "target_latency_ms": target_latency,
            "actual_latency_ms": mean_latency_ms,
            "improvement_needed_ms": improvement_needed,
            "performance_ratio": target_latency / mean_latency_ms
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmark types and provide comprehensive report."""
        print("🔍 Starting comprehensive latency benchmark...")
        print("=" * 80)
        
        # Wait for server to be ready
        print("🏥 Checking server health...")
        try:
            health_response = self.session.get(f"{self.base_url}/health", timeout=5)
            if health_response.status_code == 200:
                print("✅ Server is healthy and ready")
            else:
                print(f"⚠️ Server health check returned: {health_response.status_code}")
        except Exception as e:
            print(f"❌ Server health check failed: {e}")
            return {"error": "Server not available"}
        
        results = {}
        
        # 1. Sequential benchmark
        results["sequential"] = self.run_sequential_benchmark(100)
        
        # 2. Concurrent benchmark
        results["concurrent"] = self.run_concurrent_benchmark(100, 10)
        
        # 3. Batch benchmark
        results["batch_sizes"] = self.run_batch_benchmark([1, 2, 4, 8, 16])
        
        # 4. Overall assessment
        results["overall_assessment"] = self._generate_overall_assessment(results)
        
        return results
    
    def _generate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall performance assessment."""
        sequential_latency = results["sequential"]["latency_stats_ms"]["mean"]
        concurrent_latency = results["concurrent"]["latency_stats_ms"]["mean"]
        
        best_batch_latency = float('inf')
        best_batch_size = 1
        
        for batch_size, batch_result in results["batch_sizes"].items():
            if isinstance(batch_result, dict) and "latency_stats_ms" in batch_result:
                batch_latency = batch_result["latency_stats_ms"]["mean"]
                if batch_latency < best_batch_latency:
                    best_batch_latency = batch_latency
                    best_batch_size = batch_size
        
        target_met = all([
            sequential_latency <= 600,
            concurrent_latency <= 600,
            best_batch_latency <= 600
        ])
        
        return {
            "target_met": target_met,
            "target_latency_ms": 600,
            "best_configuration": {
                "sequential_latency_ms": sequential_latency,
                "concurrent_latency_ms": concurrent_latency,
                "best_batch_size": best_batch_size,
                "best_batch_latency_ms": best_batch_latency
            },
            "recommendation": self._get_performance_recommendation(
                sequential_latency, concurrent_latency, best_batch_latency, best_batch_size
            )
        }
    
    def _get_performance_recommendation(self, seq_latency: float, conc_latency: float,
                                      batch_latency: float, batch_size: int) -> str:
        """Get performance recommendation based on results."""
        if all(l <= 600 for l in [seq_latency, conc_latency, batch_latency]):
            return f"🎉 Performance target achieved! Optimal batch size: {batch_size}"
        elif batch_latency <= 600:
            return f"✅ Use batch size {batch_size} for optimal performance"
        elif seq_latency <= 600:
            return "🔄 Use sequential processing for best latency"
        else:
            return "⚠️ Performance optimization needed - consider model optimizations"
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results."""
        print("\n" + "=" * 80)
        print("  LATENCY BENCHMARK RESULTS")
        print("=" * 80)
        
        if "error" in results:
            print(f"❌ Benchmark failed: {results['error']}")
            return
        
        # Sequential results
        if "sequential" in results:
            seq = results["sequential"]
            print(f"\n📊 SEQUENTIAL BENCHMARK:")
            print(f"   • Requests: {seq['successful_requests']}/{seq['total_requests']}")
            print(f"   • Success Rate: {seq['success_rate']:.1f}%")
            print(f"   • Mean Latency: {seq['latency_stats_ms']['mean']:.1f}ms")
            print(f"   • P95 Latency: {seq['latency_stats_ms']['p95']:.1f}ms")
            print(f"   • P99 Latency: {seq['latency_stats_ms']['p99']:.1f}ms")
            print(f"   • Assessment: {seq['performance_assessment']['status']}")
        
        # Concurrent results
        if "concurrent" in results:
            conc = results["concurrent"]
            print(f"\n🚀 CONCURRENT BENCHMARK:")
            print(f"   • Requests: {conc['successful_requests']}/{conc['total_requests']}")
            print(f"   • Success Rate: {conc['success_rate']:.1f}%")
            print(f"   • Mean Latency: {conc['latency_stats_ms']['mean']:.1f}ms")
            print(f"   • P95 Latency: {conc['latency_stats_ms']['p95']:.1f}ms")
            print(f"   • P99 Latency: {conc['latency_stats_ms']['p99']:.1f}ms")
            print(f"   • Throughput: {conc['throughput']['avg_requests_per_second']:.1f} req/s")
            print(f"   • Assessment: {conc['performance_assessment']['status']}")
        
        # Batch results
        if "batch_sizes" in results:
            print(f"\n📊 BATCH SIZE OPTIMIZATION:")
            for batch_size, batch_result in results["batch_sizes"].items():
                if isinstance(batch_result, dict) and "latency_stats_ms" in batch_result:
                    latency = batch_result["latency_stats_ms"]["mean"]
                    print(f"   • Batch {batch_size}: {latency:.1f}ms average")
        
        # Overall assessment
        if "overall_assessment" in results:
            overall = results["overall_assessment"]
            print(f"\n🎯 OVERALL ASSESSMENT:")
            print(f"   • Target Met: {'✅ YES' if overall['target_met'] else '❌ NO'}")
            print(f"   • Target Latency: {overall['target_latency_ms']}ms")
            print(f"   • Best Config: {overall['recommendation']}")
        
        print("=" * 80)


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Latency Benchmark for PyTorch Inference")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the inference server")
    parser.add_argument("--sequential", type=int, default=100,
                       help="Number of sequential requests")
    parser.add_argument("--concurrent", type=int, default=100,
                       help="Number of concurrent requests")
    parser.add_argument("--concurrency", type=int, default=10,
                       help="Concurrency level")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16],
                       help="Batch sizes to test")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = LatencyBenchmark(args.url)
    
    try:
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark()
        
        # Print results
        benchmark.print_results(results)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Results saved to: {args.output}")
        
        # Exit with appropriate code
        overall = results.get("overall_assessment", {})
        if overall.get("target_met", False):
            print("\n🎉 Performance target achieved!")
            exit(0)
        else:
            print("\n⚠️ Performance target not met - optimization needed")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
