#!/usr/bin/env python3
"""
Advanced Benchmark Suite for Image Classification Models
Provides detailed performance metrics, statistical analysis, and visualization
"""

import json
import time
import requests
import statistics
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

class ImageClassificationBenchmark:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results_dir = Path("benchmark_results")
        self.images_dir = Path("benchmark_images")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        
        # Models to benchmark
        self.models = [
            {"id": "mobilenetv4-hybrid-large", "size": "140MB", "accuracy": "84.36%"},
            {"id": "coatnet-3-rw-224", "size": "700MB", "accuracy": "86.0%"},
            {"id": "swin-large-patch4-384", "size": "790MB", "accuracy": "87.3%"},
            {"id": "efficientnetv2-xl", "size": "850MB", "accuracy": "87.3%"},
            {"id": "eva02-large-patch14-448", "size": "1.2GB", "accuracy": "90.054%"},
        ]
        
        # Test images
        self.test_images = [
            {"name": "cat", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg"},
            {"name": "dog", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Sled_dog_Togo.jpg/400px-Sled_dog_Togo.jpg"},
        ]
        
    def check_server(self) -> bool:
        """Check if server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def download_test_images(self):
        """Download test images"""
        print("Downloading test images...")
        for img in self.test_images:
            img_path = self.images_dir / f"{img['name']}.jpg"
            if img_path.exists():
                print(f"  ✓ {img['name']}.jpg exists")
                continue
            
            try:
                response = requests.get(img['url'])
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ Downloaded {img['name']}.jpg")
            except Exception as e:
                print(f"  ✗ Failed to download {img['name']}.jpg: {e}")
    
    def warmup(self, model_id: str, iterations: int = 3):
        """Warmup model with dummy inferences"""
        print(f"  Warming up (${iterations} iterations)...", end=" ")
        img_path = self.images_dir / "cat.jpg"
        
        for _ in range(iterations):
            try:
                with open(img_path, 'rb') as f:
                    requests.post(
                        f"{self.base_url}/classify",
                        files={"image": f},
                        data={"model": model_id, "top_k": "5"},
                        timeout=30
                    )
            except:
                pass
        print("done")
    
    def benchmark_single_inference(self, model_id: str, image_path: Path) -> Dict[str, Any]:
        """Run single inference and measure performance"""
        start_time = time.time()
        
        try:
            with open(image_path, 'rb') as f:
                response = requests.post(
                    f"{self.base_url}/classify",
                    files={"image": f},
                    data={"model": model_id, "top_k": "5"},
                    timeout=30
                )
            
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                inference_time = result.get('inference_time_ms', elapsed_ms)
                
                return {
                    "success": True,
                    "elapsed_ms": elapsed_ms,
                    "inference_time_ms": inference_time,
                    "response": result
                }
            else:
                return {
                    "success": False,
                    "elapsed_ms": elapsed_ms,
                    "error": response.text
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def benchmark_model(self, model: Dict[str, str], iterations: int = 10) -> Dict[str, Any]:
        """Benchmark a single model"""
        model_id = model['id']
        print(f"\n{'='*70}")
        print(f"Benchmarking: {model_id}")
        print(f"Size: {model['size']} | Accuracy: {model['accuracy']}")
        print(f"{'='*70}\n")
        
        # Warmup
        self.warmup(model_id)
        
        # Collect timing data
        all_times = []
        image_results = {}
        
        for img in self.test_images:
            img_path = self.images_dir / f"{img['name']}.jpg"
            if not img_path.exists():
                continue
            
            print(f"  Testing with {img['name']}.jpg ({iterations} iterations)...")
            
            times = []
            for i in range(iterations):
                result = self.benchmark_single_inference(model_id, img_path)
                if result.get('success'):
                    times.append(result['inference_time_ms'])
                    all_times.append(result['inference_time_ms'])
                else:
                    print(f"    Iteration {i+1} failed: {result.get('error', 'Unknown')}")
            
            if times:
                image_results[img['name']] = {
                    "mean": statistics.mean(times),
                    "median": statistics.median(times),
                    "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                    "min": min(times),
                    "max": max(times),
                    "times": times
                }
                
                print(f"    Mean: {image_results[img['name']]['mean']:.2f}ms")
                print(f"    Min: {image_results[img['name']]['min']:.2f}ms")
                print(f"    Max: {image_results[img['name']]['max']:.2f}ms")
                print(f"    Std Dev: {image_results[img['name']]['stdev']:.2f}ms")
        
        # Overall statistics
        if all_times:
            overall = {
                "mean": statistics.mean(all_times),
                "median": statistics.median(all_times),
                "stdev": statistics.stdev(all_times) if len(all_times) > 1 else 0,
                "min": min(all_times),
                "max": max(all_times),
                "fps": 1000 / statistics.mean(all_times)
            }
            
            print(f"\n  Overall Statistics:")
            print(f"    Mean: {overall['mean']:.2f}ms")
            print(f"    Median: {overall['median']:.2f}ms")
            print(f"    Std Dev: {overall['stdev']:.2f}ms")
            print(f"    Min: {overall['min']:.2f}ms")
            print(f"    Max: {overall['max']:.2f}ms")
            print(f"    FPS: {overall['fps']:.2f}")
            
            return {
                "model_id": model_id,
                "size": model['size'],
                "accuracy": model['accuracy'],
                "overall": overall,
                "images": image_results,
                "iterations": iterations
            }
        
        return None
    
    def generate_json_report(self, results: List[Dict[str, Any]]):
        """Generate JSON report"""
        report_file = self.results_dir / f"benchmark_{self.timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                "timestamp": self.timestamp,
                "date": datetime.now().isoformat(),
                "models_tested": len(results),
                "iterations_per_test": results[0]['iterations'] if results else 0,
                "results": results
            }, f, indent=2)
        
        print(f"\n✓ JSON report: {report_file}")
        return report_file
    
    def generate_markdown_report(self, results: List[Dict[str, Any]]):
        """Generate markdown report"""
        report_file = self.results_dir / f"benchmark_{self.timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Image Classification Benchmark Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Models Tested:** {len(results)}\n")
            f.write(f"**Test Images:** {len(self.test_images)}\n\n")
            
            f.write("---\n\n")
            f.write("## Performance Comparison\n\n")
            
            # Table header
            f.write("| Model | Size | Accuracy | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | FPS |\n")
            f.write("|-------|------|----------|-----------|-------------|----------|----------|-----|\n")
            
            # Table rows
            for r in results:
                o = r['overall']
                f.write(f"| {r['model_id']} | {r['size']} | {r['accuracy']} | "
                       f"{o['mean']:.2f} | {o['median']:.2f} | {o['min']:.2f} | "
                       f"{o['max']:.2f} | {o['fps']:.2f} |\n")
            
            f.write("\n---\n\n")
            f.write("## Detailed Results\n\n")
            
            for r in results:
                f.write(f"### {r['model_id']}\n\n")
                f.write(f"- **Size:** {r['size']}\n")
                f.write(f"- **Accuracy:** {r['accuracy']}\n")
                f.write(f"- **Mean Inference Time:** {r['overall']['mean']:.2f}ms\n")
                f.write(f"- **FPS:** {r['overall']['fps']:.2f}\n\n")
                
                f.write("**Per-Image Results:**\n\n")
                for img_name, stats in r['images'].items():
                    f.write(f"- **{img_name}.jpg:** {stats['mean']:.2f}ms "
                           f"(min: {stats['min']:.2f}ms, max: {stats['max']:.2f}ms)\n")
                f.write("\n")
            
            f.write("---\n\n")
            f.write("## Recommendations\n\n")
            
            # Find best performers
            fastest = min(results, key=lambda x: x['overall']['mean'])
            most_accurate = max(results, key=lambda x: float(x['accuracy'].rstrip('%')))
            
            f.write(f"- **Fastest Model:** {fastest['model_id']} ({fastest['overall']['mean']:.2f}ms)\n")
            f.write(f"- **Most Accurate:** {most_accurate['model_id']} ({most_accurate['accuracy']})\n")
        
        print(f"✓ Markdown report: {report_file}")
        return report_file
    
    def run(self, iterations: int = 10):
        """Run complete benchmark suite"""
        print("\n" + "="*70)
        print("Image Classification Benchmark Suite")
        print("="*70 + "\n")
        
        # Check server
        if not self.check_server():
            print("✗ Server is not running!")
            print("Please start the server first:")
            print("  ./target/release/torch-inference-server")
            return
        
        print("✓ Server is running\n")
        
        # Download test images
        self.download_test_images()
        
        # Run benchmarks
        results = []
        for model in self.models:
            result = self.benchmark_model(model, iterations)
            if result:
                results.append(result)
        
        # Generate reports
        if results:
            print("\n" + "="*70)
            print("Generating Reports")
            print("="*70 + "\n")
            
            self.generate_json_report(results)
            self.generate_markdown_report(results)
            
            print("\n" + "="*70)
            print("Benchmark Complete!")
            print("="*70 + "\n")
        else:
            print("\n✗ No results to report")

if __name__ == "__main__":
    # Parse arguments
    iterations = 10
    if len(sys.argv) > 1:
        try:
            iterations = int(sys.argv[1])
        except:
            pass
    
    # Run benchmark
    benchmark = ImageClassificationBenchmark()
    benchmark.run(iterations=iterations)
