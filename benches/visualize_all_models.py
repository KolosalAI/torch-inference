#!/usr/bin/env python3
"""
Generate multi-model throughput visualization
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data(csv_path):
    """Load benchmark data from CSV"""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['model']
            if model not in data:
                data[model] = {'batch_sizes': [], 'throughput': [], 'latency': []}
            data[model]['batch_sizes'].append(int(row['batch_size']))
            data[model]['throughput'].append(float(row['throughput_img_per_sec']))
            data[model]['latency'].append(float(row['latency_ms']))
    return data

def create_multi_model_chart(data, output_path):
    """Create multi-line chart for all models"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(data)))
    
    for (model_name, model_data), color in zip(data.items(), colors):
        batch_sizes = model_data['batch_sizes']
        throughput = model_data['throughput']
        latency = model_data['latency']
        
        # Throughput plot
        ax1.plot(batch_sizes, throughput, marker='o', linewidth=2, 
                label=model_name, color=color, markersize=6)
        
        # Latency plot
        ax2.plot(batch_sizes, latency, marker='s', linewidth=2,
                label=model_name, color=color, markersize=6)
    
    # Configure throughput plot
    ax1.set_xlabel('Batch Size (Concurrency)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (images/sec)', fontsize=12, fontweight='bold')
    ax1.set_title('Ultra-Optimized Image Processing: Throughput vs Batch Size', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.set_xscale('log', base=2)
    
    # Configure latency plot
    ax2.set_xlabel('Batch Size (Concurrency)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latency per Image (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Ultra-Optimized Image Processing: Latency vs Batch Size',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Multi-model chart saved to: {output_path}")
    plt.close()

def create_throughput_scaling_chart(data, output_path):
    """Create chart showing throughput scaling"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(data)))
    
    for (model_name, model_data), color in zip(data.items(), colors):
        batch_sizes = model_data['batch_sizes']
        throughput = model_data['throughput']
        
        ax.plot(batch_sizes, throughput, marker='o', linewidth=2.5,
               label=model_name, color=color, markersize=7)
    
    # Add ideal linear scaling reference
    if data:
        first_model = list(data.values())[0]
        batch_sizes = first_model['batch_sizes']
        baseline_throughput = first_model['throughput'][0]
        ideal_scaling = [baseline_throughput * (bs / batch_sizes[0]) for bs in batch_sizes]
        ax.plot(batch_sizes, ideal_scaling, '--', color='gray', linewidth=2,
               label='Ideal Linear Scaling', alpha=0.7)
    
    ax.set_xlabel('Batch Size (Concurrency)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (images/sec)', fontsize=13, fontweight='bold')
    ax.set_title('Multi-Model Throughput Comparison\nUltra-Optimized Image Processing Pipeline',
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Throughput scaling chart saved to: {output_path}")
    plt.close()

def main():
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║         Multi-Model Visualization Generator           ║")
    print("╚════════════════════════════════════════════════════════╝\n")
    
    csv_path = Path("benches/data/all_models_throughput.csv")
    output_dir = Path("benches/data")
    
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found!")
        print("   Run benchmark first: python3 benches/run_all_models_bench.py")
        return
    
    print("Loading benchmark data...")
    data = load_data(csv_path)
    print(f"  Loaded data for {len(data)} model(s)")
    
    print("\nGenerating visualizations...")
    create_multi_model_chart(data, output_dir / "all_models_comparison.png")
    create_throughput_scaling_chart(data, output_dir / "throughput_scaling.png")
    
    print("\n✓ All visualizations generated successfully!\n")

if __name__ == "__main__":
    main()
