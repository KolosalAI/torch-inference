use std::time::Instant;
use torch_inference::ultra_optimized_processor::{UltraOptimizedProcessor, create_test_image_fast};
use std::fs::{create_dir_all, File};
use std::io::Write;

const BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

fn main() {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║        All Models Ultra-Optimized Benchmark           ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    let processor = UltraOptimizedProcessor::new(None);
    let target_size = (224, 224);

    // Create output directory
    create_dir_all("benches/data").unwrap();
    let mut csv = File::create("benches/data/all_models_throughput.csv").unwrap();
    writeln!(csv, "model,batch_size,throughput_img_per_sec,latency_ms").unwrap();

    println!("Running benchmarks for image preprocessing (ultra-optimized)...\n");

    for &batch_size in BATCH_SIZES {
        println!("═══ Batch Size: {} ═══", batch_size);
        
        // Create test images
        let images: Vec<_> = (0..batch_size)
            .map(|_| create_test_image_fast(224, 224))
            .collect();

        // Warm-up
        for _ in 0..3 {
            let _ = processor.process_batch_ultra(&images, target_size);
        }

        // Benchmark
        let iterations = if batch_size <= 64 { 10 } else { 5 };
        let mut total_time = 0.0;

        for _ in 0..iterations {
            let start = Instant::now();
            let _ = processor.process_batch_ultra(&images, target_size);
            total_time += start.elapsed().as_secs_f64();
        }

        let avg_time = total_time / iterations as f64;
        let throughput = batch_size as f64 / avg_time;
        let latency_per_image = (avg_time * 1000.0) / batch_size as f64;

        println!("  Throughput: {:.2} images/sec", throughput);
        println!("  Latency per image: {:.2} ms", latency_per_image);
        println!("  Total batch time: {:.2} ms\n", avg_time * 1000.0);

        // Write to CSV (using "image-preprocessing" as model name)
        writeln!(csv, "image-preprocessing,{},{:.2},{:.2}", 
                 batch_size, throughput, latency_per_image).unwrap();
    }

    println!("✓ Benchmark complete!");
    println!("✓ Results saved to: benches/data/all_models_throughput.csv\n");
}
