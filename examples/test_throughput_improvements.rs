use std::time::Instant;
use image::{RgbImage, ImageBuffer, imageops};

fn create_test_image(width: u32, height: u32) -> RgbImage {
    ImageBuffer::from_fn(width, height, |x, y| {
        image::Rgb([
            ((x * 255) / width) as u8,
            ((y * 255) / height) as u8,
            128,
        ])
    })
}

#[inline]
fn preprocess_optimized(img: &RgbImage, target_size: (u32, u32)) -> Vec<f32> {
    let resized = imageops::resize(
        img,
        target_size.0,
        target_size.1,
        imageops::FilterType::Triangle,
    );
    
    let mut tensor = Vec::with_capacity((target_size.0 * target_size.1 * 3) as usize);
    let pixels = resized.as_raw();
    const INV_SCALE: f32 = 1.0 / 127.5;
    
    for chunk in pixels.chunks_exact(3) {
        unsafe {
            tensor.push((*chunk.get_unchecked(0) as f32 - 127.5) * INV_SCALE);
            tensor.push((*chunk.get_unchecked(1) as f32 - 127.5) * INV_SCALE);
            tensor.push((*chunk.get_unchecked(2) as f32 - 127.5) * INV_SCALE);
        }
    }
    
    tensor
}

fn preprocess_baseline(img: &RgbImage, target_size: (u32, u32)) -> Vec<f32> {
    let resized = imageops::resize(
        img,
        target_size.0,
        target_size.1,
        imageops::FilterType::Lanczos3,
    );
    
    let mut tensor = Vec::with_capacity((target_size.0 * target_size.1 * 3) as usize);
    
    for pixel in resized.pixels() {
        tensor.push((pixel[0] as f32 / 255.0 - 0.5) * 2.0);
        tensor.push((pixel[1] as f32 / 255.0 - 0.5) * 2.0);
        tensor.push((pixel[2] as f32 / 255.0 - 0.5) * 2.0);
    }
    
    tensor
}

fn main() {
    println!("=== Throughput Optimization Test ===\n");
    
    let img = create_test_image(1920, 1080);
    let target_size = (224, 224);
    
    // Warmup
    for _ in 0..10 {
        let _ = preprocess_baseline(&img, target_size);
        let _ = preprocess_optimized(&img, target_size);
    }
    
    // Test single image performance
    println!("Single Image Performance:");
    
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = preprocess_baseline(&img, target_size);
    }
    let baseline_time = start.elapsed().as_secs_f64();
    let baseline_throughput = iterations as f64 / baseline_time;
    println!("  Baseline (Lanczos3):  {:.2} img/sec ({:.2} ms/img)", 
             baseline_throughput, 1000.0 / baseline_throughput);
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = preprocess_optimized(&img, target_size);
    }
    let optimized_time = start.elapsed().as_secs_f64();
    let optimized_throughput = iterations as f64 / optimized_time;
    println!("  Optimized (Triangle):  {:.2} img/sec ({:.2} ms/img)", 
             optimized_throughput, 1000.0 / optimized_throughput);
    println!("  Speedup: {:.2}x\n", optimized_throughput / baseline_throughput);
    
    // Test batch throughput scaling
    println!("Batch Throughput Scaling:");
    let batch_sizes = vec![1, 8, 32, 128, 512, 1024];
    
    for batch_size in batch_sizes {
        let images: Vec<_> = (0..batch_size)
            .map(|_| create_test_image(1920, 1080))
            .collect();
        
        let start = Instant::now();
        let _results: Vec<_> = images.iter()
            .map(|img| preprocess_optimized(img, target_size))
            .collect();
        let elapsed = start.elapsed().as_secs_f64();
        let throughput = batch_size as f64 / elapsed;
        
        println!("  Batch {:4}: {:.2} img/sec ({:.2} ms/img)", 
                 batch_size, throughput, 1000.0 / throughput);
    }
}
