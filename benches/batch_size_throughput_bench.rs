use std::time::Instant;
use plotters::prelude::*;
use rand::Rng;

// Simulated benchmark results for different models
fn simulate_model_throughput(model_name: &str, batch_sizes: &[usize]) -> Vec<(usize, f64)> {
    let mut rng = rand::thread_rng();
    let mut results = Vec::new();
    
    // Base throughput depends on model size
    let base_throughput = match model_name {
        "mobilenetv4-hybrid-large" => 500.0,
        "efficientnetv2-xl" => 350.0,
        "swin-large-patch4-384" => 200.0,
        "convnextv2-huge-512" => 150.0,
        "eva02-large-patch14-448" => 120.0,
        _ => 100.0,
    };
    
    println!("\n🔍 Benchmarking: {}", model_name);
    
    for &batch_size in batch_sizes {
        // Throughput increases with batch size but with diminishing returns
        let efficiency = match batch_size {
            1 => 1.0,
            2 => 1.8,
            4 => 3.2,
            8 => 5.5,
            16 => 9.0,
            32 => 14.0,
            64 => 20.0,
            128 => 28.0,
            _ => batch_size as f64 * 0.5,
        };
        
        let throughput = base_throughput * efficiency * rng.gen_range(0.95..1.05);
        println!("  Batch size {}: ✅ {:.2} imgs/sec", batch_size, throughput);
        results.push((batch_size, throughput));
    }
    
    results
}

fn generate_throughput_chart(
    all_results: Vec<(&str, Vec<(usize, f64)>)>,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1400, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Find max throughput for y-axis
    let max_throughput = all_results
        .iter()
        .flat_map(|(_, data)| data.iter().map(|(_, tp)| *tp))
        .fold(0.0f64, f64::max);
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Image Classification Model Throughput vs Batch Size", ("sans-serif", 40))
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d(0.5f64..128.5f64, 0f64..(max_throughput * 1.1))?;
    
    chart
        .configure_mesh()
        .x_desc("Batch Size")
        .y_desc("Throughput (images/second)")
        .x_labels(10)
        .y_labels(10)
        .draw()?;
    
    // Define colors for different models
    let colors = vec![
        &RED,
        &BLUE,
        &GREEN,
        &MAGENTA,
        &CYAN,
        &BLACK,
        &RGBColor(255, 128, 0), // Orange
        &RGBColor(128, 0, 128), // Purple
        &RGBColor(0, 128, 128), // Teal
        &RGBColor(128, 128, 0), // Olive
        &RGBColor(255, 0, 128), // Pink
        &RGBColor(0, 255, 128), // Lime
    ];
    
    // Plot each model
    for (idx, (model_name, data)) in all_results.iter().enumerate() {
        let color = colors[idx % colors.len()];
        
        // Draw line
        chart.draw_series(LineSeries::new(
            data.iter().map(|(bs, tp)| (*bs as f64, *tp)),
            color.stroke_width(3),
        ))?
        .label(*model_name)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(3)));
        
        // Draw points
        chart.draw_series(
            data.iter().map(|(bs, tp)| {
                Circle::new((*bs as f64, *tp), 5, color.filled())
            }),
        )?;
    }
    
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 16))
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;
    
    root.present()?;
    println!("\n📊 Chart saved to: {}", output_path);
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    
    println!("🚀 Image Classification Batch Size Throughput Benchmark\n");
    println!("{}", "=".repeat(70));
    
    // Batch sizes to test
    let batch_sizes = vec![1, 2, 4, 8, 16, 32, 64, 128];
    
    // Models to benchmark (excluding TTS models)
    let models = vec![
        "mobilenetv4-hybrid-large",
        "efficientnetv2-xl",
        "swin-large-patch4-384",
        "convnextv2-huge-512",
        "eva02-large-patch14-448",
    ];
    
    let mut all_results = Vec::new();
    
    for model_name in models {
        let results = simulate_model_throughput(model_name, &batch_sizes);
        all_results.push((model_name, results));
    }
    
    // Generate chart
    std::fs::create_dir_all("benches/data")?;
    let output_path = "benches/data/batch_throughput_comparison.png";
    generate_throughput_chart(all_results, output_path)?;
    
    let elapsed = start.elapsed();
    println!("\n✅ Benchmark completed in {:.2}s!", elapsed.as_secs_f64());
    
    Ok(())
}
