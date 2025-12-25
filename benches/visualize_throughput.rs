use plotters::prelude::*;
use std::error::Error;

#[derive(Debug)]
struct BenchmarkData {
    name: String,
    batch_sizes: Vec<usize>,
    throughputs: Vec<f64>, // images per second
}

fn main() -> Result<(), Box<dyn Error>> {
    // Benchmark data from our tests
    let baseline = BenchmarkData {
        name: "Baseline (spawn_blocking)".to_string(),
        batch_sizes: vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        throughputs: vec![77.0, 151.0, 240.0, 297.0, 338.0, 345.0, 364.0, 360.0, 354.0, 351.0, 334.0],
    };
    
    let optimized = BenchmarkData {
        name: "Optimized (Bounded+Rayon)".to_string(),
        batch_sizes: vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        throughputs: vec![81.0, 159.0, 250.0, 310.0, 350.0, 350.0, 353.0, 352.0, 341.0, 341.0, 321.0],
    };
    
    let ultra = BenchmarkData {
        name: "Ultra-Optimized".to_string(),
        batch_sizes: vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        throughputs: vec![211.0, 401.0, 597.0, 747.0, 776.0, 825.0, 845.0, 840.0, 837.0, 799.0, 798.0],
    };
    
    // Generate visualizations
    create_throughput_bar_charts(&baseline, &optimized, &ultra)?;
    create_speedup_chart(&baseline, &ultra)?;
    create_latency_chart(&baseline, &optimized, &ultra)?;
    create_efficiency_chart(&ultra)?;
    
    println!("\n✅ Generated visualization charts:");
    println!("   📊 benches/data/baseline_bar_chart.png");
    println!("   📊 benches/data/optimized_bar_chart.png");
    println!("   📊 benches/data/ultra_bar_chart.png");
    println!("   📈 benches/data/speedup_comparison.png");
    println!("   ⏱️  benches/data/latency_comparison.png");
    println!("   🎯 benches/data/parallel_efficiency.png");
    println!("\nAll charts saved successfully!");
    
    Ok(())
}

/// Create bar charts for each model's throughput
fn create_throughput_bar_charts(
    baseline: &BenchmarkData,
    optimized: &BenchmarkData,
    ultra: &BenchmarkData,
) -> Result<(), Box<dyn Error>> {
    create_single_bar_chart(baseline, "benches/data/baseline_bar_chart.png", &RED)?;
    create_single_bar_chart(optimized, "benches/data/optimized_bar_chart.png", &BLUE)?;
    create_single_bar_chart(ultra, "benches/data/ultra_bar_chart.png", &GREEN)?;
    Ok(())
}

/// Create a single bar chart for a model
fn create_single_bar_chart(
    data: &BenchmarkData,
    filename: &str,
    color: &RGBColor,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(filename, (1600, 900))
        .into_drawing_area();
    root.fill(&WHITE)?;
    
    let max_throughput = data.throughputs.iter().cloned().fold(0.0_f64, f64::max) * 1.1;
    let title = format!("{} - Throughput by Batch Size", data.name);
    
    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 50).into_font().color(&BLACK))
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(80)
        .build_cartesian_2d(
            (0..data.batch_sizes.len()).into_segmented(),
            0f64..max_throughput,
        )?;
    
    chart
        .configure_mesh()
        .x_desc("Batch Size")
        .y_desc("Throughput (images/sec)")
        .x_label_formatter(&|x| {
            if let SegmentValue::CenterOf(idx) = x {
                if *idx < data.batch_sizes.len() {
                    format!("{}", data.batch_sizes[*idx])
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        })
        .y_label_formatter(&|y| format!("{:.0}", y))
        .x_label_style(("sans-serif", 20))
        .y_label_style(("sans-serif", 20))
        .draw()?;
    
    // Draw bars
    chart.draw_series(
        data.batch_sizes.iter().enumerate().map(|(idx, _)| {
            let throughput = data.throughputs[idx];
            let mut bar = Rectangle::new(
                [(SegmentValue::CenterOf(idx), 0.0), (SegmentValue::CenterOf(idx), throughput)],
                color.filled(),
            );
            bar.set_margin(0, 0, 5, 5);
            bar
        })
    )?;
    
    // Add value labels on top of bars
    for (idx, &throughput) in data.throughputs.iter().enumerate() {
        chart.draw_series(std::iter::once(Text::new(
            format!("{:.0}", throughput),
            (SegmentValue::CenterOf(idx), throughput + max_throughput * 0.02),
            ("sans-serif", 18).into_font().color(&BLACK),
        )))?;
    }
    
    root.present()?;
    println!("✓ Generated {}", filename);
    Ok(())
}


/// Create speedup comparison chart
fn create_speedup_chart(
    baseline: &BenchmarkData,
    ultra: &BenchmarkData,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("benches/data/speedup_comparison.png", (1200, 800))
        .into_drawing_area();
    root.fill(&WHITE)?;
    
    // Calculate speedups
    let speedups: Vec<(usize, f64)> = baseline.batch_sizes.iter()
        .zip(baseline.throughputs.iter())
        .zip(ultra.throughputs.iter())
        .map(|((&size, &base), &ultra_val)| (size, ultra_val / base))
        .collect();
    
    let max_speedup = 3.5;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Performance Speedup (Ultra vs Baseline)", ("sans-serif", 40))
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (1usize..1024usize).log_scale(),
            0f64..max_speedup,
        )?;
    
    chart
        .configure_mesh()
        .x_desc("Batch Size (log scale)")
        .y_desc("Speedup (x times faster)")
        .x_label_formatter(&|x| format!("{}", x))
        .y_label_formatter(&|y| format!("{:.2}x", y))
        .draw()?;
    
    // Draw speedup line
    chart.draw_series(LineSeries::new(
        speedups.iter().map(|&(x, y)| (x, y)),
        ShapeStyle::from(&GREEN).stroke_width(3),
    ))?
    .label("Ultra vs Baseline Speedup")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.stroke_width(3)));
    
    // Draw 2x reference line
    chart.draw_series(LineSeries::new(
        vec![(1, 2.0), (1024, 2.0)],
        ShapeStyle::from(&BLACK.mix(0.3)).stroke_width(2),
    ))?
    .label("2x Speedup Reference")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.mix(0.3).stroke_width(2)));
    
    chart.configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    
    root.present()?;
    println!("✓ Generated speedup_comparison.png");
    Ok(())
}

/// Create latency comparison chart
fn create_latency_chart(
    baseline: &BenchmarkData,
    optimized: &BenchmarkData,
    ultra: &BenchmarkData,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("benches/data/latency_comparison.png", (1200, 800))
        .into_drawing_area();
    root.fill(&WHITE)?;
    
    // Convert throughput to per-image latency in ms
    let baseline_latency: Vec<(usize, f64)> = baseline.batch_sizes.iter()
        .zip(baseline.throughputs.iter())
        .map(|(&size, &throughput)| (size, 1000.0 / throughput))
        .collect();
    
    let optimized_latency: Vec<(usize, f64)> = optimized.batch_sizes.iter()
        .zip(optimized.throughputs.iter())
        .map(|(&size, &throughput)| (size, 1000.0 / throughput))
        .collect();
    
    let ultra_latency: Vec<(usize, f64)> = ultra.batch_sizes.iter()
        .zip(ultra.throughputs.iter())
        .map(|(&size, &throughput)| (size, 1000.0 / throughput))
        .collect();
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Per-Image Latency Comparison", ("sans-serif", 40))
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (1usize..1024usize).log_scale(),
            0f64..15f64,
        )?;
    
    chart
        .configure_mesh()
        .x_desc("Batch Size (log scale)")
        .y_desc("Latency per Image (ms)")
        .x_label_formatter(&|x| format!("{}", x))
        .y_label_formatter(&|y| format!("{:.1}ms", y))
        .draw()?;
    
    // Draw baseline
    chart.draw_series(LineSeries::new(
        baseline_latency.iter().map(|&(x, y)| (x, y)),
        ShapeStyle::from(&RED).stroke_width(3),
    ))?
    .label("Baseline")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
    
    // Draw optimized
    chart.draw_series(LineSeries::new(
        optimized_latency.iter().map(|&(x, y)| (x, y)),
        ShapeStyle::from(&BLUE).stroke_width(3),
    ))?
    .label("Optimized")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
    
    // Draw ultra
    chart.draw_series(LineSeries::new(
        ultra_latency.iter().map(|&(x, y)| (x, y)),
        ShapeStyle::from(&GREEN).stroke_width(3),
    ))?
    .label("Ultra-Optimized")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.stroke_width(3)));
    
    chart.configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    
    root.present()?;
    println!("✓ Generated latency_comparison.png");
    Ok(())
}

/// Create parallel efficiency chart
fn create_efficiency_chart(ultra: &BenchmarkData) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("benches/data/parallel_efficiency.png", (1200, 800))
        .into_drawing_area();
    root.fill(&WHITE)?;
    
    // Calculate speedup relative to batch size 1
    let single_throughput = ultra.throughputs[0];
    let speedups: Vec<(usize, f64)> = ultra.batch_sizes.iter()
        .zip(ultra.throughputs.iter())
        .map(|(&size, &throughput)| (size, throughput / single_throughput))
        .collect();
    
    // Calculate efficiency (speedup / theoretical max)
    let num_cores: f64 = 10.0;
    let efficiencies: Vec<(usize, f64)> = speedups.iter()
        .map(|&(size, speedup)| {
            let theoretical_max = num_cores.min(size as f64);
            (size, (speedup / theoretical_max) * 100.0)
        })
        .collect();
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Parallel Efficiency (Ultra-Optimized on 10 cores)", ("sans-serif", 40))
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (1usize..1024usize).log_scale(),
            0f64..100f64,
        )?;
    
    chart
        .configure_mesh()
        .x_desc("Batch Size (log scale)")
        .y_desc("Parallel Efficiency (%)")
        .x_label_formatter(&|x| format!("{}", x))
        .y_label_formatter(&|y| format!("{:.0}%", y))
        .draw()?;
    
    // Draw efficiency
    chart.draw_series(LineSeries::new(
        efficiencies.iter().map(|&(x, y)| (x, y)),
        ShapeStyle::from(&GREEN).stroke_width(3),
    ))?
    .label("Parallel Efficiency")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.stroke_width(3)));
    
    // Draw ideal 100% line
    chart.draw_series(LineSeries::new(
        vec![(1, 100.0), (1024, 100.0)],
        ShapeStyle::from(&BLACK.mix(0.3)).stroke_width(2).stroke_width(2),
    ))?
    .label("Ideal 100% Efficiency")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.mix(0.3).stroke_width(2)));
    
    // Draw 40% reference line (our achievement)
    chart.draw_series(LineSeries::new(
        vec![(64, 40.0), (1024, 40.0)],
        ShapeStyle::from(&RED.mix(0.5)).stroke_width(2).stroke_width(2),
    ))?
    .label("Peak 40% Efficiency (Achieved)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.5).stroke_width(2)));
    
    chart.configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    
    root.present()?;
    println!("✓ Generated parallel_efficiency.png");
    Ok(())
}
