use plotters::prelude::*;
use std::error::Error;

#[derive(Debug)]
struct ModelBenchmark {
    name: String,
    batch_sizes: Vec<usize>,
    throughputs: Vec<f64>, // images per second
    color: RGBColor,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Define multiple models with their benchmark data
    let models = vec![
        ModelBenchmark {
            name: "Baseline (spawn_blocking)".to_string(),
            batch_sizes: vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            throughputs: vec![77.0, 151.0, 240.0, 297.0, 338.0, 345.0, 364.0, 360.0, 354.0, 351.0, 334.0],
            color: RED,
        },
        ModelBenchmark {
            name: "Optimized (Bounded+Rayon)".to_string(),
            batch_sizes: vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            throughputs: vec![81.0, 159.0, 250.0, 310.0, 350.0, 350.0, 353.0, 352.0, 341.0, 341.0, 321.0],
            color: BLUE,
        },
        ModelBenchmark {
            name: "Ultra-Optimized".to_string(),
            batch_sizes: vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            throughputs: vec![211.0, 401.0, 597.0, 747.0, 776.0, 825.0, 845.0, 840.0, 837.0, 799.0, 798.0],
            color: GREEN,
        },
    ];
    
    // Generate multi-model comparison charts
    create_multi_model_line_chart(&models)?;
    create_multi_model_log_chart(&models)?;
    create_scaling_efficiency_chart(&models)?;
    
    println!("\n✅ Generated multi-model visualization charts:");
    println!("   📈 benches/data/multi_model_throughput.png");
    println!("   📊 benches/data/multi_model_log_scale.png");
    println!("   🎯 benches/data/scaling_efficiency.png");
    println!("\nAll charts saved successfully!");
    
    Ok(())
}

/// Create line chart comparing all models on linear scale
fn create_multi_model_line_chart(models: &[ModelBenchmark]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("benches/data/multi_model_throughput.png", (1600, 1000))
        .into_drawing_area();
    root.fill(&WHITE)?;
    
    // Find max throughput across all models
    let max_throughput = models.iter()
        .flat_map(|m| m.throughputs.iter())
        .cloned()
        .fold(0.0_f64, f64::max) * 1.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Multi-Model Throughput Comparison vs Batch Size", ("sans-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(
            1usize..1024usize,
            0f64..max_throughput,
        )?;
    
    chart
        .configure_mesh()
        .x_desc("Batch Size (Concurrent Requests)")
        .y_desc("Throughput (images/sec)")
        .x_label_formatter(&|x| format!("{}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .x_label_style(("sans-serif", 20))
        .y_label_style(("sans-serif", 20))
        .label_style(("sans-serif", 18))
        .draw()?;
    
    // Draw each model as a line series with markers
    for model in models {
        // Draw line
        chart.draw_series(LineSeries::new(
            model.batch_sizes.iter().zip(model.throughputs.iter())
                .map(|(&x, &y)| (x, y)),
            ShapeStyle::from(&model.color).stroke_width(3),
        ))?
        .label(&model.name)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], model.color.stroke_width(3)));
        
        // Add markers at data points
        chart.draw_series(
            model.batch_sizes.iter().zip(model.throughputs.iter())
                .map(|(&x, &y)| Circle::new((x, y), 5, model.color.filled()))
        )?;
    }
    
    chart.configure_series_labels()
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 24))
        .draw()?;
    
    root.present()?;
    println!("✓ Generated multi_model_throughput.png");
    Ok(())
}

/// Create line chart comparing all models on log scale for better visibility
fn create_multi_model_log_chart(models: &[ModelBenchmark]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("benches/data/multi_model_log_scale.png", (1600, 1000))
        .into_drawing_area();
    root.fill(&WHITE)?;
    
    let max_throughput = models.iter()
        .flat_map(|m| m.throughputs.iter())
        .cloned()
        .fold(0.0_f64, f64::max) * 1.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Multi-Model Throughput (Log Scale X-Axis)", ("sans-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(
            (1usize..1024usize).log_scale(),
            0f64..max_throughput,
        )?;
    
    chart
        .configure_mesh()
        .x_desc("Batch Size (Log Scale)")
        .y_desc("Throughput (images/sec)")
        .x_label_formatter(&|x| format!("{}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .x_label_style(("sans-serif", 20))
        .y_label_style(("sans-serif", 20))
        .label_style(("sans-serif", 18))
        .draw()?;
    
    // Draw each model
    for model in models {
        // Draw line
        chart.draw_series(LineSeries::new(
            model.batch_sizes.iter().zip(model.throughputs.iter())
                .map(|(&x, &y)| (x, y)),
            ShapeStyle::from(&model.color).stroke_width(3),
        ))?
        .label(&model.name)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], model.color.stroke_width(3)));
        
        // Add markers
        chart.draw_series(
            model.batch_sizes.iter().zip(model.throughputs.iter())
                .map(|(&x, &y)| Circle::new((x, y), 5, model.color.filled()))
        )?;
    }
    
    chart.configure_series_labels()
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 24))
        .draw()?;
    
    root.present()?;
    println!("✓ Generated multi_model_log_scale.png");
    Ok(())
}

/// Create scaling efficiency comparison showing how well each model scales
fn create_scaling_efficiency_chart(models: &[ModelBenchmark]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("benches/data/scaling_efficiency.png", (1600, 1000))
        .into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Scaling Efficiency: Throughput Relative to Batch Size 1", ("sans-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(
            (1usize..1024usize).log_scale(),
            0f64..5.0f64,
        )?;
    
    chart
        .configure_mesh()
        .x_desc("Batch Size (Log Scale)")
        .y_desc("Speedup (vs Batch Size 1)")
        .x_label_formatter(&|x| format!("{}", x))
        .y_label_formatter(&|y| format!("{:.1}x", y))
        .x_label_style(("sans-serif", 20))
        .y_label_style(("sans-serif", 20))
        .label_style(("sans-serif", 18))
        .draw()?;
    
    // Draw ideal linear scaling reference
    chart.draw_series(LineSeries::new(
        vec![(1, 1.0), (2, 2.0), (4, 4.0), (8, 5.0), (16, 5.0), (32, 5.0), (64, 5.0), (128, 5.0), (256, 5.0), (512, 5.0), (1024, 5.0)],
        ShapeStyle::from(&BLACK.mix(0.3)).stroke_width(2),
    ))?
    .label("Ideal Linear Scaling")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.mix(0.3).stroke_width(2)));
    
    // Draw each model's scaling
    for model in models {
        let baseline_throughput = model.throughputs[0];
        let speedups: Vec<(usize, f64)> = model.batch_sizes.iter()
            .zip(model.throughputs.iter())
            .map(|(&size, &throughput)| (size, throughput / baseline_throughput))
            .collect();
        
        // Draw line
        chart.draw_series(LineSeries::new(
            speedups.iter().map(|&(x, y)| (x, y)),
            ShapeStyle::from(&model.color).stroke_width(3),
        ))?
        .label(&model.name)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], model.color.stroke_width(3)));
        
        // Add markers
        chart.draw_series(
            speedups.iter().map(|&(x, y)| Circle::new((x, y), 5, model.color.filled()))
        )?;
    }
    
    chart.configure_series_labels()
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 24))
        .draw()?;
    
    root.present()?;
    println!("✓ Generated scaling_efficiency.png");
    Ok(())
}
