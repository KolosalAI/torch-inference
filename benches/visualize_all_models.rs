use plotters::prelude::*;
use std::collections::HashMap;
use std::error::Error;

// Model names and colors
const MODELS: &[(&str, RGBColor)] = &[
    ("eva02-large-patch14-448", RGBColor(255, 0, 0)),
    ("eva-giant-patch14-560", RGBColor(0, 0, 255)),
    ("convnextv2-huge-512", RGBColor(0, 255, 0)),
    ("convnext-xxlarge-clip", RGBColor(255, 165, 0)),
    ("maxvit-xlarge-512", RGBColor(128, 0, 128)),
    ("coatnet-3-rw-224", RGBColor(255, 192, 203)),
    ("efficientnetv2-xl", RGBColor(165, 42, 42)),
    ("mobilenetv4-hybrid-large", RGBColor(0, 255, 255)),
    ("vit-giant-patch14-224", RGBColor(255, 255, 0)),
    ("beit-large-patch16-512", RGBColor(128, 128, 128)),
    ("swin-large-patch4-384", RGBColor(0, 128, 128)),
    ("deit3-huge-patch14-224", RGBColor(255, 0, 255)),
];

const CONCURRENCY_LEVELS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

#[derive(Debug)]
struct BenchmarkData {
    model: String,
    concurrency: usize,
    throughput: f64,
}

fn parse_benchmark_results() -> Result<Vec<BenchmarkData>, Box<dyn Error>> {
    // This would parse from criterion output or saved data
    // For now, returning mock data structure
    Ok(Vec::new())
}

fn create_multi_model_chart(data: &[BenchmarkData]) -> Result<(), Box<dyn Error>> {
    let output_path = "benches/data/all_models_throughput_comparison.png";
    let root = BitMapBackend::new(output_path, (1600, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    // Group data by model
    let mut model_data: HashMap<String, Vec<(usize, f64)>> = HashMap::new();
    for entry in data {
        model_data
            .entry(entry.model.clone())
            .or_insert_with(Vec::new)
            .push((entry.concurrency, entry.throughput));
    }

    // Find max throughput for y-axis
    let max_throughput = data
        .iter()
        .map(|d| d.throughput)
        .fold(0.0f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Multi-Model Throughput Comparison (Ultra Optimized)",
            ("sans-serif", 40).into_font(),
        )
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(
            0usize..1100usize,
            0f64..(max_throughput * 1.1),
        )?;

    chart
        .configure_mesh()
        .x_desc("Concurrency Level")
        .y_desc("Throughput (inferences/sec)")
        .x_labels(12)
        .y_labels(10)
        .draw()?;

    // Draw lines for each model
    for (model_name, color) in MODELS {
        if let Some(points) = model_data.get(*model_name) {
            let mut sorted_points = points.clone();
            sorted_points.sort_by_key(|p| p.0);

            chart
                .draw_series(LineSeries::new(
                    sorted_points.iter().map(|&(x, y)| (x, y)),
                    color.stroke_width(2),
                ))?
                .label(*model_name)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], *color));
        }
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .label_font(("sans-serif", 14))
        .draw()?;

    root.present()?;
    println!("✓ Multi-model comparison chart saved to: {}", output_path);
    Ok(())
}

fn create_individual_charts(data: &[BenchmarkData]) -> Result<(), Box<dyn Error>> {
    let model_data: HashMap<String, Vec<(usize, f64)>> = {
        let mut map = HashMap::new();
        for entry in data {
            map.entry(entry.model.clone())
                .or_insert_with(Vec::new)
                .push((entry.concurrency, entry.throughput));
        }
        map
    };

    for (model_name, color) in MODELS {
        if let Some(points) = model_data.get(*model_name) {
            let output_path = format!("benches/data/{}_throughput.png", model_name);
            let root = BitMapBackend::new(&output_path, (1200, 800)).into_drawing_area();
            root.fill(&WHITE)?;

            let max_throughput = points.iter().map(|p| p.1).fold(0.0f64, f64::max);
            let mut sorted_points = points.clone();
            sorted_points.sort_by_key(|p| p.0);

            let mut chart = ChartBuilder::on(&root)
                .caption(
                    format!("{} - Throughput vs Concurrency", model_name),
                    ("sans-serif", 30).into_font(),
                )
                .margin(15)
                .x_label_area_size(50)
                .y_label_area_size(70)
                .build_cartesian_2d(0usize..1100usize, 0f64..(max_throughput * 1.1))?;

            chart
                .configure_mesh()
                .x_desc("Concurrency Level")
                .y_desc("Throughput (inferences/sec)")
                .draw()?;

            chart.draw_series(LineSeries::new(
                sorted_points.iter().map(|&(x, y)| (x, y)),
                color.stroke_width(3),
            ))?;

            chart.draw_series(
                sorted_points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 4, color.filled())),
            )?;

            root.present()?;
            println!("✓ Chart saved: {}", output_path);
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Generating visualization for all models...\n");

    let data = parse_benchmark_results()?;

    if data.is_empty() {
        println!("⚠️  No benchmark data found. Run benchmarks first:");
        println!("   cargo bench --bench all_models_ultra_bench");
        return Ok(());
    }

    create_multi_model_chart(&data)?;
    create_individual_charts(&data)?;

    println!("\n✓ All visualizations generated successfully!");
    Ok(())
}
