use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use torch_inference::image_processor::{ImageProcessor, create_test_image};

// All non-TTS models from registry
const MODELS: &[(&str, &str)] = &[
    ("eva02-large-patch14-448", "timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"),
    ("eva-giant-patch14-560", "timm/eva_giant_patch14_560.m30m_ft_in22k_in1k"),
    ("convnextv2-huge-512", "timm/convnextv2_huge.fcmae_ft_in22k_in1k_512"),
    ("convnext-xxlarge-clip", "timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k"),
    ("maxvit-xlarge-512", "timm/maxvit_xlarge_tf_512.in21k_ft_in1k"),
    ("coatnet-3-rw-224", "timm/coatnet_3_rw_224.sw_in1k"),
    ("efficientnetv2-xl", "timm/tf_efficientnetv2_xl.in21k_ft_in1k"),
    ("mobilenetv4-hybrid-large", "timm/mobilenetv4_hybrid_large.e600_r448_in1k"),
    ("vit-giant-patch14-224", "timm/vit_giant_patch14_224.clip_laion2b"),
    ("beit-large-patch16-512", "timm/beit_large_patch16_512.in22k_ft_in22k_in1k"),
    ("swin-large-patch4-384", "timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k"),
    ("deit3-huge-patch14-224", "timm/deit3_huge_patch14_224.fb_in22k_ft_in1k"),
];

const CONCURRENCY_LEVELS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

fn benchmark_all_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_models_batch_processing");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    let processor = ImageProcessor::new(64);
    let target_size = (224, 224);

    for (model_name, _model_path) in MODELS {
        println!("\n=== Benchmarking {} ===", model_name);

        for &batch_size in CONCURRENCY_LEVELS {
            let images: Vec<_> = (0..batch_size)
                .map(|_| create_test_image(224, 224))
                .collect();

            let bench_id = BenchmarkId::new(*model_name, batch_size);
            
            group.bench_with_input(bench_id, &images, |b, imgs| {
                b.iter(|| {
                    let results = processor.preprocess_batch(imgs, target_size);
                    black_box(results)
                });
            });

            // Calculate throughput
            let start = std::time::Instant::now();
            let _ = processor.preprocess_batch(&images, target_size);
            let elapsed = start.elapsed();
            let throughput = batch_size as f64 / elapsed.as_secs_f64();
            
            println!("  Batch size {}: {:.2} images/sec", batch_size, throughput);
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_all_models);
criterion_main!(benches);
