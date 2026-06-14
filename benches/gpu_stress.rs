/// gpu_stress — ORT execution-provider stress test.
///
/// Verifies that the CUDA → DirectML → CPU EP chain (from `core::ort_eps`) works
/// correctly under concurrent load, measures session-creation throughput and
/// per-inference latency, and reports the active EP and GPU utilisation.
///
/// Run:
///   cargo bench --bench gpu_stress --no-default-features
///   ORT_DYLIB_PATH=path/to/onnxruntime-gpu.dll cargo bench --bench gpu_stress --no-default-features
///
/// The test uses a 4-op ONNX graph embedded as bytes so no external model file
/// is required. On a CUDA-capable DLL the CUDA EP fires; on CPU-only DLLs it
/// falls through to DirectML then CPU — no crash either way.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;

// ── Minimal ONNX model (Add op, input A + B → C, shape [1, 8] f32) ──────────
//
// Hand-crafted ONNX protobuf bytes for a graph:
//   A [1×8 f32]  ─┐
//                  Add → C [1×8 f32]
//   B [1×8 f32]  ─┘
//
// Generated via: python -c "
//   import onnx, onnx.helper as h, onnx.TensorProto as tp
//   A = h.make_tensor_value_info('A', tp.FLOAT, [1,8])
//   B = h.make_tensor_value_info('B', tp.FLOAT, [1,8])
//   C = h.make_tensor_value_info('C', tp.FLOAT, [1,8])
//   node = h.make_node('Add', ['A','B'], ['C'])
//   graph = h.make_graph([node], 'add', [A,B], [C])
//   model = h.make_model(graph, opset_imports=[h.make_opsetid('',18)])
//   open('add.onnx','wb').write(model.SerializeToString())
// "
// Single Add node: A[1,8]+B[1,8]→C[1,8] f32, opset 18, IR v8 (ORT 1.22 max=10). 100 bytes.
const ADD_ONNX: &[u8] = &[
    0x8, 0x8, 0x3a, 0x5a, 0xa, 0xe, 0xa, 0x1, 0x41, 0xa, 0x1, 0x42, 0x12, 0x1, 0x43,
    0x22, 0x3, 0x41, 0x64, 0x64, 0x12, 0x9, 0x61, 0x64, 0x64, 0x5f, 0x67, 0x72, 0x61,
    0x70, 0x68, 0x5a, 0x13, 0xa, 0x1, 0x41, 0x12, 0xe, 0xa, 0xc, 0x8, 0x1, 0x12, 0x8,
    0xa, 0x2, 0x8, 0x1, 0xa, 0x2, 0x8, 0x8, 0x5a, 0x13, 0xa, 0x1, 0x42, 0x12, 0xe,
    0xa, 0xc, 0x8, 0x1, 0x12, 0x8, 0xa, 0x2, 0x8, 0x1, 0xa, 0x2, 0x8, 0x8, 0x62,
    0x13, 0xa, 0x1, 0x43, 0x12, 0xe, 0xa, 0xc, 0x8, 0x1, 0x12, 0x8, 0xa, 0x2, 0x8,
    0x1, 0xa, 0x2, 0x8, 0x8, 0x42, 0x4, 0xa, 0x0, 0x10, 0x12,
];

// ── Helpers ──────────────────────────────────────────────────────────────────

fn build_session() -> Session {
    let eps = torch_inference::core::ort_eps::build_eps(0);
    Session::builder()
        .expect("Session::builder")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("opt level")
        .with_intra_threads(1)
        .expect("intra threads")
        .with_memory_pattern(true)
        .expect("memory pattern")
        .with_execution_providers(eps)
        .expect("EPs")
        .commit_from_memory(ADD_ONNX)
        .expect("commit_from_memory — embedded ONNX invalid?")
}

/// Run one Add inference on a `&mut Session`.
fn run_inference(session: &mut Session) {
    let ta = Tensor::<f32>::from_array(([1usize, 8], vec![1.0f32; 8])).unwrap();
    let tb = Tensor::<f32>::from_array(([1usize, 8], vec![2.0f32; 8])).unwrap();
    // ort::inputs! returns Vec<(Cow<str>, SessionInputValue)> — not a Result
    let outputs = session.run(ort::inputs!["A" => ta, "B" => tb]).unwrap();
    // try_extract_tensor returns (&Shape, &[T])
    let (_, vals) = outputs[0].try_extract_tensor::<f32>().unwrap();
    debug_assert!(vals.iter().all(|&v| (v - 3.0).abs() < 1e-5));
    black_box(outputs);
}

/// Run one inference on a `Mutex<Session>` (used by concurrent bench).
fn run_inference_locked(session: &Mutex<Session>) {
    let ta = Tensor::<f32>::from_array(([1usize, 8], vec![1.0f32; 8])).unwrap();
    let tb = Tensor::<f32>::from_array(([1usize, 8], vec![2.0f32; 8])).unwrap();
    let mut guard = session.lock();
    let outputs = guard.run(ort::inputs!["A" => ta, "B" => tb]).unwrap();
    let (_, vals) = outputs[0].try_extract_tensor::<f32>().unwrap();
    debug_assert!(vals.iter().all(|&v| (v - 3.0).abs() < 1e-5));
    black_box(outputs);
}

fn report_ep_status() {
    println!("\n=== EP status on this machine ===");
    #[cfg(target_os = "windows")]
    println!("  Platform: Windows → EP chain: CUDA → DirectML → CPU");
    #[cfg(target_os = "macos")]
    println!("  Platform: macOS → EP chain: CoreML (ANE+GPU+CPU) → CPU");
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    println!("  Platform: Linux → EP chain: CUDA → CPU");

    let ort_dll = std::env::var("ORT_DYLIB_PATH")
        .unwrap_or_else(|_| "(system default — check PATH/System32)".to_string());
    println!("  ORT_DYLIB_PATH: {ort_dll}");

    #[cfg(feature = "cuda")]
    {
        if let Ok(nvml) = nvml_wrapper::Nvml::init() {
            if let Ok(dev) = nvml.device_by_index(0) {
                let name = dev.name().unwrap_or_default();
                let mem = dev.memory_info().unwrap();
                println!(
                    "  GPU: {name}  free={:.0} MB / total={:.0} MB",
                    mem.free as f64 / 1e6,
                    mem.total as f64 / 1e6
                );
            }
        }
    }

    // Try building one session to see if it panics (would mean DLL mismatch)
    println!("  Building probe session with embedded Add model...");
    let _s = build_session();
    println!("  Probe OK — ORT DLL loaded successfully\n");
}

// ── Benchmarks ───────────────────────────────────────────────────────────────

/// How long does it take to build one ORT session from scratch (EP registration
/// + graph optimization)? Amortised over 10 builds per iteration.
fn bench_session_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ort_ep");
    group.throughput(Throughput::Elements(1));
    group.bench_function("session_create_x10", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let s = build_session();
                black_box(s);
            }
        })
    });
    group.finish();
}

/// Single-threaded inference throughput: back-to-back Add ops on one session.
fn bench_single_thread_inference(c: &mut Criterion) {
    let mut session = build_session();
    let mut group = c.benchmark_group("ort_inference");
    group.throughput(Throughput::Elements(1));

    group.bench_function("single_thread_add", |b| {
        b.iter(|| run_inference(&mut session))
    });

    // Batch of 100 back-to-back inferences (measures sustained throughput).
    group.bench_function("burst_100_inferences", |b| {
        b.iter(|| {
            for _ in 0..100 {
                run_inference(&mut session)
            }
        })
    });

    group.finish();
}

/// Concurrent inference: N threads each fire 200 inferences on a shared Mutex<Session>.
fn bench_concurrent_inference(c: &mut Criterion) {
    let session = Arc::new(Mutex::new(build_session()));
    let mut group = c.benchmark_group("ort_concurrent");
    group.measurement_time(std::time::Duration::from_secs(10));

    for threads in [1usize, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &threads,
            |b, &t| {
                b.iter(|| {
                    let handles: Vec<_> = (0..t)
                        .map(|_| {
                            let sess = Arc::clone(&session);
                            std::thread::spawn(move || {
                                for _ in 0..200 {
                                    run_inference_locked(&sess);
                                }
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join().unwrap();
                    }
                })
            },
        );
    }
    group.finish();
}

/// Latency percentiles: p50 / p95 / p99 for a single inference.
fn bench_latency_percentiles(c: &mut Criterion) {
    let mut session = build_session();

    let mut group = c.benchmark_group("ort_latency");
    group.sample_size(500);

    // Warm up
    for _ in 0..20 {
        run_inference(&mut session);
    }

    let mut times: Vec<u128> = Vec::with_capacity(500);
    group.bench_function("p50_p95_p99", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let t0 = Instant::now();
                run_inference(&mut session);
                let elapsed = t0.elapsed();
                total += elapsed;
                times.push(elapsed.as_micros());
            }
            total
        });
    });

    if times.len() >= 10 {
        let mut sorted = times.clone();
        sorted.sort_unstable();
        let p50 = sorted[sorted.len() / 2];
        let p95 = sorted[sorted.len() * 95 / 100];
        let p99 = sorted[sorted.len() * 99 / 100];
        println!("\n  Latency (µs)  p50={p50}  p95={p95}  p99={p99}");
    }

    group.finish();
}

// ── Entry point ──────────────────────────────────────────────────────────────

fn gpu_stress_main(c: &mut Criterion) {
    report_ep_status();
    bench_session_creation(c);
    bench_single_thread_inference(c);
    bench_concurrent_inference(c);
    bench_latency_percentiles(c);
}

criterion_group!(benches, gpu_stress_main);
criterion_main!(benches);
