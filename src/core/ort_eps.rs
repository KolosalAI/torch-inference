/// Shared helper that returns the platform-appropriate ORT execution provider list.
///
/// Priority chain:
///   Windows  — CUDA (device_id) → DirectML (device_id) → CPU
///   macOS    — CoreML (All units, subgraphs on) → CPU
///   Other    — CUDA (device_id) → CPU
///
/// Callers register the returned vec via `Session::builder().with_execution_providers(...)`.
/// ORT silently skips any EP whose native runtime is absent, so a missing CUDA DLL just
/// falls through to DirectML, then CPU — no crash.
use ort::execution_providers::ExecutionProviderDispatch;

// `device_id` selects the CUDA/DirectML device on Windows/Linux; macOS (CoreML)
// ignores it, so the parameter is unused on that target.
#[cfg_attr(target_os = "macos", allow(unused_variables))]
pub fn build_eps(device_id: i32) -> Vec<ExecutionProviderDispatch> {
    let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();

    // --- Windows: CUDA → DirectML → CPU ---
    #[cfg(target_os = "windows")]
    {
        eps.push(
            ort::execution_providers::CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build(),
        );
        eps.push(
            ort::execution_providers::DirectMLExecutionProvider::default()
                .with_device_id(device_id)
                .build(),
        );
    }

    // --- macOS: CoreML (Neural Engine + GPU + CPU) ---
    #[cfg(target_os = "macos")]
    {
        eps.push(
            ort::execution_providers::CoreMLExecutionProvider::default()
                .with_subgraphs(true)
                .with_compute_units(
                    ort::execution_providers::coreml::CoreMLComputeUnits::All,
                )
                .build(),
        );
    }

    // --- Linux / other: CUDA → CPU ---
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        eps.push(
            ort::execution_providers::CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build(),
        );
    }

    // CPU is always the final fallback on every platform.
    eps.push(ort::execution_providers::CPUExecutionProvider::default().build());

    eps
}
