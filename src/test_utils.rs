use std::sync::OnceLock;

/// Call at the top of any test that creates an ORT Session.
///
/// On macOS, `libloading` calls `dlopen`/`dlclose` around each `Library`.
/// ORT stores its handle in a global `Arc<Library>`.  When the last Arc drops
/// (i.e. when the last ORT session is freed), `dlclose` fires the library's
/// C++ global destructors — which crash with "mutex lock failed: Invalid
/// argument" because macOS thread-local storage has already started teardown.
///
/// Calling this before any ORT session is created opens a *second* `dlopen`
/// handle to the same `.dylib` and immediately leaks it (via `mem::forget`).
/// macOS reference-counts `dlopen` calls, so ORT's eventual `dlclose` only
/// decrements the count to 1 — the library is never unloaded, and the
/// destructors never run.
pub fn ort_test_setup() {
    static GUARD: OnceLock<()> = OnceLock::new();
    GUARD.get_or_init(|| {
        let path = std::env::var("ORT_DYLIB_PATH")
            .unwrap_or_else(|_| "/opt/homebrew/lib/libonnxruntime.dylib".to_string());
        // SAFETY: opening a shared library that is already loaded.
        if let Ok(lib) = unsafe { libloading::Library::new(&path) } {
            std::mem::forget(lib); // intentional leak — see doc comment above
        }
    });
}
