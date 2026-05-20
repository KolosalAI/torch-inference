//! Regression guard: ensures llama-cpp-2 stays out of the dependency tree.

use std::fs;
use std::path::PathBuf;

#[test]
fn cargo_lock_has_no_llama_cpp() {
    let lock = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    let text = fs::read_to_string(&lock).expect("read Cargo.lock");
    assert!(
        !text.contains("\"llama-cpp-2\""),
        "llama-cpp-2 reappeared in Cargo.lock — this dep was removed by the HRM-Text swap."
    );
}
