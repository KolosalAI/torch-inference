//! Regression test for the macOS ONNX Runtime exit crash.
//!
//! ORT >= 1.21 (bundled by `ort 2.0.0-rc.10`) crashes with
//! `SIGABRT: mutex lock failed: Invalid argument` in `OrtEnv`'s static
//! destructor at process exit. `main.rs` works around it by terminating via
//! `_exit` (see `exit_skipping_ort_teardown`). This test boots the real binary
//! (which loads the 2.3 GB model and thus creates an ORT env), sends SIGINT for
//! a graceful shutdown, and asserts the process exits cleanly with code 0 —
//! i.e. it does NOT die by SIGABRT.
//!
//! `#[ignore]`d: requires the local model artifact and binds the configured
//! port. Run with: `cargo test --test clean_shutdown -- --ignored`.

use std::io::ErrorKind;
use std::net::TcpStream;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

const PORT: u16 = 8001; // matches services/llm/config.toml

#[test]
#[ignore = "boots the real server + ORT model; run explicitly with --ignored"]
fn server_exits_cleanly_on_sigint_despite_ort() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let model = Path::new(manifest).join("models/hrm-text-1b/model.onnx");
    if !model.exists() {
        eprintln!("skipping: {} not present", model.display());
        return;
    }

    let bin = env!("CARGO_BIN_EXE_llm-service");
    let mut child = Command::new(bin)
        .current_dir(manifest) // so it finds config.toml + models/
        .spawn()
        .expect("spawn llm-service");

    // Wait until the server is accepting connections (model load takes a few s).
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut listening = false;
    while Instant::now() < deadline {
        // Bail early if the child already died (e.g. bind failure).
        if let Ok(Some(status)) = child.try_wait() {
            panic!("server exited before listening: {status:?}");
        }
        if TcpStream::connect(("127.0.0.1", PORT)).is_ok() {
            listening = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(listening, "server never started listening on :{PORT}");

    // Graceful shutdown: actix-web handles SIGINT, returns from run(), and main
    // calls _exit(0). Use the `kill` CLI to avoid a libc dev-dependency.
    let killed = Command::new("kill")
        .arg("-INT")
        .arg(child.id().to_string())
        .status()
        .expect("send SIGINT");
    assert!(killed.success(), "failed to deliver SIGINT");

    // Await exit with a timeout.
    let wait_deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= wait_deadline {
                    let _ = child.kill();
                    panic!("server did not exit within 30s of SIGINT");
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => panic!("wait failed: {e}"),
        }
    };

    // The whole point: exit code 0, NOT a signal (SIGABRT = 6) from ORT teardown.
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        assert!(
            status.signal().is_none(),
            "server was killed by signal {:?} — ORT teardown crash not bypassed",
            status.signal()
        );
    }
    assert!(
        status.success(),
        "expected clean exit code 0, got {status:?}"
    );
}
