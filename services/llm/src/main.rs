mod agent;
mod config;
mod engine_lease;
mod handler;
mod hrm_engine;
mod memory_gate;
mod tokenizer;
mod vision_bridge;

use actix_web::{middleware, web, App, HttpServer};
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

use config::{HrmConfig, LlmConfig};
use hrm_engine::HrmEngine;
use handler::AppState;

/// Terminate the process via POSIX `_exit`, skipping `atexit`/C++ static
/// destructors.
///
/// ONNX Runtime (>= 1.21, bundled by `ort 2.0.0-rc.10` as 1.22) has a macOS bug
/// where `OrtEnv`'s static destructor locks an already-destroyed mutex at
/// process exit, throwing an uncaught C++ exception → `SIGABRT`, *after* all our
/// work is done. We can't patch onnxruntime's statics, and the process is
/// terminating anyway, so we bypass the broken teardown: `_exit` reclaims
/// everything via the kernel without running `OrtEnv`'s destructor.
/// Refs: pykeio/ort#409, microsoft/onnxruntime#24579, #25038.
fn exit_skipping_ort_teardown(code: i32) -> ! {
    use std::io::Write as _;
    // _exit() does not flush stdio buffers; do it ourselves first.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: immediately terminates the process; no Rust state is left dangling
    // because nothing runs after this call.
    unsafe { libc::_exit(code) }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("llm_service=info".parse().unwrap()),
        )
        .init();

    let llm_config = LlmConfig::load().unwrap_or_else(|e| {
        eprintln!("Config error: {e}");
        std::process::exit(1);
    });

    let port = llm_config.port;

    let hrm_config = llm_config.hrm.as_ref().unwrap_or_else(|| {
        eprintln!("HRM config section missing — add [hrm] to config.toml");
        std::process::exit(1);
    });

    let engine = HrmEngine::load(hrm_config).unwrap_or_else(|e| {
        eprintln!("HRM engine load failed: {e}");
        // build_session may already have spun up the ORT environment, so exit
        // the teardown-safe way even on this failure path.
        exit_skipping_ort_teardown(1);
    });

    let vision = llm_config.vision_bridge.clone().and_then(|vbcfg| {
        if vbcfg.enabled {
            Some(Arc::new(vision_bridge::VisionBridge::new(vbcfg)))
        } else { None }
    });

    let limits = llm_config.limits.clone().unwrap_or_default();
    let mg_cfg = llm_config.memory_gate.clone().unwrap_or(crate::config::MemoryGateConfig {
        high_water_mb: 4096,
        low_water_mb: 3072,
        poll_on_admit_only: true,
    });
    let lease = crate::engine_lease::EngineLease::new(limits.engine.max_concurrent);
    let gate = Arc::new(crate::memory_gate::MemoryGate::new(
        mg_cfg.high_water_mb,
        mg_cfg.low_water_mb,
    ));

    let state = web::Data::new(AppState {
        engine: Arc::new(engine),
        vision,
        lease: lease.clone(),
        gate: gate.clone(),
        limits: limits.clone(),
    });

    // Build the agent layer (if [agent] config present).
    let agent_layer: Option<web::Data<crate::agent::http::AgentLayer>> =
        if let Some(agent_cfg) = llm_config.agent.clone() {
            let planner: Arc<dyn crate::agent::planner::Planner> =
                Arc::new(crate::agent::planner::HrmPlanner::new(
                    state.engine.clone(),
                    lease.clone(),
                ));

            let per_tool_timeout = std::time::Duration::from_millis(
                agent_cfg.per_tool_ms.max(1000),
            );
            let client = reqwest::Client::builder()
                .timeout(per_tool_timeout)
                .build()
                .expect("build agent http client");

            let mut reg = crate::agent::tool::ToolRegistry::new();
            reg.insert(Arc::new(crate::agent::tools::final_tool::FinalTool));

            let tools_cfg = agent_cfg.tools.clone().unwrap_or_default();
            reg.insert(crate::agent::tools::classify::ClassifyTool::new(
                client.clone(),
                &tools_cfg.main_server_base,
                &tools_cfg.classify_endpoint,
            ));
            reg.insert(crate::agent::tools::detect::DetectTool::new(
                client.clone(),
                &tools_cfg.main_server_base,
                &tools_cfg.detect_endpoint,
            ));
            reg.insert(crate::agent::tools::tts::TtsTool::new(
                client.clone(),
                &tools_cfg.main_server_base,
                &tools_cfg.tts_endpoint,
            ));
            reg.insert(crate::agent::tools::stt::SttTool::new(
                client.clone(),
                &tools_cfg.main_server_base,
                &tools_cfg.stt_endpoint,
            ));
            if let Some(vb) = state.vision.clone() {
                reg.insert(crate::agent::tools::vision::VisionTool::new(vb));
            }
            reg.insert(crate::agent::tools::reflect::ReflectTool::new(
                planner.clone(),
                agent_cfg.reflect_max_tokens,
            ));
            let hf = agent_cfg.http_fetch.clone().unwrap_or_default();
            reg.insert(crate::agent::tools::http_fetch::HttpFetchTool::new(
                hf.allowlist,
                hf.max_bytes,
                hf.follow_redirects,
                hf.enabled,
            ));

            let layer = crate::agent::http::AgentLayer::new(
                planner,
                Arc::new(reg),
                agent_cfg,
                gate.clone(),
                limits.clone(),
            );
            Some(web::Data::new(layer))
        } else {
            None
        };

    tracing::info!("LLM microservice listening on 0.0.0.0:{}", port);

    let server = HttpServer::new(move || {
        let mut app = App::new()
            .app_data(state.clone())
            .app_data(
                web::JsonConfig::default()
                    .limit(limits.json.body_limit)
                    .error_handler(|err, _req| {
                        let msg = err.to_string();
                        actix_web::error::InternalError::from_response(
                            err,
                            actix_web::HttpResponse::BadRequest()
                                .json(serde_json::json!({"error": {"message": msg}})),
                        )
                        .into()
                    }),
            )
            .wrap(middleware::Logger::default())
            .route(
                "/v1/chat/completions",
                web::post().to(handler::chat_completions),
            )
            .route("/v1/models", web::get().to(handler::list_models));

        if let Some(layer) = agent_layer.clone() {
            app = app
                .app_data(layer)
                .route("/v1/agent/run", web::post().to(crate::agent::http::run));
        }

        app
    })
    .workers(1)
    .bind(format!("0.0.0.0:{port}"));

    let run_result = match server {
        Ok(srv) => srv.run().await,
        Err(e) => {
            tracing::error!("bind 0.0.0.0:{port} failed: {e}");
            Err(e)
        }
    };

    if let Err(e) = &run_result {
        tracing::error!("server stopped with error: {e}");
    }

    // The ORT environment is live here; returning normally would run its broken
    // macOS static destructor. Exit the teardown-safe way instead.
    exit_skipping_ort_teardown(if run_result.is_ok() { 0 } else { 1 });
}
