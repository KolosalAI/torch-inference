mod agent;
mod config;
mod handler;
mod hrm_engine;
mod tokenizer;
mod vision_bridge;

use actix_web::{middleware, web, App, HttpServer};
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

use config::{HrmConfig, LlmConfig};
use hrm_engine::HrmEngine;
use handler::AppState;

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
        std::process::exit(1);
    });

    let vision = llm_config.vision_bridge.clone().and_then(|vbcfg| {
        if vbcfg.enabled {
            Some(Arc::new(vision_bridge::VisionBridge::new(vbcfg)))
        } else { None }
    });

    let state = web::Data::new(AppState {
        engine: Arc::new(engine),
        vision,
    });

    // Build the agent layer (if [agent] config present).
    let agent_layer: Option<web::Data<crate::agent::http::AgentLayer>> =
        if let Some(agent_cfg) = llm_config.agent.clone() {
            let planner: Arc<dyn crate::agent::planner::Planner> =
                Arc::new(crate::agent::planner::HrmPlanner::new(state.engine.clone()));

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
            );
            Some(web::Data::new(layer))
        } else {
            None
        };

    tracing::info!("LLM microservice listening on 0.0.0.0:{}", port);

    HttpServer::new(move || {
        let mut app = App::new()
            .app_data(state.clone())
            .app_data(
                web::JsonConfig::default()
                    .limit(32 * 1024 * 1024)
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
    .bind(format!("0.0.0.0:{port}"))?
    .run()
    .await
}
