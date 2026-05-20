mod config;
mod engine;
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

    let state = web::Data::new(AppState {
        engine: Arc::new(engine),
    });

    tracing::info!("LLM microservice listening on 0.0.0.0:{}", port);

    HttpServer::new(move || {
        App::new()
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
            .route("/v1/models", web::get().to(handler::list_models))
    })
    .workers(1)
    .bind(format!("0.0.0.0:{port}"))?
    .run()
    .await
}
