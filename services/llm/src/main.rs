mod config;
mod engine;
mod handler;

use actix_web::{middleware, web, App, HttpServer};
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

use config::LlmConfig;
use engine::LlamaEngine;
use handler::AppState;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("llm_service=info".parse().unwrap()),
        )
        .init();

    let config = LlmConfig::load().unwrap_or_else(|e| {
        eprintln!("Config error: {e}");
        std::process::exit(1);
    });

    let engine = LlamaEngine::load(config).unwrap_or_else(|e| {
        eprintln!("Model load failed: {e}");
        std::process::exit(1);
    });

    let port = engine.config.port;
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
