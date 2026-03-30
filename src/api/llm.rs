#![allow(dead_code)]
/// OpenAI-compatible LLM inference API.
///
/// Endpoints:
/// - `GET  /v1/models`                — list available models
/// - `POST /v1/completions`           — text completion
/// - `POST /v1/chat/completions`      — chat completion (OpenAI format)
///
/// The handler delegates to the [`LlmBackend`] trait so the real inference
/// engine (scheduler + speculative decoder) can be swapped for a mock in tests.
use actix_web::{web, HttpResponse};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::core::llm::SamplingParams;
use crate::error::ApiError;

// ── OpenAI wire types ─────────────────────────────────────────────────────

/// `GET /v1/models` response.
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// `POST /v1/completions` request.
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub stop: Vec<String>,
}

fn default_max_tokens() -> usize { 256 }
fn default_temperature() -> f32 { 1.0 }
fn default_top_p() -> f32 { 1.0 }

/// `POST /v1/completions` response.
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
}

/// `POST /v1/chat/completions` request.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub stop: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// `POST /v1/chat/completions` response.
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ── Backend trait ─────────────────────────────────────────────────────────

/// Abstraction over the LLM engine (scheduler + speculative decoder).
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// List models available on this backend.
    fn list_models(&self) -> Vec<ModelInfo>;

    /// Generate text continuing `prompt`, returning `(text, completion_tokens)`.
    async fn complete(
        &self,
        model: &str,
        prompt: &str,
        params: SamplingParams,
    ) -> anyhow::Result<(String, usize)>;
}

// ── App state ─────────────────────────────────────────────────────────────

pub struct LlmState {
    pub backend: Arc<dyn LlmBackend>,
}

// ── Handlers ──────────────────────────────────────────────────────────────

/// GET /v1/models
pub async fn list_models(state: web::Data<LlmState>) -> Result<HttpResponse, ApiError> {
    Ok(HttpResponse::Ok().json(ModelListResponse {
        object: "list".to_string(),
        data: state.backend.list_models(),
    }))
}

/// POST /v1/completions
pub async fn completions(
    req: web::Json<CompletionRequest>,
    state: web::Data<LlmState>,
) -> Result<HttpResponse, ApiError> {
    if req.prompt.is_empty() {
        return Err(ApiError::BadRequest("prompt must not be empty".to_string()));
    }
    if req.max_tokens == 0 {
        return Err(ApiError::BadRequest("max_tokens must be >= 1".to_string()));
    }

    let params = SamplingParams {
        temperature: req.temperature.max(0.0),
        top_p: req.top_p.clamp(1e-4, 1.0),
        top_k: req.top_k,
        max_tokens: req.max_tokens,
        stop_token_ids: vec![],
    };

    let prompt_tokens = req.prompt.split_whitespace().count(); // rough estimate

    let (text, completion_tokens) = state
        .backend
        .complete(&req.model, &req.prompt, params)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    Ok(HttpResponse::Ok().json(CompletionResponse {
        id: new_id("cmpl"),
        object: "text_completion".to_string(),
        model: req.model.clone(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: "stop".to_string(),
        }],
        usage: UsageInfo {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    req: web::Json<ChatRequest>,
    state: web::Data<LlmState>,
) -> Result<HttpResponse, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::BadRequest("messages must not be empty".to_string()));
    }
    if req.max_tokens == 0 {
        return Err(ApiError::BadRequest("max_tokens must be >= 1".to_string()));
    }

    let params = SamplingParams {
        temperature: req.temperature.max(0.0),
        top_p: req.top_p.clamp(1e-4, 1.0),
        top_k: req.top_k,
        max_tokens: req.max_tokens,
        stop_token_ids: vec![],
    };

    // Flatten chat messages into a prompt.
    let prompt = messages_to_prompt(&req.messages);
    let prompt_tokens = prompt.split_whitespace().count();

    let (text, completion_tokens) = state
        .backend
        .complete(&req.model, &prompt, params)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    Ok(HttpResponse::Ok().json(ChatResponse {
        id: new_id("chatcmpl"),
        object: "chat.completion".to_string(),
        model: req.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage { role: "assistant".to_string(), content: text },
            finish_reason: "stop".to_string(),
        }],
        usage: UsageInfo {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

/// Configure /v1 routes.
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/v1")
            .route("/models", web::get().to(list_models))
            .route("/completions", web::post().to(completions))
            .route("/chat/completions", web::post().to(chat_completions)),
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn new_id(prefix: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{}-{}", prefix, ts)
}

/// Flatten OpenAI chat messages into a simple prompt string.
fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test as actix_test, App};

    // ── Mock backend ─────────────────────────────────────────────────────

    struct MockLlmBackend;

    #[async_trait]
    impl LlmBackend for MockLlmBackend {
        fn list_models(&self) -> Vec<ModelInfo> {
            vec![ModelInfo {
                id: "mock-llm-7b".to_string(),
                object: "model".to_string(),
                created: 1_700_000_000,
                owned_by: "torch-inference".to_string(),
            }]
        }

        async fn complete(
            &self,
            _model: &str,
            prompt: &str,
            params: SamplingParams,
        ) -> anyhow::Result<(String, usize)> {
            // Echo the prompt in reverse as the completion.
            let text: String = prompt.chars().rev().take(params.max_tokens).collect();
            let n = text.split_whitespace().count();
            Ok((text, n))
        }
    }

    fn make_state() -> web::Data<LlmState> {
        web::Data::new(LlmState {
            backend: Arc::new(MockLlmBackend),
        })
    }

    // ── GET /v1/models ────────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_list_models_returns_200() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let req = actix_test::TestRequest::get().uri("/v1/models").to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_list_models_response_shape() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let req = actix_test::TestRequest::get().uri("/v1/models").to_request();
        let body: serde_json::Value = actix_test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["object"], "list");
        assert!(body["data"].is_array());
        assert_eq!(body["data"][0]["id"], "mock-llm-7b");
    }

    // ── POST /v1/completions ──────────────────────────────────────────────

    #[actix_web::test]
    async fn test_completion_success() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({ "model": "mock-llm-7b", "prompt": "Hello world" });
        let req = actix_test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_completion_response_shape() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({ "model": "mock-llm-7b", "prompt": "test" });
        let req = actix_test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(&payload)
            .to_request();
        let body: serde_json::Value = actix_test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["object"], "text_completion");
        assert!(body["choices"][0]["text"].is_string());
        assert!(body["usage"]["total_tokens"].is_number());
        assert!(body["id"].as_str().unwrap().starts_with("cmpl-"));
    }

    #[actix_web::test]
    async fn test_completion_empty_prompt_returns_bad_request() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({ "model": "mock-llm-7b", "prompt": "" });
        let req = actix_test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_completion_zero_max_tokens_returns_bad_request() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({ "model": "m", "prompt": "hi", "max_tokens": 0 });
        let req = actix_test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_completion_default_params_applied() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({ "model": "m", "prompt": "hi there" });
        let req = actix_test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(&payload)
            .to_request();
        let body: serde_json::Value = actix_test::call_and_read_body_json(&app, req).await;
        // Defaults: max_tokens=256, temperature=1.0, top_p=1.0
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
    }

    #[actix_web::test]
    async fn test_completion_custom_params() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({
            "model": "m", "prompt": "test",
            "max_tokens": 10, "temperature": 0.7, "top_p": 0.9, "top_k": 40
        });
        let req = actix_test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // ── POST /v1/chat/completions ─────────────────────────────────────────

    #[actix_web::test]
    async fn test_chat_completion_success() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({
            "model": "mock-llm-7b",
            "messages": [{ "role": "user", "content": "Hello!" }]
        });
        let req = actix_test::TestRequest::post()
            .uri("/v1/chat/completions")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_chat_completion_response_shape() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({
            "model": "m",
            "messages": [
                { "role": "system", "content": "You are helpful." },
                { "role": "user", "content": "Hi" }
            ]
        });
        let req = actix_test::TestRequest::post()
            .uri("/v1/chat/completions")
            .set_json(&payload)
            .to_request();
        let body: serde_json::Value = actix_test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert!(body["choices"][0]["message"]["content"].is_string());
        assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
    }

    #[actix_web::test]
    async fn test_chat_completion_empty_messages_returns_bad_request() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({ "model": "m", "messages": [] });
        let req = actix_test::TestRequest::post()
            .uri("/v1/chat/completions")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_chat_completion_zero_max_tokens_returns_bad_request() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 0
        });
        let req = actix_test::TestRequest::post()
            .uri("/v1/chat/completions")
            .set_json(&payload)
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_chat_usage_totals_are_correct() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;
        let payload = serde_json::json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hello world" }]
        });
        let req = actix_test::TestRequest::post()
            .uri("/v1/chat/completions")
            .set_json(&payload)
            .to_request();
        let body: serde_json::Value = actix_test::call_and_read_body_json(&app, req).await;
        let prompt = body["usage"]["prompt_tokens"].as_u64().unwrap();
        let completion = body["usage"]["completion_tokens"].as_u64().unwrap();
        let total = body["usage"]["total_tokens"].as_u64().unwrap();
        assert_eq!(total, prompt + completion);
    }

    // ── configure_routes ──────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_configure_routes_all_endpoints_reachable() {
        let app = actix_test::init_service(
            App::new().app_data(make_state()).configure(configure_routes),
        )
        .await;

        // GET /v1/models
        let req = actix_test::TestRequest::get().uri("/v1/models").to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        // POST /v1/completions
        let req = actix_test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(serde_json::json!({ "model": "m", "prompt": "test" }))
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        // POST /v1/chat/completions
        let req = actix_test::TestRequest::post()
            .uri("/v1/chat/completions")
            .set_json(serde_json::json!({
                "model": "m",
                "messages": [{ "role": "user", "content": "hi" }]
            }))
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // ── messages_to_prompt ────────────────────────────────────────────────

    #[test]
    fn test_messages_to_prompt_single_message() {
        let msgs = vec![ChatMessage { role: "user".into(), content: "hi".into() }];
        let p = messages_to_prompt(&msgs);
        assert_eq!(p, "user: hi");
    }

    #[test]
    fn test_messages_to_prompt_multi_message() {
        let msgs = vec![
            ChatMessage { role: "system".into(), content: "You are a bot.".into() },
            ChatMessage { role: "user".into(), content: "Hello".into() },
        ];
        let p = messages_to_prompt(&msgs);
        assert!(p.contains("system: You are a bot."));
        assert!(p.contains("user: Hello"));
    }

    // ── Serde / type tests ────────────────────────────────────────────────

    #[test]
    fn test_model_info_serialization() {
        let m = ModelInfo {
            id: "test-model".into(),
            object: "model".into(),
            created: 12345,
            owned_by: "me".into(),
        };
        let json = serde_json::to_string(&m).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["id"], "test-model");
        assert_eq!(v["created"], 12345);
    }

    #[test]
    fn test_chat_message_clone_and_serde() {
        let msg = ChatMessage { role: "user".into(), content: "hello".into() };
        let msg2 = msg.clone();
        assert_eq!(msg2.content, "hello");
        let json = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "user");
    }

    #[test]
    fn test_usage_info_serialization() {
        let u = UsageInfo { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 };
        let v: serde_json::Value = serde_json::to_value(&u).unwrap();
        assert_eq!(v["total_tokens"], 15);
    }

    #[test]
    fn test_default_max_tokens() {
        assert_eq!(default_max_tokens(), 256);
    }

    #[test]
    fn test_default_temperature() {
        assert!((default_temperature() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_top_p() {
        assert!((default_top_p() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_new_id_starts_with_prefix() {
        let id = new_id("cmpl");
        assert!(id.starts_with("cmpl-"));
    }
}
