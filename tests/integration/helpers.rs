// tests/integration/helpers.rs — shared test utilities
use actix_web::web;
use std::sync::Arc;
use torch_inference::{
    api::{
        classify::{ClassificationBackend, ClassifyState, Prediction},
        system::SystemInfoState,
        tts::TTSState,
    },
    core::{
        gpu::GpuManager,
        tts_manager::{TTSManager, TTSManagerConfig},
    },
    middleware::rate_limit::RateLimiter,
    monitor::Monitor,
};

pub fn monitor() -> web::Data<Arc<Monitor>> {
    web::Data::new(Arc::new(Monitor::new()))
}

#[allow(dead_code)]
pub fn rate_limiter(max: u64) -> web::Data<Arc<RateLimiter>> {
    web::Data::new(Arc::new(RateLimiter::new(max, 60)))
}

pub fn tts_state() -> web::Data<TTSState> {
    web::Data::new(TTSState {
        manager: Arc::new(TTSManager::new(TTSManagerConfig::default())),
    })
}

pub fn system_state() -> web::Data<SystemInfoState> {
    web::Data::new(SystemInfoState {
        gpu_manager: Arc::new(GpuManager::new()),
        start_time: std::time::Instant::now(),
    })
}

pub struct NoOpBackend;

impl ClassificationBackend for NoOpBackend {
    fn classify_nchw(
        &self,
        _batch: ndarray::Array4<f32>,
        _top_k: usize,
    ) -> anyhow::Result<Vec<Vec<Prediction>>> {
        anyhow::bail!("no classification model loaded")
    }
}

pub fn classify_state() -> web::Data<ClassifyState> {
    web::Data::new(ClassifyState {
        backend: Arc::new(NoOpBackend),
    })
}
