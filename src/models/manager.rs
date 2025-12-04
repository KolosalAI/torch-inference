use dashmap::DashMap;
use serde_json::json;
use log::info;

use crate::config::Config;
use crate::error::{InferenceError, Result};

#[derive(Clone)]
pub struct BaseModel {
    pub name: String,
    pub device: String,
    pub is_loaded: bool,
}

impl BaseModel {
    pub fn new(name: String) -> Self {
        Self {
            name,
            device: "cpu".to_string(),
            is_loaded: false,
        }
    }
    
    pub async fn load(&mut self) -> Result<()> {
        info!("Loading model: {}", self.name);
        self.is_loaded = true;
        Ok(())
    }
    
    pub async fn forward(&self, inputs: &serde_json::Value) -> Result<serde_json::Value> {
        if !self.is_loaded {
            return Err(InferenceError::ModelLoadError("Model not loaded".to_string()));
        }
        Ok(inputs.clone())
    }
    
    pub fn model_info(&self) -> serde_json::Value {
        json!({
            "name": self.name,
            "device": self.device,
            "loaded": self.is_loaded
        })
    }
}

pub struct ModelManager {
    models: DashMap<String, BaseModel>,
    config: Config,
}

impl ModelManager {
    pub fn new(config: &Config) -> Self {
        Self {
            models: DashMap::new(),
            config: config.clone(),
        }
    }
    
    pub async fn register_model(&self, name: String, model: BaseModel) -> Result<()> {
        info!("Registering model: {}", name);
        self.models.insert(name, model);
        Ok(())
    }
    
    pub async fn load_model(&self, name: &str) -> Result<()> {
        if let Some(mut entry) = self.models.get_mut(name) {
            entry.load().await?;
            Ok(())
        } else {
            Err(InferenceError::ModelNotFound(name.to_string()))
        }
    }
    
    pub fn get_model(&self, name: &str) -> Result<BaseModel> {
        self.models
            .get(name)
            .map(|m| m.clone())
            .ok_or_else(|| InferenceError::ModelNotFound(name.to_string()))
    }
    
    pub fn list_available(&self) -> Vec<String> {
        self.models.iter().map(|m| m.key().clone()).collect()
    }
    
    pub async fn initialize_default_models(&self) -> Result<()> {
        info!("Initializing default models");
        
        let example_model = BaseModel::new("example".to_string());
        self.register_model("example".to_string(), example_model).await?;
        
        // Load auto-load models from config
        for model_name in &self.config.models.auto_load {
            if let Ok(model) = self.get_model(model_name) {
                let mut m = model;
                m.load().await?;
            }
        }
        
        Ok(())
    }
}
