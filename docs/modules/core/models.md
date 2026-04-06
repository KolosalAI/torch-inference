# `models` — Model Loading & Registry

`src/models/manager.rs` · `src/models/registry.rs` · `src/models/onnx_loader.rs` · `src/models/pytorch_loader.rs`

The model subsystem handles discovery, downloading, registration, loading, and inference dispatch for all file-backed models. It supports PyTorch (`.pt`/`.pth`), ONNX (`.onnx`), and SafeTensors (`.safetensors`) formats.

---

## Model Loading Pipeline

```mermaid
graph TD
    A([models.json /\nModelRegistry]) --> B[ModelRegistry\nDashMap~id, ModelMetadata~]
    B --> C[ModelManager\norchestrates loaders]
    C --> D{ModelFormat?}
    D -- ONNX --> E[OnnxModelLoader\nort::Session::builder]
    D -- PyTorch --> F[PyTorchModelLoader\ntch::CModule::load]
    D -- SafeTensors --> G[SafeTensors loader\nsafetensors crate]
    E --> H[LoadedOnnxModel\nsession + metadata]
    F --> I[LoadedPyTorchModel\nmodule + metadata]
    G --> J[SafeTensors weights\ntensor map]
    H & I & J --> K[BaseModel\nDashMap cache]
    K --> L([Ready for inference])
```

---

## Model Download & Registration Sequence

```mermaid
sequenceDiagram
    participant CLI as API / CLI
    participant MM as ModelManager
    participant REG as ModelRegistry
    participant FS as File System
    participant DL as Downloader

    CLI->>MM: register_model_from_path(path, name)
    MM->>FS: check file exists + read size
    FS-->>MM: file metadata
    MM->>MM: ModelFormat::from_extension(ext)
    MM->>REG: register(ModelMetadata { id, name, format, path, ... })
    REG->>REG: DashMap::insert(id, metadata)
    REG-->>MM: Ok(())
    MM-->>CLI: Ok(())

    note over CLI,DL: For remote models (future download flow)
    CLI->>DL: download(url, cache_dir)
    DL->>FS: stream write to cache_dir/name
    DL-->>CLI: Ok(local_path)
    CLI->>MM: register_model_from_path(local_path, name)
```

---

## Class Diagram

```mermaid
classDiagram
    class ModelRegistry {
        -models: DashMap~String, ModelMetadata~
        -model_path: PathBuf
        +new(path) Self
        +register(metadata) Result~()~
        +get(id) Option~ModelMetadata~
        +list() Vec~ModelMetadata~
    }

    class ModelMetadata {
        +id: String
        +name: String
        +format: ModelFormat
        +path: PathBuf
        +version: String
        +description: Option~String~
        +tags: Vec~String~
        +input_schema: Option~Value~
        +output_schema: Option~Value~
        +preprocessing: Option~PreprocessingConfig~
        +postprocessing: Option~PostprocessingConfig~
        +created_at: DateTime~Utc~
        +updated_at: DateTime~Utc~
        +file_size: u64
        +checksum: Option~String~
    }

    class ModelFormat {
        <<enumeration>>
        PyTorch
        ONNX
        Candle
        SafeTensors
        +from_extension(ext) Option~ModelFormat~
    }

    class PreprocessingConfig {
        +image: Option~ImagePreprocessing~
        +audio: Option~AudioPreprocessing~
        +text: Option~TextPreprocessing~
        +custom: Option~Value~
    }

    ModelRegistry --> ModelMetadata : stores
    ModelMetadata --> ModelFormat : has
    ModelMetadata --> PreprocessingConfig : optional
```

---

## Supported Model Formats

| Format | Extensions | Loader | Runtime |
|---|---|---|---|
| PyTorch TorchScript | `.pt`, `.pth` | `PyTorchModelLoader` | tch-rs 0.16 (libtorch 2.3.0) |
| ONNX | `.onnx` | `OnnxModelLoader` | ort 2.0 (ONNX Runtime) |
| SafeTensors | `.safetensors` | Direct tensor load | `safetensors` crate |
| Candle | *(future)* | — | `candle-core` |

`ModelFormat::from_extension` maps file extensions to variants:

```rust
match ext.to_lowercase().as_str() {
    "pt" | "pth"   => Some(ModelFormat::PyTorch),
    "onnx"         => Some(ModelFormat::ONNX),
    "safetensors"  => Some(ModelFormat::SafeTensors),
    _              => None,
}
```

---

## Model Lifecycle

```mermaid
graph TD
    DL([Download\nfrom URL / local path]) --> REG([Register\nModelRegistry::register])
    REG --> LOAD([Load\nOnnxModelLoader or\nPyTorchModelLoader])
    LOAD --> CACHE([Cache\nDashMap pool entry])
    CACHE --> INFER([Infer\nModelManager::infer_registered])
    INFER --> CACHE
    CACHE --> UNLOAD([Unload\nremove from DashMap pool])
    UNLOAD --> REG
```

| Stage | Action | Key Code Path |
|---|---|---|
| **Download** | Fetch remote artifact to `config.models.cache_dir`. | External downloader → filesystem |
| **Register** | Insert `ModelMetadata` into `ModelRegistry`. | `ModelRegistry::register` |
| **Load** | Open ONNX session or `tch::CModule`; push into pool DashMap. | `OnnxModelLoader` / `PyTorchModelLoader` |
| **Infer** | Pop session from pool, run, push back. | `ModelManager::infer_registered` |
| **Unload** | Drain pool Vec; drop sessions (memory freed). | `loaded_onnx_models.remove(name)` |

---

## Usage Example

```rust
use std::path::Path;
use crate::config::Config;
use crate::models::manager::ModelManager;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::load()?;
    let manager = ModelManager::new(&config, None);

    // Register an ONNX model from disk
    manager
        .register_model_from_path(
            Path::new("./models/resnet50.onnx"),
            Some("resnet50".to_string()),
        )
        .await?;

    // Run inference via the registry path
    let result = manager
        .infer_registered("resnet50", &serde_json::json!({"image": "..."}))
        .await?;

    println!("{}", result);
    Ok(())
}
```
