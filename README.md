# Kolosal Inference

High-performance multimodal inference server written in Rust (actix-web). Serves TTS synthesis, STT transcription, image classification, YOLO object detection, and LLM chat completion over HTTP.

## Quick Start

```bash
# Build (release — ~2 min)
cargo build --release

# Run server (default port 8000)
./target/release/torch-inference-server

# Open playground
open http://localhost:8000/playground
```

## Services

| Endpoint | Description |
|---|---|
| `POST /tts/stream` | Text-to-speech (Kokoro-ONNX, Bark, XTTS, StyleTTS2, Piper, VITS) |
| `POST /stt/transcribe` | Speech-to-text (Whisper) |
| `POST /classify/batch` | Image classification (ONNX) |
| `POST /detect` | Object detection (YOLO via ONNX) |
| `POST /llm/v1/chat/completions` | LLM chat (OpenAI-compatible, SSE streaming) |
| `GET  /health` | Health check |
| `GET  /system/info` | System info |
| `GET  /metrics` | Runtime metrics |
| `GET  /playground` | Interactive UI |

## Configuration

Copy and edit `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000

microservices:
  llm_host: "127.0.0.1"
  llm_port: 8001
  stt_host: "127.0.0.1"
  stt_port: 8002
```

All fields are also settable via environment variables (see `src/config.rs`).

## LLM Microservice

```bash
cd services/llm
cargo build --release
./target/release/llm-service
```

## STT Microservice

```bash
cd services/stt
pip install -r requirements.txt
python server.py
```

## Tests

```bash
# Rust unit + integration tests
cargo test

# Jest API tests
cd tests/jest && npm test

# Benchmarks
cargo bench --bench throughput_bench
cargo bench --bench tts_bench
```

## Models

Required model files are **auto-downloaded on first server start**. All downloads are non-blocking background tasks — the server is available immediately while models pull in behind it.

| Feature | Model file | Source |
|---|---|---|
| TTS | `models/kokoro-82m/kokoro-v1.0.int8.onnx` | HuggingFace `hexgrad/Kokoro-82M` |
| TTS voices | `models/kokoro-82m/voices/*.bin` | HuggingFace `hexgrad/Kokoro-82M` |
| STT | `models/whisper-onnx/encoder_model_quantized.onnx` | HuggingFace `onnx-community/whisper-base` |
| STT | `models/whisper-onnx/decoder_model_quantized.onnx` | HuggingFace `onnx-community/whisper-base` |
| Classification | `models/classify/efficientnet-lite4-11.onnx` | ONNX Model Zoo |
| Classification | `models/classify/imagenet1000.txt` | PyTorch Hub |
| Detection | `models/yolo/yolov8n.onnx` | Ultralytics assets |

To pre-download manually before starting the server:

```bash
# Kokoro TTS
python3 -c "
import urllib.request, os
base='https://huggingface.co/hexgrad/Kokoro-82M/resolve/main'
os.makedirs('models/kokoro-82m/voices', exist_ok=True)
urllib.request.urlretrieve(f'{base}/kokoro-v1.0.int8.onnx', 'models/kokoro-82m/kokoro-v1.0.int8.onnx')
for v in ['af_heart','af_bella','am_michael','bm_george']:
    urllib.request.urlretrieve(f'{base}/voices/{v}.bin', f'models/kokoro-82m/voices/{v}.bin')
"

# Whisper STT
python3 -c "
import urllib.request, os
base='https://huggingface.co/onnx-community/whisper-base/resolve/main/onnx'
os.makedirs('models/whisper-onnx', exist_ok=True)
for f in ['encoder_model_quantized.onnx','decoder_model_quantized.onnx']:
    urllib.request.urlretrieve(f'{base}/{f}', f'models/whisper-onnx/{f}')
urllib.request.urlretrieve('https://huggingface.co/onnx-community/whisper-base/resolve/main/vocab.json','models/whisper-onnx/vocab.json')
"

# YOLOv8n
mkdir -p models/yolo
curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx \
     -o models/yolo/yolov8n.onnx

# EfficientNet-Lite4
mkdir -p models/classify
curl -L https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx \
     -o models/classify/efficientnet-lite4-11.onnx
```

## Architecture

```
src/
  main.rs           — server startup, port binding, worker pool
  config.rs         — Config struct (config.yaml / env vars)
  api/
    playground.html — embedded UI (requires rebuild to update)
    tts.rs          — TTS streaming handler
    audio.rs        — STT handler
    classify.rs     — image classification
    yolo.rs         — YOLO detection
    llm_proxy.rs    — reverse proxy → LLM microservice
    stt_proxy.rs    — reverse proxy → STT microservice
    health.rs       — /health
    system.rs       — /system/info
  core/
    tts_manager.rs  — routes requests across 6 TTS engines
    kokoro_onnx.rs  — primary TTS engine
    audio.rs        — decode, resample, WAV I/O
    yolo.rs         — YoloDetector, NMS
    model_cache.rs  — FNV-1a LRU cache
```

## License

Copyright © 2025 Kolosal
