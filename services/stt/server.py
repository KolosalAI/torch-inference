#!/usr/bin/env python3
"""
Kolosal STT microservice — openai-whisper backed.
Loads model from models/whisper/ (no external download needed after first run).
Listens on 127.0.0.1:8002.
Proxied by the main server at /stt/{tail}.
"""

import os
import sys
import tempfile
import logging
import json
import cgi
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [STT] %(levelname)s %(message)s")
log = logging.getLogger("stt")

HOST = "127.0.0.1"
PORT = int(os.environ.get("STT_PORT", "8002"))
MODEL_SIZE = os.environ.get("STT_MODEL", "base")

# Resolve model directory relative to this script (i.e. project root/models/whisper)
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent.parent / "models" / "whisper"

log.info(f"Loading whisper model '{MODEL_SIZE}' from {MODEL_DIR}")

try:
    import whisper
    model = whisper.load_model(MODEL_SIZE, download_root=str(MODEL_DIR))
    log.info("Model ready.")
except Exception as e:
    log.error(f"Failed to load model: {e}")
    sys.exit(1)


def transcribe_file(path: str, timestamps: bool = False, language: str = None):
    options = {"fp16": False, "verbose": False}
    if language:
        options["language"] = language
    result = model.transcribe(path, **options)
    text = result.get("text", "").strip()
    out = {"text": text, "language": result.get("language"), "duration": result.get("duration")}
    if timestamps:
        out["segments"] = [
            {
                "text": s.get("text", "").strip(),
                "start_time": s.get("start", 0.0),
                "end_time": s.get("end", 0.0),
                "confidence": 0.0,
            }
            for s in result.get("segments", [])
        ]
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(fmt % args)

    def _send_json(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/health", "/stt/health", "/"):
            self._send_json(200, {
                "status": "ok",
                "model": MODEL_SIZE,
                "models_available": [f"STT: {MODEL_SIZE}"],
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/transcribe", "/stt/transcribe", "/audio/transcribe"):
            self._send_json(404, {"error": "not found"})
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json(400, {"error": "expected multipart/form-data"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": str(length),
        }
        form = cgi.FieldStorage(fp=io.BytesIO(body), environ=environ, keep_blank_values=True)

        audio_data = None
        for field_name in ("audio", "file"):
            if field_name in form:
                audio_data = form[field_name].file.read()
                break

        if not audio_data:
            self._send_json(400, {"error": "no audio field in request"})
            return

        timestamps = False
        if "timestamps" in form:
            timestamps = form["timestamps"].value.strip().lower() in ("true", "1", "yes")

        language = None
        if "language" in form:
            v = form["language"].value.strip()
            if v:
                language = v

        # Determine extension from content-type or default to wav
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            result = transcribe_file(tmp_path, timestamps=timestamps, language=language)
            self._send_json(200, result)
        except Exception as e:
            log.error(f"Transcription error: {e}")
            self._send_json(500, {"error": str(e)})
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), Handler)
    log.info(f"STT service listening on {HOST}:{PORT} (model={MODEL_SIZE})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
