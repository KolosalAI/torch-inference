/// WebSocket Audio Pipeline
///
/// `GET /audio/ws` — upgrades to a persistent WebSocket connection that
/// multiplexes both real-time TTS streaming and STT transcription.
///
/// ## Protocol
///
/// **Client → Server (text frames — JSON)**
/// ```json
/// {"type":"tts","text":"Hello","voice":"af_heart","speed":1.0}
/// {"type":"stt_begin","sample_rate":16000}
/// {"type":"stt_end"}
/// ```
///
/// **Client → Server (binary frames)**
/// Raw PCM f32le samples while `stt_begin` is active.
///
/// **Server → Client (text frames — JSON)**
/// ```json
/// {"type":"ready"}
/// {"type":"tts_meta","sample_rate":24000,"encoding":"pcm_f32le"}
/// {"type":"tts_done","duration_ms":1200}
/// {"type":"transcript","text":"…","confidence":0.9,"is_final":true}
/// {"type":"error","msg":"…"}
/// ```
///
/// **Server → Client (binary frames)**
/// Raw PCM f32le chunks during TTS streaming.
use crate::api::audio::AudioState;
use crate::api::tts::TTSState;
use crate::core::tts_engine::SynthesisParams;
use crate::core::tts_pipeline::StreamingTtsPipeline;
use actix_web::{web, HttpRequest, HttpResponse};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::interval;

// ── Protocol types ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMsg {
    /// Request TTS synthesis; server streams back binary PCM.
    Tts {
        text: String,
        #[serde(default)]
        voice: Option<String>,
        #[serde(default = "one_f32")]
        speed: f32,
    },
    /// Begin an STT session; subsequent binary frames are PCM f32le samples.
    SttBegin {
        #[serde(default = "default_sample_rate")]
        sample_rate: u32,
    },
    /// End the STT session; server responds with a `transcript` message.
    SttEnd,
}

fn one_f32() -> f32 {
    1.0
}
fn default_sample_rate() -> u32 {
    16000
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerMsg {
    Ready,
    /// Metadata that precedes the binary TTS audio stream.
    TtsMeta {
        sample_rate: u32,
        encoding: String,
    },
    TtsDone {
        duration_ms: u64,
    },
    Transcript {
        text: String,
        confidence: f32,
        is_final: bool,
    },
    Error {
        msg: String,
    },
}

impl ServerMsg {
    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| r#"{"type":"error","msg":"serialise"}"#.to_string())
    }
}

// ── Session state machine ─────────────────────────────────────────────────────

enum SessionState {
    Idle,
    /// Binary TTS chunks are being fanned out to the client.
    StreamingTts {
        rx: tokio::sync::mpsc::Receiver<anyhow::Result<crate::core::tts_pipeline::AudioChunk>>,
        sent_meta: bool,
        total_samples: usize,
        sample_rate: u32,
    },
    /// Binary frames from the client are being accumulated as PCM.
    AccumulatingStt {
        buf: Vec<f32>,
        sample_rate: u32,
    },
}

// ── Handler ───────────────────────────────────────────────────────────────────

pub async fn ws_audio_handler(
    req: HttpRequest,
    stream: web::Payload,
    tts_state: web::Data<TTSState>,
    audio_state: web::Data<AudioState>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, msg_stream) = actix_ws::handle(&req, stream)?;

    let tts_state = tts_state.into_inner();
    let audio_state = audio_state.into_inner();

    actix_web::rt::spawn(run_session(
        session.clone(),
        msg_stream,
        tts_state,
        audio_state,
    ));

    Ok(response)
}

async fn run_session(
    mut session: actix_ws::Session,
    mut msg_stream: actix_ws::MessageStream,
    tts_state: std::sync::Arc<TTSState>,
    audio_state: std::sync::Arc<AudioState>,
) {
    if session.text(ServerMsg::Ready.to_json()).await.is_err() {
        return;
    }

    let mut hb = interval(Duration::from_secs(20));
    hb.tick().await; // consume the immediate first tick

    let mut state = SessionState::Idle;

    loop {
        match &mut state {
            // ── Idle / STT accumulation ──────────────────────────────────────
            SessionState::Idle | SessionState::AccumulatingStt { .. } => {
                tokio::select! {
                    _ = hb.tick() => {
                        if session.ping(b"hb").await.is_err() { break; }
                    }
                    msg = msg_stream.next() => {
                        match msg {
                            Some(Ok(m)) => {
                                if !handle_incoming(m, &mut session, &mut state, &tts_state, &audio_state).await {
                                    break;
                                }
                            }
                            _ => break,
                        }
                    }
                }
            }

            // ── TTS streaming ────────────────────────────────────────────────
            SessionState::StreamingTts { rx, sent_meta, total_samples, sample_rate } => {
                tokio::select! {
                    _ = hb.tick() => {
                        if session.ping(b"hb").await.is_err() { break; }
                    }
                    // Prefer draining TTS chunks over incoming messages.
                    chunk = rx.recv() => {
                        match chunk {
                            Some(Ok(c)) => {
                                if !*sent_meta {
                                    *sample_rate = c.audio.sample_rate;
                                    let meta = ServerMsg::TtsMeta {
                                        sample_rate: c.audio.sample_rate,
                                        encoding: "pcm_f32le".to_string(),
                                    };
                                    if session.text(meta.to_json()).await.is_err() { break; }
                                    *sent_meta = true;
                                }
                                *total_samples += c.audio.samples.len();
                                let bytes = actix_web::web::Bytes::from(c.to_f32_le());
                                if session.binary(bytes).await.is_err() { break; }
                            }
                            Some(Err(e)) => {
                                let _ = session.text(ServerMsg::Error { msg: e.to_string() }.to_json()).await;
                                state = SessionState::Idle;
                            }
                            None => {
                                // TTS pipeline exhausted.
                                let dur_ms = (*total_samples as u64 * 1000)
                                    / (*sample_rate).max(1) as u64;
                                let _ = session.text(ServerMsg::TtsDone { duration_ms: dur_ms }.to_json()).await;
                                state = SessionState::Idle;
                            }
                        }
                    }
                    msg = msg_stream.next() => {
                        match msg {
                            Some(Ok(m)) => {
                                if !handle_incoming(m, &mut session, &mut state, &tts_state, &audio_state).await {
                                    break;
                                }
                            }
                            _ => break,
                        }
                    }
                }
            }
        }
    }

    let _ = session.close(None).await;
}

// ── Message dispatcher ────────────────────────────────────────────────────────

/// Returns `false` when the session should be closed.
async fn handle_incoming(
    msg: actix_ws::Message,
    session: &mut actix_ws::Session,
    state: &mut SessionState,
    tts_state: &TTSState,
    audio_state: &AudioState,
) -> bool {
    match msg {
        actix_ws::Message::Text(txt) => {
            match serde_json::from_str::<ClientMsg>(&txt) {
                Ok(ClientMsg::Tts { text, voice, speed }) => {
                    start_tts(session, state, tts_state, text, voice, speed).await;
                }
                Ok(ClientMsg::SttBegin { sample_rate }) => {
                    *state = SessionState::AccumulatingStt {
                        buf: Vec::new(),
                        sample_rate,
                    };
                }
                Ok(ClientMsg::SttEnd) => {
                    let (buf, sr) = match std::mem::replace(state, SessionState::Idle) {
                        SessionState::AccumulatingStt { buf, sample_rate } => (buf, sample_rate),
                        other => {
                            *state = other;
                            let _ = session.text(
                                ServerMsg::Error { msg: "not in STT session".to_string() }.to_json()
                            ).await;
                            return true;
                        }
                    };
                    finish_stt(session, audio_state, &buf, sr).await;
                }
                Err(e) => {
                    let _ = session.text(
                        ServerMsg::Error { msg: format!("bad message: {e}") }.to_json()
                    ).await;
                }
            }
        }
        actix_ws::Message::Binary(bin) => {
            if let SessionState::AccumulatingStt { buf, .. } = state {
                let samples: Vec<f32> = bin
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                buf.extend_from_slice(&samples);
            }
            // Binary frames while in TTS state are silently ignored.
        }
        actix_ws::Message::Ping(data) => {
            if session.pong(&data).await.is_err() {
                return false;
            }
        }
        actix_ws::Message::Close(_) => return false,
        _ => {}
    }
    true
}

// ── TTS helper ────────────────────────────────────────────────────────────────

async fn start_tts(
    session: &mut actix_ws::Session,
    state: &mut SessionState,
    tts_state: &TTSState,
    text: String,
    voice: Option<String>,
    speed: f32,
) {
    if text.trim().is_empty() {
        let _ = session.text(ServerMsg::Error { msg: "text is empty".to_string() }.to_json()).await;
        return;
    }

    // Try the requested voice as engine id; fall back to default.
    let engine = voice
        .as_deref()
        .and_then(|v| tts_state.manager.get_engine(v))
        .or_else(|| tts_state.manager.get_default_engine());

    let engine = match engine {
        Some(e) => e,
        None => {
            let _ = session.text(
                ServerMsg::Error { msg: "no TTS engine available".to_string() }.to_json()
            ).await;
            return;
        }
    };

    let params = SynthesisParams {
        speed,
        voice: voice.clone(),
        ..SynthesisParams::default()
    };

    let pipeline = StreamingTtsPipeline::new(engine);
    let rx = pipeline.synthesize_streaming(&text, params);

    // Replace any previous TTS session (old rx is dropped, stopping its synthesis).
    *state = SessionState::StreamingTts {
        rx,
        sent_meta: false,
        total_samples: 0,
        sample_rate: 24000,
    };
}

// ── STT helper ────────────────────────────────────────────────────────────────

async fn finish_stt(
    session: &mut actix_ws::Session,
    audio_state: &AudioState,
    samples: &[f32],
    sample_rate: u32,
) {
    if samples.is_empty() {
        let _ = session.text(ServerMsg::Error { msg: "no audio received".to_string() }.to_json()).await;
        return;
    }

    let audio = crate::core::audio::AudioData {
        samples: samples.to_vec(),
        sample_rate,
        channels: 1,
    };

    let result = match audio_state.model_manager.get_stt_model("default") {
        Some(model) => model.transcribe(&audio, false),
        None => {
            let _ = session.text(
                ServerMsg::Error { msg: "no STT model available".to_string() }.to_json()
            ).await;
            return;
        }
    };

    match result {
        Ok(r) => {
            let msg = ServerMsg::Transcript {
                text: r.text,
                confidence: r.confidence,
                is_final: true,
            };
            let _ = session.text(msg.to_json()).await;
        }
        Err(e) => {
            let _ = session.text(ServerMsg::Error { msg: e.to_string() }.to_json()).await;
        }
    }
}

// ── Route config ──────────────────────────────────────────────────────────────

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/audio/ws", web::get().to(ws_audio_handler));
}
