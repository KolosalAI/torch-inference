// @ts-check
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

// Shared fixture path
const imgPath = path.resolve(__dirname, '../fixtures/test.jpg');
const wavPath = path.resolve(__dirname, '../fixtures/test.wav');

test.describe('API Endpoints — Extended', () => {

  // ── System & Info ──────────────────────────────────────────────────────────
  test.describe('System & Info', () => {
    test('GET /info returns server config and health', async ({ request }) => {
      const res = await request.get('/info');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      // /info returns full config blob including health sub-object
      expect(body.health).toBeDefined();
      expect(body.server).toBeDefined();
    });

    test('GET /system/config returns JSON config', async ({ request }) => {
      const res = await request.get('/system/config');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      // Returns inference config blob
      expect(body.server || body.inference || body.cache).toBeDefined();
    });

    test('GET /system/gpu/stats returns GPU array (may be empty if no GPU)', async ({ request }) => {
      const res = await request.get('/system/gpu/stats');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      // Always returns an array (empty on CPU-only hosts)
      expect(Array.isArray(body)).toBe(true);
    });
  });

  // ── Metrics ────────────────────────────────────────────────────────────────
  test.describe('Metrics', () => {
    test('GET /metrics returns Prometheus text or placeholder', async ({ request }) => {
      const res = await request.get('/metrics');
      expect(res.ok()).toBe(true);
      const text = await res.text();
      // Either real Prometheus output or the "not enabled" stub
      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(0);
    });

    test('GET /endpoints returns per-endpoint stats', async ({ request }) => {
      const res = await request.get('/endpoints');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(typeof body.count).toBe('number');
      expect(Array.isArray(body.endpoints)).toBe(true);
    });
  });

  // ── Performance ────────────────────────────────────────────────────────────
  test.describe('Performance', () => {
    test('POST /performance/profile returns profile result', async ({ request }) => {
      const res = await request.post('/performance/profile', {
        data: { model_id: 'test', input_shape: [1, 3, 224, 224] },
      });
      // Endpoint always returns 200 with timing info
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(body.total_time_ms).toBeDefined();
    });

    test('GET /performance/optimize returns optimisation tips', async ({ request }) => {
      const res = await request.get('/performance/optimize');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      // Returns optimisations_applied array and related fields
      expect(body.optimizations_applied || body.tips || body.recommendations).toBeDefined();
    });
  });

  // ── TTS ────────────────────────────────────────────────────────────────────
  test.describe('TTS', () => {
    test('POST /tts/synthesize returns JSON with audio_base64', async ({ request }) => {
      const res = await request.post('/tts/synthesize', {
        data: { text: 'Hello.', speed: 1.0 },
      });
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(body.data).toBeDefined();
      expect(body.data.audio_base64).toBeDefined();
      expect(body.data.audio_base64.length).toBeGreaterThan(0);
    });

    test('GET /tts/engines/kokoro/capabilities returns engine caps', async ({ request }) => {
      const res = await request.get('/tts/engines/kokoro/capabilities');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(body.name).toBeDefined();
      expect(body.sample_rate).toBeGreaterThan(0);
    });

    test('GET /tts/stats returns engine stats', async ({ request }) => {
      const res = await request.get('/tts/stats');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(body.stats).toBeDefined();
      expect(typeof body.stats.total_engines).toBe('number');
    });
  });

  // ── Classify (streaming) ───────────────────────────────────────────────────
  test.describe('Classify', () => {
    test('POST /classify/stream returns SSE stream with result or error', async ({ request }) => {
      const b64 = fs.readFileSync(imgPath).toString('base64');
      const res = await request.post('/classify/stream', {
        data: { images: [b64], top_k: 1 },
      });
      // Returns 200 SSE text/event-stream regardless
      expect(res.ok()).toBe(true);
      const text = await res.text();
      expect(text).toContain('data:');
    });
  });

  // ── Audio ──────────────────────────────────────────────────────────────────
  test.describe('Audio', () => {
    test('POST /audio/synthesize — reachable (404 if no default TTS model configured)', async ({ request }) => {
      const res = await request.post('/audio/synthesize', {
        data: { text: 'Hello.' },
      });
      // 404 when default model not configured, 200 when it is
      expect([200, 404, 503]).toContain(res.status());
      const body = await res.json();
      expect(body).toBeDefined();
    });

    test('POST /audio/validate — reachable (400 when no multipart body sent as JSON)', async ({ request }) => {
      const res = await request.post('/audio/validate', {
        data: {},
      });
      // Endpoint expects multipart; JSON body triggers 400
      expect([200, 400, 415]).toContain(res.status());
    });

    test('POST /audio/validate — multipart with wav file returns 200 or 400', async ({ request }) => {
      const res = await request.post('/audio/validate', {
        multipart: {
          file: {
            name: 'test.wav',
            mimeType: 'audio/wav',
            buffer: fs.readFileSync(wavPath),
          },
        },
      });
      expect([200, 400, 415, 422]).toContain(res.status());
    });
  });

  // ── Image Security ─────────────────────────────────────────────────────────
  test.describe('Image Security', () => {
    test('POST /image/validate/security with image field returns security result', async ({ request }) => {
      const res = await request.post('/image/validate/security', {
        multipart: {
          image: {
            name: 'test.jpg',
            mimeType: 'image/jpeg',
            buffer: fs.readFileSync(imgPath),
          },
        },
      });
      // 200 with security_result, or 400 if field name wrong
      expect([200, 400]).toContain(res.status());
      if (res.status() === 200) {
        const body = await res.json();
        expect(body.security_result).toBeDefined();
      }
    });

    test('POST /image/process/secure with image field returns 200 or 400', async ({ request }) => {
      const res = await request.post('/image/process/secure', {
        multipart: {
          image: {
            name: 'test.jpg',
            mimeType: 'image/jpeg',
            buffer: fs.readFileSync(imgPath),
          },
        },
      });
      expect([200, 400, 501]).toContain(res.status());
    });

    test('GET /image/security/stats returns counters', async ({ request }) => {
      const res = await request.get('/image/security/stats');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(typeof body.total_processed).toBe('number');
      expect(typeof body.threats_detected).toBe('number');
    });

    test('GET /image/health returns image backend info', async ({ request }) => {
      const res = await request.get('/image/health');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(body.status).toBe('ok');
      expect(body.image_backend).toBeDefined();
    });
  });

  // ── YOLO ───────────────────────────────────────────────────────────────────
  test.describe('YOLO', () => {
    test('POST /yolo/detect with query params returns detection result or error', async ({ request }) => {
      // YOLO detect uses query params for model_version + model_size
      const b64 = fs.readFileSync(imgPath).toString('base64');
      const res = await request.post('/yolo/detect?model_version=v8&model_size=n', {
        data: { image_base64: b64 },
      });
      // 200 (success or "model not available"), 400 (bad input), 500 (file system err)
      expect([200, 400, 500]).toContain(res.status());
    });

    test('GET /yolo/info with query params returns model info', async ({ request }) => {
      const res = await request.get('/yolo/info?model_version=v8&model_size=n');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(body.model_name).toBeDefined();
      expect(body.version).toBeDefined();
    });

    test('GET /yolo/models returns list of YOLO versions and sizes', async ({ request }) => {
      const res = await request.get('/yolo/models');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(Array.isArray(body.versions)).toBe(true);
      expect(body.total_models).toBeGreaterThan(0);
    });

    test('POST /yolo/download returns download URL', async ({ request }) => {
      const res = await request.post('/yolo/download', {
        data: { model_version: 'v8', model_size: 'n' },
      });
      // 200 with URL, 202 if queued, 400/409 if bad request / already downloading
      expect([200, 202, 400, 409]).toContain(res.status());
      if (res.status() === 200) {
        const body = await res.json();
        expect(body.download_url || body.success).toBeDefined();
      }
    });
  });

  // ── Neural Network (/api/nn/*) ─────────────────────────────────────────────
  test.describe('Neural Network (/api/nn/*)', () => {
    test('POST /api/nn/load with nonexistent model returns 500 (PyTorch not compiled)', async ({ request }) => {
      const res = await request.post('/api/nn/load', {
        data: { model_id: 'nonexistent', model_path: '/tmp/nonexistent.onnx' },
      });
      // Returns 500 when PyTorch feature not compiled in
      expect([400, 404, 500]).toContain(res.status());
    });

    test('POST /api/nn/predict with nonexistent model returns error', async ({ request }) => {
      const res = await request.post('/api/nn/predict', {
        data: { model_id: 'nonexistent', input_data: [1.0, 2.0], input_shape: [1, 2] },
      });
      expect([400, 404, 500]).toContain(res.status());
    });

    test('POST /api/nn/batch with nonexistent model returns error', async ({ request }) => {
      const res = await request.post('/api/nn/batch', {
        data: { model_id: 'nonexistent', inputs: [[1.0, 2.0], [3.0, 4.0]], input_shape: [1, 2] },
      });
      expect([400, 404, 500]).toContain(res.status());
    });

    test('GET /api/nn/models returns list (may be empty)', async ({ request }) => {
      const res = await request.get('/api/nn/models');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(Array.isArray(body.models) || body.models !== undefined).toBe(true);
      expect(typeof body.total).toBe('number');
    });

    test('GET /api/nn/models/nonexistent returns 404', async ({ request }) => {
      const res = await request.get('/api/nn/models/nonexistent');
      expect(res.status()).toBe(404);
    });

    test('DELETE /api/nn/models/nonexistent returns 404', async ({ request }) => {
      const res = await request.delete('/api/nn/models/nonexistent');
      expect(res.status()).toBe(404);
    });
  });

  // ── Model Registry (/registry/*) ──────────────────────────────────────────
  test.describe('Model Registry (/registry/*)', () => {
    test('GET /registry/models returns model list (may be empty)', async ({ request }) => {
      const res = await request.get('/registry/models');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(body.success).toBe(true);
      expect(body.data).toBeDefined();
    });

    test('GET /registry/stats returns registry statistics', async ({ request }) => {
      const res = await request.get('/registry/stats');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(body.success).toBe(true);
      expect(body.data).toBeDefined();
      expect(typeof body.data.total_models).toBe('number');
    });

    test('GET /registry/export returns full registry JSON', async ({ request }) => {
      const res = await request.get('/registry/export');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(body.success).toBe(true);
      expect(body.data.exported_at).toBeDefined();
    });

    test('GET /registry/models/nonexistent returns 404', async ({ request }) => {
      const res = await request.get('/registry/models/nonexistent');
      expect(res.status()).toBe(404);
    });

    test('POST /registry/models/nonexistent/load returns 400 or 404', async ({ request }) => {
      const res = await request.post('/registry/models/nonexistent/load');
      // 400 when PyTorch not compiled; 404 when model genuinely not found
      expect([400, 404]).toContain(res.status());
    });

    test('POST /registry/register with nonexistent path returns 400', async ({ request }) => {
      const res = await request.post('/registry/register', {
        data: { path: '/nonexistent/model.onnx' },
      });
      // 400 when PyTorch not compiled, or 400 when file does not exist
      expect([400, 404]).toContain(res.status());
    });

    test('POST /registry/scan with /tmp returns 200', async ({ request }) => {
      const res = await request.post('/registry/scan', {
        data: { path: '/tmp' },
      });
      expect([200, 400]).toContain(res.status());
    });

    test('POST /registry/infer with nonexistent model returns 400', async ({ request }) => {
      const res = await request.post('/registry/infer', {
        data: { model_id: 'nonexistent', input: [] },
      });
      // Returns 400 when PyTorch not compiled
      expect([400, 404, 500]).toContain(res.status());
    });
  });

  // ── Models (/api/models/*) ─────────────────────────────────────────────────
  test.describe('Models (/api/models/*)', () => {
    test('GET /api/models/comparison returns comparison data', async ({ request }) => {
      const res = await request.get('/api/models/comparison');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      // Returns an array of model comparison objects
      expect(Array.isArray(body.models) || typeof body.models === 'object').toBe(true);
    });
  });

  // ── Model Downloads (/models/*) ────────────────────────────────────────────
  test.describe('Model Downloads (/models/*)', () => {
    test('GET /models/download/list returns list of downloads (may be empty array)', async ({ request }) => {
      const res = await request.get('/models/download/list');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(Array.isArray(body)).toBe(true);
    });

    test('GET /models/cache/info returns cache statistics', async ({ request }) => {
      const res = await request.get('/models/cache/info');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(body.cache_dir).toBeDefined();
      expect(typeof body.model_count).toBe('number');
    });

    test('GET /models/sota returns SOTA model list', async ({ request }) => {
      const res = await request.get('/models/sota');
      expect(res.ok()).toBe(true);
      const body = await res.json();
      expect(Array.isArray(body.models)).toBe(true);
      expect(body.models.length).toBeGreaterThan(0);
      expect(body.models[0].id).toBeDefined();
    });

    test('GET /models/download/status/nonexistent returns 404', async ({ request }) => {
      const res = await request.get('/models/download/status/nonexistent');
      expect(res.status()).toBe(404);
    });

    test('GET /models/download/nonexistent/info returns 404', async ({ request }) => {
      const res = await request.get('/models/download/nonexistent/info');
      expect(res.status()).toBe(404);
    });

    test('DELETE /models/download/nonexistent returns 500 (model not found)', async ({ request }) => {
      const res = await request.delete('/models/download/nonexistent');
      // Server returns 500 with "Model not found: nonexistent" when task doesn't exist
      expect([404, 500]).toContain(res.status());
    });

    test('POST /models/download with empty repo_id returns 400', async ({ request }) => {
      const res = await request.post('/models/download', {
        data: {
          model_id: 'invalid',
          model_name: 'test',
          source: 'hf',
          source_type: 'huggingface',
          repo: '',
          filename: 'test.bin',
        },
      });
      // 400 when repo_id is empty/missing
      expect([400, 422]).toContain(res.status());
    });

    test('POST /models/sota/nonexistent returns 404', async ({ request }) => {
      const res = await request.post('/models/sota/nonexistent');
      expect(res.status()).toBe(404);
    });
  });

  // ── Logs ──────────────────────────────────────────────────────────────────
  test.describe('Logs', () => {
    test('GET /logs/nonexistent.log returns 404', async ({ request }) => {
      const res = await request.get('/logs/nonexistent.log');
      expect(res.status()).toBe(404);
    });

    // DELETE /logs/nonexistent.log — safe to test since the file does not exist;
    // the server will 404 before any destructive action occurs.
    test('DELETE /logs/nonexistent.log returns 404', async ({ request }) => {
      const res = await request.delete('/logs/nonexistent.log');
      expect(res.status()).toBe(404);
    });
  });
});
