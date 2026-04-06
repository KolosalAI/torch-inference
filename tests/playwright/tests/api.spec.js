/**
 * Direct HTTP API tests — no browser rendering required.
 * Uses Playwright's `request` fixture which inherits baseURL from the config.
 */
const { test, expect } = require('@playwright/test');

// ── /health ──────────────────────────────────────────────────────────────────

test.describe('GET /health', () => {
  let res, body;
  test.beforeAll(async ({ request }) => {
    res = await request.get('/health');
    body = await res.json();
  });

  test('returns 200', () => { expect(res.status()).toBe(200); });

  test('returns JSON content-type', () => {
    expect(res.headers()['content-type']).toMatch(/application\/json/);
  });

  test('body has required fields', () => {
    expect(typeof body.status).toBe('string');
    expect(typeof body.uptime_seconds).toBe('number');
    expect(body.uptime_seconds).toBeGreaterThanOrEqual(0);
  });

  test('timestamp is valid ISO string', () => {
    expect(typeof body.timestamp).toBe('string');
    expect(new Date(body.timestamp).getTime()).not.toBeNaN();
  });
});

// ── /health/live ─────────────────────────────────────────────────────────────

test.describe('GET /health/live', () => {
  let res, body;
  test.beforeAll(async ({ request }) => {
    res = await request.get('/health/live');
    body = await res.json();
  });

  test('returns 200', () => { expect(res.status()).toBe(200); });

  test('body has status, version, uptime_seconds', () => {
    expect(typeof body.status).toBe('string');
    expect(body).toHaveProperty('version');
    expect(body.uptime_seconds).toBeGreaterThanOrEqual(0);
  });

  test('includes x-correlation-id response header', () => {
    expect(res.headers()).toHaveProperty('x-correlation-id');
  });
});

// ── /health/ready ─────────────────────────────────────────────────────────────

test.describe('GET /health/ready', () => {
  let res, body;
  test.beforeAll(async ({ request }) => {
    res = await request.get('/health/ready');
    body = await res.json();
  });

  test('returns 200 or 503', () => { expect([200, 503]).toContain(res.status()); });
  test('body has status field', () => { expect(body).toHaveProperty('status'); });

  test('includes x-correlation-id response header', () => {
    expect(res.headers()).toHaveProperty('x-correlation-id');
  });
});

// ── /metrics ─────────────────────────────────────────────────────────────────

test.describe('GET /metrics', () => {
  test('returns 200 or 404 (feature flag)', async ({ request }) => {
    const res = await request.get('/metrics');
    expect([200, 404]).toContain(res.status());
  });

  test('when available, content-type is text/plain', async ({ request }) => {
    const res = await request.get('/metrics');
    if (res.status() === 200) {
      expect(res.headers()['content-type']).toMatch(/text\/plain/);
    }
  });
});

// ── /models ───────────────────────────────────────────────────────────────────

test.describe('GET /models', () => {
  let body;
  test.beforeAll(async ({ request }) => {
    const res = await request.get('/models');
    expect(res.status()).toBe(200);
    body = await res.json();
  });

  test('body has models array', () => { expect(Array.isArray(body.models)).toBe(true); });
  test('body has total (number >= 0)', () => {
    expect(typeof body.total).toBe('number');
    expect(body.total).toBeGreaterThanOrEqual(0);
  });
});

// ── /stats ────────────────────────────────────────────────────────────────────

test.describe('GET /stats', () => {
  let res, body;
  test.beforeAll(async ({ request }) => {
    res = await request.get('/stats');
    body = await res.json();
  });

  test('returns 200', () => { expect(res.status()).toBe(200); });
  test('body has total_requests', () => { expect(body).toHaveProperty('total_requests'); });
});

// ── /endpoints ───────────────────────────────────────────────────────────────

test.describe('GET /endpoints', () => {
  let body;
  test.beforeAll(async ({ request }) => {
    const res = await request.get('/endpoints');
    expect(res.status()).toBe(200);
    body = await res.json();
  });

  test('body has endpoints array', () => { expect(Array.isArray(body.endpoints)).toBe(true); });
  test('body has count (number)', () => { expect(typeof body.count).toBe('number'); });
});

// ── /info ─────────────────────────────────────────────────────────────────────

test.describe('GET /info', () => {
  let body;
  test.beforeAll(async ({ request }) => {
    const res = await request.get('/info');
    expect(res.status()).toBe(200);
    body = await res.json();
  });

  test('body has server, device, health fields', () => {
    expect(body).toHaveProperty('server');
    expect(body).toHaveProperty('device');
    expect(body).toHaveProperty('health');
  });
});

// ── /system/info ──────────────────────────────────────────────────────────────

test.describe('GET /system/info', () => {
  test('returns 200 with JSON', async ({ request }) => {
    const res = await request.get('/system/info');
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(typeof body).toBe('object');
  });
});

// ── /system/config ───────────────────────────────────────────────────────────

test.describe('GET /system/config', () => {
  test('returns 200', async ({ request }) => {
    expect((await request.get('/system/config')).status()).toBe(200);
  });
});

// ── /system/gpu/stats ────────────────────────────────────────────────────────

test.describe('GET /system/gpu/stats', () => {
  test('returns 200', async ({ request }) => {
    expect((await request.get('/system/gpu/stats')).status()).toBe(200);
  });
});

// ── /performance ─────────────────────────────────────────────────────────────

test.describe('GET /performance', () => {
  test('returns 200 with JSON', async ({ request }) => {
    const res = await request.get('/performance');
    expect(res.status()).toBe(200);
    expect(typeof await res.json()).toBe('object');
  });
});

// ── /logs ─────────────────────────────────────────────────────────────────────

test.describe('GET /logs', () => {
  test('returns 200 with JSON', async ({ request }) => {
    const res = await request.get('/logs');
    expect(res.status()).toBe(200);
    expect(typeof await res.json()).toBe('object');
  });
});

// ── /tts/health ───────────────────────────────────────────────────────────────

test.describe('GET /tts/health', () => {
  test('returns 200 with JSON', async ({ request }) => {
    const res = await request.get('/tts/health');
    expect(res.status()).toBe(200);
    expect(typeof await res.json()).toBe('object');
  });
});

// ── /tts/engines ─────────────────────────────────────────────────────────────

test.describe('GET /tts/engines', () => {
  test('returns 200', async ({ request }) => {
    expect((await request.get('/tts/engines')).status()).toBe(200);
  });
});

// ── /tts/stats ────────────────────────────────────────────────────────────────

test.describe('GET /tts/stats', () => {
  test('returns 200', async ({ request }) => {
    expect((await request.get('/tts/stats')).status()).toBe(200);
  });
});

// ── /audio/health ─────────────────────────────────────────────────────────────

test.describe('GET /audio/health', () => {
  test('returns 200 with JSON', async ({ request }) => {
    const res = await request.get('/audio/health');
    expect(res.status()).toBe(200);
    expect(typeof await res.json()).toBe('object');
  });
});

// ── /image/health ─────────────────────────────────────────────────────────────

test.describe('GET /image/health', () => {
  test('returns 200', async ({ request }) => {
    expect((await request.get('/image/health')).status()).toBe(200);
  });
});

// ── /image/security/stats ─────────────────────────────────────────────────────

test.describe('GET /image/security/stats', () => {
  test('returns 200 with JSON', async ({ request }) => {
    const res = await request.get('/image/security/stats');
    expect(res.status()).toBe(200);
    expect(typeof await res.json()).toBe('object');
  });
});

// ── /models/download/list ────────────────────────────────────────────────────

test.describe('GET /models/download/list', () => {
  test('returns 200', async ({ request }) => {
    expect((await request.get('/models/download/list')).status()).toBe(200);
  });
});

// ── /models/available ────────────────────────────────────────────────────────

test.describe('GET /models/available', () => {
  test('returns 200', async ({ request }) => {
    expect((await request.get('/models/available')).status()).toBe(200);
  });
});

// ── /models/managed ──────────────────────────────────────────────────────────

test.describe('GET /models/managed', () => {
  test('returns 200', async ({ request }) => {
    expect((await request.get('/models/managed')).status()).toBe(200);
  });
});

// ── /models/cache/info ────────────────────────────────────────────────────────

test.describe('GET /models/cache/info', () => {
  test('returns 200 with JSON', async ({ request }) => {
    const res = await request.get('/models/cache/info');
    expect(res.status()).toBe(200);
    expect(typeof await res.json()).toBe('object');
  });
});

// ── /models/sota ─────────────────────────────────────────────────────────────

test.describe('GET /models/sota', () => {
  test('returns 200', async ({ request }) => {
    expect((await request.get('/models/sota')).status()).toBe(200);
  });
});

// ── /registry/models ─────────────────────────────────────────────────────────

test.describe('GET /registry/models', () => {
  test('returns 200 with JSON', async ({ request }) => {
    const res = await request.get('/registry/models');
    expect(res.status()).toBe(200);
    expect(typeof await res.json()).toBe('object');
  });
});

// ── /registry/stats ───────────────────────────────────────────────────────────

test.describe('GET /registry/stats', () => {
  test('returns 200', async ({ request }) => {
    expect((await request.get('/registry/stats')).status()).toBe(200);
  });
});

// ── /registry/export ──────────────────────────────────────────────────────────

test.describe('GET /registry/export', () => {
  test('returns 200', async ({ request }) => {
    expect((await request.get('/registry/export')).status()).toBe(200);
  });
});

// ── /api/models ───────────────────────────────────────────────────────────────

test.describe('GET /api/models', () => {
  test('returns 200 with JSON', async ({ request }) => {
    const res = await request.get('/api/models');
    expect(res.status()).toBe(200);
    expect(typeof await res.json()).toBe('object');
  });
});

// ── /api/models/downloaded ────────────────────────────────────────────────────

test.describe('GET /api/models/downloaded', () => {
  test('returns 200 with JSON', async ({ request }) => {
    const res = await request.get('/api/models/downloaded');
    expect(res.status()).toBe(200);
    const body = await res.json();
    // Body is an array (list) or an object wrapper
    expect(body !== null).toBe(true);
  });
});

// ── /api/models/comparison ────────────────────────────────────────────────────

test.describe('GET /api/models/comparison', () => {
  test('returns 200', async ({ request }) => {
    expect((await request.get('/api/models/comparison')).status()).toBe(200);
  });
});

// ── /yolo/info ────────────────────────────────────────────────────────────────

test.describe('GET /yolo/info', () => {
  test('returns 200 or 404', async ({ request }) => {
    expect([200, 404]).toContain((await request.get('/yolo/info')).status());
  });
});

// ── /yolo/models ─────────────────────────────────────────────────────────────

test.describe('GET /yolo/models', () => {
  test('returns 200 or 404', async ({ request }) => {
    expect([200, 404]).toContain((await request.get('/yolo/models')).status());
  });
});

// ── /api/nn/models ────────────────────────────────────────────────────────────

test.describe('GET /api/nn/models', () => {
  test('returns 200 or 404', async ({ request }) => {
    expect([200, 404]).toContain((await request.get('/api/nn/models')).status());
  });
});

// ── /v1/models ────────────────────────────────────────────────────────────────

test.describe('GET /v1/models', () => {
  let body;
  test.beforeAll(async ({ request }) => {
    const res = await request.get('/v1/models');
    expect(res.status()).toBe(200);
    body = await res.json();
  });

  test('body has object: "list"', () => { expect(body.object).toBe('list'); });
  test('body has data array', () => { expect(Array.isArray(body.data)).toBe(true); });
});

// ── /dashboard/stream ────────────────────────────────────────────────────────

test.describe('GET /dashboard/stream', () => {
  // SSE streams indefinitely; use a browser fetch so we can read headers then abort the body.
  test('returns 200 with text/event-stream and no-cache', async ({ page }) => {
    await page.goto('/');
    const headers = await page.evaluate(async () => {
      const controller = new AbortController();
      const res = await fetch('/dashboard/stream', { signal: controller.signal });
      const result = {
        status: res.status,
        contentType: res.headers.get('content-type'),
        cacheControl: res.headers.get('cache-control'),
      };
      controller.abort();
      return result;
    });
    expect(headers.status).toBe(200);
    expect(headers.contentType).toMatch(/text\/event-stream/);
    expect(headers.cacheControl).toMatch(/no-cache/);
  });
});

// ── GET / (root HTML) ─────────────────────────────────────────────────────────

test.describe('GET / (root)', () => {
  let res, text;
  test.beforeAll(async ({ request }) => {
    res = await request.get('/');
    text = await res.text();
  });

  test('returns 200 with text/html', () => {
    expect(res.status()).toBe(200);
    expect(res.headers()['content-type']).toMatch(/text\/html/);
  });

  test('body contains "Kolosal"', () => { expect(text).toContain('Kolosal'); });
});

// ── 404 for unknown routes ────────────────────────────────────────────────────

test.describe('Unknown routes', () => {
  test('GET /nonexistent returns 404', async ({ request }) => {
    expect((await request.get('/nonexistent-route-xyz')).status()).toBe(404);
  });

  test('GET /api/nonexistent returns 404', async ({ request }) => {
    expect((await request.get('/api/nonexistent-endpoint-xyz')).status()).toBe(404);
  });
});

// ── Wrong HTTP method ─────────────────────────────────────────────────────────

test.describe('Wrong HTTP method', () => {
  test('POST /health returns 404 or 405', async ({ request }) => {
    expect([404, 405]).toContain((await request.post('/health', { data: {} })).status());
  });

  test('DELETE /health returns 404 or 405', async ({ request }) => {
    expect([404, 405]).toContain((await request.delete('/health')).status());
  });

  test('GET /predict returns 404 or 405', async ({ request }) => {
    expect([404, 405]).toContain((await request.get('/predict')).status());
  });
});

// ── POST /predict — error cases ───────────────────────────────────────────────

test.describe('POST /predict', () => {
  test('empty body returns 400 or 422', async ({ request }) => {
    const status = (await request.post('/predict', {
      headers: { 'Content-Type': 'application/json' },
      data: {},
    })).status();
    expect([400, 422]).toContain(status);
  });

  test('valid body returns JSON with success field', async ({ request }) => {
    const res = await request.post('/predict', {
      data: { model_name: 'default', inputs: { text: 'hello' }, priority: 0 },
    });
    // Handler always returns JSON (success:false if no model loaded)
    const body = await res.json();
    expect(body).toHaveProperty('success');
  });
});

// ── POST /synthesize ─────────────────────────────────────────────────────────

test.describe('POST /synthesize', () => {
  test('empty body returns 400 or 422', async ({ request }) => {
    const status = (await request.post('/synthesize', {
      headers: { 'Content-Type': 'application/json' },
      data: {},
    })).status();
    expect([400, 422]).toContain(status);
  });
});

// ── POST /v1/completions ─────────────────────────────────────────────────────

test.describe('POST /v1/completions', () => {
  test('valid request returns 200 or 500', async ({ request }) => {
    const status = (await request.post('/v1/completions', {
      data: { model: 'default', prompt: 'Hello', max_tokens: 16 },
    })).status();
    expect([200, 500]).toContain(status);
  });

  test('missing prompt returns 400, 422, or 500', async ({ request }) => {
    expect([400, 422, 500]).toContain(
      (await request.post('/v1/completions', { data: { model: 'default' } })).status()
    );
  });
});

// ── POST /v1/chat/completions ─────────────────────────────────────────────────

test.describe('POST /v1/chat/completions', () => {
  test('valid request returns 200 or 500', async ({ request }) => {
    const status = (await request.post('/v1/chat/completions', {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hello' }],
        max_tokens: 16,
      },
    })).status();
    expect([200, 500]).toContain(status);
  });

  test('missing messages returns 400, 422, or 500', async ({ request }) => {
    expect([400, 422, 500]).toContain(
      (await request.post('/v1/chat/completions', { data: { model: 'default' } })).status()
    );
  });

  test('non-JSON body returns 400 or 415', async ({ request }) => {
    expect([400, 415]).toContain(
      (await request.post('/v1/chat/completions', {
        headers: { 'Content-Type': 'application/json' },
        data: 'not-json',
      })).status()
    );
  });
});

// ── POST /classify/batch ──────────────────────────────────────────────────────

test.describe('POST /classify/batch', () => {
  test('empty body returns 400, 415, or 422', async ({ request }) => {
    expect([400, 415, 422]).toContain(
      (await request.post('/classify/batch', {
        headers: { 'Content-Type': 'application/json' },
        data: {},
      })).status()
    );
  });
});

// ── POST /models/download ─────────────────────────────────────────────────────

test.describe('POST /models/download', () => {
  test('missing body fields returns 400 or 422', async ({ request }) => {
    expect([400, 422]).toContain(
      (await request.post('/models/download', { data: {} })).status()
    );
  });
});

// ── Correlation ID middleware ─────────────────────────────────────────────────

test.describe('Correlation ID middleware', () => {
  test('custom x-correlation-id is echoed back', async ({ request }) => {
    const customId = 'test-correlation-id-12345';
    const res = await request.get('/health/live', {
      headers: { 'x-correlation-id': customId },
    });
    expect(res.headers()['x-correlation-id']).toBe(customId);
  });
});
