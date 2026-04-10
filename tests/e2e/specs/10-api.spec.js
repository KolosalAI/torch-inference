// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('API Endpoints (direct fetch)', () => {
  test('GET /health returns healthy JSON', async ({ request }) => {
    const res = await request.get('/health');
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(body.status).toBe('healthy');
    expect(typeof body.uptime_seconds).toBe('number');
    expect(body.checks).toBeDefined();
  });

  test('GET /health/live returns 200', async ({ request }) => {
    const res = await request.get('/health/live');
    expect(res.ok()).toBe(true);
  });

  test('GET /health/ready returns 200', async ({ request }) => {
    const res = await request.get('/health/ready');
    expect(res.ok()).toBe(true);
  });

  test('GET /system/info returns system data', async ({ request }) => {
    const res = await request.get('/system/info');
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(body.system).toBeDefined();
    expect(body.system.os).toBeDefined();
    expect(body.system.cpu_count).toBeGreaterThan(0);
  });

  test('GET /performance returns metrics', async ({ request }) => {
    const res = await request.get('/performance');
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(body.system_info || body.process_info || body.runtime_info).toBeDefined();
  });

  test('GET /stats returns request statistics', async ({ request }) => {
    const res = await request.get('/stats');
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(typeof body.total_requests).toBe('number');
    expect(typeof body.uptime_seconds).toBe('number');
  });

  test('GET /tts/engines returns engine list', async ({ request }) => {
    const res = await request.get('/tts/engines');
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(Array.isArray(body.engines)).toBe(true);
    expect(body.engines.length).toBeGreaterThan(0);
    expect(body.engines[0].id).toBeDefined();
  });

  test('GET /tts/health returns engine count', async ({ request }) => {
    const res = await request.get('/tts/health');
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(body.engines_loaded).toBeGreaterThan(0);
    expect(body.status).toBe('healthy');
  });

  test('GET /tts/engines/:id/voices returns voices', async ({ request }) => {
    const res = await request.get('/tts/engines/kokoro/voices');
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(Array.isArray(body.voices)).toBe(true);
    expect(body.voices.length).toBeGreaterThan(0);
  });

  test('POST /tts/stream returns PCM audio bytes', async ({ request }) => {
    const res = await request.post('/tts/stream', {
      data: { text: 'Hello.', speed: 1.0 },
    });
    expect(res.ok()).toBe(true);
    const buf = await res.body();
    // PCM data should be non-empty and even-byte-aligned (16-bit)
    expect(buf.length).toBeGreaterThan(0);
    expect(buf.length % 2).toBe(0);
  });

  test('POST /tts/stream with explicit engine and voice', async ({ request }) => {
    const res = await request.post('/tts/stream', {
      data: { text: 'Hi.', engine: 'kokoro', voice: 'af', speed: 1.0 },
    });
    expect(res.ok()).toBe(true);
    const buf = await res.body();
    expect(buf.length).toBeGreaterThan(0);
  });

  test('POST /tts/stream with empty text returns 400', async ({ request }) => {
    const res = await request.post('/tts/stream', {
      data: { text: '', speed: 1.0 },
    });
    expect(res.status()).toBe(400);
  });

  test('POST /classify/batch returns predictions', async ({ request }) => {
    const fs = require('fs');
    const path = require('path');
    const imgPath = path.resolve(__dirname, '../fixtures/test.jpg');
    const b64 = fs.readFileSync(imgPath).toString('base64');

    const res = await request.post('/classify/batch', {
      data: { images: [b64], top_k: 5 },
    });
    expect(res.ok()).toBe(true);
    const body = await res.json();
    const results = body?.data?.results ?? body?.results ?? [];
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0][0].label).toBeDefined();
    expect(results[0][0].confidence).toBeGreaterThan(0);
  });

  test('GET /audio/health returns audio backend info', async ({ request }) => {
    const res = await request.get('/audio/health');
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(body.status).toBe('ok');
    expect(body.audio_backend).toBeDefined();
  });

  test('GET /models/available returns model catalogue', async ({ request }) => {
    const res = await request.get('/models/available');
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(body.models).toBeDefined();
    expect(body.categories).toBeDefined();
  });

  test('GET /models/managed returns managed model list', async ({ request }) => {
    const res = await request.get('/models/managed');
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(Array.isArray(body.models)).toBe(true);
  });

  test('GET /dashboard/stream sends SSE data', async ({ page }) => {
    const sseMessages = [];
    page.on('response', async (resp) => {
      if (resp.url().includes('/dashboard/stream')) {
        sseMessages.push(resp.status());
      }
    });
    await page.goto('/');
    // Click Dashboard to trigger SSE connection
    await page.locator('.nav-item', { hasText: 'Dashboard' }).first().click();
    await page.waitForTimeout(4_000);
    // Tile should have been populated by SSE
    const uptime = await page.locator('#d-uptime').textContent();
    expect(uptime).not.toBe('—');
  });
});
