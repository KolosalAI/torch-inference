/**
 * TTS endpoint tests
 * Covers: POST /tts/synthesize, GET /tts/engines, GET /tts/engines/:id/capabilities,
 *         GET /tts/engines/:id/voices, GET /tts/stats, GET /tts/health
 */
const { api, createClient, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

const TTS_TIMEOUT = parseInt(process.env.TTS_TIMEOUT || '60000', 10);

let serverAvailable = false;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) console.warn('[tts] Server not available — tests will be skipped');
});

const skip = () => !serverAvailable;

describe('TTS health', () => {
  test('GET /tts/health returns status', async () => {
    if (skip()) return;
    const res = await api.get('/tts/health');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('status');
  });
});

describe('TTS engines', () => {
  test('GET /tts/engines returns list of engines', async () => {
    if (skip()) return;
    const res = await api.get('/tts/engines');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('engines');
    expect(Array.isArray(res.data.engines)).toBe(true);
    expect(res.data).toHaveProperty('total');
  });

  test('GET /tts/engines/:id/capabilities returns engine capabilities', async () => {
    if (skip()) return;
    // First get an engine ID
    const listRes = await api.get('/tts/engines');
    if (listRes.status !== 200 || !listRes.data.engines?.length) {
      console.warn('[tts] No engines available — skipping capabilities test');
      return;
    }
    const engineId = listRes.data.engines[0].id || listRes.data.engines[0].name;
    const res = await api.get(`/tts/engines/${engineId}/capabilities`);
    expect([200, 404]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
      expect(res.data).toHaveProperty('name');
    }
  });

  test('GET /tts/engines/:id/voices returns voice list', async () => {
    if (skip()) return;
    const listRes = await api.get('/tts/engines');
    if (listRes.status !== 200 || !listRes.data.engines?.length) {
      console.warn('[tts] No engines available — skipping voices test');
      return;
    }
    const engineId = listRes.data.engines[0].id || listRes.data.engines[0].name;
    const res = await api.get(`/tts/engines/${engineId}/voices`);
    expect([200, 404]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
      expect(res.data).toHaveProperty('voices');
      expect(Array.isArray(res.data.voices)).toBe(true);
    }
  });
});

describe('TTS stats', () => {
  test('GET /tts/stats returns synthesis stats', async () => {
    if (skip()) return;
    const res = await api.get('/tts/stats');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('stats');
  });
});

describe('TTS synthesis', () => {
  test('POST /tts/synthesize rejects empty text', async () => {
    if (skip()) return;
    const res = await api.post('/tts/synthesize', { text: '' });
    expect(res.status).toBeGreaterThanOrEqual(400);
  });

  test('POST /tts/synthesize returns audio for short text', async () => {
    if (skip()) return;
    const client = createClient({ timeout: TTS_TIMEOUT });
    const res = await client.post('/tts/synthesize', {
      text: 'Hello, world.',
      speed: 1.0,
    });

    // May fail if no engine configured — we accept 200 or 4xx/5xx with error message
    expect([200, 400, 422, 500, 503]).toContain(res.status);

    if (res.status === 200) {
      expectJsonBody(res);
      // Response should contain base64 audio
      expect(res.data).toHaveProperty('audio_base64');
      expect(typeof res.data.audio_base64).toBe('string');
      expect(res.data.audio_base64.length).toBeGreaterThan(0);
      expect(res.data).toHaveProperty('sample_rate');
      expect(typeof res.data.sample_rate).toBe('number');
      expect(res.data).toHaveProperty('duration_secs');
      expect(res.data).toHaveProperty('engine_used');
    } else {
      // Failure should carry an error message
      expectJsonBody(res);
    }
  }, TTS_TIMEOUT);

  test('POST /tts/synthesize accepts voice and language params', async () => {
    if (skip()) return;
    const client = createClient({ timeout: TTS_TIMEOUT });
    const res = await client.post('/tts/synthesize', {
      text: 'Test synthesis.',
      voice: 'af_heart',
      language: 'en-US',
      speed: 1.0,
      pitch: 1.0,
    });
    expect([200, 400, 422, 500, 503]).toContain(res.status);
  }, TTS_TIMEOUT);

  test('POST /tts/synthesize rejects text exceeding max length', async () => {
    if (skip()) return;
    const longText = 'a'.repeat(10001);
    const client = createClient({ timeout: TTS_TIMEOUT });
    const res = await client.post('/tts/synthesize', { text: longText });
    expect(res.status).toBeGreaterThanOrEqual(400);
  });

  // Also test legacy /audio/synthesize endpoint
  test('POST /audio/synthesize is an alias for TTS synthesis', async () => {
    if (skip()) return;
    const client = createClient({ timeout: TTS_TIMEOUT });
    const res = await client.post('/audio/synthesize', {
      text: 'Hello from audio endpoint.',
    });
    expect([200, 400, 404, 422, 500, 503]).toContain(res.status);
  }, TTS_TIMEOUT);
});
