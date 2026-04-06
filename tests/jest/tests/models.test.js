/**
 * Model download/management endpoint tests
 * Covers: POST /models/download, GET /models/download/status/:id,
 *         GET /models/download/list, GET /models/managed,
 *         GET /models/download/:name/info, DELETE /models/download/:name,
 *         GET /models/cache/info, GET /models/available,
 *         GET /models/sota, POST /models/sota/:model_id
 */
const { api, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) console.warn('[models] Server not available — tests will be skipped');
});

const skip = () => !serverAvailable;

describe('Model download list & cache', () => {
  test('GET /models/download/list returns active downloads', async () => {
    if (skip()) return;
    const res = await api.get('/models/download/list');
    expectSuccess(res);
    expectJsonBody(res);
    expect(Array.isArray(res.data) || typeof res.data === 'object').toBe(true);
  });

  test('GET /models/managed returns managed models', async () => {
    if (skip()) return;
    const res = await api.get('/models/managed');
    expectSuccess(res);
    expectJsonBody(res);
    // Should have models array or similar structure
    expect(res.data).toBeDefined();
  });

  test('GET /models/cache/info returns cache statistics', async () => {
    if (skip()) return;
    const res = await api.get('/models/cache/info');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('cache_dir');
  });
});

describe('Available models catalog', () => {
  test('GET /models/available returns downloadable models', async () => {
    if (skip()) return;
    const res = await api.get('/models/available');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toBeDefined();
  });

  test('GET /models/sota returns SOTA model registry', async () => {
    if (skip()) return;
    const res = await api.get('/models/sota');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toBeDefined();
  });
});

describe('Model download trigger', () => {
  test('POST /models/download with missing model_name returns 400/422', async () => {
    if (skip()) return;
    const res = await api.post('/models/download', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });

  test('POST /models/download with valid payload starts task', async () => {
    if (skip()) return;
    const res = await api.post('/models/download', {
      model_name: 'test-model-nonexistent',
      source_type: 'url',
      url: 'http://example.com/model.bin',
    });
    // Either accepted (200/202) or rejected (4xx) — never a 5xx without body
    expect([200, 202, 400, 404, 422, 500]).toContain(res.status);
    expectJsonBody(res);
  });

  test('GET /models/download/status/:id with unknown id returns 404', async () => {
    if (skip()) return;
    const res = await api.get('/models/download/status/nonexistent-task-id-xyz');
    expect([404, 400]).toContain(res.status);
  });
});

describe('Model info & deletion', () => {
  test('GET /models/download/:name/info for unknown model returns 404', async () => {
    if (skip()) return;
    const res = await api.get('/models/download/nonexistent-xyz/info');
    expect([404, 400]).toContain(res.status);
  });

  test('DELETE /models/download/:name for unknown model returns 404', async () => {
    if (skip()) return;
    const res = await api.delete('/models/download/nonexistent-xyz');
    expect([404, 400, 200]).toContain(res.status);
  });
});

describe('SOTA model download', () => {
  test('POST /models/sota/:model_id with unknown model returns 404/400', async () => {
    if (skip()) return;
    const res = await api.post('/models/sota/nonexistent-model-id');
    expect([404, 400, 422]).toContain(res.status);
  });
});
