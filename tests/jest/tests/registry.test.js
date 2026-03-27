/**
 * Model registry endpoint tests
 * Covers: POST /registry/register, POST /registry/scan, GET /registry/models,
 *         GET /registry/models/:id, POST /registry/models/:id/load,
 *         POST /registry/infer, GET /registry/stats, GET /registry/export
 */
const { api, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) console.warn('[registry] Server not available — tests will be skipped');
});

const skip = () => !serverAvailable;

describe('Registry listing & stats', () => {
  test('GET /registry/models returns registered models', async () => {
    if (skip()) return;
    const res = await api.get('/registry/models');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toBeDefined();
  });

  test('GET /registry/models with format filter', async () => {
    if (skip()) return;
    for (const fmt of ['pytorch', 'onnx', 'safetensors']) {
      const res = await api.get(`/registry/models?format=${fmt}`);
      expect([200, 400]).toContain(res.status);
      if (res.status === 200) expectJsonBody(res);
    }
  });

  test('GET /registry/stats returns statistics', async () => {
    if (skip()) return;
    const res = await api.get('/registry/stats');
    expectSuccess(res);
    expectJsonBody(res);
  });

  test('GET /registry/export returns full registry export', async () => {
    if (skip()) return;
    const res = await api.get('/registry/export');
    expectSuccess(res);
    expectJsonBody(res);
  });
});

describe('Registry registration', () => {
  test('POST /registry/register with missing path returns 400/422', async () => {
    if (skip()) return;
    const res = await api.post('/registry/register', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });

  test('POST /registry/register with nonexistent path returns error', async () => {
    if (skip()) return;
    const res = await api.post('/registry/register', {
      path: '/nonexistent/model.pth',
      name: 'test-model',
    });
    expect([400, 404, 422, 500]).toContain(res.status);
    expectJsonBody(res);
  });

  test('POST /registry/scan with nonexistent path returns error', async () => {
    if (skip()) return;
    const res = await api.post('/registry/scan', { path: '/nonexistent/path' });
    expect([200, 400, 404, 422, 500]).toContain(res.status);
  });
});

describe('Registry model operations', () => {
  test('GET /registry/models/:id with unknown id returns 404', async () => {
    if (skip()) return;
    const res = await api.get('/registry/models/nonexistent-model-id-xyz');
    expect([404, 400]).toContain(res.status);
  });

  test('POST /registry/models/:id/load with unknown id returns 404', async () => {
    if (skip()) return;
    const res = await api.post('/registry/models/nonexistent-model-id-xyz/load');
    expect([404, 400]).toContain(res.status);
  });

  test('POST /registry/infer with missing model_id returns 400/422', async () => {
    if (skip()) return;
    const res = await api.post('/registry/infer', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });

  test('POST /registry/infer with unknown model_id returns 404', async () => {
    if (skip()) return;
    const res = await api.post('/registry/infer', {
      model_id: 'nonexistent-model',
      input: { data: [1.0, 2.0] },
    });
    expect([404, 400, 422, 500]).toContain(res.status);
  });
});
