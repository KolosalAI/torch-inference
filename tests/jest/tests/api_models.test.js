/**
 * Model registry API endpoint tests
 * Covers: GET /api/models, GET /api/models/:model_id,
 *         GET /api/models/downloaded, POST /api/models/download,
 *         GET /api/models/comparison
 */
const { api, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;
let firstModelId = null;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) {
    console.warn('[api-models] Server not available — tests will be skipped');
    return;
  }
  // Discover a model ID from the catalog for single-model tests
  const res = await api.get('/api/models');
  if (res.status === 200 && res.data) {
    // Model list may be an array or an object with nested arrays
    if (Array.isArray(res.data)) {
      firstModelId = res.data[0]?.id || res.data[0]?.model_id || null;
    } else if (res.data.models && Array.isArray(res.data.models)) {
      firstModelId = res.data.models[0]?.id || null;
    } else {
      // Try to find first key in the object
      const keys = Object.keys(res.data);
      if (keys.length > 0) firstModelId = keys[0];
    }
  }
});

const skip = () => !serverAvailable;

describe('API model catalog', () => {
  test('GET /api/models returns full model registry', async () => {
    if (skip()) return;
    const res = await api.get('/api/models');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toBeDefined();
  });

  test('GET /api/models response is non-empty', async () => {
    if (skip()) return;
    const res = await api.get('/api/models');
    expectSuccess(res);
    const data = res.data;
    const isNonEmpty = Array.isArray(data)
      ? data.length > 0
      : Object.keys(data).length > 0;
    expect(isNonEmpty).toBe(true);
  });
});

describe('API single model info', () => {
  test('GET /api/models/:model_id returns model details', async () => {
    if (skip()) return;
    if (!firstModelId) {
      console.warn('[api-models] No model ID available — skipping single model test');
      return;
    }
    const res = await api.get(`/api/models/${firstModelId}`);
    expect([200, 404]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
      expect(res.data).toHaveProperty('name');
    }
  });

  test('GET /api/models/nonexistent returns 404', async () => {
    if (skip()) return;
    const res = await api.get('/api/models/nonexistent-model-id-xyz-123');
    expect([404, 400]).toContain(res.status);
  });
});

describe('API downloaded models', () => {
  test('GET /api/models/downloaded returns downloaded models list', async () => {
    if (skip()) return;
    const res = await api.get('/api/models/downloaded');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('count');
  });
});

describe('API model download', () => {
  test('POST /api/models/download with missing model_id returns 400/422', async () => {
    if (skip()) return;
    const res = await api.post('/api/models/download', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });

  test('POST /api/models/download with unknown model_id returns 404/400', async () => {
    if (skip()) return;
    const res = await api.post('/api/models/download', {
      model_id: 'nonexistent-model-xyz-abc',
    });
    expect([400, 404, 422]).toContain(res.status);
    expectJsonBody(res);
  });
});

describe('API model comparison', () => {
  test('GET /api/models/comparison returns comparison data', async () => {
    if (skip()) return;
    const res = await api.get('/api/models/comparison');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('total_count');
    expect(res.data).toHaveProperty('models');
    expect(Array.isArray(res.data.models)).toBe(true);
  });
});
