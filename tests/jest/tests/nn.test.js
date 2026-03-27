/**
 * Neural network inference endpoint tests
 * Covers: POST /api/nn/load, POST /api/nn/infer,
 *         GET /api/nn/list, DELETE /api/nn/:model_id
 */
const { api, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) console.warn('[nn] Server not available — tests will be skipped');
});

const skip = () => !serverAvailable;

describe('NN model listing', () => {
  test('GET /api/nn/list returns loaded models', async () => {
    if (skip()) return;
    const res = await api.get('/api/nn/list');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('models');
    expect(Array.isArray(res.data.models)).toBe(true);
    expect(res.data).toHaveProperty('total');
  });
});

describe('NN model loading', () => {
  test('POST /api/nn/load with missing fields returns 400/422', async () => {
    if (skip()) return;
    const res = await api.post('/api/nn/load', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });

  test('POST /api/nn/load with nonexistent model path returns error', async () => {
    if (skip()) return;
    const res = await api.post('/api/nn/load', {
      model_id: 'test-nn-model',
      model_path: '/nonexistent/model.pth',
      device: 'cpu',
    });
    expect([400, 404, 422, 500]).toContain(res.status);
    expectJsonBody(res);
    const hasError = res.data.success === false || res.data.error !== undefined || res.data.message !== undefined;
    expect(hasError).toBe(true);
  });
});

describe('NN model inference', () => {
  test('POST /api/nn/infer with missing model_id returns 400/422', async () => {
    if (skip()) return;
    const res = await api.post('/api/nn/infer', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });

  test('POST /api/nn/infer with unloaded model returns 404/400', async () => {
    if (skip()) return;
    const res = await api.post('/api/nn/infer', {
      model_id: 'nonexistent-nn-model-xyz',
      input_data: [1.0, 2.0, 3.0],
      input_shape: [1, 3],
    });
    expect([400, 404, 422, 500]).toContain(res.status);
    expectJsonBody(res);
  });
});

describe('NN model unloading', () => {
  test('DELETE /api/nn/:model_id with unknown model returns 404/400', async () => {
    if (skip()) return;
    const res = await api.delete('/api/nn/nonexistent-model-xyz');
    expect([400, 404]).toContain(res.status);
  });
});
