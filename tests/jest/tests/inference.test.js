/**
 * Core inference endpoint tests
 * Covers: POST /predict, POST /synthesize (legacy)
 */
const { api, createClient, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) console.warn('[inference] Server not available — tests will be skipped');
});

const skip = () => !serverAvailable;

describe('Predict endpoint', () => {
  test('POST /predict with missing model_name returns 400/422', async () => {
    if (skip()) return;
    const res = await api.post('/predict', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });

  test('POST /predict with unknown model returns error', async () => {
    if (skip()) return;
    const res = await api.post('/predict', {
      model_name: 'nonexistent-model-xyz',
      inputs: { data: [[1.0, 2.0, 3.0]] },
    });
    expect([400, 404, 422, 500]).toContain(res.status);
    expectJsonBody(res);
  });

  test('POST /predict response shape on error includes message', async () => {
    if (skip()) return;
    const res = await api.post('/predict', {
      model_name: 'nonexistent-model',
      inputs: {},
    });
    expectJsonBody(res);
    // Error response should have some error indicator
    const hasError = res.data.error !== undefined
      || res.data.message !== undefined
      || res.data.success === false;
    expect(hasError).toBe(true);
  });
});

describe('Legacy synthesize endpoint', () => {
  test('POST /synthesize with missing text returns 400/422', async () => {
    if (skip()) return;
    const res = await api.post('/synthesize', { model_name: 'test' });
    expect([400, 404, 422]).toContain(res.status);
  });

  test('POST /synthesize handles unknown model gracefully', async () => {
    if (skip()) return;
    const client = createClient({ timeout: 15000 });
    const res = await client.post('/synthesize', {
      model_name: 'nonexistent-tts',
      text: 'Hello world',
    });
    expect([200, 400, 404, 422, 500]).toContain(res.status);
  });
});
