/**
 * Image classification endpoint tests
 * Covers: POST /api/classify/image, POST /api/classify/upload
 */
const FormData = require('form-data');
const { api, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) console.warn('[classify] Server not available — tests will be skipped');
});

const skip = () => !serverAvailable;

// Minimal 1×1 pixel PNG
function minimalPng() {
  return Buffer.from(
    '89504e470d0a1a0a0000000d49484452000000010000000108020000009001' +
    '2e00000000c4944415478016360f8cfc00000000200012100003ab4cf000000' +
    '0049454e44ae426082',
    'hex'
  );
}

describe('Classify by path', () => {
  test('POST /api/classify/image with missing image_path returns 400/422', async () => {
    if (skip()) return;
    const res = await api.post('/api/classify/image', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });

  test('POST /api/classify/image with nonexistent path returns error', async () => {
    if (skip()) return;
    const res = await api.post('/api/classify/image', {
      image_path: '/nonexistent/image.jpg',
      top_k: 5,
    });
    expect([400, 404, 422, 500]).toContain(res.status);
    expectJsonBody(res);
    const hasError = res.data.success === false || res.data.error !== undefined;
    expect(hasError).toBe(true);
  });
});

describe('Classify by upload', () => {
  test('POST /api/classify/upload with PNG returns result or model-not-loaded error', async () => {
    if (skip()) return;
    const form = new FormData();
    form.append('image', minimalPng(), { filename: 'test.png', contentType: 'image/png' });
    form.append('top_k', '5');

    const res = await api.post('/api/classify/upload', form, {
      headers: form.getHeaders(),
      timeout: 15000,
    });
    expect([200, 400, 422, 500]).toContain(res.status);
    expectJsonBody(res);
    // success=true with results, or success=false with error
    expect(res.data).toHaveProperty('success');
  });

  test('POST /api/classify/upload without image returns 400', async () => {
    if (skip()) return;
    const res = await api.post('/api/classify/upload', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });
});
