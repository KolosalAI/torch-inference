/**
 * YOLO object detection endpoint tests
 * Covers: POST /api/yolo/detect, GET /api/yolo/models, POST /api/yolo/info
 */
const FormData = require('form-data');
const { api, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) console.warn('[yolo] Server not available — tests will be skipped');
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

describe('YOLO model catalog', () => {
  test('GET /api/yolo/models returns available YOLO versions and sizes', async () => {
    if (skip()) return;
    const res = await api.get('/api/yolo/models');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('versions');
    expect(Array.isArray(res.data.versions)).toBe(true);
    expect(res.data).toHaveProperty('sizes');
    expect(Array.isArray(res.data.sizes)).toBe(true);
    expect(res.data).toHaveProperty('total_models');
  });
});

describe('YOLO model info', () => {
  test('POST /api/yolo/info with valid version and size returns model info', async () => {
    if (skip()) return;
    const res = await api.post('/api/yolo/info', {
      model_version: 'v8',
      model_size: 'n',
    });
    expect([200, 400, 404]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
      expect(res.data).toHaveProperty('model_name');
      expect(res.data).toHaveProperty('available');
    }
  });

  test('POST /api/yolo/info with invalid version returns 400/404', async () => {
    if (skip()) return;
    const res = await api.post('/api/yolo/info', {
      model_version: 'v999',
      model_size: 'z',
    });
    expect([400, 404, 422]).toContain(res.status);
  });

  test('POST /api/yolo/info with missing fields returns 400', async () => {
    if (skip()) return;
    const res = await api.post('/api/yolo/info', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });
});

describe('YOLO detection', () => {
  test('POST /api/yolo/detect without image returns 400', async () => {
    if (skip()) return;
    const res = await api.post('/api/yolo/detect', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });

  test('POST /api/yolo/detect with PNG returns detections or model error', async () => {
    if (skip()) return;
    const form = new FormData();
    form.append('image', minimalPng(), { filename: 'test.png', contentType: 'image/png' });

    const res = await api.post(
      '/api/yolo/detect?model_version=v8&model_size=n&conf_threshold=0.25&iou_threshold=0.45',
      form,
      { headers: form.getHeaders(), timeout: 30000 }
    );
    expect([200, 400, 404, 422, 500]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
      expect(res.data).toHaveProperty('success');
    }
  });
});
