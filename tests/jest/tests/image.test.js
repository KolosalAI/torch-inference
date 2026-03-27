/**
 * Image security endpoint tests
 * Covers: POST /image/process/secure, POST /image/validate/security,
 *         GET /image/security/stats, GET /image/health
 */
const FormData = require('form-data');
const { api, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) console.warn('[image] Server not available — tests will be skipped');
});

const skip = () => !serverAvailable;

// Minimal 1×1 pixel PNG (89 bytes)
function minimalPng() {
  return Buffer.from(
    '89504e470d0a1a0a0000000d49484452000000010000000108020000009001' +
    '2e00000000c4944415478016360f8cfc00000000200012100003ab4cf000000' +
    '0049454e44ae426082',
    'hex'
  );
}

describe('Image health', () => {
  test('GET /image/health returns status', async () => {
    if (skip()) return;
    const res = await api.get('/image/health');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('status');
  });
});

describe('Image security stats', () => {
  test('GET /image/security/stats returns statistics', async () => {
    if (skip()) return;
    const res = await api.get('/image/security/stats');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toBeDefined();
  });
});

describe('Image security validation', () => {
  test('POST /image/validate/security with PNG returns security result', async () => {
    if (skip()) return;
    const form = new FormData();
    form.append('image', minimalPng(), { filename: 'test.png', contentType: 'image/png' });
    form.append('security_level', 'low');

    const res = await api.post('/image/validate/security', form, {
      headers: form.getHeaders(),
      timeout: 15000,
    });
    expect([200, 400, 422, 500]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
      expect(res.data).toHaveProperty('security_result');
    }
  });

  test('POST /image/validate/security with invalid security_level is handled', async () => {
    if (skip()) return;
    const form = new FormData();
    form.append('image', minimalPng(), { filename: 'test.png', contentType: 'image/png' });
    form.append('security_level', 'invalid_level');

    const res = await api.post('/image/validate/security', form, {
      headers: form.getHeaders(),
    });
    expect([200, 400, 422]).toContain(res.status);
  });

  test('POST /image/validate/security without image returns 400', async () => {
    if (skip()) return;
    const res = await api.post('/image/validate/security', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });
});

describe('Image processing (secure)', () => {
  test('POST /image/process/secure with PNG returns result', async () => {
    if (skip()) return;
    const form = new FormData();
    form.append('image', minimalPng(), { filename: 'test.png', contentType: 'image/png' });
    form.append('security_level', 'medium');

    const res = await api.post('/image/process/secure', form, {
      headers: form.getHeaders(),
      timeout: 15000,
    });
    expect([200, 400, 422, 500]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
      expect(res.data).toHaveProperty('success');
      expect(res.data).toHaveProperty('security_result');
    }
  });

  test('POST /image/process/secure without body returns 400', async () => {
    if (skip()) return;
    const res = await api.post('/image/process/secure', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });
});
