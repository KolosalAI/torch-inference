/**
 * System information endpoint tests
 * Covers: GET /system/info, GET /system/config, GET /system/gpu/stats
 */
const { api, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) console.warn('[system] Server not available — tests will be skipped');
});

const skip = () => !serverAvailable;

describe('System info', () => {
  test('GET /system/info returns OS, CPU, memory info', async () => {
    if (skip()) return;
    const res = await api.get('/system/info');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('system');
    const sys = res.data.system;
    expect(sys).toHaveProperty('os');
    expect(sys).toHaveProperty('arch');
  });

  test('GET /system/info includes GPU availability', async () => {
    if (skip()) return;
    const res = await api.get('/system/info');
    expectSuccess(res);
    expect(res.data).toHaveProperty('gpu');
    expect(res.data.gpu).toHaveProperty('available');
  });

  test('GET /system/info includes runtime info', async () => {
    if (skip()) return;
    const res = await api.get('/system/info');
    expectSuccess(res);
    expect(res.data).toHaveProperty('runtime');
  });

  test('GET /system/info includes feature flags', async () => {
    if (skip()) return;
    const res = await api.get('/system/info');
    expectSuccess(res);
    expect(res.data).toHaveProperty('features');
  });
});

describe('System config', () => {
  test('GET /system/config returns server configuration', async () => {
    if (skip()) return;
    const res = await api.get('/system/config');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('server');
    expect(res.data.server).toHaveProperty('port');
  });

  test('GET /system/config includes inference settings', async () => {
    if (skip()) return;
    const res = await api.get('/system/config');
    expectSuccess(res);
    expect(res.data).toHaveProperty('inference');
  });
});

describe('GPU stats', () => {
  test('GET /system/gpu/stats returns GPU info (even if no GPU)', async () => {
    if (skip()) return;
    const res = await api.get('/system/gpu/stats');
    // 200 with stats or 404/503 if GPU not available
    expect([200, 404, 503]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
    }
  });
});
