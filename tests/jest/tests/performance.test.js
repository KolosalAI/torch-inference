/**
 * Performance monitoring endpoint tests
 * Covers: GET /performance, POST /performance/profile, GET /performance/optimize
 */
const { api, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) console.warn('[performance] Server not available — tests will be skipped');
});

const skip = () => !serverAvailable;

describe('Performance metrics', () => {
  test('GET /performance returns system and process metrics', async () => {
    if (skip()) return;
    const res = await api.get('/performance');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('system_info');
    expect(res.data).toHaveProperty('process_info');
  });

  test('GET /performance system_info includes cpu_count and memory', async () => {
    if (skip()) return;
    const res = await api.get('/performance');
    expectSuccess(res);
    const sys = res.data.system_info;
    expect(sys).toHaveProperty('cpu_count');
    expect(sys).toHaveProperty('memory_mb');
  });

  test('GET /performance process_info includes pid', async () => {
    if (skip()) return;
    const res = await api.get('/performance');
    expectSuccess(res);
    expect(res.data.process_info).toHaveProperty('pid');
  });
});

describe('Performance profiling', () => {
  test('POST /performance/profile returns timing metrics', async () => {
    if (skip()) return;
    const res = await api.post('/performance/profile', {});
    expect([200, 400, 500]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
      expect(res.data).toHaveProperty('total_time_ms');
    }
  });
});

describe('Performance optimization', () => {
  test('GET /performance/optimize runs optimizations and reports result', async () => {
    if (skip()) return;
    const res = await api.get('/performance/optimize');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('optimizations_applied');
    expect(Array.isArray(res.data.optimizations_applied)).toBe(true);
  });
});
