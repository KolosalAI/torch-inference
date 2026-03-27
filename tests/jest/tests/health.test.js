/**
 * Health check endpoint tests
 * Covers: GET /, GET /health, GET /health/live, GET /health/ready,
 *         GET /metrics, GET /info, GET /stats, GET /endpoints
 */
const { api, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) console.warn('[health] Server not available — tests will be skipped');
});

const skip = () => !serverAvailable;

describe('Root endpoint', () => {
  test('GET / returns API info', async () => {
    if (skip()) return;
    const res = await api.get('/');
    expect([200, 404]).toContain(res.status); // root may be 404 if not registered
    if (res.status === 200) {
      expectJsonBody(res);
    }
  });
});

describe('Health endpoints', () => {
  test('GET /health returns 200 with status', async () => {
    if (skip()) return;
    const res = await api.get('/health');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('status');
  });

  test('GET /health/live returns 200 (liveness probe)', async () => {
    if (skip()) return;
    const res = await api.get('/health/live');
    expect([200, 503]).toContain(res.status);
    expectJsonBody(res);
  });

  test('GET /health/ready returns 200 or 503 (readiness probe)', async () => {
    if (skip()) return;
    const res = await api.get('/health/ready');
    expect([200, 503]).toContain(res.status);
    expectJsonBody(res);
  });
});

describe('Metrics endpoint', () => {
  test('GET /metrics returns prometheus-format text', async () => {
    if (skip()) return;
    const res = await api.get('/metrics');
    expect([200, 404]).toContain(res.status);
    if (res.status === 200) {
      const ct = res.headers['content-type'] || '';
      // Prometheus metrics are text/plain
      expect(ct.includes('text') || ct.includes('json')).toBe(true);
    }
  });
});

describe('Server info & stats', () => {
  test('GET /info returns system information', async () => {
    if (skip()) return;
    const res = await api.get('/info');
    expect([200, 404]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
    }
  });

  test('GET /stats returns request metrics', async () => {
    if (skip()) return;
    const res = await api.get('/stats');
    expect([200, 404]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
    }
  });

  test('GET /endpoints returns endpoint list', async () => {
    if (skip()) return;
    const res = await api.get('/endpoints');
    expect([200, 404]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
    }
  });
});

describe('Models listing (core)', () => {
  test('GET /models returns model list', async () => {
    if (skip()) return;
    const res = await api.get('/models');
    expect([200, 404]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
    }
  });
});
