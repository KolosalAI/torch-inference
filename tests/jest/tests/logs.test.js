/**
 * Logging endpoint tests
 * Covers: GET /logs, GET /logs/:log_file, DELETE /logs/:log_file
 */
const { api, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;
let availableLogFile = null;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) {
    console.warn('[logs] Server not available — tests will be skipped');
    return;
  }
  // Discover an available log file for use in subsequent tests
  const res = await api.get('/logs');
  if (res.status === 200 && Array.isArray(res.data.available_log_files) && res.data.available_log_files.length > 0) {
    availableLogFile = res.data.available_log_files[0];
  }
});

const skip = () => !serverAvailable;

describe('Logs listing', () => {
  test('GET /logs returns log directory info', async () => {
    if (skip()) return;
    const res = await api.get('/logs');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('log_directory');
    expect(res.data).toHaveProperty('available_log_files');
    expect(Array.isArray(res.data.available_log_files)).toBe(true);
  });

  test('GET /logs includes log level', async () => {
    if (skip()) return;
    const res = await api.get('/logs');
    expectSuccess(res);
    expect(res.data).toHaveProperty('log_level');
  });
});

describe('Log file read', () => {
  test('GET /logs/:log_file returns file content', async () => {
    if (skip()) return;
    if (!availableLogFile) {
      console.warn('[logs] No log files available — skipping read test');
      return;
    }
    const res = await api.get(`/logs/${availableLogFile}`);
    expect([200, 404]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
      expect(res.data).toHaveProperty('file_name');
      expect(res.data).toHaveProperty('content');
      expect(res.data).toHaveProperty('line_count');
    }
  });

  test('GET /logs/nonexistent-file returns 404', async () => {
    if (skip()) return;
    const res = await api.get('/logs/nonexistent-log-file-xyz.log');
    expect([404, 400]).toContain(res.status);
  });
});

describe('Log file deletion', () => {
  test('DELETE /logs/nonexistent-file returns 404', async () => {
    if (skip()) return;
    const res = await api.delete('/logs/nonexistent-log-file-xyz.log');
    expect([404, 400]).toContain(res.status);
  });

  // NOTE: We intentionally do NOT delete real log files in tests to preserve server logs.
});
