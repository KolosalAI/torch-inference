/**
 * Axios client and helper utilities for API tests.
 */
const axios = require('axios');

const BASE_URL = process.env.BASE_URL || 'http://localhost:8000';

/**
 * Create an axios instance pre-configured for the server.
 * @param {object} options
 * @param {string} [options.token] - Bearer token for Authorization header
 * @param {number} [options.timeout] - Request timeout in ms
 */
function createClient({ token, timeout } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  return axios.create({
    baseURL: BASE_URL,
    timeout: timeout || parseInt(process.env.DEFAULT_TIMEOUT || '10000', 10),
    headers,
    validateStatus: () => true, // never throw on HTTP errors — let tests assert
  });
}

/**
 * Default unauthenticated client for tests that don't need auth.
 */
const api = createClient();

/**
 * Generate a JWT token using the configured secret.
 * Requires the server to issue tokens, or builds one via jsonwebtoken if available.
 */
async function getAuthToken() {
  try {
    const jwt = require('jsonwebtoken');
    const secret = process.env.JWT_SECRET || 'your-secret-key-here-change-in-production';
    return jwt.sign({ sub: 'test-user', role: 'admin' }, secret, {
      algorithm: 'HS256',
      expiresIn: '1h',
    });
  } catch {
    // jsonwebtoken not available — tests that need auth will skip
    return null;
  }
}

/**
 * Assert common response shape expectations.
 */
function expectSuccess(response) {
  expect(response.status).toBeGreaterThanOrEqual(200);
  expect(response.status).toBeLessThan(300);
}

function expectError(response, statusCode) {
  if (statusCode !== undefined) {
    expect(response.status).toBe(statusCode);
  } else {
    expect(response.status).toBeGreaterThanOrEqual(400);
  }
}

function expectJsonBody(response) {
  const ct = response.headers['content-type'] || '';
  expect(ct).toMatch(/json/);
  expect(response.data).toBeDefined();
}

/**
 * Safely get a nested property without throwing.
 */
function get(obj, path, fallback = undefined) {
  return path.split('.').reduce((acc, key) => (acc == null ? fallback : acc[key]), obj) ?? fallback;
}

module.exports = { api, createClient, getAuthToken, expectSuccess, expectError, expectJsonBody, get, BASE_URL };
