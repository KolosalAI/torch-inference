'use strict';
const axios = require('axios').default;

const BASE_URL = process.env.BASE_URL || 'http://localhost:8000';

const client = axios.create({ baseURL: BASE_URL, validateStatus: () => true, timeout: 30000 });

const api = {
  get:    (path, config) => client.get(path, config),
  post:   (path, data, config) => client.post(path, data, config),
  put:    (path, data, config) => client.put(path, data, config),
  delete: (path, config) => client.delete(path, config),
};

function createClient(baseURLOrConfig) {
  const config = typeof baseURLOrConfig === 'string'
    ? { baseURL: baseURLOrConfig }
    : (baseURLOrConfig || {});
  return axios.create({ baseURL: BASE_URL, validateStatus: () => true, timeout: 30000, ...config });
}

function expectSuccess(res) {
  expect(res.status).toBeGreaterThanOrEqual(200);
  expect(res.status).toBeLessThan(300);
}

function expectJsonBody(res) {
  expect(res.headers['content-type']).toMatch(/application\/json/);
  expect(res.data).toBeDefined();
}

module.exports = { api, createClient, expectSuccess, expectJsonBody };
