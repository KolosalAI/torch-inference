'use strict';
const http = require('http');
const { execSync, spawn } = require('child_process');
const path = require('path');

const BASE_URL = process.env.BASE_URL || 'http://localhost:8000';
const url = new URL(BASE_URL);
const HOST = url.hostname;
const PORT = parseInt(url.port, 10) || 8000;

let _serverProcess = null;

function isServerRunning() {
  return new Promise(resolve => {
    const req = http.get({ host: HOST, port: PORT, path: '/health', timeout: 2000 }, res => {
      res.resume();
      resolve(res.statusCode < 500);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

async function startServer() {
  const root = path.resolve(__dirname, '../../..');
  const bin = path.join(root, 'target', 'release', 'torch-inference-server');
  _serverProcess = spawn(bin, [], { cwd: root, stdio: 'ignore', detached: false });
  _serverProcess.unref();
  await waitForServer(30000);
}

function waitForServer(timeoutMs = 30000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const poll = async () => {
      if (await isServerRunning()) return resolve();
      if (Date.now() - start > timeoutMs) return reject(new Error('Server did not start within ' + timeoutMs + 'ms'));
      setTimeout(poll, 500);
    };
    poll();
  });
}

async function stopServer() {
  if (_serverProcess) {
    _serverProcess.kill('SIGTERM');
    _serverProcess = null;
  }
}

module.exports = { isServerRunning, startServer, waitForServer, stopServer };
