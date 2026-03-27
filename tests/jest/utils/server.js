/**
 * Server lifecycle utilities — health-check and optional auto-start.
 */
const http = require('http');
const { execFile } = require('child_process');
const path = require('path');

const BASE_URL = process.env.BASE_URL || 'http://localhost:8000';
const [serverHost, serverPort] = BASE_URL.replace('http://', '').split(':');
const PORT = parseInt(serverPort || '8000', 10);
const HOST = serverHost || 'localhost';

let serverProcess = null;

/**
 * Check if the server is reachable.
 * @returns {Promise<boolean>}
 */
async function isServerRunning() {
  return new Promise((resolve) => {
    const req = http.get(`${BASE_URL}/health`, { timeout: 3000 }, (res) => {
      resolve(res.statusCode < 500);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

/**
 * Wait for the server to become healthy, polling every 500ms.
 * @param {number} maxWaitMs
 * @returns {Promise<boolean>}
 */
async function waitForServer(maxWaitMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    if (await isServerRunning()) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

/**
 * Attempt to start the server binary.
 * Only called when AUTO_START_SERVER=true and server is not running.
 * @returns {Promise<void>}
 */
async function startServer() {
  const binaryPath = path.resolve(
    process.env.SERVER_BINARY ||
    path.join(__dirname, '../../../target/release/torch-inference')
  );

  console.log(`[server] Starting server binary: ${binaryPath}`);
  serverProcess = execFile(binaryPath, [], {
    cwd: path.join(__dirname, '../../..'),
    env: { ...process.env },
    detached: false,
  });

  serverProcess.stdout?.on('data', (d) => process.stdout.write(`[server] ${d}`));
  serverProcess.stderr?.on('data', (d) => process.stderr.write(`[server] ${d}`));
  serverProcess.on('exit', (code) => {
    if (code !== null) console.log(`[server] exited with code ${code}`);
    serverProcess = null;
  });

  const ready = await waitForServer(20000);
  if (!ready) throw new Error('Server failed to start within 20s');
  console.log('[server] Server is ready');
}

/**
 * Stop the server process if it was started by us.
 */
async function stopServer() {
  if (serverProcess) {
    serverProcess.kill('SIGTERM');
    await new Promise((r) => setTimeout(r, 1000));
    serverProcess = null;
    console.log('[server] Server stopped');
  }
}

module.exports = { isServerRunning, waitForServer, startServer, stopServer };
