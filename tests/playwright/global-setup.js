require('dotenv').config({ path: __dirname + '/.env.test' });
const { execFile, execFileSync } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const BASE_URL = process.env.BASE_URL || 'http://localhost:8000';
const DASHBOARD_DIR = path.resolve(__dirname, '../../.worktrees/dashboard');
const PID_FILE = path.join(__dirname, '.server.pid');

function checkHealth() {
  return new Promise(resolve => {
    const req = http.get(`${BASE_URL}/health`, { timeout: 2000 }, res => {
      resolve(res.statusCode < 500);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

async function waitForServer(maxWaitMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    if (await checkHealth()) return true;
    await new Promise(r => setTimeout(r, 500));
  }
  return false;
}

module.exports = async function globalSetup() {
  const resultsDir = path.join(__dirname, 'results');
  if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

  const alreadyRunning = await checkHealth();
  if (alreadyRunning) {
    console.log(`[playwright-setup] Server already running at ${BASE_URL}`);
    return;
  }

  const buildServer = process.env.BUILD_SERVER !== 'false';
  if (buildServer) {
    console.log('[playwright-setup] Building dashboard server (cargo build --release)…');
    execFileSync('cargo', ['build', '--release'], {
      cwd: DASHBOARD_DIR,
      stdio: 'inherit',
    });
    console.log('[playwright-setup] Build complete.');
  }

  const binaryPath = process.env.SERVER_BINARY
    ? path.resolve(process.env.SERVER_BINARY)
    : path.join(DASHBOARD_DIR, 'target', 'release', 'torch-inference');

  if (!fs.existsSync(binaryPath)) {
    throw new Error(
      `Server binary not found at ${binaryPath}. ` +
      `Set BUILD_SERVER=true or provide SERVER_BINARY in .env.test.`
    );
  }

  console.log(`[playwright-setup] Starting server: ${binaryPath}`);
  const serverProcess = execFile(binaryPath, [], {
    cwd: DASHBOARD_DIR,
    env: { ...process.env },
    detached: false,
  });
  serverProcess.stdout?.on('data', d => process.stdout.write(`[server] ${d}`));
  serverProcess.stderr?.on('data', d => process.stderr.write(`[server] ${d}`));
  serverProcess.on('exit', code => {
    if (code !== null && code !== 0) {
      console.error(`[server] exited with code ${code}`);
    }
  });

  fs.writeFileSync(PID_FILE, String(serverProcess.pid));

  const ready = await waitForServer(30000);
  if (!ready) throw new Error('Server failed to become healthy within 30s');
  console.log('[playwright-setup] Server is ready');
};
