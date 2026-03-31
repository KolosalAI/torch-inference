const fs = require('fs');
const path = require('path');

const PID_FILE = path.join(__dirname, '.server.pid');

module.exports = async function globalTeardown() {
  if (!fs.existsSync(PID_FILE)) return;
  const pid = parseInt(fs.readFileSync(PID_FILE, 'utf8'), 10);
  try {
    process.kill(pid, 'SIGTERM');
    console.log(`[playwright-teardown] Sent SIGTERM to PID ${pid}`);
  } catch (e) {
    console.warn(`[playwright-teardown] Could not kill PID ${pid}: ${e.message}`);
  }
  fs.unlinkSync(PID_FILE);
};
