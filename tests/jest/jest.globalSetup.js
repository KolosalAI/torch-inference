require('dotenv').config({ path: __dirname + '/.env.test' });
const { isServerRunning, startServer, waitForServer } = require('./utils/server');
const fs = require('fs');
const path = require('path');

module.exports = async function globalSetup() {
  // Ensure results directory exists
  const resultsDir = path.join(__dirname, 'results');
  if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

  const running = await isServerRunning();

  if (!running) {
    if (process.env.AUTO_START_SERVER === 'true') {
      await startServer();
    } else {
      console.warn(
        '\n[globalSetup] WARNING: Server is not running at ' + (process.env.BASE_URL || 'http://localhost:8000') +
        '\n  Start the server with: cargo run --release' +
        '\n  Or set AUTO_START_SERVER=true in .env.test to auto-start.\n' +
        '  Tests will be skipped or fail if the server is unavailable.\n'
      );
    }
  } else {
    console.log('[globalSetup] Server is running at ' + (process.env.BASE_URL || 'http://localhost:8000'));
  }
};
