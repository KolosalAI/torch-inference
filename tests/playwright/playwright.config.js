require('dotenv').config({ path: __dirname + '/.env.test' });
const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.js',
  timeout: 30000,
  expect: { timeout: 5000 },
  fullyParallel: false,
  workers: 1,
  reporter: [
    ['html',  { outputFolder: 'results/report', open: 'never' }],
    ['junit', { outputFile: 'results/junit.xml' }],
    ['list'],
  ],
  use: {
    baseURL:    process.env.BASE_URL || 'http://localhost:8000',
    headless:   process.env.HEADED !== '1',
    screenshot: 'only-on-failure',
  },
  globalSetup:    './global-setup.js',
  globalTeardown: './global-teardown.js',
  outputDir: 'results/test-results',
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
});
