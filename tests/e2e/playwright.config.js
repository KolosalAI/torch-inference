// @ts-check
const { defineConfig, devices } = require('@playwright/test');

// All test output (per-test artifacts, HTML report, JSON summary, JUnit XML)
// is consolidated under ./results/ so the suite has a single artifact root.
// The ./results/ directory is gitignored; see .gitignore at repo root.
module.exports = defineConfig({
  testDir: './specs',
  outputDir: './results/artifacts',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  retries: 1,
  reporter: [
    ['list'],
    ['html', { open: 'never', outputFolder: 'results/html-report' }],
    ['json', { outputFile: 'results/run.json' }],
    ['junit', { outputFile: 'results/junit.xml' }],
  ],
  use: {
    baseURL: 'http://localhost:8000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'off',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
