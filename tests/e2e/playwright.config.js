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
    // Custom reporter emits results/summary.{md,json} with test id,
    // what's being tested, reproduction command, and any captured images.
    ['./reporters/detailed-reporter.js'],
  ],
  // Screenshot on EVERY test (not just failures) so the summary markdown
  // has visual context for passing tests. trace still only on retry to
  // keep artifact size reasonable.
  use: {
    baseURL: 'http://localhost:8000',
    trace: 'on-first-retry',
    screenshot: { mode: 'on', fullPage: false },
    video: 'off',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
