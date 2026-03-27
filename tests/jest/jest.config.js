require('dotenv').config({ path: __dirname + '/.env.test' });

/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'node',
  rootDir: __dirname,
  testMatch: ['**/tests/**/*.test.js'],
  testTimeout: 30000,
  globalSetup: './jest.globalSetup.js',
  globalTeardown: './jest.globalTeardown.js',
  reporters: [
    'default',
    ['./reporters/json-reporter.js', {
      outputFile: './results/test-results.json',
    }],
    ['jest-junit', {
      outputDirectory: './results',
      outputName: 'junit.xml',
      classNameTemplate: '{classname}',
      titleTemplate: '{title}',
    }],
  ],
  coverageDirectory: './results/coverage',
  collectCoverageFrom: ['utils/**/*.js'],
  verbose: true,
};
