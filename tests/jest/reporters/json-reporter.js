'use strict';
const fs = require('fs');
const path = require('path');

class JsonReporter {
  constructor(globalConfig, options) {
    this._options = options || {};
    this._outputFile = options && options.outputFile
      ? path.resolve(options.outputFile)
      : path.resolve('results/test-results.json');
  }

  onRunComplete(_contexts, results) {
    const output = {
      numTotalTests: results.numTotalTests,
      numPassedTests: results.numPassedTests,
      numFailedTests: results.numFailedTests,
      numPendingTests: results.numPendingTests,
      startTime: results.startTime,
      success: results.success,
      testResults: results.testResults.map(suite => ({
        testFilePath: suite.testFilePath,
        numPassingTests: suite.numPassingTests,
        numFailingTests: suite.numFailingTests,
        testResults: suite.testResults.map(t => ({
          fullName: t.fullName,
          status: t.status,
          duration: t.duration,
          failureMessages: t.failureMessages,
        })),
      })),
    };
    fs.mkdirSync(path.dirname(this._outputFile), { recursive: true });
    fs.writeFileSync(this._outputFile, JSON.stringify(output, null, 2));
  }
}

module.exports = JsonReporter;
