/**
 * Custom Jest reporter that writes full test results to a JSON file.
 * Output: results/test-results.json
 *
 * Schema:
 * {
 *   "timestamp": "ISO8601",
 *   "summary": { total, passed, failed, skipped, duration_ms },
 *   "suites": [ { name, file, status, duration_ms, tests: [ ... ] } ]
 * }
 */
const fs = require('fs');
const path = require('path');

class JsonReporter {
  constructor(globalConfig, options) {
    this._globalConfig = globalConfig;
    this._outputFile = (options && options.outputFile)
      ? options.outputFile
      : path.join(path.dirname(__dirname), 'results', 'test-results.json');
  }

  onRunComplete(_contexts, results) {
    const startTime = results.startTime;
    const endTime = Date.now();

    const summary = {
      total: results.numTotalTests,
      passed: results.numPassedTests,
      failed: results.numFailedTests,
      skipped: results.numPendingTests + results.numTodoTests,
      duration_ms: endTime - startTime,
    };

    const suites = (results.testResults || []).map((suite) => {
      const tests = (suite.testResults || []).map((t) => ({
        name: t.fullName,
        status: t.status,               // "passed" | "failed" | "pending" | "todo"
        duration_ms: t.duration || 0,
        error: t.status === 'failed'
          ? (t.failureMessages || []).join('\n')
          : null,
        ancestorTitles: t.ancestorTitles || [],
      }));

      return {
        name: path.basename(suite.testFilePath),
        file: suite.testFilePath,
        status: suite.testExecError ? 'error' : (suite.numFailingTests > 0 ? 'failed' : 'passed'),
        duration_ms: suite.perfStats
          ? suite.perfStats.end - suite.perfStats.start
          : 0,
        tests,
        exec_error: suite.testExecError ? suite.testExecError.message : null,
      };
    });

    const output = {
      timestamp: new Date(startTime).toISOString(),
      config: {
        base_url: process.env.BASE_URL || 'http://localhost:8000',
        node_version: process.version,
      },
      summary,
      suites,
    };

    const dir = path.dirname(this._outputFile);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    fs.writeFileSync(this._outputFile, JSON.stringify(output, null, 2), 'utf8');
    console.log(`\n[json-reporter] Results written to: ${this._outputFile}`);
  }
}

module.exports = JsonReporter;
