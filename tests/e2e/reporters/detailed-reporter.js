// @ts-check
/**
 * Custom Playwright reporter that emits a rich, human-readable summary
 * alongside a structured JSON copy. Every test entry includes:
 *   - test id (stable across runs, used by Playwright internally)
 *   - what the test is exercising (full describe path + title)
 *   - all available details (status, duration, retries, error, stdout/stderr)
 *   - exact reproduction command
 *   - relative paths to any captured screenshots/traces/videos
 *
 * Outputs (relative to this file's parent, resolved via testDir config):
 *   results/summary.md   — markdown report, grouped by spec file
 *   results/summary.json — machine-readable equivalent
 */
const fs = require('fs');
const path = require('path');

class DetailedReporter {
  constructor(options = {}) {
    this.options = options;
    this.entries = [];
    this.startTime = null;
    this.config = null;
    this.rootDir = null;
  }

  onBegin(config, suite) {
    this.startTime = new Date();
    this.config = config;
    // Playwright's config.rootDir points at the test directory (specs/),
    // not the config file's directory. Anchor on this reporter's own
    // location instead: tests/e2e/reporters/ → tests/e2e/.
    this.projectRoot = path.resolve(__dirname, '..');
    this.totalTests = suite.allTests().length;
  }

  onTestEnd(test, result) {
    const location = test.location || { file: '', line: 0, column: 0 };
    const relFile = location.file
      ? path.relative(this.projectRoot, location.file)
      : '';

    // Title path: ['', <project>, <spec-file>, <describe...>, <test title>]
    // Playwright prepends an empty root + the project name + the spec file
    // name. Strip leading empties, the project name, and the spec file name
    // (already surfaced as the Spec field) so describePath reflects only the
    // user-visible describe() nesting.
    const projectNames = new Set(
      (this.config?.projects || []).map((p) => p.name).filter(Boolean),
    );
    const specBasename = relFile ? path.basename(relFile) : '';
    const titlePath = test
      .titlePath()
      .filter((seg) => seg && !projectNames.has(seg) && seg !== specBasename);
    const describePath = titlePath.slice(0, -1);
    const testTitle = titlePath[titlePath.length - 1] || test.title;

    // Reproduction command — points Playwright at the exact file:line.
    // Run from the playwright.config.js directory (tests/e2e).
    const reproduceCmd = relFile
      ? `npx playwright test ${relFile}:${location.line} --reporter=list`
      : '';

    // Attachments: screenshots, traces, videos. Playwright stores these in
    // outputDir (results/artifacts). Emit paths relative to the results/
    // directory so the summary stays portable.
    const resultsDir = path.resolve(this.projectRoot, 'results');
    const attachments = (result.attachments || []).map((a) => ({
      name: a.name,
      contentType: a.contentType,
      path: a.path ? path.relative(resultsDir, a.path) : null,
    }));

    const errorMessage = result.error
      ? (result.error.message || String(result.error)).slice(0, 2000)
      : null;
    const errorStack = result.error?.stack
      ? String(result.error.stack).slice(0, 4000)
      : null;

    this.entries.push({
      testId: test.id,
      title: testTitle,
      describePath,
      fullTitle: titlePath.join(' › '),
      spec: relFile,
      line: location.line,
      column: location.column,
      status: result.status, // 'passed' | 'failed' | 'skipped' | 'timedOut' | 'interrupted'
      expectedStatus: test.expectedStatus,
      duration: result.duration,
      retry: result.retry,
      workerIndex: result.workerIndex,
      reproduce: reproduceCmd,
      error: errorMessage,
      errorStack,
      stdout: (result.stdout || []).map((c) => c.toString()).join('').slice(0, 2000),
      stderr: (result.stderr || []).map((c) => c.toString()).join('').slice(0, 2000),
      attachments,
    });
  }

  onEnd(result) {
    const resultsDir = path.resolve(this.projectRoot, 'results');
    fs.mkdirSync(resultsDir, { recursive: true });

    const totalDurationMs = this.startTime
      ? new Date().getTime() - this.startTime.getTime()
      : 0;

    const summary = {
      startedAt: this.startTime?.toISOString() || null,
      finishedAt: new Date().toISOString(),
      totalDurationMs,
      overallStatus: result.status, // 'passed' | 'failed' | 'timedout' | 'interrupted'
      counts: this._countByStatus(),
      totalTests: this.entries.length,
      tests: this.entries,
    };

    const jsonPath = path.join(resultsDir, 'summary.json');
    fs.writeFileSync(jsonPath, JSON.stringify(summary, null, 2));

    const mdPath = path.join(resultsDir, 'summary.md');
    fs.writeFileSync(mdPath, this._toMarkdown(summary));

    // Human-visible notice on stdout so runners see where to look.
    process.stdout.write(
      `\nDetailedReporter wrote:\n  ${path.relative(this.projectRoot, jsonPath)}\n  ${path.relative(this.projectRoot, mdPath)}\n`,
    );
  }

  _countByStatus() {
    const counts = { passed: 0, failed: 0, skipped: 0, timedOut: 0, interrupted: 0 };
    for (const e of this.entries) {
      counts[e.status] = (counts[e.status] || 0) + 1;
    }
    return counts;
  }

  _toMarkdown(summary) {
    const lines = [];
    lines.push('# Playwright E2E — Detailed Test Summary');
    lines.push('');
    lines.push(`- **Started:** ${summary.startedAt}`);
    lines.push(`- **Finished:** ${summary.finishedAt}`);
    lines.push(`- **Duration:** ${(summary.totalDurationMs / 1000).toFixed(1)}s`);
    lines.push(`- **Overall status:** \`${summary.overallStatus}\``);
    lines.push(`- **Total tests:** ${summary.totalTests}`);
    const c = summary.counts;
    lines.push(
      `- **Counts:** ${c.passed} passed · ${c.failed} failed · ${c.skipped} skipped · ${c.timedOut} timedOut · ${c.interrupted} interrupted`,
    );
    lines.push('');
    lines.push('---');
    lines.push('');

    // Group by spec file.
    const bySpec = new Map();
    for (const e of summary.tests) {
      if (!bySpec.has(e.spec)) bySpec.set(e.spec, []);
      bySpec.get(e.spec).push(e);
    }
    const specs = [...bySpec.keys()].sort();

    for (const spec of specs) {
      lines.push(`## ${spec}`);
      lines.push('');
      const specEntries = bySpec.get(spec).sort((a, b) => a.line - b.line);
      for (const e of specEntries) {
        const statusIcon = this._statusIcon(e.status);
        lines.push(`### ${statusIcon} \`${e.testId}\` — ${e.title}`);
        lines.push('');
        if (e.describePath.length) {
          lines.push(`- **Testing:** ${e.describePath.join(' › ')} › ${e.title}`);
        } else {
          lines.push(`- **Testing:** ${e.title}`);
        }
        lines.push(`- **Spec:** \`${e.spec}:${e.line}:${e.column}\``);
        lines.push(`- **Status:** \`${e.status}\` (expected: \`${e.expectedStatus}\`)`);
        lines.push(`- **Duration:** ${e.duration}ms`);
        if (e.retry > 0) lines.push(`- **Retry:** ${e.retry}`);
        lines.push(`- **Reproduce:** \`${e.reproduce}\``);

        if (e.error) {
          lines.push('- **Error:**');
          lines.push('  ```');
          lines.push('  ' + e.error.split('\n').join('\n  '));
          lines.push('  ```');
        }
        if (e.errorStack && e.error) {
          lines.push('  <details><summary>Stack</summary>');
          lines.push('');
          lines.push('  ```');
          lines.push('  ' + e.errorStack.split('\n').join('\n  '));
          lines.push('  ```');
          lines.push('  </details>');
        }
        if (e.stdout) {
          lines.push('- **Stdout:**');
          lines.push('  ```');
          lines.push('  ' + e.stdout.trimEnd().split('\n').join('\n  '));
          lines.push('  ```');
        }
        if (e.stderr) {
          lines.push('- **Stderr:**');
          lines.push('  ```');
          lines.push('  ' + e.stderr.trimEnd().split('\n').join('\n  '));
          lines.push('  ```');
        }
        if (e.attachments.length) {
          lines.push('- **Attachments:**');
          for (const a of e.attachments) {
            if (!a.path) continue;
            const isImage =
              a.contentType?.startsWith('image/') ||
              /\.(png|jpe?g|webp|gif)$/i.test(a.path);
            if (isImage) {
              // Markdown image embed — path is relative to results/summary.md
              lines.push(`  - ${a.name} (${a.contentType}): ![${a.name}](${a.path})`);
            } else {
              lines.push(`  - ${a.name} (${a.contentType}): \`${a.path}\``);
            }
          }
        }
        lines.push('');
      }
      lines.push('---');
      lines.push('');
    }

    return lines.join('\n');
  }

  _statusIcon(status) {
    switch (status) {
      case 'passed': return '✅';
      case 'failed': return '❌';
      case 'skipped': return '⏭️';
      case 'timedOut': return '⏱️';
      case 'interrupted': return '⚠️';
      default: return '•';
    }
  }
}

module.exports = DetailedReporter;
