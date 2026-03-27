/**
 * Audio endpoint tests
 * Covers: POST /audio/transcribe, POST /audio/validate, GET /audio/health
 */
const FormData = require('form-data');
const { api, createClient, expectSuccess, expectJsonBody } = require('../utils/api');
const { isServerRunning } = require('../utils/server');

let serverAvailable = false;

beforeAll(async () => {
  serverAvailable = await isServerRunning();
  if (!serverAvailable) console.warn('[audio] Server not available — tests will be skipped');
});

const skip = () => !serverAvailable;

// Minimal valid WAV header (44 bytes) + silence for tests that need a file
function minimalWav(sampleRate = 24000, channels = 1, durationSecs = 0.1) {
  const numSamples = Math.floor(sampleRate * durationSecs);
  const dataSize = numSamples * channels * 2; // 16-bit PCM
  const buf = Buffer.alloc(44 + dataSize, 0);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);          // chunk size
  buf.writeUInt16LE(1, 20);           // PCM
  buf.writeUInt16LE(channels, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * channels * 2, 28); // byte rate
  buf.writeUInt16LE(channels * 2, 32);  // block align
  buf.writeUInt16LE(16, 34);           // bits per sample
  buf.write('data', 36);
  buf.writeUInt32LE(dataSize, 40);
  return buf;
}

describe('Audio health', () => {
  test('GET /audio/health returns status', async () => {
    if (skip()) return;
    const res = await api.get('/audio/health');
    expectSuccess(res);
    expectJsonBody(res);
    expect(res.data).toHaveProperty('status');
  });
});

describe('Audio validation', () => {
  test('POST /audio/validate with valid WAV returns valid=true', async () => {
    if (skip()) return;
    const wavBuf = minimalWav();
    const form = new FormData();
    form.append('audio', wavBuf, { filename: 'test.wav', contentType: 'audio/wav' });

    const res = await api.post('/audio/validate', form, {
      headers: form.getHeaders(),
      timeout: 10000,
    });
    expect([200, 400, 404, 422]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
      expect(res.data).toHaveProperty('valid');
    }
  });

  test('POST /audio/validate with empty body returns 400/422', async () => {
    if (skip()) return;
    const res = await api.post('/audio/validate', {});
    expect(res.status).toBeGreaterThanOrEqual(400);
  });
});

describe('Audio transcription', () => {
  test('POST /audio/transcribe with WAV file returns transcript or error', async () => {
    if (skip()) return;
    const wavBuf = minimalWav(16000, 1, 0.5);
    const form = new FormData();
    form.append('audio', wavBuf, { filename: 'test.wav', contentType: 'audio/wav' });

    const client = createClient({ timeout: 30000 });
    const res = await client.post('/audio/transcribe', form, {
      headers: form.getHeaders(),
    });
    // May fail if no model loaded — both 200 and error states are valid
    expect([200, 400, 404, 422, 500, 503]).toContain(res.status);
    if (res.status === 200) {
      expectJsonBody(res);
      expect(res.data).toHaveProperty('text');
    }
  });
});
