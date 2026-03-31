/**
 * Shared Playwright selectors for playground.html.
 * All IDs are taken directly from the HTML source in
 * .worktrees/dashboard/src/api/playground.html.
 */
module.exports = {
  // ── Topbar ──────────────────────────────────────────────
  healthDot:  '#health-dot',
  healthText: '#health-text',

  // ── Sidebar nav — uses Playwright :has-text() pseudo-class
  navStatus:     'button.nav-item:has-text("Status")',
  navTTS:        'button.nav-item:has-text("TTS Stream")',
  navClassify:   'button.nav-item:has-text("Classify")',
  navLLM:        'button.nav-item:has-text("LLM Chat")',
  navCompletion: 'button.nav-item:has-text("Completion")',
  navDashboard:  'button.nav-item:has-text("Dashboard")',
  navEndpoints:  'button.nav-item:has-text("Endpoints")',

  // ── Panels ───────────────────────────────────────────────
  panelStatus:     '#panel-status',
  panelTTS:        '#panel-tts',
  panelClassify:   '#panel-classify',
  panelLLM:        '#panel-llm',
  panelCompletion: '#panel-completion',
  panelDashboard:  '#panel-dashboard',
  panelEndpoints:  '#panel-endpoints',

  // ── Status panel ────────────────────────────────────────
  sStatus:   '#s-status',
  sUptime:   '#s-uptime',
  sActive:   '#s-active',
  sTotal:    '#s-total',
  sLatency:  '#s-latency',
  sErrors:   '#s-errors',
  healthRaw: '#health-raw',

  // ── TTS panel ───────────────────────────────────────────
  ttsText:      '#tts-text',
  ttsEngine:    '#tts-engine',
  ttsVoice:     '#tts-voice',
  ttsSpeed:     '#tts-speed',
  ttsBtn:       '#tts-btn',
  ttsStatus:    '#tts-status',
  ttsAudioWrap: '#tts-audio-wrap',
  ttsAudio:     '#tts-audio',

  // ── Classify panel ──────────────────────────────────────
  dropzone:   '#dropzone',
  imgFile:    '#img-file',
  imgPreview: '#img-preview',
  clsTopk:    '#cls-topk',
  clsW:       '#cls-w',
  clsH:       '#cls-h',
  clsBtn:     '#cls-btn',
  clsOut:     '#cls-out',

  // ── LLM Chat panel ──────────────────────────────────────
  chatHistory: '#chat-history',
  chatInput:   '#chat-input',
  chatBtn:     '#chat-btn',
  chatModel:   '#chat-model',
  chatMaxtok:  '#chat-maxtok',
  chatTemp:    '#chat-temp',

  // ── Completion panel ────────────────────────────────────
  cmpPrompt: '#cmp-prompt',
  cmpModel:  '#cmp-model',
  cmpMaxtok: '#cmp-maxtok',
  cmpTemp:   '#cmp-temp',
  cmpTopp:   '#cmp-topp',
  cmpBtn:    '#cmp-btn',
  cmpOut:    '#cmp-out',

  // ── Dashboard panel ─────────────────────────────────────
  dashTabOverview:   '#dash-tab-overview',
  dashTabPlayground: '#dash-tab-playground',
  dashOverview:      '#dash-overview',
  dashPlayground:    '#dash-playground',
  dUptime:       '#d-uptime',
  dTotal:        '#d-total',
  dActive:       '#d-active',
  dLatency:      '#d-latency',
  dErrors:       '#d-errors',
  dRps:          '#d-rps',
  dashSpark:     '#dash-spark',
  dashLiveBadge: '#dash-live-badge',
  barCpu:  '#bar-cpu',
  barRam:  '#bar-ram',
  barGpu:  '#bar-gpu',
  valCpu:  '#val-cpu',
  valRam:  '#val-ram',
  valGpu:  '#val-gpu',
  gpuDeviceInfo: '#gpu-device-info',

  // ── Model download manager ──────────────────────────────
  dlSource:       '#dl-source',
  dlRepo:         '#dl-repo',
  dlRevision:     '#dl-revision',
  dlRevisionWrap: '#dl-revision-wrap',
  dlBtn:          '#dl-btn',
  dlError:        '#dl-error',
  dlTasksEmpty:   '#dl-tasks-empty',
  dlTasksList:    '#dl-tasks-list',

  // ── Playground sub-tabs ─────────────────────────────────
  pgTabTTS:        '#pg-tab-tts',
  pgTabClassify:   '#pg-tab-classify',
  pgTabLLM:        '#pg-tab-llm',
  pgTabCompletion: '#pg-tab-completion',

  // ── Playground TTS ──────────────────────────────────────
  pgTTS:          '#pg-tts',
  pgTtsText:      '#pg-tts-text',
  pgTtsEngine:    '#pg-tts-engine',
  pgTtsVoice:     '#pg-tts-voice',
  pgTtsSpeed:     '#pg-tts-speed',
  pgTtsBtn:       '#pg-tts-btn',
  pgTtsStatus:    '#pg-tts-status',
  pgTtsAudioWrap: '#pg-tts-audio-wrap',

  // ── Playground Classify ─────────────────────────────────
  pgClassify:   '#pg-classify',
  pgDropzone:   '#pg-dropzone',
  pgImgFile:    '#pg-img-file',
  pgImgPreview: '#pg-img-preview',
  pgClsTopk:    '#pg-cls-topk',
  pgClsW:       '#pg-cls-w',
  pgClsH:       '#pg-cls-h',
  pgClsBtn:     '#pg-cls-btn',
  pgClsOut:     '#pg-cls-out',

  // ── Playground LLM Chat ─────────────────────────────────
  pgLLM:         '#pg-llm',
  pgChatHistory: '#pg-chat-history',
  pgChatInput:   '#pg-chat-input',
  pgChatBtn:     '#pg-chat-btn',
  pgChatModel:   '#pg-chat-model',
  pgChatMaxtok:  '#pg-chat-maxtok',
  pgChatTemp:    '#pg-chat-temp',

  // ── Playground Completion ───────────────────────────────
  pgCompletion: '#pg-completion',
  pgCmpPrompt:  '#pg-cmp-prompt',
  pgCmpModel:   '#pg-cmp-model',
  pgCmpMaxtok:  '#pg-cmp-maxtok',
  pgCmpTemp:    '#pg-cmp-temp',
  pgCmpTopp:    '#pg-cmp-topp',
  pgCmpBtn:     '#pg-cmp-btn',
  pgCmpOut:     '#pg-cmp-out',
};
