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
  navTTS:        'button.nav-item:has-text("TTS")',
  navClassify:   'button.nav-item:has-text("Classify")',
  navLLM:        'button.nav-item:has-text("Assistant")',
  navCompletion: 'button.nav-item:has-text("Completion")',
  navDashboard:  'button.nav-item:has-text("Dashboard")',
  navEndpoints:  'button.nav-item:has-text("Endpoints")',

  // ── Panels ───────────────────────────────────────────────
  panelStatus:     '#panel-status',
  panelTTS:        '#panel-tts',
  panelClassify:   '#panel-classify',
  panelLLM:        '#panel-assistant',
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

  // ── TTS panel (WebSocket streaming) ─────────────────────
  ttsText:      '#ws-tts-text',
  ttsEngine:    '#tts-engine',
  ttsVoice:     '#ws-tts-voice',
  ttsSpeed:     '#ws-tts-speed',
  ttsBtn:       '#ws-tts-btn',
  ttsStatus:    '#ws-tts-status',
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

  // ── LLM Chat panel (Assistant) ──────────────────────────
  chatHistory: '#llm-chat-box',
  chatInput:   '#llm-input',
  chatBtn:     '#llm-send-btn',
  chatModel:   '#llm-model',
  chatMaxtok:  '#llm-max-tokens',
  chatTemp:    '#llm-temp',

  // ── Completion panel (not implemented) ──────────────────
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

  // ── Logs panel ───────────────────────────────────────────
  navLogs:           'button.nav-item:has-text("Logs")',
  panelLogs:         '#panel-logs',
  logsFileList:      '#logs-file-list',
  logsViewerHeader:  '#logs-viewer-header',
  logsViewerContent: '#logs-viewer-content',

  // ── System panel ─────────────────────────────────────────
  navSystem:        'button.nav-item:has-text("System")',
  panelSystem:      '#panel-system',
  sysTabInfo:       '#sys-tab-info',
  sysTabPerf:       '#sys-tab-perf',
  sysTabConfig:     '#sys-tab-config',
  sysInfoPane:      '#sys-info-pane',
  sysPerfPane:      '#sys-perf-pane',
  sysConfigPane:    '#sys-config-pane',
  sysInfoCards:     '#sys-info-cards',
  sysPerfTiles:     '#sys-perf-tiles',
  sysOptTips:       '#sys-opt-tips',
  sysConfigContent: '#sys-config-content',

  // ── Audio panel ─────────────────────────────────────────
  navAudio:         'button.nav-item:has-text("STT")',
  panelAudio:       '#panel-audio',
  audioHealthBadge: '#audio-health-badge',
  audioDropzone:    '#audio-dropzone',
  audioFile:        '#audio-file',
  audioModel:       '#audio-model',
  audioTimestamps:  '#audio-timestamps',
  audioBtn:         '#audio-btn',
  audioResultText:  '#audio-result-text',
  audioSegments:    '#audio-segments',
  audioMeta:        '#audio-meta',

  // ── Models panel ────────────────────────────────────────
  navModels:   'button.nav-item:has-text("Models")',
  panelModels: '#panel-models',

  // Models sub-tabs
  mdlTabAvailable:  '#mdl-tab-available',
  mdlTabDownloaded: '#mdl-tab-downloaded',
  mdlTabSota:       '#mdl-tab-sota',
  mdlTabDownload:   '#mdl-tab-download',

  // Catalog sub-tab
  mdlAvailable:    '#mdl-available',
  mdlSearch:       '#mdl-search',
  mdlFilterTask:   '#mdl-filter-task',
  mdlAvailGrid:    '#mdl-avail-grid',
  mdlAvailLoading: '#mdl-avail-loading',
  mdlAvailError:   '#mdl-avail-error',

  // Downloaded sub-tab
  mdlDownloaded:     '#mdl-downloaded',
  mdlDownloadedList: '#mdl-downloaded-list',
  mdlDownLoading:    '#mdl-down-loading',

  // SOTA sub-tab
  mdlSota:      '#mdl-sota',
  mdlSotaGrid:  '#mdl-sota-grid',

  // Download form sub-tab
  mdlDownload:       '#mdl-download',
  mdlDlSource:       '#mdl-dl-source',
  mdlDlRepo:         '#mdl-dl-repo',
  mdlDlRevisionWrap: '#mdl-dl-revision-wrap',
  mdlDlRevision:     '#mdl-dl-revision',
  mdlDlBtn:          '#mdl-dl-btn',
  mdlDlError:        '#mdl-dl-error',
  mdlDlTasksList:    '#mdl-dl-tasks-list',
};
