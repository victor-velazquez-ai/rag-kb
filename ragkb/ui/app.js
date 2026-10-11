/* rag-kb UI: plain JavaScript, no build step, no external resources.
 *
 * Every string from the server or the user is rendered as text (textContent). The only HTML built from
 * strings is the answer's markdown, which is escaped first and then formatted by a small renderer.
 *
 * Loaded synchronously in <head> (not deferred): the saved theme is applied while the script runs, before the
 * first paint, and everything else waits for DOMContentLoaded.
 *
 * Addresses: #/c/<collection id>/<tab>, #/<tab> when no collection exists, #/answer/<answer id>. Back and
 * Forward move between tabs and collections; a reload opens the same view.
 */
(() => {
  'use strict';

  const TABS = ['documents', 'ask', 'settings', 'connect', 'help'];
  // Upload rows that are still moving: waiting in this browser, uploading, or a job on the server.
  const ACTIVE = new Set(['waiting', 'uploading', 'queued', 'processing', 'cancelling']);
  // Server-side states worth polling.
  const TRACKED = new Set(['queued', 'processing', 'cancelling']);
  const BAR_STATES = new Set(['waiting', 'uploading', 'queued', 'processing', 'cancelling', 'ready']);
  const JOB_ACTIVE = new Set(['queued', 'processing']);
  const WORK_STAGES = new Set(['parsing', 'chunking', 'embedding', 'writing']);
  const ARM_MS = 4000;
  const PENDING_NOTE_MS = 2000;
  const HEALTH_EVERY_MS = 20000;
  const JOB_POLL_MS = 1000;
  const DOCS_POLL_MS = 2000;
  const PROVIDER_LABELS = { claude_code: 'Claude Code', codex: 'Codex', extractive: 'Extractive mode (no AI)' };
  const AUTO_ORDER = ['claude_code', 'codex', 'extractive'];
  const CLAUDE_MODELS = ['sonnet', 'opus', 'haiku'];
  const STATUS_LABELS = {
    queued: 'Queued', processing: 'Processing', ready: 'Ready', failed: 'Failed', cancelled: 'Cancelled',
    waiting: 'Queued', uploading: 'Uploading', cancelling: 'Cancelling', known: 'Already indexed',
    rejected: 'Rejected', unpacked: 'Unpacked', skipped: 'Skipped',
  };
  const CHIP_TONES = { waiting: 'queued', cancelling: 'warn', unpacked: 'known', skipped: 'queued', rejected: 'failed' };
  const STAGE_LABELS = {
    parsing: 'Reading the file', chunking: 'Splitting into passages', embedding: 'Embedding passages',
    writing: 'Writing to the index',
  };
  const TYPE_LABELS = {
    'application/pdf': 'PDF',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'Word',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'Excel',
    'text/markdown': 'Markdown', 'text/html': 'HTML', 'text/plain': 'Text', 'text/csv': 'CSV',
    'text/tab-separated-values': 'TSV', 'application/zip': 'Zip',
  };
  const MCP_TOOLS = {
    list_collections: 'Lists the collections with their ids and how many searchable documents each holds.',
    list_documents: 'Lists the documents of a collection with pages, passages and when each was indexed.',
    search_knowledge_base: 'Returns the best passages for a query, with title, page, section and score. No answer is written.',
    ask_knowledge_base: 'Returns a cited brief for a question. Your assistant writes the final answer from it.',
  };
  const OUT_OF_SCOPE_PROBE = 'What is the boiling point of mercury?';
  const ARCHIVE = '.zip';
  const SHELLS = [['bash', 'bash'], ['powershell', 'PowerShell'], ['cmd', 'cmd']];
  const BACKSLASH = String.fromCharCode(92);

  const state = {
    tab: 'documents',
    health: null,
    healthFresh: false,
    offline: false,
    providers: [],
    providersError: null,
    providersLoadedAt: 0,
    collections: [],
    collectionsError: null,
    current: null,
    docs: [],
    docsLoaded: false,
    docsSig: '',
    docsError: null,
    stats: null,
    summary: null,
    expanded: new Set(),
    pending: new Map(),
    uploads: [],
    answerToken: 0,
    answerId: null,
    asking: false,
  };
  const chunkCache = new Map();
  const jobCache = new Map();
  let rowCache = new Map();
  let drawerCache = new Map();
  let uploadRowCache = new Map();
  let docsTimer = null;
  let jobTimer = null;
  let jobBusy = false;
  let uploadSeq = 0;
  let activeUploads = 0;
  let shellSeq = 0;
  let lastHash = null;
  const uploadQueue = [];

  // ------------------------------------------------------------------ small helpers

  const $ = (id) => document.getElementById(id);
  const enc = encodeURIComponent;

  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') el.className = value;
      else if (key === 'text') el.textContent = value;
      else if (key === 'dataset') Object.assign(el.dataset, value);
      else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2).toLowerCase(), value);
      else if (value === true) el.setAttribute(key, '');
      else el.setAttribute(key, String(value));
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid === null || kid === undefined || kid === false) continue;
      el.append(kid instanceof Node ? kid : String(kid));
    }
    return el;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function store(key, value) {
    try { localStorage.setItem('ragkb.' + key, value); } catch (e) { /* storage may be blocked */ }
  }

  function recall(key, fallback) {
    try {
      const value = localStorage.getItem('ragkb.' + key);
      return value === null ? fallback : value;
    } catch (e) {
      return fallback;
    }
  }

  function storeJson(key, value) {
    try { store(key, JSON.stringify(value)); } catch (e) { /* a value that cannot be saved is simply not cached */ }
  }

  function recallJson(key) {
    try {
      const raw = recall(key, '');
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  function finite(value) {
    return typeof value === 'number' && Number.isFinite(value);
  }

  // ------------------------------------------------------------------ theme

  const THEMES = ['system', 'light', 'dark'];

  // System removes the attribute so the stylesheet follows prefers-color-scheme.
  function applyTheme(choice) {
    const root = document.documentElement;
    if (choice === 'light' || choice === 'dark') root.setAttribute('data-theme', choice);
    else root.removeAttribute('data-theme');
  }

  // Runs as the script loads, before the first paint, so a saved theme never flashes the other one.
  applyTheme(recall('theme', 'system'));

  function setupTheme() {
    const saved = recall('theme', 'system');
    const current = THEMES.includes(saved) ? saved : 'system';
    for (const input of document.querySelectorAll('#theme-switch input[name="theme"]')) {
      input.checked = input.value === current;
      input.addEventListener('change', () => {
        if (!input.checked) return;
        applyTheme(input.value);
        store('theme', input.value);
      });
    }
  }

  function announce(text) {
    const region = $('sr-status');
    region.textContent = '';
    window.setTimeout(() => { region.textContent = text; }, 30);
  }

  function providerLabel(name) {
    return PROVIDER_LABELS[name] || name || 'The provider';
  }

  function fmtBytes(n) {
    if (n === null || n === undefined) return '';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(n < 10240 ? 1 : 0) + ' KB';
    return (n / 1024 / 1024).toFixed(1) + ' MB';
  }

  function fmtType(mediaType) {
    return TYPE_LABELS[mediaType] || (mediaType ? mediaType.split('/').pop() : '');
  }

  const relative = typeof Intl !== 'undefined' && Intl.RelativeTimeFormat
    ? new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' }) : null;

  function fmtWhen(iso) {
    if (!iso) return { text: '', title: '' };
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return { text: '', title: '' };
    const full = date.toLocaleString();
    const seconds = Math.round((date.getTime() - Date.now()) / 1000);
    const abs = Math.abs(seconds);
    if (relative && abs < 86400) {
      if (abs < 45) return { text: 'just now', title: full };
      if (abs < 3600) return { text: relative.format(Math.round(seconds / 60), 'minute'), title: full };
      return { text: relative.format(Math.round(seconds / 3600), 'hour'), title: full };
    }
    return { text: date.toLocaleDateString(undefined, { dateStyle: 'medium' }), title: full };
  }

  function whenCell(iso, extraClass) {
    const w = fmtWhen(iso);
    return h('td', { class: 'when' + (extraClass ? ' ' + extraClass : ''), title: w.title || null }, w.text || '');
  }

  function fmtSeconds(ms) {
    return (ms / 1000).toFixed(ms < 10000 ? 1 : 0) + ' s';
  }

  function fmtEta(seconds) {
    if (!finite(seconds) || seconds <= 0) return '';
    if (seconds < 60) return 'about ' + Math.max(1, Math.round(seconds)) + ' s left';
    return 'about ' + plural(Math.round(seconds / 60), 'minute') + ' left';
  }

  // The fused rank score: small numbers, three decimals.
  function fmtScore(value) {
    if (!finite(value)) return '';
    return value >= 1 ? value.toFixed(2) : value.toFixed(3);
  }

  // Similarity and reranker scores: two decimals.
  function fmtRelevance(value) {
    return finite(value) ? value.toFixed(2) : '';
  }

  function pagesLabel(page, pageEnd) {
    if (page === null || page === undefined) return '';
    if (pageEnd !== null && pageEnd !== undefined && pageEnd !== page) return 'pages ' + page + ' to ' + pageEnd;
    return 'page ' + page;
  }

  function suffixOf(name) {
    const dot = name.lastIndexOf('.');
    return dot > 0 ? name.slice(dot).toLowerCase() : '';
  }

  function plural(n, one, many) {
    return n + ' ' + (n === 1 ? one : (many || one + 's'));
  }

  function spinner() {
    return h('span', { class: 'spinner spinner-small', 'aria-hidden': 'true' });
  }

  /* Puts `nodes` into `parent` in order, keeping nodes that are already there in place, so a row that did not
   * change keeps its focus and its armed state while its neighbours update. */
  function reconcile(parent, nodes) {
    const wanted = new Set(nodes);
    let cursor = parent.firstChild;
    for (const node of nodes) {
      while (cursor && !wanted.has(cursor)) {
        const next = cursor.nextSibling;
        cursor.remove();
        cursor = next;
      }
      if (node === cursor) {
        cursor = cursor.nextSibling;
        continue;
      }
      parent.insertBefore(node, cursor);
    }
    while (cursor) {
      const next = cursor.nextSibling;
      cursor.remove();
      cursor = next;
    }
  }

  /* Remembers which keyed control has the focus inside `parent`, and puts it back after a re-render. */
  function keepFocus(parent, render) {
    const active = document.activeElement;
    const holder = active && parent.contains(active) ? active.closest('[data-key]') : null;
    const memo = holder && active.dataset.action ? { key: holder.dataset.key, action: active.dataset.action } : null;
    render();
    if (memo && !parent.contains(active)) {
      const target = parent.querySelector('[data-key="' + CSS.escape(memo.key) + '"] [data-action="' + CSS.escape(memo.action) + '"]');
      if (target && !target.disabled) target.focus();
    }
  }

  // ------------------------------------------------------------------ API

  class ApiError extends Error {
    constructor(status, code, message, retryAfter) {
      super(message);
      this.status = status;
      this.code = code;
      this.retryAfter = retryAfter;
    }
  }

  async function api(method, path, body) {
    const options = { method, headers: { Accept: 'application/json' } };
    if (body !== undefined) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await fetch(path, options);
    } catch (e) {
      throw new ApiError(0, 'network', 'The rag-kb server cannot be reached. Check that it is still running.');
    }
    const text = await response.text();
    let data = null;
    if (text) {
      try { data = JSON.parse(text); } catch (e) { data = null; }
    }
    if (!response.ok) {
      const error = data && data.error ? data.error : {};
      const retry = Number(response.headers.get('Retry-After')) || null;
      throw new ApiError(response.status, error.code || 'error',
        error.message || 'The server answered with status ' + response.status + '.', retry);
    }
    return data;
  }

  // ------------------------------------------------------------------ copy and two-click buttons

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (e) {
      try {
        const area = h('textarea', { class: 'sr-only', readonly: true });
        area.value = text;
        document.body.append(area);
        area.select();
        const ok = document.execCommand('copy');
        area.remove();
        return ok;
      } catch (err) {
        return false;
      }
    }
  }

  function copyButton(text, label) {
    const idle = label || 'Copy';
    const button = h('button', { type: 'button', class: 'btn btn-ghost btn-small', 'aria-label': idle + ': ' + text }, idle);
    button.addEventListener('click', async () => {
      const ok = await copyText(text);
      button.textContent = ok ? 'Copied' : 'Select and copy';
      announce(ok ? 'Copied to the clipboard.' : 'Copy did not work. Select the text and copy it.');
      window.setTimeout(() => { button.textContent = idle; }, 1600);
    });
    return button;
  }

  function codeBlock(text, label) {
    return h('div', { class: 'code-copy' }, h('pre', {}, h('code', {}, text)), copyButton(text, label));
  }

  /* A destructive button that needs a second click within ARM_MS; it disarms itself after that. */
  function armButton(button, options) {
    let timer = null;
    const idle = options.idle;
    button.textContent = idle;
    const disarm = () => {
      window.clearTimeout(timer);
      button.classList.remove('armed');
      button.textContent = idle;
    };
    button.addEventListener('click', async () => {
      if (!button.classList.contains('armed')) {
        button.classList.add('armed');
        button.textContent = typeof options.armed === 'function' ? options.armed() : options.armed;
        announce(button.textContent + '. Click again within 4 seconds to confirm.');
        timer = window.setTimeout(disarm, ARM_MS);
        return;
      }
      disarm();
      button.disabled = true;
      try {
        await options.onConfirm();
      } finally {
        button.disabled = false;
      }
    });
    return button;
  }

  /* Shows a spinner and a label on a button while an action runs, and a longer note after two seconds. */
  function busyButton(button, label, slowLabel) {
    const idle = button.textContent;
    button.replaceChildren(spinner(), ' ' + label);
    button.setAttribute('aria-busy', 'true');
    const timer = window.setTimeout(() => {
      button.replaceChildren(spinner(), ' ' + slowLabel);
      announce(slowLabel);
    }, PENDING_NOTE_MS);
    return () => {
      window.clearTimeout(timer);
      button.removeAttribute('aria-busy');
      button.textContent = idle;
    };
  }

  // ------------------------------------------------------------------ markdown (escape first, then format)

  function citeButton(n) {
    return '<button type="button" class="cite" data-marker="' + n + '" aria-label="Show source ' + n + '">' + n + '</button>';
  }

  function formatInline(escaped, markers) {
    return escaped.split(/(`[^`]+`)/g).map((part, index) => {
      if (index % 2 === 1) return '<code>' + part.slice(1, -1) + '</code>';
      let text = part;
      text = text.replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, '<strong>$1</strong>');
      text = text.replace(/__(?=\S)([\s\S]*?\S)__/g, '<strong>$1</strong>');
      text = text.replace(/(^|[^*\w])\*(?=\S)([^*]*?\S)\*(?!\*)/g, '$1<em>$2</em>');
      text = text.replace(/(^|[^_\w])_(?=\S)([^_]*?\S)_(?![_\w])/g, '$1<em>$2</em>');
      text = text.replace(/\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]/g, (all, list) => {
        const numbers = list.split(',').map((s) => Number(s.trim()));
        if (!numbers.every((n) => markers.has(n))) return all;
        return numbers.map(citeButton).join('');
      });
      return text;
    }).join('');
  }

  function renderMarkdown(source, markers) {
    const lines = escapeHtml(String(source || '')).replace(/\r\n?/g, '\n').split('\n');
    const out = [];
    let paragraph = [];
    let list = null;
    let code = null;
    const inline = (s) => formatInline(s, markers);
    const flushParagraph = () => {
      if (paragraph.length) out.push('<p>' + inline(paragraph.join(' ')) + '</p>');
      paragraph = [];
    };
    const flushList = () => {
      if (!list) return;
      const start = list.type === 'ol' && list.start !== 1 ? ' start="' + list.start + '"' : '';
      out.push('<' + list.type + start + '>' + list.items.map((item) => '<li>' + inline(item) + '</li>').join('') + '</' + list.type + '>');
      list = null;
    };
    for (const raw of lines) {
      if (code) {
        if (/^\s*```/.test(raw)) {
          out.push('<pre><code>' + code.join('\n') + '</code></pre>');
          code = null;
        } else {
          code.push(raw);
        }
        continue;
      }
      if (/^\s*```/.test(raw)) { flushParagraph(); flushList(); code = []; continue; }
      const line = raw.replace(/\s+$/, '');
      if (!line.trim()) { flushParagraph(); flushList(); continue; }
      let m = /^\s{0,3}(#{1,6})\s+(.+?)\s*#*$/.exec(line);
      if (m) {
        flushParagraph(); flushList();
        const level = Math.min(6, m[1].length + 2);
        out.push('<h' + level + '>' + inline(m[2]) + '</h' + level + '>');
        continue;
      }
      if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { flushParagraph(); flushList(); out.push('<hr>'); continue; }
      m = /^\s*[-*+•]\s+(.*)$/.exec(line);
      if (m) {
        flushParagraph();
        if (!list || list.type !== 'ul') { flushList(); list = { type: 'ul', items: [] }; }
        list.items.push(m[1]);
        continue;
      }
      m = /^\s*(\d{1,3})[.)]\s+(.*)$/.exec(line);
      if (m) {
        flushParagraph();
        if (!list || list.type !== 'ol') { flushList(); list = { type: 'ol', items: [], start: Number(m[1]) }; }
        list.items.push(m[2]);
        continue;
      }
      if (list && /^\s{2,}\S/.test(raw)) {
        list.items[list.items.length - 1] += ' ' + line.trim();
        continue;
      }
      flushList();
      paragraph.push(line.trim().replace(/^&gt;\s?/, ''));
    }
    if (code) out.push('<pre><code>' + code.join('\n') + '</code></pre>');
    flushParagraph();
    flushList();
    return out.join('');
  }

  // ------------------------------------------------------------------ routing: #/c/<id>/<tab>, #/<tab>, #/answer/<id>

  function hashFor(collectionId, tab) {
    return collectionId ? '#/c/' + enc(collectionId) + '/' + tab : '#/' + tab;
  }

  function parseHash(hash) {
    const raw = String(hash || '').replace(/^#\/?/, '');
    if (!raw) return null;
    const parts = raw.split('/').map((part) => {
      try { return decodeURIComponent(part); } catch (e) { return part; }
    });
    if (parts[0] === 'c' && parts[1]) return { cid: parts[1], tab: TABS.includes(parts[2]) ? parts[2] : 'documents' };
    if (parts[0] === 'answer' && parts[1]) return { answer: parts[1] };
    // The older forms: #ask/<answer id> and #<tab>.
    if (parts[0] === 'ask' && parts[1]) return { answer: parts[1] };
    if (TABS.includes(parts[0])) return { tab: parts[0] };
    return null;
  }

  /* mode: 'push' adds a history entry (Back returns here), 'replace' corrects the address, 'none' only records it. */
  function setRoute(hash, mode) {
    if (mode !== 'none' && location.hash !== hash) {
      if (mode === 'replace') history.replaceState(null, '', hash);
      else history.pushState(null, '', hash);
    }
    lastHash = location.hash;
  }

  async function onLocationChange() {
    if (location.hash === lastHash) return;
    lastHash = location.hash;
    const route = parseHash(location.hash);
    if (!route) return;
    if (route.answer) {
      // 'replace' turns the older #ask/<id> into #/answer/<id>; the current form is left as it is.
      await loadStoredAnswer(route.answer, 'replace');
      return;
    }
    if (route.tab && route.tab !== state.tab) selectTab(route.tab, { route: 'none' });
    if (route.cid && route.cid !== state.current && state.collections.some((c) => c.id === route.cid)) {
      await selectCollection(route.cid, false, 'none');
    }
    // An older or partial address (#ask, #/help, an unknown collection) is corrected in place.
    if (location.hash === lastHash) setRoute(hashFor(state.current, state.tab), 'replace');
  }

  // ------------------------------------------------------------------ tabs

  function selectTab(name, options) {
    const opts = options || {};
    if (!TABS.includes(name)) name = 'documents';
    for (const tab of TABS) {
      const on = tab === name;
      const button = $('tab-' + tab);
      button.setAttribute('aria-selected', String(on));
      button.tabIndex = on ? 0 : -1;
      $('panel-' + tab).hidden = !on;
    }
    state.tab = name;
    store('tab', name);
    if (opts.focus) $('tab-' + name).focus();
    setRoute(hashFor(state.current, name), opts.route || 'push');
    if (name === 'settings') {
      renderSettings(true);
      renderFeedbackPanel();
    }
    if (name === 'connect') renderConnect();
    if (name === 'ask') loadProviders(false);
  }

  function setupTabs() {
    const list = document.querySelector('[role="tablist"]');
    list.addEventListener('click', (event) => {
      const tab = event.target.closest('[role="tab"]');
      if (tab) selectTab(tab.id.replace('tab-', ''));
    });
    list.addEventListener('keydown', (event) => {
      const index = TABS.indexOf(state.tab);
      let next = null;
      if (event.key === 'ArrowRight') next = TABS[(index + 1) % TABS.length];
      else if (event.key === 'ArrowLeft') next = TABS[(index - 1 + TABS.length) % TABS.length];
      else if (event.key === 'Home') next = TABS[0];
      else if (event.key === 'End') next = TABS[TABS.length - 1];
      if (next) {
        event.preventDefault();
        selectTab(next, { focus: true });
      }
    });
  }

  function isTyping(target) {
    return target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
  }

  // ------------------------------------------------------------------ health, banner, status bar

  /* The last known health and provider states, shown at once on a reload until the fresh ones arrive. */
  function restoreCached() {
    const health = recallJson('health');
    if (health && typeof health === 'object' && health.status) {
      state.health = health;
      renderHealth();
    }
    const providers = recallJson('providers');
    if (Array.isArray(providers) && providers.length) {
      state.providers = providers;
      renderProviderBadge();
      renderProviderControls();
    }
  }

  async function loadHealth() {
    try {
      state.health = await api('GET', '/api/health');
      state.healthFresh = true;
      state.offline = false;
      storeJson('health', state.health);
    } catch (e) {
      state.offline = true;
    }
    renderHealth();
  }

  function acceptedTypes() {
    const list = (state.health && state.health.supported_types) || [];
    const types = new Set(list.map((t) => String(t).toLowerCase()));
    if (types.size) types.add(ARCHIVE);
    return types;
  }

  function renderHealth() {
    const badge = $('health-badge');
    const banner = $('banner');
    const health = state.health;
    if (state.offline) {
      badge.className = 'badge badge-err';
      badge.textContent = 'Offline';
      badge.title = 'The server cannot be reached.';
      banner.className = 'banner';
      banner.textContent = 'The rag-kb server cannot be reached. Check that "ragkb serve" is still running, then reload.';
      banner.hidden = false;
    } else if (health) {
      const ok = health.status === 'ok';
      badge.className = 'badge ' + (ok ? 'badge-ok' : 'badge-warn');
      badge.textContent = ok ? 'Healthy' : 'Degraded';
      badge.title = ok ? 'All parts are working.' : 'Some parts are not working: ' + (health.problems || []).join('; ');
      const storeDown = health.store && (health.store.metadata !== 'ok' || health.store.vector !== 'ok');
      if (storeDown && state.healthFresh) {
        banner.className = 'banner';
        banner.textContent = 'The document store is not available. Uploads, search and answers can fail until it is back. See Settings for details.';
        banner.hidden = false;
      } else if (!banner.classList.contains('warn')) {
        banner.hidden = true;
      }
    }
    if (health) {
      $('status-version').textContent = health.knowledge_version || 'none yet';
      $('status-data').textContent = health.data_dir || '';
      $('status-data').title = health.data_dir || '';
      $('status-app-version').textContent = health.version || '';
      const types = Array.from(acceptedTypes());
      $('dropzone-hint').textContent = (types.length ? types.join(' ') + ', ' : '') + 'up to ' + health.max_upload_mb + ' MB each';
      $('file-input').accept = types.join(',');
      const topk = $('topk-input');
      if (!topk.dataset.touched && !recall('topk', '')) topk.value = String(health.top_k || 6);
    }
    if (state.healthFresh || state.offline) {
      $('status-checked').textContent = 'checked ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    renderProviderBadge();
  }

  // ------------------------------------------------------------------ providers

  async function loadProviders(refresh) {
    if (!refresh && state.providers.length && state.providersLoadedAt && Date.now() - state.providersLoadedAt < 30000) {
      return;
    }
    try {
      state.providers = await api('GET', '/api/providers' + (refresh ? '?refresh=1' : ''));
      state.providersError = null;
      state.providersLoadedAt = Date.now();
      storeJson('providers', state.providers);
    } catch (e) {
      state.providers = [];
      state.providersError = e.message;
    }
    renderProviderBadge();
    renderProviderControls();
  }

  function providerStatus(name) {
    return state.providers.find((p) => p.name === name) || null;
  }

  function providerReady(name) {
    const status = providerStatus(name);
    return Boolean(status && status.installed && status.signed_in);
  }

  function autoProvider() {
    const setting = state.health ? state.health.provider_setting : 'auto';
    if (setting && setting !== 'auto') return setting;
    for (const name of AUTO_ORDER) if (providerReady(name)) return name;
    return 'extractive';
  }

  function chosenProvider() {
    const value = $('provider-select').value || 'auto';
    return value === 'auto' ? autoProvider() : value;
  }

  function renderProviderBadge() {
    const badge = $('provider-badge');
    if (!state.providers.length) {
      // Before the first answer from the server there is nothing better to show; afterwards the cache is used.
      badge.className = 'badge badge-muted';
      badge.textContent = state.providersError ? 'Providers unknown' : 'Provider';
      badge.title = state.providersError || 'Checking the answer providers';
      return;
    }
    const name = autoProvider();
    badge.className = 'badge ' + (name === 'extractive' ? 'badge-warn' : 'badge-info');
    badge.textContent = 'Answers: ' + (PROVIDER_LABELS[name] || name).replace(' (no AI)', '');
    badge.title = name === 'extractive'
      ? 'No AI provider is signed in, so answers are extractive digests. See Settings.'
      : 'Answers are written by ' + providerLabel(name) + ' on your own subscription.';
  }

  function fixFor(status) {
    if (!status || (status.installed && status.signed_in)) return null;
    return status.fix_command || null;
  }

  function renderProviderControls() {
    const select = $('provider-select');
    const saved = recall('provider', 'auto');
    const previous = select.value || saved;
    select.replaceChildren();
    select.append(h('option', { value: 'auto' }, 'Auto (' + (PROVIDER_LABELS[autoProvider()] || autoProvider()) + ')'));
    for (const status of state.providers) {
      const ready = status.installed && status.signed_in;
      const reason = !status.installed ? 'not installed' : 'not signed in';
      const fix = fixFor(status);
      select.append(h('option', {
        value: status.name,
        disabled: !ready,
        title: ready ? (status.detail || '') : (fix ? reason + '. Run: ' + fix : reason),
      }, providerLabel(status.name) + (ready ? '' : ' (' + reason + ')')));
    }
    const options = Array.from(select.options);
    const keep = options.find((o) => o.value === previous && !o.disabled);
    select.value = keep ? previous : 'auto';

    const notes = $('provider-notes');
    notes.replaceChildren();
    if (state.providersError) {
      notes.append(h('div', { class: 'provider-note error-text' }, 'Providers could not be checked: ' + state.providersError));
    }
    for (const status of state.providers) {
      const fix = fixFor(status);
      if (!fix) continue;
      const reason = status.installed ? 'is not signed in' : 'is not installed';
      notes.append(h('div', { class: 'provider-note' },
        h('span', {}, providerLabel(status.name) + ' ' + reason + '. Run'),
        h('code', {}, fix),
        copyButton(fix, 'Copy')));
    }
    renderModelControl();
  }

  /* The Model menu stays in place for every provider; it is disabled when the provider has no models. */
  function renderModelControl() {
    const select = $('model-select');
    const provider = chosenProvider();
    const status = providerStatus(provider);
    let models = status && Array.isArray(status.models) ? status.models.slice() : [];
    let fallback = '';
    if (provider === 'claude_code') {
      if (!models.length) models = CLAUDE_MODELS.slice();
      fallback = (state.health && state.health.claude_model) || 'sonnet';
    } else if (provider === 'codex') {
      fallback = (state.health && state.health.codex_model) || 'the CLI default';
    }
    if (provider === 'extractive' || !models.length) {
      const label = provider === 'codex' ? 'Default (' + fallback + ')' : 'no model';
      select.replaceChildren(h('option', { value: '' }, label));
      select.value = '';
      select.disabled = true;
      select.title = provider === 'extractive'
        ? 'Extractive mode uses no model.'
        : providerLabel(provider) + ' lists no models here; its default is used.';
      return;
    }
    const previous = select.value || recall('model.' + provider, '');
    select.replaceChildren(h('option', { value: '' }, 'Default (' + fallback + ')'));
    for (const model of models) select.append(h('option', { value: model }, model));
    select.value = models.includes(previous) ? previous : '';
    select.disabled = false;
    select.title = '';
  }

  // ------------------------------------------------------------------ counts: searchable documents first

  function statusCounts(byStatus) {
    const b = byStatus || {};
    const n = (key) => Number(b[key]) || 0;
    return { ready: n('ready'), processing: n('queued') + n('processing'), failed: n('failed'), cancelled: n('cancelled') };
  }

  function countsFromDocs(docs) {
    const byStatus = {};
    for (const doc of docs) byStatus[doc.status] = (byStatus[doc.status] || 0) + 1;
    return statusCounts(byStatus);
  }

  /* GET /api/collections: `documents` (ready), `documents_total` and `documents_by_status` on every row. */
  function collectionCounts(collection) {
    if (collection.id === state.current && state.docsLoaded) return countsFromDocs(state.docs);
    return statusCounts(collection.documents_by_status);
  }

  /* GET /api/collections/{cid}/stats: `documents_ready` and `documents_by_status`. */
  function statsCounts(stats) {
    const counts = statusCounts(stats.documents_by_status);
    counts.ready = Number(stats.documents_ready) || 0;
    return counts;
  }

  /* The secondary note: what is not searchable yet, or will not be. */
  function countsNote(counts) {
    const parts = [];
    if (counts.processing) parts.push(counts.processing + ' processing');
    if (counts.failed) parts.push(counts.failed + ' failed');
    if (counts.cancelled) parts.push(counts.cancelled + ' cancelled');
    return parts.join(', ');
  }

  function searchableText(counts) {
    const note = countsNote(counts);
    return plural(counts.ready, 'searchable document') + (note ? ' (' + note + ')' : '');
  }

  // ------------------------------------------------------------------ collections

  function currentCollection() {
    return state.collections.find((c) => c.id === state.current) || null;
  }

  async function loadCollections(preferId, route) {
    try {
      state.collections = await api('GET', '/api/collections');
      state.collectionsError = null;
    } catch (e) {
      state.collections = [];
      state.collectionsError = e.message;
    }
    let target = preferId || state.current || recall('collection', '');
    if (!state.collections.some((c) => c.id === target)) target = state.collections.length ? state.collections[0].id : null;
    renderCollections();
    await selectCollection(target, true, route || 'replace');
  }

  let collectionsSig = '';

  function renderCollections() {
    const list = $('collection-list');
    // The list is rebuilt only when something in it changed, so a focused button keeps its focus during polling.
    const sig = JSON.stringify([state.current, state.collectionsError,
      state.collections.map((c) => [c.id, c.name, c.description, collectionCounts(c)])]);
    if (sig === collectionsSig && list.childElementCount === state.collections.length) return;
    collectionsSig = sig;
    list.replaceChildren();
    for (const collection of state.collections) {
      const current = collection.id === state.current;
      const counts = collectionCounts(collection);
      const note = countsNote(counts);
      const nameId = 'collection-label-' + collection.id;
      const descId = 'collection-desc-' + collection.id;
      const description = [collection.description, searchableText(counts)].filter(Boolean).join('. ');
      // The accessible name is the collection's name alone; the description and counts are its description.
      list.append(h('li', {},
        h('button', {
          type: 'button',
          class: 'collection-item',
          'aria-current': current ? 'true' : null,
          'aria-labelledby': nameId,
          'aria-describedby': descId,
          title: collection.description || null,
          onclick: () => selectCollection(collection.id, false),
        },
        h('span', { class: 'collection-text' },
          h('span', { class: 'collection-name', id: nameId }, collection.name),
          note ? h('span', { class: 'collection-note', 'aria-hidden': 'true' }, note) : null),
        h('span', { class: 'count', 'aria-hidden': 'true' }, String(counts.ready)),
        h('span', { class: 'sr-only', id: descId }, description))));
    }
    const empty = $('collection-empty');
    empty.hidden = state.collections.length > 0;
    empty.textContent = state.collectionsError
      ? 'Collections could not be loaded: ' + state.collectionsError
      : 'No collections yet. Create one below.';
  }

  async function selectCollection(id, force, route) {
    if (id === state.current && !force) return;
    const changed = id !== state.current;
    state.current = id || null;
    store('collection', id || '');
    if (changed) {
      state.docs = [];
      state.docsLoaded = false;
      state.docsSig = '';
      state.stats = null;
      state.summary = null;
      state.docsError = null;
      state.expanded.clear();
      state.pending.clear();
      clearAnswer();
    }
    setRoute(hashFor(state.current, state.tab), route || 'push');
    renderCollections();
    renderCollectionHeader();
    renderDocs();
    renderSummary();
    renderUploads();
    renderAskAvailability();
    if (state.current) await loadDocs();
    if (changed && state.tab === 'settings') {
      renderSettings(false);
      renderFeedbackPanel();
    }
  }

  function renderCollectionHeader() {
    const collection = currentCollection();
    const title = $('collection-title');
    const meta = $('collection-meta');
    const remove = $('collection-delete');
    if (!collection) {
      title.textContent = 'Knowledge base';
      meta.textContent = state.collections.length ? 'Select a collection in the sidebar.' : 'Create a collection to get started.';
      remove.hidden = true;
      document.title = 'rag-kb';
      return;
    }
    title.textContent = collection.name;
    document.title = collection.name + ' | rag-kb';
    const parts = [];
    if (collection.description) parts.push(collection.description);
    const counts = state.docsLoaded ? countsFromDocs(state.docs) : (state.stats ? statsCounts(state.stats) : null);
    if (counts) parts.push(searchableText(counts));
    if (state.stats) {
      parts.push(plural(state.stats.chunks || 0, 'passage'));
      if (state.stats.last_indexed_at) parts.push('last indexed ' + fmtWhen(state.stats.last_indexed_at).text);
    }
    meta.textContent = parts.join('  |  ');
    remove.hidden = false;
  }

  function setupCollectionForm() {
    const form = $('collection-form');
    const error = $('collection-form-error');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      error.hidden = true;
      const name = $('collection-name').value.trim();
      const description = $('collection-desc').value.trim();
      if (!name) {
        error.textContent = 'Give the collection a name.';
        error.hidden = false;
        $('collection-name').focus();
        return;
      }
      const button = form.querySelector('button[type="submit"]');
      button.disabled = true;
      try {
        const created = await api('POST', '/api/collections', { name, description });
        form.reset();
        announce('Collection ' + created.name + ' created.');
        await loadCollections(created.id, 'push');
        selectTab('documents', { route: 'replace' });
      } catch (e) {
        error.textContent = e.message;
        error.hidden = false;
      } finally {
        button.disabled = false;
      }
    });

    const remove = $('collection-delete');
    armButton(remove, {
      idle: 'Delete collection',
      armed: () => {
        const n = state.docsLoaded ? state.docs.length : (state.stats ? Number(state.stats.documents_total) || 0 : 0);
        return n ? 'Click again to delete it and ' + plural(n, 'document') : 'Click again to delete';
      },
      onConfirm: async () => {
        const collection = currentCollection();
        if (!collection) return;
        const done = busyButton(remove, 'Deleting', 'Still deleting: a large collection takes a few seconds');
        try {
          const result = await api('DELETE', '/api/collections/' + enc(collection.id));
          announce('Collection ' + collection.name + (result && result.pending ? ': ' + result.message : ' deleted.'));
          state.current = null;
          await loadCollections(null, 'replace');
          loadHealth();
        } catch (e) {
          showBannerMessage('The collection could not be deleted: ' + e.message);
        } finally {
          done();
        }
      },
    });
  }

  function showBannerMessage(text) {
    const banner = $('banner');
    banner.className = 'banner warn';
    banner.textContent = text;
    banner.hidden = false;
    window.setTimeout(() => {
      if (banner.textContent === text) {
        banner.hidden = true;
        banner.className = 'banner';
      }
    }, 8000);
  }

  // ------------------------------------------------------------------ jobs: stage, progress, cancel

  /* Queued while the job waits for a worker, processing once a worker has started it. */
  function jobDisplayStatus(job) {
    const status = job.status;
    if (status === 'ready' || status === 'failed' || status === 'cancelled') return status;
    if (job.cancel_requested) return 'cancelling';
    return status === 'processing' || WORK_STAGES.has(job.stage) ? 'processing' : 'queued';
  }

  function jobProgressText(job) {
    const status = jobDisplayStatus(job);
    if (status === 'queued') return 'Waiting for a free worker';
    if (status === 'cancelling') return 'Cancelling: it stops after the current step';
    if (status !== 'processing') return '';
    const stage = WORK_STAGES.has(job.stage) ? job.stage : null;
    let text;
    if (stage === 'embedding' && job.chunks_total) {
      text = 'Embedding ' + (job.chunks_done || 0) + ' of ' + plural(job.chunks_total, 'passage');
    } else if (stage === 'writing' && job.chunks_total) {
      text = 'Writing ' + plural(job.chunks_total, 'passage') + ' to the index';
    } else {
      text = stage ? STAGE_LABELS[stage] : 'Processing';
    }
    const eta = fmtEta(job.eta_seconds);
    return eta ? text + ', ' + eta : text;
  }

  /* Overall progress of a job in percent, or null when the stage is unknown (an indeterminate bar). */
  function jobPercent(job) {
    const status = jobDisplayStatus(job);
    if (status === 'ready' || status === 'failed' || status === 'cancelled') return 100;
    if (status === 'queued') return 0;
    if (job.stage === 'parsing') return 5;
    if (job.stage === 'chunking') return 15;
    if (job.stage === 'embedding') {
      const share = job.chunks_total ? Math.min(1, (job.chunks_done || 0) / job.chunks_total) : 0;
      return Math.round(20 + 75 * share);
    }
    if (job.stage === 'writing') return 97;
    return null;
  }

  function progressBar(percent, label, extraClass) {
    const bar = h('progress', { max: '100', class: extraClass || null, 'aria-label': label });
    if (percent !== null && percent !== undefined) bar.value = percent;
    return bar;
  }

  async function refreshJobs(cid) {
    try {
      const jobs = await api('GET', '/api/collections/' + enc(cid) + '/jobs?limit=500');
      if (!Array.isArray(jobs)) return null;
      for (const job of jobs) if (job && job.id) jobCache.set(job.id, job);
      return jobs;
    } catch (e) {
      return null;
    }
  }

  function activeJobFor(docId) {
    let best = null;
    for (const job of jobCache.values()) {
      if (job.doc_id !== docId || !JOB_ACTIVE.has(job.status)) continue;
      if (!best || String(job.created_at) > String(best.created_at)) best = job;
    }
    return best;
  }

  async function cancelJob(jobId, button) {
    if (button) button.disabled = true;
    try {
      const job = await api('POST', '/api/jobs/' + enc(jobId) + '/cancel');
      if (job && job.id) {
        jobCache.set(job.id, job);
        for (const item of state.uploads) if (item.jobId === job.id) applyJob(item, job);
      }
      announce('Cancel requested.');
      ensureJobPolling();
    } catch (e) {
      showBannerMessage('The job could not be cancelled: ' + e.message);
      if (button) button.disabled = false;
    }
    renderUploads();
    scheduleDocs(0);
  }

  // ------------------------------------------------------------------ documents

  function scheduleDocs(delay) {
    window.clearTimeout(docsTimer);
    docsTimer = window.setTimeout(loadDocs, delay);
  }

  function jobSignature(job) {
    return job ? [job.id, job.status, job.stage, job.chunks_done, job.chunks_total, job.eta_seconds, job.cancel_requested].join(':') : '';
  }

  async function loadDocs() {
    const cid = state.current;
    if (!cid) return;
    window.clearTimeout(docsTimer);
    docsTimer = null;
    try {
      const base = '/api/collections/' + enc(cid);
      const [docs, stats, summary] = await Promise.all([
        api('GET', base + '/documents'),
        api('GET', base + '/stats').catch(() => null),
        api('GET', base + '/jobs/summary').catch(() => null),
      ]);
      if (cid !== state.current) return;
      if (docs.some((d) => JOB_ACTIVE.has(d.status))) await refreshJobs(cid);
      if (cid !== state.current) return;
      state.docsError = null;
      state.stats = stats;
      state.summary = summary && typeof summary === 'object' && !Array.isArray(summary) ? summary : null;
      state.docsLoaded = true;
      const sig = JSON.stringify(docs) + docs.map((d) => jobSignature(activeJobFor(d.id))).join('|');
      if (sig !== state.docsSig) {
        state.docs = docs;
        state.docsSig = sig;
        renderDocs();
        renderSuggestions();
      }
      renderSummary();
      renderCollections();
      renderCollectionHeader();
    } catch (e) {
      if (cid !== state.current) return;
      if (e.status === 404) {
        await loadCollections(null, 'replace');
        return;
      }
      state.docsError = e.message;
      state.docsSig = '';
      renderDocs();
    }
    if (state.current === cid && state.docs.some((d) => JOB_ACTIVE.has(d.status))) {
      window.clearTimeout(docsTimer);
      docsTimer = window.setTimeout(loadDocs, DOCS_POLL_MS);
    }
  }

  function chip(status, label) {
    return h('span', { class: 'chip chip-' + (CHIP_TONES[status] || status) }, label || STATUS_LABELS[status] || status);
  }

  /* The queue line: "42 ready, 3 failed, 1 replaced, 2 duplicates", from the job summary when the server has it. */
  function renderSummary() {
    const box = $('docs-summary');
    const summary = state.summary;
    const parts = [];
    let lead = '';
    if (state.current && summary && finite(summary.total)) {
      lead = 'Uploads: ';
      const add = (key, one, many, tone) => {
        const n = Number(summary[key]) || 0;
        if (n) parts.push([plural(n, one, many), tone]);
      };
      add('processing', 'processing', 'processing', 'info');
      add('queued', 'queued', 'queued', '');
      add('ready', 'ready', 'ready', '');
      add('failed', 'failed', 'failed', 'err');
      add('cancelled', 'cancelled', 'cancelled', '');
      add('replaced', 'replaced', 'replaced', '');
      add('duplicates', 'duplicate', 'duplicates', '');
    } else if (state.current && state.docs.length) {
      lead = 'Documents: ';
      const counts = countsFromDocs(state.docs);
      if (counts.processing) parts.push([counts.processing + ' processing', 'info']);
      parts.push([counts.ready + ' ready', '']);
      if (counts.failed) parts.push([counts.failed + ' failed', 'err']);
      if (counts.cancelled) parts.push([counts.cancelled + ' cancelled', '']);
    }
    box.replaceChildren();
    box.hidden = !parts.length;
    if (!parts.length) return;
    box.append(h('span', { class: 'muted' }, lead));
    parts.forEach(([text, tone], i) => {
      if (i) box.append(h('span', { class: 'muted' }, ', '));
      box.append(h('span', { class: tone === 'err' ? 'error-text' : (tone === 'info' ? 'info-text' : null) }, text));
    });
  }

  function visibleColumns() {
    let n = 0;
    for (const th of document.querySelectorAll('#docs-table thead th')) {
      if (getComputedStyle(th).display !== 'none') n += 1;
    }
    return n || 7;
  }

  /* Columns drop out as the table gets narrower (see the container queries in styles.css); full-width cells
   * follow so the passages drawer always spans the whole table. */
  function setupDocsLayout() {
    const wrap = document.querySelector('.table-wrap');
    if (!wrap || typeof ResizeObserver === 'undefined') return;
    let last = 0;
    new ResizeObserver(() => {
      const n = visibleColumns();
      if (n === last) return;
      last = n;
      for (const td of document.querySelectorAll('#docs-tbody td.span-all')) td.colSpan = n;
    }).observe(wrap);
  }

  function rowSignature(doc) {
    const pending = state.pending.get(doc.id);
    return JSON.stringify([doc, jobSignature(JOB_ACTIVE.has(doc.status) ? activeJobFor(doc.id) : null),
      pending ? pending.action + (pending.slow ? '+' : '') : '', state.expanded.has(doc.id)]);
  }

  function renderDocs() {
    const has = Boolean(state.current);
    $('docs-no-collection').hidden = has;
    $('docs-body').hidden = !has;
    const body = $('docs-tbody');
    if (!has) {
      body.replaceChildren();
      rowCache = new Map();
      drawerCache = new Map();
      return;
    }
    const span = visibleColumns();
    if (state.docsError) {
      body.replaceChildren(h('tr', {}, h('td', { colspan: span, class: 'span-all table-message error-text' },
        'The documents could not be loaded: ' + state.docsError)));
      rowCache = new Map();
      return;
    }
    if (!state.docs.length) {
      body.replaceChildren(h('tr', {}, h('td', { colspan: span, class: 'span-all table-message' },
        h('strong', {}, state.docsLoaded ? 'No documents yet. ' : 'Loading the documents. '),
        state.docsLoaded ? 'Drop files above, choose files or add a folder to build this knowledge base.' : '')));
      rowCache = new Map();
      return;
    }
    const nextRows = new Map();
    const nextDrawers = new Map();
    const nodes = [];
    for (const doc of state.docs) {
      const sig = rowSignature(doc);
      const cached = rowCache.get(doc.id);
      const row = cached && cached.sig === sig ? cached.row : docRow(doc);
      nextRows.set(doc.id, { sig, row });
      nodes.push(row);
      if (state.expanded.has(doc.id)) {
        const key = doc.id + '@' + (doc.indexed_at || '');
        const drawer = drawerCache.get(key) || drawerRow(doc, span);
        drawer.firstChild.colSpan = span;
        nextDrawers.set(key, drawer);
        nodes.push(drawer);
      }
    }
    rowCache = nextRows;
    drawerCache = nextDrawers;
    keepFocus(body, () => reconcile(body, nodes));
  }

  function actionButton(action, label, props) {
    return h('button', Object.assign({ type: 'button', class: 'btn btn-ghost btn-small', dataset: { action } }, props || {}), label);
  }

  function docRow(doc) {
    const pending = state.pending.get(doc.id) || null;
    const job = JOB_ACTIVE.has(doc.status) ? activeJobFor(doc.id) : null;
    const status = job ? jobDisplayStatus(job) : doc.status;
    const expanded = state.expanded.has(doc.id);
    const label = doc.source_path || doc.filename || '';
    const indexed = fmtWhen(doc.indexed_at);

    const sub = h('div', { class: 'doc-sub' },
      label ? h('a', {
        class: 'doc-file', href: '/api/documents/' + enc(doc.id) + '/file', download: doc.filename || '',
        title: 'Download the original file', 'aria-label': 'Download ' + label,
      }, label) : null,
      doc.media_type ? h('span', {}, fmtType(doc.media_type)) : null,
      finite(doc.size_bytes) ? h('span', {}, fmtBytes(doc.size_bytes)) : null,
      doc.pages ? h('span', { class: 'sub-pages' }, plural(doc.pages, 'page')) : null,
      doc.uploaded_at ? h('span', { class: 'sub-uploaded' }, 'uploaded ' + fmtWhen(doc.uploaded_at).text) : null,
      doc.indexed_at ? h('span', { class: 'sub-indexed' }, 'indexed ' + indexed.text) : null);

    let progressText = null;
    if (job) progressText = jobProgressText(job);
    else if (doc.status === 'queued') progressText = 'Waiting for a free worker';
    else if (doc.status === 'processing') progressText = 'Processing';

    const titleCell = h('td', { class: 'col-title' },
      h('div', { class: 'doc-title' }, doc.title),
      sub,
      progressText ? h('div', { class: 'doc-progress' }, progressText) : null,
      doc.status === 'failed' ? h('div', { class: 'doc-error' }, doc.error || 'Indexing failed.') : null,
      doc.status === 'cancelled' ? h('div', { class: 'doc-note' }, doc.error || 'Indexing was cancelled.') : null);

    const statusCell = h('td', { class: 'col-status' }, chip(status),
      job && status !== 'queued' ? progressBar(jobPercent(job), 'Indexing progress for ' + doc.title, 'row-progress') : null);

    const actions = h('div', { class: 'row-actions' });
    if (pending) {
      actions.append(pendingView(pending));
    } else if (doc.status === 'ready') {
      actions.append(
        actionButton('passages', expanded ? 'Hide passages' : 'Passages', {
          'aria-expanded': String(expanded), 'aria-controls': 'drawer-' + doc.id, onclick: () => toggleDrawer(doc.id),
        }),
        actionButton('reindex', 'Re-index', {
          title: 'Parse and embed again from the kept original', onclick: () => reindexDocument(doc),
        }),
        deleteButton(doc, 'Delete', 'Confirm delete', 'delete'));
    } else if (doc.status === 'failed' || doc.status === 'cancelled') {
      actions.append(
        actionButton('retry', 'Retry', { title: 'Index it again from the kept original', onclick: () => retryDocument(doc) }),
        deleteButton(doc, 'Dismiss', 'Confirm dismiss', 'dismiss'));
    } else {
      if (job && !job.cancel_requested) {
        actions.append(actionButton('cancel', 'Cancel', {
          title: 'Stop indexing this document', onclick: (event) => cancelJob(job.id, event.currentTarget),
        }));
      }
      actions.append(deleteButton(doc, 'Delete', 'Confirm delete', 'delete'));
    }

    return h('tr', { dataset: { doc: doc.id, key: doc.id } },
      statusCell,
      titleCell,
      h('td', { class: 'col-pages num' }, doc.pages === null || doc.pages === undefined ? '' : String(doc.pages)),
      h('td', { class: 'col-chunks num' }, doc.status === 'ready' ? String(doc.chunk_count) : ''),
      whenCell(doc.uploaded_at, 'col-uploaded'),
      whenCell(doc.indexed_at, 'col-indexed'),
      h('td', { class: 'col-actions' }, actions));
  }

  function deleteButton(doc, idle, armed, action) {
    const button = h('button', { type: 'button', class: 'btn btn-danger-ghost btn-small', dataset: { action } });
    return armButton(button, {
      idle,
      armed,
      onConfirm: () => withPending(doc, action, () => deleteDocument(doc, action)),
    });
  }

  const PENDING_LABELS = { delete: 'Deleting', dismiss: 'Removing', reindex: 'Starting re-index', retry: 'Retrying' };

  function pendingView(pending) {
    return h('div', { class: 'pending', role: 'status' },
      spinner(),
      h('span', {}, PENDING_LABELS[pending.action] || 'Working'),
      pending.slow ? h('span', { class: 'pending-note' }, 'Still working: a large document takes a few seconds.') : null);
  }

  /* Shows the row's pending state while `run` works, with a note when it takes longer than two seconds. */
  async function withPending(doc, action, run) {
    const entry = { action, slow: false };
    state.pending.set(doc.id, entry);
    renderDocs();
    const timer = window.setTimeout(() => {
      if (state.pending.get(doc.id) !== entry) return;
      entry.slow = true;
      renderDocs();
    }, PENDING_NOTE_MS);
    try {
      return await run();
    } finally {
      window.clearTimeout(timer);
      if (state.pending.get(doc.id) === entry) state.pending.delete(doc.id);
      renderDocs();
    }
  }

  function toggleDrawer(docId) {
    if (state.expanded.has(docId)) state.expanded.delete(docId);
    else state.expanded.add(docId);
    renderDocs();
    const button = document.querySelector('tr[data-doc="' + CSS.escape(docId) + '"] [data-action="passages"]');
    if (button) button.focus();
  }

  async function loadChunks(docId, version) {
    const key = docId + '@' + (version || '');
    if (!chunkCache.has(key)) {
      chunkCache.set(key, api('GET', '/api/documents/' + enc(docId) + '/chunks').catch((e) => {
        chunkCache.delete(key);
        throw e;
      }));
    }
    return chunkCache.get(key);
  }

  function drawerRow(doc, span) {
    const cell = h('td', { colspan: span, class: 'span-all' }, h('p', { class: 'muted small' }, 'Loading passages...'));
    const row = h('tr', { class: 'drawer-row', id: 'drawer-' + doc.id }, cell);
    loadChunks(doc.id, doc.indexed_at).then((chunks) => {
      cell.replaceChildren(
        h('div', { class: 'drawer-head' }, h('strong', {}, plural(chunks.length, 'passage')), ' of ', doc.title, ', in reading order'),
        h('ol', { class: 'passage-list' }, chunks.map((c) => h('li', { class: 'passage' },
          h('div', { class: 'passage-meta' },
            h('span', { class: 'badge badge-muted plain' }, '#' + c.ordinal),
            c.section ? h('span', {}, c.section) : null,
            pagesLabel(c.page_start !== null && c.page_start !== undefined ? c.page_start : c.page, c.page_end)
              ? h('span', {}, pagesLabel(c.page_start !== null && c.page_start !== undefined ? c.page_start : c.page, c.page_end)) : null,
            c.token_estimate ? h('span', {}, 'about ' + c.token_estimate + ' tokens') : null),
          h('p', { class: 'passage-text' }, c.text)))));
    }).catch((e) => {
      cell.replaceChildren(h('p', { class: 'error-text small' }, 'The passages could not be loaded: ' + e.message));
    });
    return row;
  }

  function trackJob(job, base) {
    const item = Object.assign({ status: 'queued', progress: 100, note: null }, base);
    applyJob(item, job);
    addUploadItem(item);
    ensureJobPolling();
  }

  async function reindexDocument(doc) {
    await withPending(doc, 'reindex', async () => {
      try {
        const job = await api('POST', '/api/documents/' + enc(doc.id) + '/reindex');
        trackJob(job, { name: 'Re-index: ' + doc.title, size: doc.size_bytes, collectionId: doc.collection_id, kind: 'reindex' });
        await loadDocs();
      } catch (e) {
        showBannerMessage('Re-index could not start: ' + e.message);
      }
    });
  }

  async function retryDocument(doc) {
    await withPending(doc, 'retry', async () => {
      let job = null;
      try {
        job = await api('POST', '/api/documents/' + enc(doc.id) + '/retry');
      } catch (e) {
        if (e.status !== 409) {
          showBannerMessage('Retry could not start: ' + e.message);
          return;
        }
        // Retry is for failed rows only; a cancelled row answers 409, and re-indexing it from the kept
        // original does the same thing.
        try {
          job = await api('POST', '/api/documents/' + enc(doc.id) + '/reindex');
        } catch (err) {
          showBannerMessage('Retry could not start: ' + err.message);
          return;
        }
      }
      if (job && job.id) {
        trackJob(job, { name: 'Retry: ' + (doc.source_path || doc.filename || doc.title), size: doc.size_bytes, collectionId: doc.collection_id, kind: 'retry' });
      }
      await loadDocs();
    });
  }

  async function deleteDocument(doc, action) {
    try {
      const result = await api('DELETE', '/api/documents/' + enc(doc.id));
      state.expanded.delete(doc.id);
      state.docs = state.docs.filter((d) => d.id !== doc.id);
      if (result && result.pending) {
        // 202: the delete goes on in the background; the list catches up on a later refresh.
        announce(doc.title + ': ' + result.message);
        scheduleDocs(DOCS_POLL_MS * 2);
      } else {
        announce(doc.title + (action === 'dismiss' ? ' dismissed.' : ' deleted.'));
        await loadDocs();
      }
      loadHealth();
    } catch (e) {
      showBannerMessage('The document could not be ' + (action === 'dismiss' ? 'dismissed: ' : 'deleted: ') + e.message);
    }
  }

  // ------------------------------------------------------------------ uploads: files, folders, zips

  function setupUploads() {
    const zone = $('dropzone');
    const input = $('file-input');
    const folder = $('folder-input');
    zone.addEventListener('click', () => { if (state.current) input.click(); });
    zone.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        if (state.current) input.click();
      }
    });
    $('add-folder').addEventListener('click', () => { if (state.current) folder.click(); });
    input.addEventListener('change', () => {
      handleFiles(Array.from(input.files || []).map((file) => ({ file, path: '' })), false);
      input.value = '';
    });
    folder.addEventListener('change', () => {
      handleFiles(Array.from(folder.files || []).map((file) => ({ file, path: file.webkitRelativePath || '' })), true);
      folder.value = '';
    });
    let depth = 0;
    zone.addEventListener('dragenter', (event) => { event.preventDefault(); depth += 1; zone.classList.add('dragging'); });
    zone.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) zone.classList.remove('dragging'); });
    zone.addEventListener('dragover', (event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; });
    zone.addEventListener('drop', (event) => {
      event.preventDefault();
      depth = 0;
      zone.classList.remove('dragging');
      filesFromDrop(event.dataTransfer).then(({ entries, folder: fromFolder }) => handleFiles(entries, fromFolder));
    });
    // A file dropped next to the zone must not replace the page.
    window.addEventListener('dragover', (event) => event.preventDefault());
    window.addEventListener('drop', (event) => event.preventDefault());
    $('uploads-clear').addEventListener('click', () => {
      state.uploads = state.uploads.filter((u) => ACTIVE.has(u.status));
      renderUploads();
    });
  }

  /* Files and folders from a drop. Folders are walked, and each file keeps its path inside the folder. The entries
   * are taken before the first await: the browser empties the drop data when the event handler returns. */
  function filesFromDrop(dataTransfer) {
    const items = Array.from((dataTransfer && dataTransfer.items) || []);
    const entries = items.map((item) => (item.kind === 'file' && item.webkitGetAsEntry ? item.webkitGetAsEntry() : null));
    if (!entries.some((entry) => entry && entry.isDirectory)) {
      const files = Array.from((dataTransfer && dataTransfer.files) || []).map((file) => ({ file, path: '' }));
      return Promise.resolve({ entries: files, folder: false });
    }
    const loose = items.map((item, i) => (entries[i] ? null : item.getAsFile()));
    const out = [];
    const walk = async (entry) => {
      if (entry.isFile) {
        const file = await new Promise((resolve) => entry.file(resolve, () => resolve(null)));
        if (file) out.push({ file, path: String(entry.fullPath || file.name).replace(/^\/+/, '') });
        return;
      }
      if (!entry.isDirectory) return;
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((resolve) => reader.readEntries(resolve, () => resolve([])));
        if (!batch.length) break;
        for (const child of batch) await walk(child);
      }
    };
    return (async () => {
      for (const entry of entries) if (entry) await walk(entry);
      for (const file of loose) if (file) out.push({ file, path: '' });
      return { entries: out, folder: true };
    })();
  }

  function handleFiles(entries, fromFolder) {
    if (!state.current || !entries.length) return;
    const health = state.health || {};
    const limitMb = health.max_upload_mb || 50;
    const maxBytes = limitMb * 1024 * 1024;
    const types = acceptedTypes();
    const titleInput = $('upload-title');
    const title = titleInput.value.trim();
    let skipped = 0;
    const chosen = [];
    for (const entry of entries) {
      const name = entry.file.name;
      // A folder holds files nobody meant to upload (pictures, hidden system files): leave them out quietly.
      if (fromFolder && (name.startsWith('.') || (types.size && !types.has(suffixOf(name))))) {
        skipped += 1;
        continue;
      }
      chosen.push(entry);
    }
    const single = chosen.length === 1 && !fromFolder && suffixOf(chosen[0].file.name) !== ARCHIVE;
    const useTitle = Boolean(title) && single;
    let titleNote = title && !useTitle ? 'The title was not used: it applies to one file only, not to several files, a folder or a zip.' : null;
    if (titleNote) announce(titleNote);
    for (const entry of chosen) {
      const file = entry.file;
      const suffix = suffixOf(file.name);
      const item = {
        name: entry.path || file.name, path: entry.path || '', size: file.size, collectionId: state.current, file,
        status: 'waiting', progress: 0, kind: 'upload', title: useTitle ? title : null, note: titleNote,
        archive: suffix === ARCHIVE,
      };
      titleNote = null;
      let reason = null;
      if (types.size && !types.has(suffix)) reason = 'this file type (' + (suffix || 'no extension') + ') is not supported.';
      else if (file.size > maxBytes && suffix !== ARCHIVE) reason = 'larger than the ' + limitMb + ' MB limit per file.';
      else if (file.size === 0) reason = 'the file is empty.';
      if (reason) {
        item.status = 'rejected';
        item.note = 'Rejected: ' + reason;
        item.file = null;
      }
      addUploadItem(item);
      if (item.status === 'waiting') uploadQueue.push(item);
    }
    if (skipped) {
      addUploadItem({
        name: plural(skipped, 'file') + ' left out of the folder', size: null, collectionId: state.current,
        status: 'skipped', kind: 'note', note: 'Hidden files and file types rag-kb cannot read are not uploaded.',
      });
    }
    if (useTitle) titleInput.value = '';
    pumpUploads();
  }

  function addUploadItem(item) {
    uploadSeq += 1;
    item.key = uploadSeq;
    state.uploads.push(item);
    renderUploads();
  }

  function pumpUploads() {
    while (activeUploads < 2 && uploadQueue.length) {
      const item = uploadQueue.shift();
      if (item.status !== 'waiting') continue;
      activeUploads += 1;
      runUpload(item).finally(() => {
        activeUploads -= 1;
        pumpUploads();
      });
    }
  }

  function xhrUpload(item) {
    return new Promise((resolve) => {
      const form = new FormData();
      form.append('files', item.file, item.file.name);
      // The path inside a folder or zip is the document's identity; a plain file is known by its name.
      if (item.path) form.append('relative_paths', item.path);
      if (item.title) form.append('title', item.title);
      const xhr = new XMLHttpRequest();
      item.xhr = xhr;
      xhr.open('POST', '/api/collections/' + enc(item.collectionId) + '/documents');
      xhr.setRequestHeader('Accept', 'application/json');
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          item.progress = Math.round((event.loaded / event.total) * 100);
          renderUploadsSoon();
        }
      });
      xhr.addEventListener('load', () => {
        let data = null;
        try { data = JSON.parse(xhr.responseText); } catch (e) { data = null; }
        resolve({ status: xhr.status, data });
      });
      xhr.addEventListener('error', () => resolve({ status: 0, data: null }));
      xhr.addEventListener('abort', () => resolve({ status: -1, data: null }));
      xhr.send(form);
    });
  }

  /* The 202 upload response: `jobs` (one Job per accepted file; a zip gives one per file inside it),
   * `rejected` (one {filename, source_path, code, message} per file refused up front, which left no row) and
   * `skipped` (zip entries left out on purpose: folders and hidden files). Each file is in exactly one list. */
  function uploadEntries(data) {
    if (!data || !Array.isArray(data.jobs)) return [];
    return data.jobs.map((job) => ({ job })).concat((data.rejected || []).map((rejected) => ({ rejected })));
  }

  async function runUpload(item) {
    item.status = 'uploading';
    renderUploads();
    const { status, data } = await xhrUpload(item);
    item.file = null;
    item.xhr = null;
    if (status === -1) {
      item.status = 'cancelled';
      item.note = 'Upload cancelled before it reached the server.';
      renderUploads();
      return;
    }
    const entries = status === 202 ? uploadEntries(data) : [];
    if (entries.length) {
      applyUploadEntries(item, entries, Number(data.skipped) || 0);
    } else if (data && data.error) {
      item.status = 'rejected';
      item.note = 'Rejected: ' + (data.error.message || 'the server did not accept this file.');
    } else {
      item.status = 'failed';
      item.note = status === 0 ? 'The upload did not reach the server.' : 'The upload failed with status ' + status + '.';
    }
    renderUploads();
    if (state.uploads.some((u) => u.jobId && TRACKED.has(u.status))) ensureJobPolling();
    if (item.collectionId === state.current) scheduleDocs(300);
  }

  function entryName(entry) {
    const source = entry.job || entry.rejected;
    return source.source_path || source.filename;
  }

  function applyUploadEntries(item, entries, skipped) {
    if (entries.length === 1 && !item.archive) {
      applyEntry(item, entries[0]);
      return;
    }
    // A zip: the archive's row stays as a header, followed by one row per file inside it.
    const children = entries.map((entry) => {
      const child = {
        name: entryName(entry), size: entry.job && finite(entry.job.bytes_total) ? entry.job.bytes_total : null,
        collectionId: item.collectionId, status: 'queued', progress: 100, kind: 'upload', note: null,
      };
      applyEntry(child, entry);
      uploadSeq += 1;
      child.key = uploadSeq;
      return child;
    });
    const rejected = children.filter((c) => c.status === 'rejected').length;
    item.status = 'unpacked';
    item.progress = 100;
    item.note = 'Unpacked into ' + plural(children.length - rejected, 'file') + (rejected ? '; ' + rejected + ' rejected' : '')
      + (skipped ? '; ' + plural(skipped, 'entry', 'entries') + ' left out (folders and hidden files)' : '') + '.';
    state.uploads.splice(state.uploads.indexOf(item) + 1, 0, ...children);
  }

  function applyEntry(item, entry) {
    if (entry.rejected) {
      item.status = 'rejected';
      item.note = 'Rejected: ' + entry.rejected.message;
      return;
    }
    applyJob(item, entry.job);
  }

  function jobNote(item, job) {
    const status = jobDisplayStatus(job);
    const replaced = job.replaced_title ? '"' + job.replaced_title + '"' : null;
    if (status === 'failed') return (job.error || 'Indexing failed.') + (replaced ? ' ' + replaced + ' was kept as it was.' : '');
    if (status === 'cancelled') return 'Cancelled.' + (replaced ? ' ' + replaced + ' was kept as it was.' : '');
    if (status === 'ready') {
      if (replaced) return 'Replaced ' + replaced + '.';
      if (item.kind === 'reindex') return 'Re-indexed.';
      if (item.kind === 'retry') return 'Indexed on retry.';
      return 'Indexed and searchable.';
    }
    const progress = jobProgressText(job);
    return replaced ? progress + '. Replaces ' + replaced + ' once indexed.' : progress;
  }

  /* Updates an upload row from its job. Returns true when what the row shows has changed. */
  function applyJob(item, job) {
    if (!job || typeof job !== 'object') return false;
    if (job.id) {
      item.jobId = job.id;
      jobCache.set(job.id, job);
    }
    item.job = job;
    if (job.doc_id) item.docId = job.doc_id;
    item.progress = 100;
    const before = item.status + '|' + item.note + '|' + jobSignature(job);
    if (job.already_indexed) {
      item.status = 'known';
      item.note = 'Same content as a document already in this collection, so it was kept once.';
    } else {
      item.status = jobDisplayStatus(job);
      item.note = jobNote(item, job);
    }
    return before !== item.status + '|' + item.note + '|' + jobSignature(item.job);
  }

  function ensureJobPolling() {
    if (!jobTimer) jobTimer = window.setInterval(pollJobs, JOB_POLL_MS);
  }

  async function pollJobs() {
    if (jobBusy) return;
    const active = state.uploads.filter((u) => u.jobId && TRACKED.has(u.status));
    if (!active.length) {
      window.clearInterval(jobTimer);
      jobTimer = null;
      return;
    }
    jobBusy = true;
    let changed = false;
    const finished = [];
    try {
      const byCollection = new Map();
      for (const item of active) {
        if (!byCollection.has(item.collectionId)) byCollection.set(item.collectionId, []);
        byCollection.get(item.collectionId).push(item);
      }
      for (const [cid, items] of byCollection) {
        const listed = await refreshJobs(cid);
        const seen = new Set(Array.isArray(listed) ? listed.map((j) => j.id) : []);
        const gone = new Set();
        // Jobs the list did not include (an older server, or a very long queue): ask for a few one by one.
        await Promise.all(items.filter((i) => !seen.has(i.jobId)).slice(0, 8).map(async (item) => {
          try {
            jobCache.set(item.jobId, await api('GET', '/api/jobs/' + enc(item.jobId)));
          } catch (e) {
            if (e.status === 404) gone.add(item);
          }
        }));
        for (const item of items) {
          if (gone.has(item)) {
            item.status = 'failed';
            item.note = 'The server no longer knows this job.';
            changed = true;
            finished.push(item);
            continue;
          }
          const job = jobCache.get(item.jobId);
          if (!job) continue;
          if (applyJob(item, job)) changed = true;
          if (!TRACKED.has(item.status)) finished.push(item);
        }
      }
    } finally {
      jobBusy = false;
    }
    if (changed) renderUploads();
    const ready = finished.filter((i) => i.status === 'ready');
    if (ready.length === 1) announce(ready[0].name + ' is ready.');
    else if (ready.length > 1) announce(ready.length + ' documents are ready.');
    if (finished.length) {
      scheduleDocs(0);
      loadHealth();
    }
  }

  let uploadsFrame = 0;
  function renderUploadsSoon() {
    if (uploadsFrame) return;
    uploadsFrame = window.requestAnimationFrame(() => {
      uploadsFrame = 0;
      renderUploads();
    });
  }

  function itemPercent(item) {
    if (item.status === 'waiting') return 0;
    // An upload's bar covers the transfer (the first 30%) and then the indexing; a re-index only the indexing.
    const scale = (p) => (item.kind === 'upload' ? Math.round(30 + 0.7 * p) : p);
    if (item.status === 'uploading') return item.kind === 'upload' ? Math.round(0.3 * (item.progress || 0)) : 0;
    if (item.status === 'queued') return scale(0);
    if (item.status === 'processing' || item.status === 'cancelling') {
      const p = item.job ? jobPercent(item.job) : null;
      return p === null ? null : scale(p);
    }
    return 100;
  }

  function cancellable(item) {
    if (item.status === 'waiting' || item.status === 'uploading') return true;
    return Boolean(item.jobId) && (item.status === 'queued' || item.status === 'processing');
  }

  async function cancelUpload(item, button) {
    if (item.status === 'waiting') {
      const at = uploadQueue.indexOf(item);
      if (at >= 0) uploadQueue.splice(at, 1);
      item.status = 'cancelled';
      item.note = 'Not uploaded.';
      item.file = null;
      renderUploads();
      return;
    }
    if (item.status === 'uploading' && item.xhr) {
      item.xhr.abort();
      return;
    }
    if (item.jobId) await cancelJob(item.jobId, button);
  }

  function uploadRow(item) {
    const percent = itemPercent(item);
    const failed = item.status === 'failed' || item.status === 'rejected';
    let note = item.note;
    if (!note && item.status === 'waiting') note = 'Waiting to upload';
    else if (!note && item.status === 'uploading') note = 'Uploading, ' + (item.progress || 0) + '%';
    return h('li', { class: 'upload-item', dataset: { key: 'u' + item.key, status: item.status } },
      h('span', { class: 'upload-name', title: item.name }, item.name,
        finite(item.size) ? h('span', { class: 'muted small' }, '  ' + fmtBytes(item.size)) : null),
      // A bar while the file moves, and a full one once it is ready; a rejected or cancelled row has none.
      BAR_STATES.has(item.status) ? progressBar(percent, 'Progress for ' + item.name) : h('span'),
      chip(item.status),
      cancellable(item)
        ? h('button', {
          type: 'button', class: 'btn btn-ghost btn-small', dataset: { action: 'cancel' }, 'aria-label': 'Cancel ' + item.name,
          onclick: (event) => cancelUpload(item, event.currentTarget),
        }, 'Cancel')
        : h('span'),
      note ? h('span', { class: 'upload-note' + (failed ? ' error-text' : '') }, note) : null);
  }

  function renderUploads() {
    const items = state.uploads.filter((u) => u.collectionId === state.current);
    $('upload-panel').hidden = !items.length;
    const list = $('upload-list');
    const next = new Map();
    const nodes = items.map((item) => {
      const sig = [item.status, item.note, itemPercent(item), item.name, item.size, cancellable(item)].join('|');
      const cached = uploadRowCache.get(item.key);
      const node = cached && cached.sig === sig ? cached.node : uploadRow(item);
      next.set(item.key, { sig, node });
      return node;
    });
    uploadRowCache = next;
    keepFocus(list, () => reconcile(list, nodes));
    const files = items.filter((u) => u.kind !== 'note' && u.status !== 'unpacked');
    const moving = files.filter((u) => ACTIVE.has(u.status)).length;
    const problems = files.filter((u) => u.status === 'failed' || u.status === 'rejected').length;
    const parts = [plural(files.length, 'file')];
    if (moving) parts.push(moving + ' in progress');
    if (problems) parts.push(problems + ' with problems');
    $('upload-count').textContent = files.length ? '(' + parts.join(', ') + ')' : '';
  }

  // ------------------------------------------------------------------ ask

  function renderAskAvailability() {
    const has = Boolean(state.current);
    $('ask-no-collection').hidden = has;
    $('ask-form').hidden = !has;
    $('answer-area').hidden = !has;
    renderSuggestions();
  }

  function renderSuggestions() {
    const box = $('suggestions');
    box.replaceChildren();
    const ready = state.docs.filter((d) => d.status === 'ready');
    if (!ready.length) return;
    const fill = (text) => () => {
      $('question').value = text;
      $('question').focus();
    };
    for (const doc of ready.slice(0, 2)) {
      const text = 'What does "' + doc.title + '" cover?';
      box.append(h('button', { type: 'button', class: 'suggestion', onclick: fill(text) }, text));
    }
    // The accessible name is the visible text; the explanation is the description.
    box.append(h('button', {
      type: 'button', class: 'suggestion probe', 'aria-describedby': 'probe-hint', onclick: fill(OUT_OF_SCOPE_PROBE),
    }, OUT_OF_SCOPE_PROBE, ' ', h('span', { class: 'probe-tag' }, '(out of scope test)')));
  }

  function setupAsk() {
    const form = $('ask-form');
    const question = $('question');
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      ask();
    });
    question.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        ask();
      }
    });
    $('provider-select').addEventListener('change', () => {
      store('provider', $('provider-select').value);
      renderModelControl();
    });
    $('model-select').addEventListener('change', () => store('model.' + chosenProvider(), $('model-select').value));
    const topk = $('topk-input');
    const savedTopk = recall('topk', '');
    if (savedTopk) topk.value = savedTopk;
    topk.addEventListener('change', () => {
      topk.dataset.touched = '1';
      const n = Math.max(1, Math.min(20, Number(topk.value) || 6));
      topk.value = String(n);
      store('topk', String(n));
    });
    $('answer-area').addEventListener('click', onAnswerClick);
    $('answer-area').addEventListener('mouseover', onAnswerHover);
    $('answer-area').addEventListener('mouseout', onAnswerHover);
    document.addEventListener('keydown', (event) => {
      if (event.key === '/' && !isTyping(event.target) && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault();
        selectTab('ask');
        question.focus();
      }
    });
  }

  function clearAnswer() {
    state.answerToken += 1;
    state.answerId = null;
    const area = $('answer-area');
    if (area) area.replaceChildren();
  }

  async function ask() {
    const question = $('question').value.trim();
    if (!question || state.asking || !state.current) {
      if (!question) $('question').focus();
      return;
    }
    const cid = state.current;
    const provider = $('provider-select').value || 'auto';
    const modelSelect = $('model-select');
    const model = modelSelect.disabled ? '' : modelSelect.value;
    const topK = Math.max(1, Math.min(20, Number($('topk-input').value) || 6));
    const body = { question, top_k: topK };
    if (provider !== 'auto') body.provider = provider;
    if (model) body.model = model;

    state.asking = true;
    const token = ++state.answerToken;
    const button = $('ask-button');
    button.disabled = true;
    const started = performance.now();
    const area = $('answer-area');
    const ticker = h('span', { class: 'muted small' }, '0 s');
    const who = providerLabel(provider === 'auto' ? autoProvider() : provider);
    area.replaceChildren(h('article', { class: 'answer-card state-card state-loading', 'aria-busy': 'true' },
      h('div', { class: 'state-head' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }),
        h('h3', {}, 'Searching the documents and writing the answer'), ticker),
      h('p', { class: 'state-hint' }, who.startsWith('Extractive')
        ? 'Extractive mode answers in a moment.'
        : who + ' usually takes 10 to 30 seconds.')));
    const tick = window.setInterval(() => {
      ticker.textContent = Math.round((performance.now() - started) / 1000) + ' s';
    }, 1000);

    try {
      const answer = await api('POST', '/api/collections/' + enc(cid) + '/ask', body);
      if (token !== state.answerToken) return;
      const elapsed = performance.now() - started;
      area.replaceChildren(renderAnswer(answer, { elapsedMs: elapsed, request: body }));
      if (answer.id) {
        state.answerId = answer.id;
        setRoute('#/answer/' + enc(answer.id), 'push');
      }
      announce(answer.status === 'answered' ? 'Answer ready.' : 'No answer: ' + (answer.message || answer.status));
    } catch (e) {
      if (token !== state.answerToken) return;
      if (e.status === 429) {
        area.replaceChildren(renderAnswer(pseudoAnswer(question, cid, 'rate_limited', e.message), { retryAfter: e.retryAfter }));
      } else {
        area.replaceChildren(errorCard('The question could not be answered', e.message));
      }
    } finally {
      window.clearInterval(tick);
      state.asking = false;
      button.disabled = false;
    }
  }

  async function loadStoredAnswer(id, route) {
    selectTab('ask', { route: 'none' });
    if (state.answerId === id && $('answer-area').firstChild) {
      setRoute('#/answer/' + enc(id), route);
      return;
    }
    const token = ++state.answerToken;
    $('answer-area').replaceChildren(h('article', { class: 'answer-card state-card state-loading', 'aria-busy': 'true' },
      h('div', { class: 'state-head' }, spinner(), h('h3', {}, 'Opening the stored answer'))));
    try {
      const answer = await api('GET', '/api/answers/' + enc(id));
      if (token !== state.answerToken) return;
      if (answer.collection_id !== state.current && state.collections.some((c) => c.id === answer.collection_id)) {
        await selectCollection(answer.collection_id, false, 'none');
      }
      $('question').value = answer.question;
      $('answer-area').replaceChildren(renderAnswer(answer, {}));
      state.answerId = answer.id;
      setRoute('#/answer/' + enc(answer.id), route);
    } catch (e) {
      if (token !== state.answerToken) return;
      $('answer-area').replaceChildren(errorCard('That answer could not be loaded', e.message));
      setRoute('#/answer/' + enc(id), route);
    }
  }

  function pseudoAnswer(question, cid, status, message) {
    return {
      id: '', question, collection_id: cid, status, answer: null, message, citations: [], closest: [],
      passages_considered: 0, provider: null, knowledge_version: '', language: null, created_at: new Date().toISOString(),
    };
  }

  function errorCard(title, message) {
    return h('article', { class: 'answer-card state-card state-error' },
      h('div', { class: 'state-head' }, h('span', { class: 'state-icon', 'aria-hidden': 'true' }, '!'), h('h3', {}, title)),
      h('p', { class: 'state-message' }, message));
  }

  function fixFromMessage(message) {
    const match = /Run:\s*(.+?)\.?\s*$/.exec(message || '');
    return match ? match[1].trim() : null;
  }

  /* The distinct look of every non-answer state. */
  function stateParts(answer, options) {
    const message = answer.message || '';
    const status = answer.status;
    if (status === 'no_answer') {
      const checked = answer.passages_considered
        ? ' ' + plural(answer.passages_considered, 'passage') + ' were checked and none answers it.' : '';
      const closest = Array.isArray(answer.closest) && answer.closest.length
        ? ' The closest passages are below, so you can judge for yourself.' : '';
      return {
        tone: 'warn', icon: '?', title: 'Not covered by these documents', message,
        hint: 'The knowledge base does not guess.' + checked + closest + ' Try other words, or upload a document that covers it.',
      };
    }
    if (status === 'no_documents') {
      return {
        tone: 'info', icon: 'i', title: 'This collection has no documents yet', message,
        hint: 'Upload documents on the Documents tab; questions are answered once they show Ready.',
        action: options.preview ? null : h('button', { type: 'button', class: 'btn btn-primary btn-small', onclick: () => selectTab('documents') }, 'Go to Documents'),
      };
    }
    if (status === 'provider_unavailable') {
      const name = answer.provider ? answer.provider.name : null;
      const fix = fixFromMessage(message) || fixFor(providerStatus(name));
      return {
        tone: 'error', icon: '!', title: 'The answer provider is not available', message,
        fix,
        hint: 'Run the command in a terminal, then ask again. Or pick another provider, such as Extractive mode (no AI), above.',
      };
    }
    if (status === 'rate_limited') {
      const wait = options.retryAfter ? ' Try again in about ' + options.retryAfter + ' seconds.' : '';
      return {
        tone: 'warn', icon: '!', title: 'Rate limited', message,
        hint: 'Too many questions in a short time, or the provider plan reached its limit.' + wait + ' You can also switch to another provider.',
      };
    }
    if (status === 'retrieval_unavailable') {
      return {
        tone: 'error', icon: '!', title: 'Search is unavailable', message,
        hint: 'The document store did not respond. Check the health badge and the Settings tab; restarting the server usually helps.',
      };
    }
    return { tone: 'error', icon: '!', title: 'Unexpected state: ' + status, message, hint: '' };
  }

  function renderAnswer(answer, options) {
    const opts = options || {};
    const prefix = opts.idPrefix || '';
    const answered = answer.status === 'answered' && answer.answer;
    const card = h('article', {
      class: 'answer-card' + (answered ? ' state-answered' : ' state-card state-' + stateParts(answer, opts).tone),
      dataset: { status: answer.status },
    });
    if (answer.question) card.append(h('p', { class: 'answer-question' }, answer.question));

    const markers = new Set((answer.citations || []).map((c) => c.marker));
    if (answered) {
      const body = h('div', { class: 'markdown answer-body', dataset: { prefix } });
      body.innerHTML = renderMarkdown(answer.answer, markers);
      card.append(body);
    } else {
      const parts = stateParts(answer, opts);
      card.append(h('div', { class: 'state-head' },
        h('span', { class: 'state-icon', 'aria-hidden': 'true' }, parts.icon), h('h3', {}, parts.title)));
      if (parts.message) card.append(h('p', { class: 'state-message' }, parts.message));
      if (parts.fix) card.append(codeBlock(parts.fix, 'Copy command'));
      if (parts.hint) card.append(h('p', { class: 'state-hint' }, parts.hint));
      if (parts.action) card.append(parts.action);
    }

    const meta = [];
    if (answer.provider && answer.provider.name) {
      const model = answer.provider.model ? ' (' + answer.provider.model + ')' : '';
      meta.push((answered ? 'Answered by ' : 'Provider: ') + providerLabel(answer.provider.name) + model);
      if (answer.provider.elapsed_ms) meta.push('model ' + fmtSeconds(answer.provider.elapsed_ms));
    }
    if (opts.elapsedMs) meta.push('total ' + fmtSeconds(opts.elapsedMs));
    if (answer.passages_considered || answer.status !== 'rate_limited') {
      meta.push(plural(answer.passages_considered || 0, 'passage') + ' considered');
    }
    if (answer.knowledge_version) meta.push('knowledge ' + answer.knowledge_version);
    if (answer.language) meta.push('language ' + answer.language);
    if (meta.length) card.append(h('div', { class: 'answer-meta' }, meta.map((m) => h('span', {}, m))));

    if (answer.citations && answer.citations.length) {
      card.append(renderSources(answer.citations, { prefix, preview: opts.preview }));
    }
    if (!answered && answer.status === 'no_answer' && Array.isArray(answer.closest) && answer.closest.length) {
      card.append(renderSources(answer.closest, { prefix: prefix + 'closest-', preview: opts.preview, closest: true }));
    }
    if (!opts.preview && answer.id) {
      if (answered || answer.status === 'no_answer') card.append(renderFeedback(answer));
      card.append(renderRaw(answer, opts.request));
    }
    return card;
  }

  function renderSources(citations, options) {
    const opts = options || {};
    const list = h('ol', { class: 'source-list' });
    for (const c of citations) list.append(sourceCard(c, opts.prefix || '', opts.preview));
    if (opts.closest) {
      return h('section', { class: 'sources closest-passages', 'aria-label': 'Closest passages' },
        h('h3', {}, 'Closest passages'),
        h('p', { class: 'muted small' }, 'None of these answers the question. They are the nearest passages that were found, shown so you can judge for yourself.'),
        list);
    }
    return h('section', { class: 'sources', 'aria-label': 'Sources' },
      h('h3', {}, 'Sources'), list,
      h('p', { class: 'muted small' }, 'Similarity (the bar) says how close each passage is in meaning to the question, from 0 to 1. The rank score sets the order. See Help for how to read the scores.'));
  }

  function sourceCard(c, prefix, preview) {
    const where = [];
    const pages = pagesLabel(c.page, c.page_end);
    if (pages) where.push(pages);
    if (c.section) where.push(c.section);
    const passage = h('div', { class: 'source-passage', hidden: true });
    const open = preview ? null : h('button', {
      type: 'button', class: 'btn btn-ghost btn-small', 'aria-expanded': 'false',
      onclick: () => togglePassage(c, passage, open),
    }, 'Open passage');
    return h('li', { class: 'source-card', id: prefix + 'source-' + c.marker, tabindex: '-1' },
      h('div', { class: 'source-head' },
        h('span', { class: 'cite-badge' }, String(c.marker)),
        h('span', { class: 'source-title' }, c.title),
        where.length ? h('span', { class: 'muted small' }, where.join('  |  ')) : null),
      c.snippet ? h('p', { class: 'snippet' }, c.snippet) : null,
      h('div', { class: 'source-foot' }, scoreView(c), open),
      passage);
  }

  /* The similarity bar is absolute (the raw cosine, 0 to 1), so a full bar means a close match in any answer.
   * The reranker score, when a reranker ran, says how relevant the passage is; the fused rank score only orders. */
  function scoreView(c) {
    const parts = [];
    const cosine = finite(c.vector_score) ? c.vector_score : null;
    const fill = h('span', { class: 'scorebar-fill' });
    const share = cosine === null ? 0 : Math.max(0, Math.min(1, cosine));
    fill.style.width = (cosine === null ? 0 : Math.max(2, Math.round(share * 100))) + '%';
    parts.push(h('span', {
      class: 'scorebar' + (cosine === null ? ' scorebar-none' : ''), role: 'img',
      'aria-label': cosine === null ? 'No similarity score' : 'Similarity ' + fmtRelevance(cosine) + ' on a scale from 0 to 1',
    }, fill));
    parts.push(h('span', {
      class: 'score',
      title: cosine === null
        ? 'This passage was found by the word search alone, so it has no meaning score.'
        : 'Cosine similarity between the meaning of the question and of the passage, from 0 to 1. Higher is closer.',
    }, cosine === null ? 'no similarity score' : 'similarity ' + fmtRelevance(cosine)));
    if (finite(c.rerank_score)) {
      parts.push(h('span', {
        class: 'score', title: 'Relevance from the local reranker, which reads the question and the passage together. Higher is more relevant.',
      }, 'reranker ' + fmtRelevance(c.rerank_score)));
    }
    if (finite(c.score)) {
      parts.push(h('span', {
        class: 'score muted', title: 'Rank fusion score: it sets the order of the sources. It is not a measure of relevance.',
      }, 'rank ' + fmtScore(c.score)));
    }
    return h('span', { class: 'score-group' }, parts);
  }

  async function togglePassage(citation, box, button) {
    if (!box.hidden) {
      box.hidden = true;
      button.setAttribute('aria-expanded', 'false');
      button.textContent = 'Open passage';
      return;
    }
    box.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    button.textContent = 'Hide passage';
    box.textContent = 'Loading the passage...';
    try {
      const chunks = await loadChunks(citation.doc_id, 'answer');
      const chunk = chunks.find((c) => c.id === citation.chunk_id);
      box.textContent = chunk ? chunk.text : 'This passage is no longer in the index (the document was changed or deleted).';
    } catch (e) {
      box.textContent = e.status === 404
        ? 'This document is no longer in the collection.'
        : 'The passage could not be loaded: ' + e.message;
    }
  }

  function highlightSource(card) {
    card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    card.classList.remove('flash');
    void card.offsetWidth;
    card.classList.add('flash');
    card.focus({ preventScroll: true });
    window.setTimeout(() => card.classList.remove('flash'), 1800);
  }

  function sourceFor(target) {
    const cite = target.closest('.cite');
    if (!cite) return null;
    const article = cite.closest('.answer-card');
    const body = cite.closest('.answer-body');
    const prefix = body ? body.dataset.prefix || '' : '';
    return article ? article.querySelector('#' + CSS.escape(prefix + 'source-' + cite.dataset.marker)) : null;
  }

  function onAnswerClick(event) {
    const card = sourceFor(event.target);
    if (card) highlightSource(card);
  }

  function onAnswerHover(event) {
    const card = sourceFor(event.target);
    if (card) card.classList.toggle('hover', event.type === 'mouseover');
  }

  const THUMB_UP = '<svg class="thumb" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 10v11H3V10z"/><path d="M7 10l4-8a3 3 0 0 1 3 3v4h6a2 2 0 0 1 2 2.3l-1.4 8A2 2 0 0 1 18.6 21H7"/></svg>';
  const THUMB_DOWN = '<svg class="thumb" viewBox="0 0 24 24" aria-hidden="true"><path d="M17 14V3h4v11z"/><path d="M17 14l-4 8a3 3 0 0 1-3-3v-4H4a2 2 0 0 1-2-2.3l1.4-8A2 2 0 0 1 5.4 3H17"/></svg>';

  function renderFeedback(answer) {
    const box = h('div', { class: 'feedback' });
    const note = h('textarea', { maxlength: '2000', placeholder: 'What was right or wrong? (optional)', 'aria-label': 'Feedback note', hidden: true });
    const send = h('button', { type: 'button', class: 'btn btn-small', hidden: true }, 'Send feedback');
    const result = h('span', { class: 'muted small', role: 'status' });
    let rating = null;
    const thumb = (value, svg, label) => {
      const button = h('button', { type: 'button', class: 'btn btn-small', 'aria-pressed': 'false', 'aria-label': label, title: label });
      button.innerHTML = svg;
      button.append(value === 'up' ? ' Helpful' : ' Not helpful');
      button.addEventListener('click', () => {
        rating = value;
        for (const b of box.querySelectorAll('[aria-pressed]')) b.setAttribute('aria-pressed', 'false');
        button.setAttribute('aria-pressed', 'true');
        note.hidden = false;
        send.hidden = false;
        result.textContent = '';
        note.focus();
      });
      return button;
    };
    send.addEventListener('click', async () => {
      if (!rating) return;
      send.disabled = true;
      try {
        await api('POST', '/api/answers/' + enc(answer.id) + '/feedback', { rating, note: note.value.trim() });
        result.textContent = 'Thanks, your feedback was saved. Curators see it on the Settings tab.';
        note.hidden = true;
        send.hidden = true;
      } catch (e) {
        result.textContent = 'Feedback could not be saved: ' + e.message;
      } finally {
        send.disabled = false;
      }
    });
    box.append(
      h('div', { class: 'feedback-row' }, h('span', { class: 'muted small' }, 'Was this helpful?'),
        thumb('up', THUMB_UP, 'Helpful'), thumb('down', THUMB_DOWN, 'Not helpful'), result),
      note, h('div', {}, send));
    return box;
  }

  // ------------------------------------------------------------------ the same request from a terminal

  // Letters, digits and punctuation that no supported shell treats specially inside its quotes.
  const SAFE_CHAR = /^[A-Za-z0-9 .,:;?@#()*+=_/-]$/;

  /* A JSON string in which every other character is a JSON escape, so the text is plain ASCII and contains no
   * quote, backslash, dollar, percent or caret that bash, PowerShell or cmd would interpret. */
  function shellSafeString(text) {
    let out = '"';
    for (const ch of String(text)) {
      if (SAFE_CHAR.test(ch)) {
        out += ch;
        continue;
      }
      for (let i = 0; i < ch.length; i += 1) out += BACKSLASH + 'u' + ch.charCodeAt(i).toString(16).padStart(4, '0');
    }
    return out + '"';
  }

  function shellSafeJson(value) {
    if (value === null || value === undefined) return 'null';
    if (typeof value === 'number') return Number.isFinite(value) ? String(value) : 'null';
    if (typeof value === 'boolean') return String(value);
    if (typeof value === 'string') return shellSafeString(value);
    if (Array.isArray(value)) return '[' + value.map(shellSafeJson).join(',') + ']';
    return '{' + Object.entries(value)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => shellSafeString(k) + ':' + shellSafeJson(v)).join(',') + '}';
  }

  function shellCommand(shell, url, body) {
    const json = shellSafeJson(body);
    const safeUrl = url.replace(/'/g, '%27').replace(/"/g, '%22');
    if (shell === 'powershell') {
      // Invoke-RestMethod, because curl is an alias with other options in Windows PowerShell 5.1.
      return "Invoke-RestMethod -Method Post -Uri '" + safeUrl + "' -ContentType 'application/json' -Body '" + json
        + "' | ConvertTo-Json -Depth 8";
    }
    if (shell === 'cmd') {
      // curl.exe ships with Windows 10 and later; inside double quotes a quote is written as a backslash and a quote.
      return 'curl -s -X POST "' + safeUrl + '" -H "Content-Type: application/json" -d "' + json.replace(/"/g, BACKSLASH + '"') + '"';
    }
    return "curl -s -X POST '" + safeUrl + "' -H 'Content-Type: application/json' -d '" + json + "'";
  }

  function defaultShell() {
    const platform = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || navigator.userAgent || '';
    return /win/i.test(platform) ? 'powershell' : 'bash';
  }

  /* A bash, PowerShell or cmd version of a request, with a switch this browser remembers. */
  function shellSnippet(url, body) {
    shellSeq += 1;
    const group = 'shell-' + shellSeq;
    const saved = recall('shell', '');
    let current = SHELLS.some(([value]) => value === saved) ? saved : defaultShell();
    const code = h('div', { class: 'shell-code' });
    const paint = () => code.replaceChildren(codeBlock(shellCommand(current, url, body), 'Copy'));
    const switcher = h('div', { class: 'segmented shell-switch', role: 'radiogroup', 'aria-label': 'Shell' });
    for (const [value, label] of SHELLS) {
      const id = group + '-' + value;
      const input = h('input', { type: 'radio', name: group, id, value, autocomplete: 'off' });
      input.checked = value === current;
      input.addEventListener('change', () => {
        if (!input.checked) return;
        current = value;
        store('shell', value);
        paint();
      });
      switcher.append(input, h('label', { for: id }, label));
    }
    paint();
    return h('div', { class: 'shell-snippet' }, switcher, code);
  }

  function renderRaw(answer, request) {
    const details = h('details', { class: 'raw' }, h('summary', {}, 'Raw response and request'));
    details.append(h('pre', {}, JSON.stringify(answer, null, 2)));
    if (request) {
      const url = location.origin + '/api/collections/' + enc(answer.collection_id) + '/ask';
      details.append(h('p', { class: 'muted small' }, 'The same request from a terminal:'), shellSnippet(url, request));
    }
    return details;
  }

  // ------------------------------------------------------------------ settings

  function kv(pairs) {
    const list = h('dl', { class: 'kv' });
    for (const [key, value] of pairs) {
      if (value === null || value === undefined || value === '') continue;
      list.append(h('dt', {}, key), h('dd', {}, value instanceof Node ? value : String(value)));
    }
    return list;
  }

  function stateChip(ok, yes, no) {
    return h('span', { class: 'chip ' + (ok ? 'chip-ready' : 'chip-failed') }, ok ? yes : no);
  }

  function setupSettings() {
    $('settings-refresh').addEventListener('click', () => {
      renderSettings(true);
      renderFeedbackPanel();
    });
    $('feedback-refresh').addEventListener('click', () => renderFeedbackPanel());
  }

  async function renderSettings(refresh) {
    const body = $('settings-body');
    if (refresh) {
      body.replaceChildren(h('p', { class: 'muted' }, 'Checking...'));
      await Promise.all([loadHealth(), loadProviders(true), state.current ? loadDocs() : null]);
    }
    if (state.tab !== 'settings') return;
    const health = state.health;
    body.replaceChildren();
    if (!health) {
      body.append(h('p', { class: 'error-text' }, 'The server cannot be reached.'));
      return;
    }

    const providers = h('section', { class: 'card wide' }, h('h3', {}, 'Answer providers'));
    if (state.providersError) providers.append(h('p', { class: 'error-text' }, state.providersError));
    if (state.providers.length) {
      const table = h('table', { class: 'simple-table' },
        h('thead', {}, h('tr', {}, ['Provider', 'State', 'Version', 'Details', 'Fix', 'Models'].map((t) => h('th', { scope: 'col' }, t)))));
      const tbody = h('tbody');
      for (const p of state.providers) {
        const ready = p.installed && p.signed_in;
        const label = ready ? 'Ready' : (p.installed ? 'Not signed in' : 'Not installed');
        const fix = fixFor(p);
        tbody.append(h('tr', {},
          h('td', {}, h('strong', {}, providerLabel(p.name))),
          h('td', {}, h('span', { class: 'chip ' + (ready ? 'chip-ready' : (p.installed ? 'chip-warn' : 'chip-queued')) }, label)),
          h('td', {}, p.version || ''),
          h('td', {}, p.detail || ''),
          h('td', {}, fix ? codeBlock(fix, 'Copy') : ''),
          h('td', {}, (p.models || []).join(', '))));
      }
      table.append(tbody);
      providers.append(h('div', { class: 'table-scroll' }, table));
      providers.append(h('p', { class: 'muted small' },
        'Setting: ' + (health.provider_setting || 'auto') + '. With auto, answers use ' + providerLabel(autoProvider()) + '.'));
    }
    body.append(providers);

    const counts = health.counts || {};
    let documents = null;
    if (finite(counts.documents_ready)) {
      const processing = Number(counts.documents_processing) || 0;
      const failed = Number(counts.documents_failed) || 0;
      // The health counts have no cancelled figure: it is what the total holds beyond the other three.
      const cancelled = Math.max(0, (Number(counts.documents_total) || 0) - counts.documents_ready - processing - failed);
      documents = searchableText({ ready: counts.documents_ready, processing, failed, cancelled });
    }
    body.append(h('section', { class: 'card' }, h('h3', {}, 'Index'), kv([
      ['Collections', counts.collections],
      ['Documents', documents],
      ['Passages', counts.chunks],
      ['Metadata store', stateChip(health.store && health.store.metadata === 'ok', 'OK', 'Unavailable')],
      ['Vector store', stateChip(health.store && health.store.vector === 'ok', 'OK', 'Unavailable')],
      ['Knowledge version', health.knowledge_version || 'none yet'],
    ])));

    body.append(h('section', { class: 'card' }, h('h3', {}, 'Retrieval'), kv([
      ['Embedding model', health.embed_model + (health.fake_embedder ? ' (fake, for tests)' : '')],
      ['Dimensions', health.embed_dims],
      ['Reranker', health.reranker === 'local' ? 'local (' + health.rerank_model + ')' : 'off'],
      ['Passages per answer', health.top_k],
      ['Minimum vector score', health.min_score],
      ['Context budget', health.max_context_chars + ' characters'],
    ])));

    const stats = state.stats;
    if (stats) {
      const byType = Object.entries(stats.by_media_type || {}).map(([k, v]) => v + ' ' + fmtType(k)).join(', ');
      const collectionCountsNow = state.docsLoaded ? countsFromDocs(state.docs) : statsCounts(stats);
      body.append(h('section', { class: 'card' }, h('h3', {}, 'This collection'), kv([
        ['Name', stats.name],
        ['Documents', searchableText(collectionCountsNow)],
        ['Types', byType || 'none'],
        ['Passages', stats.chunks],
        ['Pages', stats.pages || null],
        ['Size', fmtBytes(stats.size_bytes)],
        ['Last indexed', stats.last_indexed_at ? fmtWhen(stats.last_indexed_at).title : 'never'],
      ])));
    }

    const limit = health.ask_rate_limit || {};
    body.append(h('section', { class: 'card wide' }, h('h3', {}, 'Storage and limits'), kv([
      ['Data folder', codeBlock(health.data_dir || '', 'Copy')],
      ['Upload limit', health.max_upload_mb + ' MB per file'],
      ['File types', Array.from(acceptedTypes()).join(' ')],
      ['Questions', limit.per_minute ? limit.per_minute + ' per minute (' + limit.scope + ')' : null],
      ['Version', 'rag-kb ' + health.version],
    ])));

    if (health.problems && health.problems.length) {
      body.append(h('section', { class: 'card wide' }, h('h3', {}, 'Problems'),
        h('ul', { class: 'problem-list' }, health.problems.map((p) => h('li', {}, p))),
        h('p', { class: 'muted small' }, 'Run "ragkb doctor" in a terminal for a full check.')));
    }
  }

  /* Feedback review: the collection's votes, newest first, each with a link that opens the answer. */
  async function renderFeedbackPanel() {
    const body = $('feedback-body');
    const cid = state.current;
    if (!cid) {
      body.replaceChildren(h('p', { class: 'muted small' }, 'Pick a collection to see its feedback.'));
      return;
    }
    body.replaceChildren(h('p', { class: 'muted small' }, 'Loading feedback...'));
    let items;
    try {
      items = await api('GET', '/api/collections/' + enc(cid) + '/feedback?limit=100');
    } catch (e) {
      if (cid === state.current) body.replaceChildren(h('p', { class: 'error-text small' }, 'Feedback could not be loaded: ' + e.message));
      return;
    }
    if (cid !== state.current) return;
    items = Array.isArray(items) ? items : [];
    if (!items.length) {
      body.replaceChildren(h('p', { class: 'muted small' }, 'No feedback yet. Votes left under answers on the Ask tab appear here.'));
      return;
    }
    const up = items.filter((f) => f.rating === 'up').length;
    const rows = items.map((f) => h('tr', {},
      whenCell(f.created_at),
      h('td', {}, h('span', { class: 'chip ' + (f.rating === 'up' ? 'chip-ready' : 'chip-failed') }, f.rating === 'up' ? 'Helpful' : 'Not helpful')),
      h('td', { class: 'wrap' }, f.question ? f.question : h('span', { class: 'muted' }, 'not recorded')),
      h('td', { class: 'wrap' }, f.note || ''),
      h('td', {}, f.answer_id ? h('a', {
        href: '#/answer/' + enc(f.answer_id),
        'aria-label': f.question ? 'Open answer to: ' + f.question : 'Open answer',
      }, 'Open answer') : '')));
    body.replaceChildren(
      h('p', { class: 'small' }, plural(items.length, 'vote') + ': ' + up + ' helpful, ' + (items.length - up) + ' not helpful'
        + (items.length >= 100 ? ' (the latest 100)' : '') + '.'),
      h('div', { class: 'table-scroll' }, h('table', { class: 'simple-table feedback-table' },
        h('thead', {}, h('tr', {}, ['When', 'Rating', 'Question', 'Note', 'Answer'].map((t) => h('th', { scope: 'col' }, t)))),
        h('tbody', {}, rows))));
  }

  // ------------------------------------------------------------------ connect

  function renderConnect() {
    const body = $('connect-body');
    const mcp = (state.health && state.health.mcp) || { enabled: true, mounted: true, tools: [] };
    const url = location.origin + '/mcp';
    body.replaceChildren();

    const status = !mcp.enabled
      ? h('p', { class: 'error-text' }, 'MCP is turned off on this server. Start it with RAGKB_MCP_ENABLED=true to connect an assistant.')
      : (!mcp.mounted
        ? h('p', { class: 'error-text' }, 'MCP is enabled but not running on this server. Run "ragkb doctor" for details.')
        : h('p', {}, stateChip(true, 'Running', ''), ' ',
          mcp.public ? 'Answers on any host name (RAGKB_MCP_PUBLIC=true).' : 'Answers on this machine only (localhost).'));

    body.append(h('section', { class: 'card' }, h('h3', {}, 'MCP address'), status, codeBlock(url, 'Copy address')));

    body.append(h('section', { class: 'card' }, h('h3', {}, 'Claude Code'),
      h('p', {}, 'Run this once in a terminal; the tools are then available in every Claude Code session:'),
      codeBlock('claude mcp add --transport http rag-kb ' + url, 'Copy')));

    const desktopConfig = JSON.stringify({
      mcpServers: { 'rag-kb': { command: 'npx', args: ['-y', 'mcp-remote', url] } },
    }, null, 2);
    body.append(h('section', { class: 'card' }, h('h3', {}, 'Claude Desktop'),
      h('ol', { class: 'steps' },
        h('li', {}, 'Open Claude Desktop, then Settings, Developer, Edit Config. This opens claude_desktop_config.json.'),
        h('li', {}, 'Add this entry under "mcpServers" (keep any servers already there). It uses the mcp-remote bridge, which needs Node.js:',
          codeBlock(desktopConfig, 'Copy JSON')),
        h('li', {}, 'Restart Claude Desktop. The rag-kb tools appear in the tools menu of a new chat.'))));

    const tunnelPort = location.port || (location.protocol === 'https:' ? '443' : '80');
    body.append(h('section', { class: 'card' }, h('h3', {}, 'claude.ai custom connector'),
      h('p', {}, 'claude.ai connects from the internet, so it cannot reach localhost. It needs a public HTTPS address for this server, for example through a tunnel.'),
      h('ol', { class: 'steps' },
        h('li', {}, 'Restart the server with RAGKB_MCP_PUBLIC=true so /mcp answers on the tunnel host name.'),
        h('li', {}, 'Start a tunnel to this port, for example:', codeBlock('cloudflared tunnel --url ' + location.origin, 'Copy'),
          h('span', { class: 'muted small' }, 'or: ngrok http ' + tunnelPort)),
        h('li', {}, 'In claude.ai open Settings, Connectors, Add custom connector, and paste the tunnel address followed by /mcp.')),
      h('p', { class: 'state-hint' }, 'There is no sign-in on this connector yet: anyone with the tunnel address can search these documents through /mcp. The web page and its API keep answering only on this machine unless RAGKB_API_PUBLIC=true. Stop the tunnel when you are done.')));

    const names = mcp.tools && mcp.tools.length ? mcp.tools : Object.keys(MCP_TOOLS);
    body.append(h('section', { class: 'card' }, h('h3', {}, 'Tools'),
      h('ul', { class: 'tool-list' }, names.map((name) => h('li', {}, h('code', {}, name), h('span', {}, MCP_TOOLS[name] || '')))),
      h('p', { class: 'muted small' }, 'The tools never call a model: your assistant writes the final answer from the brief, on your own subscription.')));
  }

  // ------------------------------------------------------------------ help: what every state looks like

  function renderHelpStates() {
    const box = $('help-states');
    const base = {
      id: '', question: 'How many days of annual leave do employees get?', collection_id: 'sample', citations: [], closest: [],
      passages_considered: 6, provider: { name: 'claude_code', model: 'sonnet', elapsed_ms: 11800 },
      knowledge_version: 'v20261010.3', language: 'en', created_at: new Date().toISOString(),
    };
    const samples = [
      ['answered', 'Answered', 'A direct answer first, then details, with [n] markers that point at the sources.', {
        status: 'answered', message: null,
        answer: 'Employees get **25 days** of annual leave per calendar year [1].\n\n- Up to five unused days carry over to the next year [1].\n- Carried-over days expire at the end of March [2].',
        citations: [
          { marker: 1, chunk_id: 's#1', doc_id: 's', title: 'Leave policy', page: 2, page_end: 3, section: 'Leave policy > Annual leave', snippet: 'Every employee gets 25 days of annual leave per calendar year. Up to five unused days carry over...', score: 0.032, vector_score: 0.81 },
          { marker: 2, chunk_id: 's#2', doc_id: 's', title: 'Leave policy', page: 3, section: 'Leave policy > Carry-over', snippet: 'Days carried over expire at the end of March of the following year.', score: 0.024, vector_score: 0.64 },
        ],
      }],
      ['no_answer', 'Not covered', 'None of the documents answers the question. Nothing is guessed; the closest passages are shown so you can judge.', {
        status: 'no_answer', answer: null, message: 'The knowledge base does not cover this question.',
        question: 'Can I bring my dog to work?',
        closest: [
          { marker: 1, chunk_id: 'w#4', doc_id: 'w', title: 'Workplace guidelines', page: 4, section: 'Workplace guidelines > Visitors', snippet: 'Visitors must sign in at reception and be accompanied at all times...', score: 0.016, vector_score: 0.42 },
        ],
      }],
      ['no_documents', 'No documents', 'The collection is empty, or nothing in it is indexed yet.', {
        status: 'no_documents', answer: null, passages_considered: 0, message: 'This collection has no documents yet. Upload some to get started.',
      }],
      ['provider_unavailable', 'Provider unavailable', 'The chosen provider is not installed, not signed in, timed out or failed. The fix command is shown.', {
        status: 'provider_unavailable', answer: null, message: 'Claude Code is not signed in. Run: claude /login',
      }],
      ['rate_limited', 'Rate limited', 'Too many questions in a minute on this server, or the provider plan reached its limit.', {
        status: 'rate_limited', answer: null, message: 'Codex reached its usage limit. Try again later.',
        provider: { name: 'codex', model: null, elapsed_ms: 0 },
      }],
      ['retrieval_unavailable', 'Search unavailable', 'The document store did not respond, so no passages could be found.', {
        status: 'retrieval_unavailable', answer: null, provider: null, passages_considered: 0, message: 'Search is unavailable right now.',
      }],
    ];
    const tones = { answered: 'ready', no_answer: 'warn', no_documents: 'known', provider_unavailable: 'failed', rate_limited: 'warn', retrieval_unavailable: 'failed' };
    box.replaceChildren();
    for (const [status, label, meaning, extra] of samples) {
      const details = h('details', {},
        h('summary', {}, h('span', { class: 'chip chip-' + tones[status] }, label), meaning));
      details.addEventListener('toggle', () => {
        if (details.open && details.childElementCount === 1) {
          details.append(renderAnswer(Object.assign({}, base, extra), { preview: true, idPrefix: 'help-' + status + '-' }));
        }
      }, { once: false });
      box.append(details);
    }
    box.addEventListener('click', onAnswerClick);
  }

  // ------------------------------------------------------------------ start

  async function start() {
    setupTheme();
    restoreCached();
    setupTabs();
    setupCollectionForm();
    setupUploads();
    setupAsk();
    setupSettings();
    setupDocsLayout();
    renderHelpStates();
    const initial = parseHash(location.hash);
    let firstTab = recall('tab', 'documents');
    if (initial && initial.answer) firstTab = 'ask';
    else if (initial && initial.tab) firstTab = initial.tab;
    selectTab(firstTab, { route: 'none' });
    window.addEventListener('popstate', onLocationChange);
    window.addEventListener('hashchange', onLocationChange);
    const startHash = location.hash;
    await loadHealth();
    // The provider check can take seconds (it asks the CLIs); the page is usable before it ends.
    const providers = loadProviders(false);
    await loadCollections(initial ? initial.cid : null, 'none');
    // Correct the address once, unless the person has already moved on (asked a question, changed tab).
    if (location.hash === startHash) {
      if (initial && initial.answer) await loadStoredAnswer(initial.answer, 'replace');
      else setRoute(hashFor(state.current, state.tab), 'replace');
    }
    window.setInterval(loadHealth, HEALTH_EVERY_MS);
    await providers;
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
