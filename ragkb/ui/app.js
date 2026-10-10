/* rag-kb UI: plain JavaScript, no build step, no external resources.
 *
 * Every string from the server or the user is rendered as text (textContent). The only HTML built from
 * strings is the answer's markdown, which is escaped first and then formatted by a small renderer.
 */
(() => {
  'use strict';

  const TABS = ['documents', 'ask', 'settings', 'connect', 'help'];
  const ACTIVE = new Set(['queued', 'processing', 'uploading', 'waiting']);
  const ARM_MS = 4000;
  const HEALTH_EVERY_MS = 20000;
  const PROVIDER_LABELS = { claude_code: 'Claude Code', codex: 'Codex', extractive: 'Extractive mode (no AI)' };
  const AUTO_ORDER = ['claude_code', 'codex', 'extractive'];
  const CLAUDE_MODELS = ['sonnet', 'opus', 'haiku'];
  const STATUS_LABELS = {
    queued: 'Queued', processing: 'Processing', ready: 'Ready', failed: 'Failed',
    uploading: 'Uploading', waiting: 'Waiting', known: 'Already indexed',
  };
  const TYPE_LABELS = {
    'application/pdf': 'PDF',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'Word',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'Excel',
    'text/markdown': 'Markdown', 'text/html': 'HTML', 'text/plain': 'Text', 'text/csv': 'CSV',
    'text/tab-separated-values': 'TSV',
  };
  const MCP_TOOLS = {
    list_collections: 'Lists the collections with their ids and how many documents each holds.',
    list_documents: 'Lists the documents of a collection with pages, passages and when each was indexed.',
    search_knowledge_base: 'Returns the best passages for a query, with title, page, section and score. No answer is written.',
    ask_knowledge_base: 'Returns a cited brief for a question. Your assistant writes the final answer from it.',
  };
  const OUT_OF_SCOPE_PROBE = 'What is the boiling point of mercury?';

  const state = {
    tab: 'documents',
    health: null,
    offline: false,
    providers: [],
    providersError: null,
    collections: [],
    collectionsError: null,
    current: null,
    docs: [],
    docsSig: '',
    docsError: null,
    stats: null,
    expanded: new Set(),
    uploads: [],
    answerToken: 0,
    asking: false,
  };
  const chunkCache = new Map();
  let docsTimer = null;
  let jobTimer = null;
  let jobBusy = false;
  let uploadSeq = 0;
  let activeUploads = 0;
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

  function whenCell(iso) {
    const w = fmtWhen(iso);
    return h('td', { class: 'when', title: w.title || null }, w.text || '');
  }

  function fmtSeconds(ms) {
    return (ms / 1000).toFixed(ms < 10000 ? 1 : 0) + ' s';
  }

  function fmtScore(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '';
    return value >= 1 ? value.toFixed(2) : value.toFixed(3);
  }

  function suffixOf(name) {
    const dot = name.lastIndexOf('.');
    return dot > 0 ? name.slice(dot).toLowerCase() : '';
  }

  function plural(n, one, many) {
    return n + ' ' + (n === 1 ? one : (many || one + 's'));
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

  // ------------------------------------------------------------------ tabs and routing

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
    if (opts.hash !== false) {
      const current = location.hash.replace(/^#/, '').split('/')[0];
      if (current !== name) history.replaceState(null, '', '#' + name);
    }
    if (name === 'settings') renderSettings(true);
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

  /* #documents, #ask, #settings, #connect, #help, and #ask/<answer id> for a stored answer. */
  function routeFromHash(withAnswer) {
    const [tab, arg] = location.hash.replace(/^#/, '').split('/');
    if (TABS.includes(tab)) {
      selectTab(tab, { hash: false });
      if (withAnswer && tab === 'ask' && arg) loadStoredAnswer(decodeURIComponent(arg));
      return true;
    }
    return false;
  }

  function isTyping(target) {
    return target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
  }

  // ------------------------------------------------------------------ health, banner, status bar

  async function loadHealth() {
    try {
      state.health = await api('GET', '/api/health');
      state.offline = false;
    } catch (e) {
      state.offline = true;
    }
    renderHealth();
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
      if (storeDown) {
        banner.className = 'banner';
        banner.textContent = 'The document store is not available. Uploads, search and answers can fail until it is back. See Settings for details.';
        banner.hidden = false;
      } else {
        banner.hidden = true;
      }
    }
    if (health) {
      $('status-version').textContent = health.knowledge_version || 'none yet';
      $('status-data').textContent = health.data_dir || '';
      $('status-data').title = health.data_dir || '';
      $('status-app-version').textContent = health.version || '';
      const hint = $('dropzone-hint');
      const types = (health.supported_types || []).join(' ');
      hint.textContent = (types ? types + ', ' : '') + 'up to ' + health.max_upload_mb + ' MB each';
      $('file-input').accept = (health.supported_types || []).join(',');
      const topk = $('topk-input');
      if (!topk.dataset.touched && !recall('topk', '')) topk.value = String(health.top_k || 6);
    }
    $('status-checked').textContent = 'checked ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
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

  function renderModelControl() {
    const field = $('model-field');
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
    if (!models.length || provider === 'extractive') {
      field.hidden = true;
      select.replaceChildren();
      return;
    }
    const previous = select.value || recall('model.' + provider, '');
    select.replaceChildren(h('option', { value: '' }, 'Default (' + fallback + ')'));
    for (const model of models) select.append(h('option', { value: model }, model));
    select.value = models.includes(previous) ? previous : '';
    field.hidden = false;
  }

  // ------------------------------------------------------------------ collections

  function currentCollection() {
    return state.collections.find((c) => c.id === state.current) || null;
  }

  async function loadCollections(preferId) {
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
    await selectCollection(target, true);
  }

  function renderCollections() {
    const list = $('collection-list');
    list.replaceChildren();
    for (const collection of state.collections) {
      const current = collection.id === state.current;
      list.append(h('li', {},
        h('button', {
          type: 'button',
          class: 'collection-item',
          'aria-current': current ? 'true' : null,
          title: collection.description || collection.name,
          onclick: () => selectCollection(collection.id, false),
        },
        h('span', { class: 'collection-name' }, collection.name),
        h('span', { class: 'count', 'aria-label': plural(collection.documents || 0, 'document') }, String(collection.documents || 0)))));
    }
    const empty = $('collection-empty');
    empty.hidden = state.collections.length > 0;
    empty.textContent = state.collectionsError
      ? 'Collections could not be loaded: ' + state.collectionsError
      : 'No collections yet. Create one below.';
  }

  async function selectCollection(id, force) {
    if (id === state.current && !force) return;
    const changed = id !== state.current;
    state.current = id || null;
    store('collection', id || '');
    if (changed) {
      state.docs = [];
      state.docsSig = '';
      state.stats = null;
      state.docsError = null;
      state.expanded.clear();
      clearAnswer();
    }
    renderCollections();
    renderCollectionHeader();
    renderDocs();
    renderUploads();
    renderAskAvailability();
    if (state.current) await loadDocs();
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
    if (state.stats) {
      parts.push(plural(state.stats.documents, 'document'));
      parts.push(plural(state.stats.chunks, 'passage'));
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
        await loadCollections(created.id);
        selectTab('documents');
      } catch (e) {
        error.textContent = e.message;
        error.hidden = false;
      } finally {
        button.disabled = false;
      }
    });

    armButton($('collection-delete'), {
      idle: 'Delete collection',
      armed: () => {
        const n = state.stats ? state.stats.documents : (currentCollection() || {}).documents || 0;
        return n ? 'Click again to delete it and ' + plural(n, 'document') : 'Click again to delete';
      },
      onConfirm: async () => {
        const collection = currentCollection();
        if (!collection) return;
        try {
          await api('DELETE', '/api/collections/' + enc(collection.id));
          announce('Collection ' + collection.name + ' deleted.');
          state.current = null;
          await loadCollections(null);
          loadHealth();
        } catch (e) {
          showBannerMessage('The collection could not be deleted: ' + e.message);
        }
      },
    });
  }

  function showBannerMessage(text) {
    const banner = $('banner');
    banner.className = 'banner warn';
    banner.textContent = text;
    banner.hidden = false;
    window.setTimeout(() => { if (banner.textContent === text) banner.hidden = true; }, 8000);
  }

  // ------------------------------------------------------------------ documents

  async function loadDocs() {
    const cid = state.current;
    if (!cid) return;
    window.clearTimeout(docsTimer);
    try {
      const [docs, stats] = await Promise.all([
        api('GET', '/api/collections/' + enc(cid) + '/documents'),
        api('GET', '/api/collections/' + enc(cid) + '/stats').catch(() => null),
      ]);
      if (cid !== state.current) return;
      state.docsError = null;
      state.stats = stats;
      const sig = JSON.stringify(docs);
      if (sig !== state.docsSig) {
        state.docs = docs;
        state.docsSig = sig;
        renderDocs();
        renderSuggestions();
      }
      const collection = currentCollection();
      if (collection && collection.documents !== docs.length) {
        collection.documents = docs.length;
        renderCollections();
      }
      renderCollectionHeader();
    } catch (e) {
      if (cid !== state.current) return;
      if (e.status === 404) {
        await loadCollections(null);
        return;
      }
      state.docsError = e.message;
      state.docsSig = '';
      renderDocs();
    }
    if (state.docs.some((d) => ACTIVE.has(d.status))) docsTimer = window.setTimeout(loadDocs, 2000);
  }

  function chip(status, label) {
    return h('span', { class: 'chip chip-' + status }, label || STATUS_LABELS[status] || status);
  }

  function renderDocs() {
    const has = Boolean(state.current);
    $('docs-no-collection').hidden = has;
    $('docs-body').hidden = !has;
    const body = $('docs-tbody');
    body.replaceChildren();
    if (!has) return;
    if (state.docsError) {
      body.append(h('tr', {}, h('td', { colspan: 9, class: 'table-message error-text' },
        'The documents could not be loaded: ' + state.docsError)));
      return;
    }
    if (!state.docs.length) {
      body.append(h('tr', {}, h('td', { colspan: 9, class: 'table-message' },
        h('strong', {}, 'No documents yet. '),
        'Drop files above, or choose files, to build this knowledge base.')));
      return;
    }
    for (const doc of state.docs) {
      body.append(docRow(doc));
      if (state.expanded.has(doc.id)) body.append(drawerRow(doc));
    }
  }

  function docRow(doc) {
    const busy = ACTIVE.has(doc.status);
    const expanded = state.expanded.has(doc.id);
    const titleCell = h('td', { class: 'title-cell' },
      h('div', { class: 'doc-title' }, doc.title),
      doc.filename && doc.filename !== doc.title ? h('div', { class: 'doc-file' }, doc.filename) : null,
      doc.status === 'failed' ? h('div', { class: 'doc-error' }, doc.error || 'Indexing failed.') : null);

    const passages = h('button', {
      type: 'button', class: 'btn btn-ghost btn-small',
      'aria-expanded': String(expanded), 'aria-controls': 'drawer-' + doc.id,
      disabled: doc.status !== 'ready',
      onclick: () => toggleDrawer(doc.id),
    }, expanded ? 'Hide passages' : 'Passages');
    const download = h('a', {
      class: 'btn btn-ghost btn-small', href: '/api/documents/' + enc(doc.id) + '/file',
      download: doc.filename || '', title: 'Download the original file',
    }, 'Download');
    const reindex = h('button', {
      type: 'button', class: 'btn btn-ghost btn-small', disabled: busy,
      title: 'Parse and embed again from the kept original',
      onclick: () => reindexDocument(doc, reindex),
    }, 'Re-index');
    const remove = armButton(h('button', { type: 'button', class: 'btn btn-danger-ghost btn-small' }), {
      idle: 'Delete',
      armed: 'Confirm delete',
      onConfirm: () => deleteDocument(doc),
    });

    return h('tr', { dataset: { doc: doc.id } },
      h('td', {}, chip(doc.status)),
      titleCell,
      h('td', {}, fmtType(doc.media_type)),
      h('td', { class: 'num' }, doc.pages === null || doc.pages === undefined ? '' : String(doc.pages)),
      h('td', { class: 'num' }, doc.status === 'ready' ? String(doc.chunk_count) : ''),
      h('td', { class: 'num' }, fmtBytes(doc.size_bytes)),
      whenCell(doc.uploaded_at),
      whenCell(doc.indexed_at),
      h('td', {}, h('div', { class: 'row-actions' }, passages, download, reindex, remove)));
  }

  function toggleDrawer(docId) {
    if (state.expanded.has(docId)) state.expanded.delete(docId);
    else state.expanded.add(docId);
    renderDocs();
    const button = document.querySelector('tr[data-doc="' + CSS.escape(docId) + '"] [aria-controls]');
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

  function drawerRow(doc) {
    const cell = h('td', { colspan: 9 }, h('p', { class: 'muted small' }, 'Loading passages...'));
    const row = h('tr', { class: 'drawer-row', id: 'drawer-' + doc.id }, cell);
    loadChunks(doc.id, doc.indexed_at).then((chunks) => {
      cell.replaceChildren(
        h('div', { class: 'drawer-head' }, h('strong', {}, plural(chunks.length, 'passage')), ' in reading order'),
        h('ol', { class: 'passage-list' }, chunks.map((c) => h('li', { class: 'passage' },
          h('div', { class: 'passage-meta' },
            h('span', { class: 'badge badge-muted plain' }, '#' + c.ordinal),
            c.section ? h('span', {}, c.section) : null,
            c.page !== null && c.page !== undefined ? h('span', {}, 'page ' + c.page) : null,
            c.token_estimate ? h('span', {}, 'about ' + c.token_estimate + ' tokens') : null),
          h('p', { class: 'passage-text' }, c.text)))));
    }).catch((e) => {
      cell.replaceChildren(h('p', { class: 'error-text small' }, 'The passages could not be loaded: ' + e.message));
    });
    return row;
  }

  async function reindexDocument(doc, button) {
    button.disabled = true;
    try {
      const job = await api('POST', '/api/documents/' + enc(doc.id) + '/reindex');
      addUploadItem({
        name: 'Re-index: ' + doc.title, size: doc.size_bytes, collectionId: doc.collection_id,
        status: job.status, jobId: job.id, docId: doc.id, progress: 100, kind: 'reindex',
      });
      ensureJobPolling();
      loadDocs();
    } catch (e) {
      button.disabled = false;
      showBannerMessage('Re-index could not start: ' + e.message);
    }
  }

  async function deleteDocument(doc) {
    try {
      await api('DELETE', '/api/documents/' + enc(doc.id));
      state.expanded.delete(doc.id);
      announce(doc.title + ' deleted.');
      await loadDocs();
      loadHealth();
    } catch (e) {
      showBannerMessage('The document could not be deleted: ' + e.message);
    }
  }

  // ------------------------------------------------------------------ uploads

  function setupUploads() {
    const zone = $('dropzone');
    const input = $('file-input');
    zone.addEventListener('click', () => { if (state.current) input.click(); });
    zone.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        if (state.current) input.click();
      }
    });
    input.addEventListener('change', () => {
      handleFiles(Array.from(input.files || []));
      input.value = '';
    });
    let depth = 0;
    zone.addEventListener('dragenter', (event) => { event.preventDefault(); depth += 1; zone.classList.add('dragging'); });
    zone.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) zone.classList.remove('dragging'); });
    zone.addEventListener('dragover', (event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; });
    zone.addEventListener('drop', (event) => {
      event.preventDefault();
      depth = 0;
      zone.classList.remove('dragging');
      handleFiles(Array.from((event.dataTransfer && event.dataTransfer.files) || []));
    });
    // A file dropped next to the zone must not replace the page.
    window.addEventListener('dragover', (event) => event.preventDefault());
    window.addEventListener('drop', (event) => event.preventDefault());
    $('uploads-clear').addEventListener('click', () => {
      state.uploads = state.uploads.filter((u) => ACTIVE.has(u.status));
      renderUploads();
    });
  }

  function handleFiles(files) {
    if (!state.current || !files.length) return;
    const health = state.health || {};
    const maxBytes = (health.max_upload_mb || 50) * 1024 * 1024;
    const types = new Set(health.supported_types || []);
    const titleInput = $('upload-title');
    const title = titleInput.value.trim();
    const valid = files.filter((f) => (!types.size || types.has(suffixOf(f.name))) && f.size <= maxBytes && f.size > 0);
    const useTitle = title && files.length === 1 && valid.length === 1;
    let titleNote = title && files.length > 1 ? 'The title was not used because several files were chosen.' : null;
    if (titleNote) announce(titleNote);
    for (const file of files) {
      const item = {
        name: file.name, size: file.size, collectionId: state.current, file,
        status: 'waiting', progress: 0, kind: 'upload', title: useTitle ? title : null, note: titleNote,
      };
      titleNote = null;
      if (types.size && !types.has(suffixOf(file.name))) {
        item.status = 'failed';
        item.note = 'This file type (' + (suffixOf(file.name) || 'no extension') + ') is not supported.';
      } else if (file.size > maxBytes) {
        item.status = 'failed';
        item.note = 'Larger than the ' + (health.max_upload_mb || 50) + ' MB limit per file.';
      } else if (file.size === 0) {
        item.status = 'failed';
        item.note = 'The file is empty.';
      }
      addUploadItem(item);
      if (item.status === 'waiting') {
        uploadQueue.push(item);
      }
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
      form.append('files', item.file, item.name);
      if (item.title) form.append('title', item.title);
      const xhr = new XMLHttpRequest();
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
      xhr.send(form);
    });
  }

  async function runUpload(item) {
    item.status = 'uploading';
    renderUploads();
    const { status, data } = await xhrUpload(item);
    item.file = null;
    const job = data && Array.isArray(data.jobs) ? data.jobs[0] : null;
    if (status === 202 && job) {
      item.jobId = job.id;
      item.docId = job.doc_id || null;
      item.progress = 100;
      if (job.already_indexed) {
        item.status = 'known';
        item.note = 'Same content as a document already in this collection, so it was kept once.';
        if (item.collectionId === state.current) loadDocs();
      } else {
        item.status = job.status;
        ensureJobPolling();
      }
    } else {
      item.status = 'failed';
      item.note = data && data.error ? data.error.message
        : (status === 0 ? 'The upload did not reach the server.' : 'The upload failed with status ' + status + '.');
    }
    renderUploads();
  }

  function ensureJobPolling() {
    if (!jobTimer) jobTimer = window.setInterval(pollJobs, 1000);
  }

  async function pollJobs() {
    if (jobBusy) return;
    const active = state.uploads.filter((u) => u.jobId && (u.status === 'queued' || u.status === 'processing'));
    if (!active.length) {
      window.clearInterval(jobTimer);
      jobTimer = null;
      return;
    }
    jobBusy = true;
    let changed = false;
    try {
      await Promise.all(active.map(async (item) => {
        try {
          const job = await api('GET', '/api/jobs/' + enc(item.jobId));
          if (job.doc_id) item.docId = job.doc_id;
          if (job.status !== item.status) {
            item.status = job.status;
            changed = true;
            if (job.status === 'failed') item.note = job.error || 'Indexing failed.';
            if (job.status === 'ready') {
              item.note = item.kind === 'reindex' ? 'Re-indexed.' : 'Indexed and searchable.';
              announce(item.name + ' is ready.');
            }
          }
        } catch (e) {
          if (e.status === 404) {
            item.status = 'failed';
            item.note = 'The server no longer knows this job.';
            changed = true;
          }
        }
      }));
    } finally {
      jobBusy = false;
    }
    if (changed) {
      renderUploads();
      loadDocs();
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

  function renderUploads() {
    const items = state.uploads.filter((u) => u.collectionId === state.current);
    $('upload-panel').hidden = !items.length;
    const list = $('upload-list');
    list.replaceChildren();
    for (const item of items) {
      const progress = h('progress', { max: '100', 'aria-label': 'Upload progress for ' + item.name });
      if (item.status === 'uploading' || item.status === 'waiting') progress.value = item.progress || 0;
      else progress.value = 100;
      list.append(h('li', { class: 'upload-item' },
        h('span', { class: 'upload-name', title: item.name }, item.name, h('span', { class: 'muted small' }, '  ' + fmtBytes(item.size))),
        progress,
        chip(item.status),
        item.note ? h('span', { class: 'upload-note' + (item.status === 'failed' ? ' error-text' : '') }, item.note) : null));
    }
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
    const pick = (text, probe) => h('button', {
      type: 'button', class: 'suggestion' + (probe ? ' probe' : ''),
      title: probe ? 'A question the documents should not cover: a good knowledge base says so instead of guessing.' : text,
      onclick: () => {
        $('question').value = text;
        $('question').focus();
      },
    }, probe ? text + ' (out of scope test)' : text);
    for (const doc of ready.slice(0, 2)) box.append(pick('What does "' + doc.title + '" cover?', false));
    box.append(pick(OUT_OF_SCOPE_PROBE, true));
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
    const modelField = $('model-field');
    const model = !modelField.hidden ? $('model-select').value : '';
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
      history.replaceState(null, '', '#ask/' + enc(answer.id));
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

  async function loadStoredAnswer(id) {
    try {
      const answer = await api('GET', '/api/answers/' + enc(id));
      if (answer.collection_id !== state.current && state.collections.some((c) => c.id === answer.collection_id)) {
        await selectCollection(answer.collection_id, false);
      }
      $('question').value = answer.question;
      $('answer-area').replaceChildren(renderAnswer(answer, {}));
    } catch (e) {
      $('answer-area').replaceChildren(errorCard('That answer could not be loaded', e.message));
    }
  }

  function pseudoAnswer(question, cid, status, message) {
    return {
      id: '', question, collection_id: cid, status, answer: null, message, citations: [],
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
      return {
        tone: 'warn', icon: '?', title: 'Not covered by these documents', message,
        hint: 'The knowledge base does not guess.' + checked + ' Try other words, or upload a document that covers it.',
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

    if (answer.citations && answer.citations.length) card.append(renderSources(answer.citations, prefix, opts.preview));
    if (!opts.preview && answer.id) {
      if (answered || answer.status === 'no_answer') card.append(renderFeedback(answer));
      card.append(renderRaw(answer, opts.request));
    }
    return card;
  }

  function renderSources(citations, prefix, preview) {
    const top = Math.max(...citations.map((c) => (typeof c.score === 'number' ? c.score : 0)), 0);
    const list = h('ol', { class: 'source-list' });
    for (const c of citations) {
      const where = [];
      if (c.page !== null && c.page !== undefined) where.push('page ' + c.page);
      if (c.section) where.push(c.section);
      const fill = h('span', { class: 'scorebar-fill' });
      const ratio = top > 0 && typeof c.score === 'number' ? Math.max(0.04, c.score / top) : 0;
      fill.style.width = Math.round(ratio * 100) + '%';
      const passage = h('div', { class: 'source-passage', hidden: true });
      const open = preview ? null : h('button', {
        type: 'button', class: 'btn btn-ghost btn-small', 'aria-expanded': 'false',
        onclick: () => togglePassage(c, passage, open),
      }, 'Open passage');
      list.append(h('li', { class: 'source-card', id: prefix + 'source-' + c.marker, tabindex: '-1' },
        h('div', { class: 'source-head' },
          h('span', { class: 'cite-badge' }, String(c.marker)),
          h('span', { class: 'source-title' }, c.title),
          where.length ? h('span', { class: 'muted small' }, where.join('  |  ')) : null),
        c.snippet ? h('p', { class: 'snippet' }, c.snippet) : null,
        h('div', { class: 'source-foot' },
          h('span', { class: 'scorebar', role: 'img', 'aria-label': 'Relative score ' + Math.round(ratio * 100) + ' percent' }, fill),
          h('span', { class: 'muted small' }, 'score ' + fmtScore(c.score)),
          open),
        passage));
    }
    return h('section', { class: 'sources', 'aria-label': 'Sources' },
      h('h3', {}, 'Sources'), list,
      h('p', { class: 'muted small' }, 'Bars are relative to the strongest source in this answer. See Help for how to read the scores.'));
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
        result.textContent = 'Thanks, your feedback was saved.';
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

  function shellQuote(text) {
    return "'" + String(text).replace(/'/g, "'\\''") + "'";
  }

  function renderRaw(answer, request) {
    const details = h('details', { class: 'raw' }, h('summary', {}, 'Raw response and request'));
    details.append(h('pre', {}, JSON.stringify(answer, null, 2)));
    if (request) {
      const url = location.origin + '/api/collections/' + enc(answer.collection_id) + '/ask';
      const curl = 'curl -s -X POST ' + shellQuote(url) + ' -H ' + shellQuote('Content-Type: application/json') + ' -d ' + shellQuote(JSON.stringify(request));
      details.append(h('p', { class: 'muted small' }, 'The same request from a terminal:'), codeBlock(curl, 'Copy'));
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
    body.append(h('section', { class: 'card' }, h('h3', {}, 'Index'), kv([
      ['Collections', counts.collections],
      ['Documents', counts.documents],
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
      const byStatus = Object.entries(stats.by_status || {}).map(([k, v]) => v + ' ' + k).join(', ');
      const byType = Object.entries(stats.by_media_type || {}).map(([k, v]) => v + ' ' + fmtType(k)).join(', ');
      body.append(h('section', { class: 'card' }, h('h3', {}, 'This collection'), kv([
        ['Name', stats.name],
        ['Documents', stats.documents + (byStatus ? ' (' + byStatus + ')' : '')],
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
      ['File types', (health.supported_types || []).join(' ')],
      ['Questions', limit.per_minute ? limit.per_minute + ' per minute (' + limit.scope + ')' : null],
      ['Version', 'rag-kb ' + health.version],
    ])));

    if (health.problems && health.problems.length) {
      body.append(h('section', { class: 'card wide' }, h('h3', {}, 'Problems'),
        h('ul', { class: 'problem-list' }, health.problems.map((p) => h('li', {}, p))),
        h('p', { class: 'muted small' }, 'Run "ragkb doctor" in a terminal for a full check.')));
    }
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
      id: '', question: 'How many days of annual leave do employees get?', collection_id: 'sample', citations: [],
      passages_considered: 6, provider: { name: 'claude_code', model: 'sonnet', elapsed_ms: 11800 },
      knowledge_version: 'v20261010.3', language: 'en', created_at: new Date().toISOString(),
    };
    const samples = [
      ['answered', 'Answered', 'A direct answer first, then details, with [n] markers that point at the sources.', {
        status: 'answered', message: null,
        answer: 'Employees get **25 days** of annual leave per calendar year [1].\n\n- Up to five unused days carry over to the next year [1].\n- Carried-over days expire at the end of March [2].',
        citations: [
          { marker: 1, chunk_id: 's#1', doc_id: 's', title: 'Leave policy', page: 2, section: 'Leave policy > Annual leave', snippet: 'Every employee gets 25 days of annual leave per calendar year. Up to five unused days carry over...', score: 0.032 },
          { marker: 2, chunk_id: 's#2', doc_id: 's', title: 'Leave policy', page: 3, section: 'Leave policy > Carry-over', snippet: 'Days carried over expire at the end of March of the following year.', score: 0.024 },
        ],
      }],
      ['no_answer', 'Not covered', 'The documents were searched but none of them answers the question. Nothing is guessed.', {
        status: 'no_answer', answer: null, message: 'The knowledge base does not cover this question.',
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
    setupTabs();
    setupCollectionForm();
    setupUploads();
    setupAsk();
    renderHelpStates();
    if (!routeFromHash(false)) selectTab(recall('tab', 'documents'), { hash: false });
    window.addEventListener('hashchange', () => routeFromHash(true));
    await loadHealth();
    await Promise.all([loadCollections(null), loadProviders(false)]);
    if (/^#ask\/./.test(location.hash)) routeFromHash(true);
    window.setInterval(loadHealth, HEALTH_EVERY_MS);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
