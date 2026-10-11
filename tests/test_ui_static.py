"""The single-page UI: served at /, self-contained (no external scripts or styles), safe headers."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragkb.config import Settings
from ragkb.server.app import create_app

UI_DIR = Path(__file__).resolve().parents[1] / "ragkb" / "ui"
TABS = ("documents", "ask", "settings", "connect", "help")
# The API answers only loopback hosts unless RAGKB_API_PUBLIC is set.
LOCAL_BASE = "http://127.0.0.1:8765"


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("RAGKB_FAKE_EMBEDDER", "1")
    return Settings(_env_file=None, DATA_DIR=tmp_path / "data", MCP_ENABLED=False, LOG_LEVEL="WARNING")


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings), base_url=LOCAL_BASE, raise_server_exceptions=False) as client:
        yield client


def test_index_is_served_at_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    for tab in TABS:
        assert f'id="tab-{tab}"' in html
        assert f'id="panel-{tab}"' in html
        assert f'aria-controls="panel-{tab}"' in html
    assert 'role="tablist"' in html
    assert "/ui/app.js" in html and "/ui/styles.css" in html


def test_security_headers_on_the_page(client):
    headers = client.get("/").headers
    csp = headers["content-security-policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp
    assert headers["x-content-type-options"] == "nosniff"


def test_assets_are_served_with_their_types(client):
    js = client.get("/ui/app.js")
    css = client.get("/ui/styles.css")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert css.status_code == 200 and "text/css" in css.headers["content-type"]


def test_deep_links_serve_the_page_and_missing_assets_are_404(client):
    deep = client.get("/ui/ask")
    assert deep.status_code == 200 and 'id="tab-ask"' in deep.text
    missing = client.get("/ui/missing.js")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_path_traversal_is_blocked(client):
    for path in ("/ui/%2e%2e/config.py", "/ui/..%2f..%2fpyproject.toml", "/ui/%2e%2e/%2e%2e/pyproject.toml"):
        response = client.get(path)
        assert "BaseSettings" not in response.text and "[project]" not in response.text


def test_ui_can_be_disabled(settings):
    app = create_app(settings.model_copy(update={"UI_ENABLED": False}))
    with TestClient(app, base_url=LOCAL_BASE, raise_server_exceptions=False) as client:
        response = client.get("/")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
        assert client.get("/api/health").status_code == 200


def test_no_external_scripts_or_stylesheets():
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    for tag in re.findall(r"<script\b[^>]*>", html, flags=re.IGNORECASE):
        src = re.search(r"\bsrc\s*=\s*[\"']([^\"']+)", tag, flags=re.IGNORECASE)
        assert src is not None, "inline scripts are blocked by the page's CSP"
        assert not re.match(r"(https?:)?//", src.group(1), flags=re.IGNORECASE), tag
    for tag in re.findall(r"<link\b[^>]*>", html, flags=re.IGNORECASE):
        href = re.search(r"\bhref\s*=\s*[\"']([^\"']+)", tag, flags=re.IGNORECASE)
        if href:
            assert not re.match(r"(https?:)?//", href.group(1), flags=re.IGNORECASE), tag
    assert not re.search(r"\bstyle\s*=", html, flags=re.IGNORECASE), "inline style attributes are blocked by the CSP"

    css = (UI_DIR / "styles.css").read_text(encoding="utf-8")
    assert "@import" not in css
    assert not re.search(r"url\(\s*[\"']?(https?:)?//", css, flags=re.IGNORECASE)

    js = (UI_DIR / "app.js").read_text(encoding="utf-8")
    assert not re.search(r"createElement\(\s*[\"']script", js)
    assert not re.search(r"\bimport\s*\(", js)
    assert not re.search(r"https?://(?!127\.0\.0\.1|localhost)[\w.-]+\.[a-z]{2,}", js), "no external hosts"


def test_ui_files_have_no_long_dashes():
    for name in ("index.html", "app.js", "styles.css"):
        text = (UI_DIR / name).read_text(encoding="utf-8")
        assert "\u2014" not in text and "\u2013" not in text, name


def _custom_properties(block: str) -> dict[str, str]:
    return dict(re.findall(r"--([\w-]+)\s*:\s*([^;]+);", block))


def _rule_body(css: str, selector_pattern: str) -> str:
    match = re.search(selector_pattern + r"\s*\{([^{}]*)\}", css)
    assert match is not None, selector_pattern
    return match.group(1)


def test_theme_switch_is_in_the_page():
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    switch = re.search(r'<fieldset class="theme-switch">(.*?)</fieldset>', html, flags=re.DOTALL)
    assert switch is not None, "the sidebar has a theme switch"
    assert re.search(r"<legend[^>]*>Theme</legend>", switch.group(1))
    for value, label in (("system", "System"), ("light", "Light"), ("dark", "Dark")):
        assert re.search(rf'<input type="radio" name="theme" id="theme-{value}" value="{value}"', switch.group(1))
        assert re.search(rf'<label for="theme-{value}"[^>]*>{label}</label>', switch.group(1))
    # The theme is applied before the first paint, so the script that does it must not be deferred.
    head = html.split("</head>", 1)[0]
    script = re.search(r'<script src="/ui/app\.js"([^>]*)>', head)
    assert script is not None and "defer" not in script.group(1) and "async" not in script.group(1)

    js = (UI_DIR / "app.js").read_text(encoding="utf-8")
    assert "recall('theme'" in js and "store('theme'" in js
    assert "setAttribute('data-theme'" in js and "removeAttribute('data-theme')" in js


def test_css_defines_light_system_dark_and_forced_dark():
    css = (UI_DIR / "styles.css").read_text(encoding="utf-8")
    light = _custom_properties(_rule_body(css, r"(?m)^:root"))
    assert re.search(r':root\[data-theme="light"\]\s*\{[^}]*color-scheme:\s*light', css)
    media = re.search(r"@media \(prefers-color-scheme: dark\)\s*\{\s*:root:not\(\[data-theme=\"light\"\]\)\s*\{([^{}]*)\}",
                      css)
    assert media is not None, "the system dark theme must step aside when Light is picked"
    system_dark = _custom_properties(media.group(1))
    forced_dark = _custom_properties(_rule_body(css, r':root\[data-theme="dark"\]'))
    assert system_dark and system_dark == forced_dark, "the two dark sets must stay identical"
    assert set(system_dark) <= set(light), "every dark colour needs a light counterpart"


def _html() -> str:
    return (UI_DIR / "index.html").read_text(encoding="utf-8")


def _js() -> str:
    return (UI_DIR / "app.js").read_text(encoding="utf-8")


def test_folder_and_zip_upload_controls():
    html = _html()
    folder = re.search(r"<input\b[^>]*\bid=\"folder-input\"[^>]*>", html)
    assert folder is not None, "a folder picker exists"
    assert re.search(r"\bwebkitdirectory\b", folder.group(0)) and re.search(r"\bmultiple\b", folder.group(0))
    assert re.search(r'<button\b[^>]*\bid="add-folder"[^>]*>Add folder</button>', html)
    js = _js()
    # Each folder file is sent with its path, aligned with the file in the same request.
    assert "webkitRelativePath" in js and "form.append('relative_paths'" in js
    # Dropped folders are walked, and .zip is accepted by the picker and the drop zone.
    assert "webkitGetAsEntry" in js and "readEntries" in js
    assert "const ARCHIVE = '.zip'" in js and "types.add(ARCHIVE)" in js


def test_upload_rows_read_the_real_response():
    js = _js()
    # The 202 body is {jobs, rejected, skipped}: a Job per accepted file, {filename, source_path, code,
    # message} per refused one, and the number of zip entries left out. No other shape is guessed at.
    assert "data.jobs.map((job) => ({ job }))" in js
    assert "(data.rejected || []).map((rejected) => ({ rejected }))" in js and "Number(data.skipped)" in js
    assert "source.source_path || source.filename" in js
    for guess in ("'results'", "'items'", "'code' in entry", "entry.error", "relative_path ||"):
        assert guess not in js, guess
    # /retry answers 409 for a row that did not fail; only then the page falls back to /reindex.
    assert "if (e.status !== 409)" in js and "e.status !== 405" not in js
    # Counts come from the documented keys only.
    assert "stats.by_status" not in js and "Number(collection.documents)" not in js
    for wording in ("'Rejected: '", "'Replaced '", "known: 'Already indexed'"):
        assert wording in js, wording
    hint = re.search(r'<p id="identity-hint"[^>]*>(.*?)</p>', _html(), flags=re.DOTALL)
    assert hint is not None
    text = " ".join(hint.group(1).split())
    assert "file name" in text and "path" in text and "may repeat" in text


def test_job_progress_cancel_and_queue_summary():
    js = _js()
    assert "'/api/jobs/' + enc(jobId) + '/cancel'" in js
    assert "dataset: { action: 'cancel' }" in js and "actionButton('cancel', 'Cancel'" in js
    for field in ("chunks_done", "chunks_total", "eta_seconds", "cancel_requested", "stage"):
        assert f"job.{field}" in js, field
    assert "'/jobs/summary'" in js and 'id="docs-summary"' in _html()
    # Delete, dismiss, re-index and retry show a pending state, with a note after two seconds.
    assert "const PENDING_NOTE_MS = 2000" in js and "function withPending" in js
    assert "'/retry'" in js and "'Dismiss'" in js


def test_documents_table_fits_its_container():
    css = (UI_DIR / "styles.css").read_text(encoding="utf-8")
    table = _rule_body(css, r"\n\.docs-table")
    assert "table-layout: fixed" in table and "820px" not in table
    for column in ("col-status", "col-actions"):
        assert re.search(rf"\.docs-table \.{column} \{{ width: \d+px; \}}", css), column
    assert "container-type: inline-size" in css and "@container docs" in css
    assert re.search(r"\.main \{[^}]*min-width: 0", css)
    # The passages drawer spans every visible column, so it never sits inside a scrolled-off column.
    assert "td.span-all" in _js() and "visibleColumns()" in _js()


def test_scores_closest_passages_and_counts():
    js = _js()
    assert "c.vector_score" in js and "c.rerank_score" in js
    assert "'similarity '" in js and "'reranker '" in js and "'rank '" in js
    assert "closest-passages" in js and "answer.closest" in js and "'Closest passages'" in js
    assert "'searchable document'" in js and "documents_by_status" in js and "documents_ready" in js
    help_text = " ".join(_html().split())
    assert "The rank score orders the sources; the similarity and reranker scores say how relevant each one is." in help_text


def test_model_menu_is_disabled_without_models():
    html = _html()
    assert re.search(r'<label class="control" id="model-field">', html), "the Model field is never hidden"
    js = _js()
    assert "'no model'" in js and "select.disabled = true" in js


def test_accessible_names():
    js = _js()
    assert "'aria-labelledby': nameId" in js and "'aria-describedby': descId" in js
    assert "'aria-describedby': 'probe-hint'" in js and 'id="probe-hint"' in _html()


def test_shell_switch_for_the_terminal_request():
    js = _js()
    assert "const SHELLS = [['bash', 'bash'], ['powershell', 'PowerShell'], ['cmd', 'cmd']]" in js
    assert "recall('shell'" in js and "store('shell'" in js
    assert "Invoke-RestMethod" in js and "curl -s -X POST" in js
    # A literal backslash-u in the source would be easy to corrupt; the escape is built from the character code.
    assert "String.fromCharCode(92)" in js


def test_feedback_review_panel():
    html = _html()
    panel = re.search(r'<section id="feedback-panel"[^>]*>(.*?)</section>', html, flags=re.DOTALL)
    assert panel is not None and 'id="feedback-body"' in panel.group(1)
    assert "/feedback?limit=100" in _js() and "'#/answer/' + enc(f.answer_id)" in _js()


def test_hash_routing_and_cached_status():
    js = _js()
    assert "'#/c/' + enc(collectionId) + '/' + tab" in js and "'#/answer/'" in js
    assert "history.pushState" in js and "history.replaceState" in js
    assert "addEventListener('popstate'" in js and "addEventListener('hashchange'" in js
    # The last known health and providers render at once on a reload.
    assert "recallJson('health')" in js and "recallJson('providers')" in js


def test_css_colours_live_only_in_the_theme_variables():
    css = (UI_DIR / "styles.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    for body in re.findall(r"\{([^{}]*)\}", css):
        body = re.sub(r"--[\w-]+\s*:[^;]+;", "", body)
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\b(rgba?|hsla?)\(|:\s*(white|black)\b", body), body.strip()
