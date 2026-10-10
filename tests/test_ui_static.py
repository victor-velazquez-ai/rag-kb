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


def test_css_colours_live_only_in_the_theme_variables():
    css = (UI_DIR / "styles.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    for body in re.findall(r"\{([^{}]*)\}", css):
        body = re.sub(r"--[\w-]+\s*:[^;]+;", "", body)
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\b(rgba?|hsla?)\(|:\s*(white|black)\b", body), body.strip()
