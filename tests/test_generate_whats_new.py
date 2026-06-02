from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import urllib.error


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_whats_new.py"
SPEC = importlib.util.spec_from_file_location("generate_whats_new", SCRIPT_PATH)
assert SPEC and SPEC.loader
generate_whats_new = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_whats_new)


def test_resolve_raw_notes_prefers_release_body(monkeypatch):
    monkeypatch.setattr(generate_whats_new, "_fetch_release_body_from_github", lambda tag, token: "api-body")
    monkeypatch.setattr(generate_whats_new, "_read_changelog_section", lambda tag: "changelog-body")

    result = generate_whats_new._resolve_raw_notes(
        tag="v1.2.0-dev.3",
        release_body="event-body",
        token="token",
    )

    assert result == "event-body"


def test_resolve_raw_notes_uses_github_api_before_changelog(monkeypatch):
    monkeypatch.setattr(generate_whats_new, "_fetch_release_body_from_github", lambda tag, token: "api-body")
    monkeypatch.setattr(generate_whats_new, "_read_changelog_section", lambda tag: "changelog-body")

    result = generate_whats_new._resolve_raw_notes(
        tag="v1.2.0-dev.3",
        release_body="",
        token="token",
    )

    assert result == "api-body"


def test_fetch_release_body_from_github_returns_empty_and_warns_on_failure(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_REPOSITORY", "RussellSB/pytrendy")

    def _raise_urlerror(*_args, **_kwargs):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(
        generate_whats_new.urllib.request,
        "urlopen",
        _raise_urlerror,
    )

    result = generate_whats_new._fetch_release_body_from_github("v1.2.0-dev.3", "token")

    assert result == ""
    assert "Warning: failed to fetch release notes" in capsys.readouterr().err


def test_fetch_release_body_from_github_returns_empty_and_warns_on_timeout(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_REPOSITORY", "RussellSB/pytrendy")

    def _raise_timeout(*_args, **_kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        generate_whats_new.urllib.request,
        "urlopen",
        _raise_timeout,
    )

    result = generate_whats_new._fetch_release_body_from_github("v1.2.0-dev.3", "token")

    assert result == ""
    assert "Warning: failed to fetch release notes" in capsys.readouterr().err


def test_fetch_release_body_from_github_reads_body(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "RussellSB/pytrendy")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"body": "notes from api"}).encode("utf-8")

    monkeypatch.setattr(
        generate_whats_new.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(),
    )

    result = generate_whats_new._fetch_release_body_from_github("v1.2.0-dev.3", "token")

    assert result == "notes from api"
