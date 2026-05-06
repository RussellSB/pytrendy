#!/usr/bin/env python3
"""Generate or update docs/whats-new.md using the GitHub Models API (Claude 3.5 Haiku).

The script:
  1. Reads the latest release notes (from the RELEASE_BODY env var or CHANGELOG.md).
  2. Calls the GitHub Models API (claude-3-5-haiku) to produce a user-friendly
     What's New section in MkDocs-compatible Markdown.
  3. Prepends the new section into docs/whats-new.md between the sentinel
     comment markers, preserving the rest of the file.

Environment variables
---------------------
GITHUB_TOKEN   – required; used to authenticate against the GitHub Models API.
RELEASE_TAG    – the semantic-release tag (e.g. "v1.2.0" or "v1.2.0-dev.1").
RELEASE_NAME   – human-readable release title.
RELEASE_BODY   – raw Markdown body of the GitHub Release (release notes).
IS_PRERELEASE  – "true" / "false"; controls the "Coming in…" vs "Released in…" framing.
BRANCH         – the branch being released from (default: "develop").
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
WHATS_NEW_PATH = REPO_ROOT / "docs" / "whats-new.md"
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"

CONTENT_START = "<!-- WHATS_NEW_CONTENT_START -->"
CONTENT_END = "<!-- WHATS_NEW_CONTENT_END -->"

GITHUB_MODELS_URL = "https://models.inference.ai.azure.com/chat/completions"
MODEL = "claude-3-5-haiku"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _read_changelog_section(tag: str) -> str:
    """Extract the changelog block for a given tag from CHANGELOG.md."""
    if not CHANGELOG_PATH.exists():
        return ""
    text = CHANGELOG_PATH.read_text(encoding="utf-8")
    # Match the heading line containing the version number
    version = tag.lstrip("v").split("-")[0]  # strip "v" and pre-release suffix
    pattern = re.compile(
        rf"(?:^|\n)(#{1,2}\s+\[?{re.escape(version)}\]?.*?)(?=\n#{1,2}\s|\Z)",
        re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _call_github_models(prompt: str, token: str) -> str | None:
    """Call the GitHub Models API (Claude 3.5 Haiku) and return the generated text, or None on failure."""
    system = textwrap.dedent("""\
        You are a technical writer for PyTrendy, a Python library for time-series trend
        detection. Your task is to turn raw CHANGELOG / release-note Markdown into a
        concise, user-friendly "What's New" entry for the library's MkDocs documentation
        site.

        Guidelines:
        - Focus on user impact, not internal implementation details.
        - Use clear, plain language aimed at data scientists and analysts.
        - Start with a one-sentence summary of the release.
        - Wrap each individual change in a `??? note "Change title"` collapsible block
          (MkDocs pymdownx.details syntax) so the page stays scannable.
        - Inside collapsible blocks, use `??? example "Code"` for code samples so code
          is hidden by default — visuals and notes lead.
        - Use `=== "Before"` / `=== "After"` tabbed blocks for behaviour-changing fixes.
        - Group small patch releases (1–2 minor fixes each) into a single combined section
          rather than giving each its own heading.
        - Reference issue or PR numbers from the CHANGELOG where available (e.g. `[#12](url)`).
        - Avoid emoji in headings. Keep emoji use minimal overall.
        - Do NOT include the top-level ## heading; the caller will add it.
        - Do NOT wrap the output in a code fence.
    """)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.5,
    }

    req = urllib.request.Request(
        GITHUB_MODELS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"[whats-new] GitHub Models API unavailable: {exc}", file=sys.stderr)
        return None


def _build_fallback_section(
    tag: str,
    raw_notes: str,
    is_prerelease: bool,
) -> str:
    """Build a minimal What's New section without AI, from raw release notes."""
    lines: list[str] = []

    # Parse raw_notes into Bug Fixes / Features buckets, skipping Markdown headings
    fixes: list[str] = []
    features: list[str] = []
    other: list[str] = []
    for line in raw_notes.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Strip leading list markers and asterisks
        content = stripped.lstrip("*- ").strip()
        if not content:
            continue
        lower = content.lower()
        if any(k in lower for k in ("feat", "feature", "new", "add")):
            # Strip "feat: " / "feat(scope): " prefixes from conventional commits
            content = re.sub(r"^feat(?:\([^)]*\))?!?:\s*", "", content, flags=re.IGNORECASE)
            features.append(f"    - {content}")
        elif any(k in lower for k in ("fix", "bug", "resolv", "patch", "correct")):
            content = re.sub(r"^fix(?:\([^)]*\))?!?:\s*", "", content, flags=re.IGNORECASE)
            fixes.append(f"    - {content}")
        else:
            other.append(f"    - {content}")

    if features:
        lines.append('??? note "New Features"\n')
        lines.extend(features)
        lines.append("")

    if fixes:
        lines.append('??? note "Bug Fixes"\n')
        lines.extend(fixes)
        lines.append("")

    if other:
        lines.append('??? note "Improvements"\n')
        lines.extend(other)
        lines.append("")

    if not features and not fixes and not other:
        # Dump raw notes as-is when we cannot parse them
        lines.append(raw_notes)

    return "\n".join(lines).strip()


def _build_section(
    tag: str,
    raw_notes: str,
    is_prerelease: bool,
    token: str,
) -> str:
    """Return the full Markdown block for one release (without top-level heading)."""
    prompt = textwrap.dedent(f"""\
        PyTrendy release {tag} ({'pre-release / develop' if is_prerelease else 'stable'}).

        Raw release notes (Markdown):
        ---
        {raw_notes}
        ---

        Write a "What's New" documentation entry for this release.
        Omit the top-level ## heading — I will add it myself.
    """)

    ai_content = _call_github_models(prompt, token) if token else None
    return ai_content or _build_fallback_section(tag, raw_notes, is_prerelease)


def _make_heading(tag: str, is_prerelease: bool, date_str: str) -> str:
    version = tag.lstrip("v")
    if is_prerelease:
        base = version.split("-")[0]
        return (
            f'## Coming in v{base} <span class="version-prerelease">pre-release</span>\n\n'
            f"*Staged on the `develop` branch — will land in the next stable release.*"
        )
    return f"## Released in v{version}\n\n> Released {date_str}"


def _inject_section(file_path: Path, new_block: str) -> None:
    """Prepend `new_block` inside the sentinel markers in `file_path`."""
    text = file_path.read_text(encoding="utf-8")

    start_idx = text.find(CONTENT_START)
    end_idx = text.find(CONTENT_END)

    if start_idx == -1 or end_idx == -1:
        print(
            f"[whats-new] Sentinel markers not found in {file_path}. Appending instead.",
            file=sys.stderr,
        )
        file_path.write_text(
            text.rstrip() + "\n\n" + new_block + "\n",
            encoding="utf-8",
        )
        return

    before = text[: start_idx + len(CONTENT_START)]
    after = text[end_idx:]

    # Extract existing content (between the markers)
    existing = text[start_idx + len(CONTENT_START) : end_idx].strip()

    updated = (
        before
        + "\n\n"
        + new_block.strip()
        + ("\n\n---\n\n" + existing if existing else "")
        + "\n\n"
        + after
    )

    file_path.write_text(updated, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    token = _env("GITHUB_TOKEN")
    tag = _env("RELEASE_TAG") or "v0.0.0"
    release_body = _env("RELEASE_BODY")
    is_prerelease = _env("IS_PRERELEASE", "false").lower() == "true"
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # Prefer the GitHub Release body; fall back to CHANGELOG.md
    raw_notes = release_body or _read_changelog_section(tag)

    if not raw_notes:
        print(
            f"[whats-new] No release notes found for {tag}. Nothing to do.",
            file=sys.stderr,
        )
        sys.exit(0)

    print(f"[whats-new] Generating entry for {tag} (prerelease={is_prerelease})…")

    body = _build_section(tag, raw_notes, is_prerelease, token)
    heading = _make_heading(tag, is_prerelease, date_str)
    new_block = heading + "\n\n" + body

    _inject_section(WHATS_NEW_PATH, new_block)
    print(f"[whats-new] Updated {WHATS_NEW_PATH}")


if __name__ == "__main__":
    main()
