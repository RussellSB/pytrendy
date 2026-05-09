#!/usr/bin/env python3
"""Generate or update docs/whats-new.md using the GitHub Models API (Claude Sonnet 4).

The script:
  1. Reads the latest release notes (from the RELEASE_BODY env var or CHANGELOG.md).
  2. Calls the GitHub Models API (claude-sonnet-4) to produce a user-friendly
     What's New section in MkDocs-compatible Markdown.
  3. Prepends the new section into docs/whats-new.md between the sentinel
     comment markers, preserving the rest of the file.

Agent instructions (run before each refresh):
  - Verify that CSV file URLs referenced in code examples still resolve.
    Files live in tests/tests_crashes_edgecases/data/ on both develop and main.
    Use the develop branch URL for pre-releases, main branch URL for stable releases.
    Example check: curl -sI <raw_url> | grep "200"
  - Before/after plot comparisons must be visually consistent. For bug fixes or features
    directly related to time-series trend detection, generate both images via the same
    `detect_trends()` + `plot_pytrendy()` pipeline so that figsize, grid style, legend,
    and color scheme are identical between the two images.

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

NOTE_START = "<!-- WHATS_NEW_NOTE_START -->"
NOTE_END = "<!-- WHATS_NEW_NOTE_END -->"

GITHUB_MODELS_URL = "https://models.inference.ai.azure.com/chat/completions"
MODEL = "claude-sonnet-4"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _base_version(tag: str) -> str:
    """Return the base semver string from a tag, stripping the leading 'v' and any pre-release suffix.

    Examples:
        "v1.2.0"        → "1.2.0"
        "v1.3.0-dev.1"  → "1.3.0"
    """
    return tag.lstrip("v").split("-")[0]


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
    """Call the GitHub Models API (Claude Sonnet 4) and return the generated text, or None on failure."""
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
        - Show before/after images stacked vertically (one column) inside a
          `<div class="before-after-grid">` with two `<div class="before-after-panel">` children,
          each labelled with `<span class="before-after-label before-label">Before</span>` or
          `<span class="before-after-label after-label">After</span>`.
        - Group small patch releases (1–2 minor fixes each) into a single combined section
          rather than giving each its own heading.
        - Reference issue or PR numbers from the CHANGELOG where available (e.g. `[#12](url)`).
        - In code examples, load CSV test data via GitHub raw URLs rather than local paths.
          Use the develop branch URL for pre-releases:
          `https://raw.githubusercontent.com/RussellSB/pytrendy/develop/tests/tests_crashes_edgecases/data/<file>.csv`
          Use the main branch URL for stable releases:
          `https://raw.githubusercontent.com/RussellSB/pytrendy/main/tests/tests_crashes_edgecases/data/<file>.csv`
          Bundled datasets (series_synthetic, classes_signals) may use `pt.load_data(name)` directly.
        - Avoid emoji in headings and admonition titles. Keep emoji use minimal overall.
        - Do NOT include the top-level ## heading; the caller will add it.
        - Do NOT wrap the output in a code fence.
        - Before/after images must be visually comparable. For any bug fix or feature
          directly related to time-series trend detection, produce both the "before" and
          "after" images through the same `detect_trends()` + `plot_pytrendy()` pipeline
          so that figsize, grid style, legend, and color scheme are identical.
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
    base = _base_version(tag)
    if is_prerelease:
        return (
            f'## Coming in v{base} <span class="version-prerelease">pre-release</span>\n\n'
            f"*Staged on the `develop` branch — will land in the next stable release.*"
        )
    return f"## Released in v{base}\n\n> Released {date_str}"


def _remove_prerelease_for_version_in_file(file_path: Path, base_version: str) -> None:
    """Remove any pre-release section for *base_version* from the sentinel-delimited block.

    Matches headings such as:
      ## Coming in v1.2.0 <span class="version-prerelease">pre-release</span>
      ## Upcoming Changes (v1.2.0 pre-release) <span class="version-prerelease">in development</span>
    """
    if not file_path.exists():
        return
    text = file_path.read_text(encoding="utf-8")
    start_idx = text.find(CONTENT_START)
    end_idx = text.find(CONTENT_END)
    if start_idx == -1 or end_idx == -1:
        return

    after_start = start_idx + len(CONTENT_START)
    content = text[after_start:end_idx]

    escaped = re.escape(base_version)
    # Match any ## heading line that references this version as pre-release.
    # Use [^\n]* (any non-newline chars) to keep the match strictly on one line.
    header_pat = re.compile(
        rf"^## [^\n]*v{escaped}[^\n]*(?:pre-release|version-prerelease)[^\n]*$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = header_pat.search(content)
    if not m:
        print(
            f"[whats-new] No pre-release section found for v{base_version}; nothing removed.",
            file=sys.stderr,
        )
        return

    sec_start = m.start()
    rest = content[m.end():]

    # The section ends at the next `---` separator or the next `## ` heading.
    end_pat = re.compile(r"^(?:---\s*|##\s+)", re.MULTILINE)
    end_m = end_pat.search(rest)
    sec_end = m.end() + (end_m.start() if end_m else len(rest))

    # Rebuild content without the pre-release section; collapse extra blank lines
    # (including lines that contain only whitespace).
    new_content = re.sub(
        r"\n[ \t]*\n[ \t]*\n+",
        "\n\n",
        content[:sec_start].rstrip("\n") + content[sec_end:].lstrip("\n"),
    ).strip()

    new_text = (
        text[:after_start]
        + ("\n\n" + new_content if new_content else "")
        + "\n\n"
        + text[end_idx:]
    )
    file_path.write_text(new_text, encoding="utf-8")
    print(f"[whats-new] Removed pre-release section for v{base_version}.")


def _update_develop_note(file_path: Path, is_prerelease: bool, tag: str) -> None:
    """Replace the content between the note sentinels (if present) with appropriate text.

    Pre-release pending:
        !!! note "Pre-release documentation"
            The section at the top reflects changes staged for the next stable release.
    No pre-release (after stable sync):
        !!! note "Develop build"
            Currently aligned with stable release vX.Y.Z.

    On the ``main`` branch the sentinels are absent, so this is a no-op.
    """
    if not file_path.exists():
        return
    text = file_path.read_text(encoding="utf-8")
    ns = text.find(NOTE_START)
    ne = text.find(NOTE_END)
    if ns == -1 or ne == -1:
        return  # No note sentinels — nothing to update (e.g. main branch).

    version = _base_version(tag)
    if is_prerelease:
        note = (
            '!!! note "Pre-release documentation"\n'
            "    You are viewing the **develop** (pre-release) build.  \n"
            "    The section at the top reflects changes staged for the next stable release.  \n"
            "    Switch to the **main** docs via the badge in the header to see only stable content."
        )
    else:
        note = (
            '!!! note "Develop build"\n'
            f"    You are viewing the **develop** build, currently aligned with stable release **v{version}**.  \n"
            "    Switch to the **main** docs via the badge in the header to see the stable documentation."
        )

    new_text = (
        text[: ns + len(NOTE_START)]
        + "\n"
        + note
        + "\n"
        + text[ne:]
    )
    file_path.write_text(new_text, encoding="utf-8")
    print(f"[whats-new] Updated develop note (is_prerelease={is_prerelease}).")


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

    # When a stable release is published, remove any existing pre-release section
    # for the same base version to avoid duplication (dev branch has the pre-release
    # entry; syncing the stable version should replace it, not duplicate it).
    if not is_prerelease:
        _remove_prerelease_for_version_in_file(WHATS_NEW_PATH, _base_version(tag))

    body = _build_section(tag, raw_notes, is_prerelease, token)
    heading = _make_heading(tag, is_prerelease, date_str)
    new_block = heading + "\n\n" + body

    _inject_section(WHATS_NEW_PATH, new_block)

    # Update the develop-branch note block (no-op when sentinels are absent, e.g. main).
    _update_develop_note(WHATS_NEW_PATH, is_prerelease, tag)

    print(f"[whats-new] Updated {WHATS_NEW_PATH}")


if __name__ == "__main__":
    main()
