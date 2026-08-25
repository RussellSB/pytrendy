#!/usr/bin/env python3
"""Generate or update docs/whats-new.md using OpenCode (opencode-go/deepseek-v4-flash).

The script:
  1. Reads the latest release notes (from RELEASE_BODY, then the GitHub Releases API by
     tag, then CHANGELOG.md).
  2. Runs OpenCode in agentic mode so the agent updates docs/whats-new.md directly:
     it reads the file, decides where the new release entry belongs (merging into an
     existing "Coming in ..." pre-release section, converting one to stable, or adding
     a fresh section at the top), writes the entry, and generates before/after images
     under docs/img/whats-new/ when useful.
  3. The agent's stdout is never written into the file; its edits are the only output.
     Afterwards the script re-asserts the pre-release/stable note banner and validates
     that referenced image files exist.

Nothing is committed here; the caller opens a PR from the resulting working tree.

Environment variables
---------------------
OPENCODE_API_KEY – required; authenticates against the OpenCode API.
OPENCODE_MODEL   – optional; model ID passed to `opencode run` (default:
                   "opencode-go/deepseek-v4-flash").
GITHUB_TOKEN     – required; used for the GitHub Releases API fallback.
RELEASE_TAG      – the semantic-release tag (e.g. "v1.2.0" or "v1.2.0-dev.1").
RELEASE_NAME     – human-readable release title (informational).
RELEASE_BODY     – raw Markdown body of the GitHub Release (release notes).
IS_PRERELEASE    – "true" / "false"; controls the "Coming in…" vs "Released in…" framing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
WHATS_NEW_PATH = REPO_ROOT / "docs" / "whats-new.md"
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"

NOTE_START = "<!-- WHATS_NEW_NOTE_START -->"
NOTE_END = "<!-- WHATS_NEW_NOTE_END -->"

DEFAULT_OPENCODE_MODEL = "opencode-go/deepseek-v4-flash"

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a technical writer for PyTrendy, a Python library for time-series trend
    detection. Your task is to turn raw CHANGELOG / release-note Markdown into a concise,
    user-friendly "What's New" entry on the library's MkDocs documentation site, editing
    docs/whats-new.md directly with your tools.

    You are working in a git checkout of the pytrendy repository. Open and read
    docs/whats-new.md first, follow its existing structure, and apply your edits there.
    Do NOT print the entry and stop: edit the file. The only output that matters is the
    file diff.

    Styling conventions:
    - One-sentence summary at the top of the section.
    - Wrap each individual change in a `??? note "Change title"` collapsible block
      (MkDocs pymdownx.details syntax). The first sentence inside is the concise summary;
      subsequent lines add detail.
    - Inside collapsible blocks use `??? example "Code"` for code samples so code is
      hidden by default: visuals and notes lead.
    - Reference issue or PR numbers from the release notes where available
      (e.g. `[#12](https://github.com/RussellSB/pytrendy/pull/12)`).
    - Code examples load CSV test data via GitHub raw URLs, never local paths. Pre-release:
      `https://raw.githubusercontent.com/RussellSB/pytrendy/develop/tests/tests_crashes_edgecases/data/<file>.csv`
      Stable:
      `https://raw.githubusercontent.com/RussellSB/pytrendy/main/tests/tests_crashes_edgecases/data/<file>.csv`
      Bundled datasets (series_synthetic, classes_signals) may use `pt.load_data(name)` directly.
      If a code example references a CSV, confirm the file exists in
      tests/tests_crashes_edgecases/data/ (e.g. `ls`) so the raw URL is valid.
    - No emoji in headings or admonition titles; minimal emoji overall.
    - No em dashes anywhere; use colons or semicolons instead.
    - Group small patch releases (1-2 minor fixes each) into a single combined section.
    - Write in a natural, human voice. No filler, no marketing tone, no rule-of-three
      list rhythm.

    Before/after images:
    - Use them for bug fixes or features directly related to trend detection. Stack the two
      images vertically inside a `<div class="before-after-grid" markdown>` with two
      `<div class="before-after-panel" markdown>` children, each labelled with
      `<span class="before-after-label before-label">Before: ...</span>` /
      `<span class="before-after-label after-label">After: ...</span>`, separated by a
      blank line, matching the pattern already used in the file.
    - Generate both images yourself with the same `detect_trends()` + `plot_pytrendy()`
      pipeline (install the package with `pip install -e .` if needed) so figsize, grid,
      legend, and colors match between the two images. Base the scenario on the related
      tests or the scenarios specified in the PR/issue rather than a synthetic example, and
      use the same value_col, method_params, and data file as the test.
    - Save images under docs/img/whats-new/pre-release/ (pre-release) or
      docs/img/whats-new/v<version>/ (stable) and reference them with docs-relative paths.
      Never link to GitHub user-attachments URLs. After writing, verify every referenced
      image file exists before finishing; broken image links are worse than no images at all.

    File-scope rules:
    - Keep every existing section intact; only add, merge, or convert the entry you were
      asked to write. Do not reformat or rewrite anything else.
    - Do not touch anything outside the `<!-- WHATS_NEW_CONTENT_START -->` and
      `<!-- WHATS_NEW_CONTENT_END -->` markers, in particular not the pre-release note
      banner between `<!-- WHATS_NEW_NOTE_START -->` and `<!-- WHATS_NEW_NOTE_END -->`.
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str = "") -> str:
    """Return the value of environment variable ``key``, stripped, or ``default``."""
    return os.environ.get(key, default).strip()


def _base_version(tag: str) -> str:
    """Return the base semver string from a tag, stripping the leading 'v' and any pre-release suffix.

    Examples:
        "v1.2.0"        → "1.2.0"
        "v1.3.0-dev.1"  → "1.3.0"
    """
    return tag.lstrip("v").split("-")[0]


def _target_prerelease_version(tag: str) -> str:
    """Return the upcoming stable version for a pre-release tag when derivable.

    For tags in the form ``vMAJOR.MINOR.PATCH-dev.N``, use ``MAJOR.MINOR.N`` so
    pre-release streams are grouped by the latest dev index (e.g. ``v1.2.0-dev.4``
    becomes ``1.2.4``).
    """
    match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)-dev\.(\d+)$", tag.strip())
    if not match:
        return _base_version(tag)
    major, minor, _patch, dev_index = match.groups()
    return f"{major}.{minor}.{dev_index}"


def _read_changelog_section(tag: str) -> str:
    """Extract the changelog block for a given tag from CHANGELOG.md."""
    if not CHANGELOG_PATH.exists():
        return ""
    text = CHANGELOG_PATH.read_text(encoding="utf-8")
    # Match the heading line containing the version number
    version = tag.lstrip("v").split("-")[0]  # strip "v" and pre-release suffix
    pattern = re.compile(
        rf"(?:^|\n)(#{{1,2}}\s+\[?{re.escape(version)}\]?.*?)(?=\n#{{1,2}}\s|\Z)",
        re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _fetch_release_body_from_github(tag: str, token: str) -> str:
    """Fetch release notes body from the GitHub Release API by tag."""
    repository = _env("GITHUB_REPOSITORY")
    if not token or not repository or "/" not in repository:
        return ""

    # Normalise tag: GitHub releases are conventionally prefixed with 'v'
    normalised_tag = tag if tag.startswith("v") else f"v{tag}"
    url_encoded_tag = urllib.parse.quote(normalised_tag, safe="")
    url = f"https://api.github.com/repos/{repository}/releases/tags/{url_encoded_tag}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + token,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            body = data.get("body", "")
            return body.strip() if isinstance(body, str) else ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(
            f"[whats-new] Warning: failed to fetch release notes for {tag} from GitHub API: {exc}",
            file=sys.stderr,
        )
        return ""


def _resolve_raw_notes(tag: str, release_body: str, token: str) -> str:
    """Resolve release notes in priority order: env body, GitHub API, changelog."""
    if release_body:
        return release_body
    return _fetch_release_body_from_github(tag, token) or _read_changelog_section(tag)


# ---------------------------------------------------------------------------
# Agentic generation
# ---------------------------------------------------------------------------


def _build_agent_prompt(tag: str, raw_notes: str, is_prerelease: bool, date_str: str) -> str:
    """Build the full agent prompt: system guidance + placement task + release notes."""
    base = _base_version(tag)
    target = _target_prerelease_version(tag) if is_prerelease else base
    prerelease_heading = (
        f'## Coming in v{target} <span class="version-prerelease">pre-release</span>'
    )
    stable_heading = f"## Released in v{base}"

    placement = textwrap.dedent(f"""\
        Task
        ----
        PyTrendy release {tag} ({'pre-release / develop' if is_prerelease else 'stable'}).

        Add a "What's New" entry for this release to docs/whats-new.md. Decide for yourself
        where the entry belongs, following the existing file structure:

        - Pre-release: use the heading `{prerelease_heading}`, followed by the staged-on-develop
          note and `pip install --pre pytrendy` block used by existing pre-release sections.
          If a `## Coming in vX.Y.Z ... pre-release` section for the same major.minor stream
          already exists, merge the new notes into it: refresh the heading to the latest target
          version and keep any older `??? note` blocks. Otherwise add the new section at the top
          of the content block.
        - Stable: use the heading `{stable_heading}`, a `> Released {date_str}` line, and a
          one-sentence summary paragraph. If a matching "Coming in" pre-release section for this
          version exists, convert it to stable in place (change the heading, drop the staged /
          pip-install framing, add the date and summary) and keep its note blocks. Otherwise add
          the new section at the top of the content block.
        - If a section for this exact version already exists, update it instead of duplicating.

        Raw release notes (Markdown):
        ---
        {raw_notes}
        ---
    """)

    return SYSTEM_PROMPT + "\n\n" + placement


def _call_opencode(prompt: str, model: str) -> None:
    """Run OpenCode agentically; the agent's file edits are the result.

    Exits with code 1 if the CLI is missing, errors, or times out. The agent's stdout
    is a session log only and is never written into docs/whats-new.md.
    """
    env = os.environ.copy()
    env.setdefault(
        "OPENCODE_PERMISSION",
        '{"bash": "allow", "edit": "allow", "webfetch": "allow", "websearch": "allow", '
        '"external_directory": "deny", "task": "allow"}',
    )

    try:
        result = subprocess.run(
            ["opencode", "run", "-m", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[whats-new] OpenCode CLI unavailable: {exc}", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        print(
            f"[whats-new] OpenCode CLI failed (exit {result.returncode}):",
            file=sys.stderr,
        )
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    tail = result.stdout.strip().splitlines()[-10:]
    print("[whats-new] OpenCode session finished. Last lines:")
    for line in tail:
        print(f"  {line}")


# ---------------------------------------------------------------------------
# Post-run checks
# ---------------------------------------------------------------------------


def _update_develop_note(file_path: Path, is_prerelease: bool, tag: str) -> None:
    """Replace the content between the note sentinels (if present) with appropriate text.

    Pre-release pending:
        !!! note "Pre-release documentation"
            The section at the top reflects changes staged for the next stable release.
    No pre-release (after stable sync):
        The note is wrapped in an HTML comment so it is invisible when rendered.
        This is intentional: when develop is merged into main for a stable release the
        commented-out block is carried along but produces no visible output on either branch.
        The comment acts as "optional html" — it can be uncommented to view the note, and
        the script restores it as a visible admonition automatically on the next pre-release.

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
        # Wrap in an HTML comment so the banner is invisible on both develop and main after a
        # stable release merge. The script restores a visible note on the next pre-release.
        note = (
            "<!--\n"
            '!!! note "Develop build"\n'
            f"    You are viewing the **develop** build, currently aligned with stable release **v{version}**.  \n"
            "    Switch to the **main** docs via the badge in the header to see the stable documentation.\n"
            "-->"
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


def _validate_image_refs(file_path: Path) -> list[str]:
    """Check for image references pointing to non-existent files.

    Returns a list of broken image paths (empty if all valid).
    """
    if not file_path.exists():
        return []
    text = file_path.read_text(encoding="utf-8")
    # Find all markdown image references: ![alt](path)
    img_pattern = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
    broken = []
    for match in img_pattern.finditer(text):
        img_path = match.group(1)
        # Skip URLs (only validate local paths)
        if img_path.startswith(("http://", "https://")):
            continue
        # Resolve relative to the docs directory
        docs_dir = file_path.parent
        full_path = docs_dir / img_path
        if not full_path.exists():
            broken.append(img_path)
    return broken


def _git_file_modified(file_path: Path) -> bool:
    """Return True if the file has uncommitted working-tree changes."""
    result = subprocess.run(
        ["git", "diff", "--quiet", "--exit-code", "--", str(file_path)],
        capture_output=True,
        text=True,
    )
    return result.returncode != 0


def _entry_already_exists(file_path: Path, tag: str, is_prerelease: bool) -> bool:
    """Return True if a section for this release already exists in the file."""
    if not file_path.exists():
        return False
    text = file_path.read_text(encoding="utf-8")
    version = _target_prerelease_version(tag) if is_prerelease else _base_version(tag)
    heading = f"## Coming in v{version}" if is_prerelease else f"## Released in v{version}"
    return heading in text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the agentic generator; the agent edits docs/whats-new.md directly."""
    token = _env("GITHUB_TOKEN")
    tag = _env("RELEASE_TAG") or "v0.0.0"
    release_body = _env("RELEASE_BODY")
    is_prerelease = _env("IS_PRERELEASE", "false").lower() == "true"
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # Prefer event body, then GitHub API by tag, then changelog fallback.
    raw_notes = _resolve_raw_notes(tag, release_body, token)

    if not raw_notes:
        print(
            f"[whats-new] No release notes found for {tag}. Nothing to do.",
            file=sys.stderr,
        )
        sys.exit(0)

    print(f"[whats-new] Generating entry for {tag} (prerelease={is_prerelease})…")

    api_key = _env("OPENCODE_API_KEY")
    if not api_key:
        print(
            "[whats-new] Error: OPENCODE_API_KEY is not set. Cannot call OpenCode.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = _env("OPENCODE_MODEL", DEFAULT_OPENCODE_MODEL)
    prompt = _build_agent_prompt(tag, raw_notes, is_prerelease, date_str)
    _call_opencode(prompt, model)

    if not _git_file_modified(WHATS_NEW_PATH):
        if _entry_already_exists(WHATS_NEW_PATH, tag, is_prerelease):
            print(
                f"[whats-new] Entry for {tag} already exists in {WHATS_NEW_PATH.name}; nothing to do.",
                file=sys.stderr,
            )
            sys.exit(0)
        print(
            f"[whats-new] Error: the agent did not modify {WHATS_NEW_PATH.name}. "
            "Nothing to open a PR for.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Re-assert the develop/main note banner (deterministic, agent-independent).
    _update_develop_note(WHATS_NEW_PATH, is_prerelease, tag)

    # Validate image references
    broken_images = _validate_image_refs(WHATS_NEW_PATH)
    if broken_images:
        print(
            f"[whats-new] WARNING: Found {len(broken_images)} broken image reference(s):",
            file=sys.stderr,
        )
        for img in broken_images:
            print(f"  - {img}", file=sys.stderr)
        print(
            "[whats-new] Broken images will render as placeholders in the docs. "
            "Add the missing image files or remove the references.",
            file=sys.stderr,
        )

    print(f"[whats-new] Updated {WHATS_NEW_PATH}")


if __name__ == "__main__":
    main()