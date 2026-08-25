#!/usr/bin/env python3
"""Generate or update docs/whats-new.md using OpenCode (opencode-go/deepseek-v4-flash).

The script:
  1. Reads the latest release notes (from RELEASE_BODY, then GitHub Releases API by
     tag, then CHANGELOG.md).
  2. Calls OpenCode (opencode-go/deepseek-v4-flash) to produce a user-friendly
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
  - For bug-fix before/after images, derive the scenario from the PR's test cases
    (look in tests/tests_crashes_edgecases/ for the relevant test), not from an
    arbitrary synthetic example. The "before" image must reproduce the actual broken
    output (e.g. by constructing the segment list that the pre-fix code produced) and
    the "after" image must use the current fixed `detect_trends()` output. Use the same
    `value_col`, `method_params`, and data file as the test so users can reproduce it.

Environment variables
---------------------
OPENCODE_API_KEY – required; authenticates against the OpenCode API.
OPENCODE_MODEL   – optional; model ID passed to `opencode run` (default:
                   "opencode-go/deepseek-v4-flash").
GITHUB_TOKEN     – required; used for the GitHub Releases API fallback and by the
                   workflow to open pull requests.
RELEASE_TAG      – the semantic-release tag (e.g. "v1.2.0" or "v1.2.0-dev.1").
RELEASE_NAME   – human-readable release title.
RELEASE_BODY   – raw Markdown body of the GitHub Release (release notes).
IS_PRERELEASE  – "true" / "false"; controls the "Coming in…" vs "Released in…" framing.
BRANCH         – the branch being released from (default: "develop").
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

CONTENT_START = "<!-- WHATS_NEW_CONTENT_START -->"
CONTENT_END = "<!-- WHATS_NEW_CONTENT_END -->"

NOTE_START = "<!-- WHATS_NEW_NOTE_START -->"
NOTE_END = "<!-- WHATS_NEW_NOTE_END -->"

DEFAULT_OPENCODE_MODEL = "opencode-go/deepseek-v4-flash"

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


def _major_minor(version: str) -> str:
    """Return the ``major.minor`` prefix of a dotted version string."""
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version


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


def _call_opencode(prompt: str, model: str) -> str | None:
    """Call the OpenCode CLI and return the generated text, or None on failure."""
    system = textwrap.dedent("""\
        You are a technical writer for PyTrendy, a Python library for time-series trend
        detection. Your task is to turn raw CHANGELOG / release-note Markdown into a
        concise, user-friendly "What's New" entry for the library's MkDocs documentation
        site.

        Guidelines:
        - Focus on user impact, not internal implementation details.
        - Use clear, plain language aimed at data scientists and analysts.
        - Do NOT include any AI chat preamble, thinking traces, or meta-commentary
          in the output. The output is documentation, not a conversation. Every line
          must be user-facing docs content — no "Let me look at…", "Now I have all
          the context…", or similar internal reasoning.
        - Start with a one-sentence summary of the release.
        - Keep all change-specific prose inside `??? note` blocks. Do not place
          standalone change descriptions outside those blocks.
        - Wrap each individual change in a `??? note "Change title"` collapsible block
          (MkDocs pymdownx.details syntax) so the page stays scannable.
        - In each note block, use the first sentence as a concise summary, then
          add details in subsequent lines.
        - Inside collapsible blocks, use `??? example "Code"` for code samples so code
          is hidden by default — visuals and notes lead.
        - Show before/after images stacked vertically (one column) inside a
          `<div class="before-after-grid">` with two `<div class="before-after-panel">` children,
          each labelled with `<span class="before-after-label before-label">Before</span>` or
          `<span class="before-after-label after-label">After</span>`.
          For bug fixes, always label with version numbers: `Before — vX.Y.Z` (the last
          stable release before the fix) and `After — vX.Y.Z` (the release introducing
          the fix). For feature toggles or configuration comparisons, a short descriptive
          label (e.g. `Before — \\`avoid_noise=True\\` (default)`) is acceptable instead.
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
        - For images, always use local docs-relative paths under
          `img/whats-new/v<version>/...` (or `img/whats-new/pre-release/...` when a
          versioned path is not available). Never link to GitHub user-attachments URLs.
          **CRITICAL: Do NOT reference image files that do not exist.** Before adding any
          image reference, verify the file exists by reading the docs/img/whats-new/
          directory. If images are needed for a new feature or fix but don't exist yet,
          omit the image references entirely — do not invent paths to non-existent files.
          Broken image links in the docs are worse than no images at all.
        - Do NOT include the top-level ## heading; the caller will add it.
        - Do NOT wrap the output in a code fence.
        - Before/after images must be visually comparable. For any bug fix or feature
          directly related to time-series trend detection, produce both the "before" and
          "after" images through the same `detect_trends()` + `plot_pytrendy()` pipeline
          so that figsize, grid style, legend, and color scheme are identical.
        - For bug-fix before/after comparisons, base them on the exact test case from the
          related PR, not on an arbitrary synthetic example. Look up the PR referenced in
          the CHANGELOG, find its regression/edge-case tests (e.g. in
          `tests/tests_crashes_edgecases/`), and reproduce the failing scenario for the
          "before" image and the passing scenario for the "after" image. The code example
          must use the same `value_col`, `method_params`, and data file as the test.
          Never fabricate a "before" by hand-crafting segments unless you have verified
          via the test or the PR description what the actual broken output was.

        Here is a high-quality example of a finished entry (for a feature that added
        `avoid_noise` support). Match this level of depth, structure, and detail:

        ---
        Four updates in v1.2.0: an agentic docs generator, a new noise toggle, and two fixes to trend metrics and normalised input handling.

        ??? note "Noise detection control (`avoid_noise`)"
            A new `avoid_noise` parameter in `method_params` lets users opt out of noise detection entirely.
            When set to `False`, spikes and noisy regions are ignored and trend detection proceeds straight through them.
            Introduced: [#110](https://github.com/RussellSB/pytrendy/pull/110)

            Useful for modelling a new-market launch or quasi-experiment where the signal is zero before and
            after the activation window. With `avoid_noise=False`, boundary artifacts around step-changes are
            suppressed, yielding clean Up/Down segments.

            <div class="before-after-grid" markdown>
            <div class="before-after-panel" markdown>
            <span class="before-after-label before-label">Before — `avoid_noise=True` (default)</span>

            ![New-market case — Noise artifacts at step boundaries](img/whats-new/pre-release/whats_new_avoid_noise_abrupt_before_pr110.png)

            </div>
            <div class="before-after-panel" markdown>
            <span class="before-after-label after-label">After — `avoid_noise=False`</span>

            ![New-market case — clean Up/Down with avoid_noise=False](img/whats-new/pre-release/whats_new_avoid_noise_abrupt_after_pr110.png)

            </div>
            </div>

            ??? example "Code"
                ```python
                import pytrendy as pt

                df = pt.load_data("series_synthetic")
                df.set_index("date", inplace=True)
                # Simulate a new-market / quasi-experiment: zero activity before and after activation
                df.loc["2025-01-01":"2025-02-27", "abrupt"] = 0
                df.loc["2025-05-05":"2025-06-30", "abrupt"] = 0
                df = df.reset_index()

                result = pt.detect_trends(
                    df, date_col="date", value_col="abrupt",
                    method_params=dict(is_abrupt_padded=True, avoid_noise=False)
                )
                print(result.df[["direction", "start", "end"]])
                ```
        ---
    """)

    env = os.environ.copy()
    env.setdefault(
        "OPENCODE_PERMISSION",
        '{"bash": "allow", "edit": "allow", "webfetch": "allow", "websearch": "allow", '
        '"external_directory": "deny", "task": "allow"}',
    )

    full_prompt = system + "\n\n" + prompt

    try:
        result = subprocess.run(
            ["opencode", "run", "-m", model],
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"[whats-new] OpenCode CLI failed (exit {result.returncode}):",
                file=sys.stderr,
            )
            print(result.stderr, file=sys.stderr)
            return None
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[whats-new] OpenCode CLI unavailable: {exc}", file=sys.stderr)
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
) -> str:
    """Return the full Markdown block for one release (without top-level heading)."""
    api_key = _env("OPENCODE_API_KEY")
    if not api_key:
        print(
            "[whats-new] Error: OPENCODE_API_KEY is not set. Cannot call OpenCode.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = _env("OPENCODE_MODEL", DEFAULT_OPENCODE_MODEL)

    prompt = textwrap.dedent(f"""\
        PyTrendy release {tag} ({'pre-release / develop' if is_prerelease else 'stable'}).

        Raw release notes (Markdown):
        ---
        {raw_notes}
        ---

        Write a "What's New" documentation entry for this release.
        Omit the top-level ## heading — I will add it myself.
    """)

    ai_content = _call_opencode(prompt, model)
    if ai_content is None:
        print(
            "[whats-new] Error: OpenCode call failed. See above for details. "
            "Failing the workflow rather than writing a low-quality fallback entry.",
            file=sys.stderr,
        )
        sys.exit(1)
    return ai_content


def _make_heading(tag: str, is_prerelease: bool, date_str: str) -> str:
    """Return the top-level Markdown heading for a release section."""
    base = _base_version(tag)
    if is_prerelease:
        target = _target_prerelease_version(tag)
        return (
            f'## Coming in v{target} <span class="version-prerelease">pre-release</span>\n\n'
            "*Staged on the `develop` branch — will land in the next stable release. "
            "Currently available as the latest pre-release:*\n\n"
            "```bash\n"
            "pip install --pre pytrendy\n"
            "```"
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


def _note_blocks(markdown: str) -> list[tuple[str, str]]:
    """Return (`title`, `block`) tuples for top-level `??? note` blocks."""
    pattern = re.compile(
        r'(^\?\?\? note "([^"]+)"[\s\S]*?)(?=^\?\?\? note "|\Z)',
        re.MULTILINE,
    )
    return [(m.group(2), m.group(1).strip()) for m in pattern.finditer(markdown)]


def _convert_prerelease_to_stable(
    file_path: Path,
    tag: str,
    date_str: str,
    summary: str,
) -> bool:
    """Convert an existing pre-release section to a stable release section.

    When a stable release is published and a pre-release section already exists
    for the same base version (e.g. ``## Coming in v1.3.0``), this function
    converts it in place rather than discarding the carefully structured content:

    1. Changes the heading to ``## Released in vX.Y.Z``
    2. Adds the release date line
    3. Removes the pre-release framing (staged note + pip install)
    4. Inserts a summary paragraph right after the date line
    5. Preserves **all** content after the first ``??? note`` block unchanged

    Returns ``True`` if converted, ``False`` if no pre-release section was found.
    """
    if not file_path.exists():
        return False
    text = file_path.read_text(encoding="utf-8")
    start_idx = text.find(CONTENT_START)
    end_idx = text.find(CONTENT_END)
    if start_idx == -1 or end_idx == -1:
        return False

    after_start = start_idx + len(CONTENT_START)
    content = text[after_start:end_idx]

    base = _base_version(tag)
    escaped = re.escape(base)
    header_pat = re.compile(
        rf"^## [^\n]*v{escaped}[^\n]*(?:pre-release|version-prerelease)[^\n]*$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = header_pat.search(content)
    if not m:
        print(
            f"[whats-new] No pre-release section found for v{base}; nothing to convert.",
            file=sys.stderr,
        )
        return False

    # Find the full section boundaries (heading to next --- or ## or end)
    sec_start = m.start()
    rest = content[m.end():]
    end_pat = re.compile(r"^(?:---\s*|##\s+)", re.MULTILINE)
    end_m = end_pat.search(rest)
    sec_end = m.end() + (end_m.start() if end_m else len(rest))

    old_section = content[sec_start:sec_end]

    # Split the pre-release section into:
    #   (a) framing — everything before the first ??? note block
    #   (b) rest    — from the first ??? note to the section end
    notes_start_pat = re.compile(r"^\?\?\? note ", re.MULTILINE)
    notes_start_m = notes_start_pat.search(old_section)
    if notes_start_m:
        notes_to_end = old_section[notes_start_m.start():]
    else:
        notes_to_end = ""

    # Build the stable release section: new heading + date + summary + preserved content
    new_section = (
        f"## Released in v{base}\n\n"
        f"> Released {date_str}\n\n"
        f"{summary}\n\n"
        f"{notes_to_end}"
    ).strip()

    # Rebuild content: before section + new section + after section
    before_content = content[:sec_start].rstrip()
    after_content = content[sec_end:].lstrip()
    new_content = (
        before_content + "\n\n" + new_section + "\n\n" + after_content
    ).strip()

    updated = text[:after_start] + "\n\n" + new_content + "\n\n" + text[end_idx:]
    file_path.write_text(updated, encoding="utf-8")
    print(f"[whats-new] Converted pre-release section for v{base} to stable.")
    return True


def _upsert_prerelease_stream_section(file_path: Path, new_block: str, tag: str) -> bool:
    """Merge into an existing pre-release stream section when one already exists.

    This keeps a single "Coming in ..." section per major.minor stream, updates it to
    the latest target version, and appends any older note blocks not already present.
    """
    text = file_path.read_text(encoding="utf-8")
    start_idx = text.find(CONTENT_START)
    end_idx = text.find(CONTENT_END)
    if start_idx == -1 or end_idx == -1:
        return False

    after_start = start_idx + len(CONTENT_START)
    content = text[after_start:end_idx].strip()
    if not content:
        return False

    stream = _major_minor(_target_prerelease_version(tag))
    heading_pattern = re.compile(
        r'^## Coming in v(?P<version>\d+\.\d+\.\d+)\s+<span class="version-prerelease">pre-release</span>$',
        re.MULTILINE,
    )

    match = None
    for m in heading_pattern.finditer(content):
        if _major_minor(m.group("version")) == stream:
            match = m
            break
    if not match:
        return False

    section_start = match.start()
    separator = re.search(r"^\s*---\s*$", content[match.end():], re.MULTILINE)
    if separator:
        section_end = match.end() + separator.start()
        post_separator = match.end() + separator.end()
    else:
        section_end = len(content)
        post_separator = len(content)

    existing_section = content[section_start:section_end].strip()
    existing_notes = _note_blocks(existing_section)
    new_notes = _note_blocks(new_block)
    new_titles = {title for title, _ in new_notes}

    carryover = [block for title, block in existing_notes if title not in new_titles]
    merged_block = new_block.strip()
    if carryover:
        merged_block += "\n\n" + "\n\n".join(carryover)

    remaining = (content[:section_start].rstrip() + "\n\n" + content[post_separator:].lstrip()).strip()
    merged_content = merged_block + (f"\n\n---\n\n{remaining}" if remaining else "")

    updated = text[:after_start] + "\n\n" + merged_content + "\n\n" + text[end_idx:]
    file_path.write_text(updated, encoding="utf-8")
    print(f"[whats-new] Updated existing pre-release stream section for v{stream}.")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _generate_summary_block(raw_notes: str, tag: str) -> str:
    """Generate a concise 2-3 line summary paragraph from release notes.

    Tries OpenCode first; falls back to a simple auto-generated summary.
    """
    api_key = _env("OPENCODE_API_KEY")
    if api_key:
        model = _env("OPENCODE_MODEL", DEFAULT_OPENCODE_MODEL)
        prompt = textwrap.dedent(f"""\
            PyTrendy release {tag}.

            Raw release notes (Markdown):
            ---
            {raw_notes}
            ---

            Write a concise 2-3 sentence summary paragraph for this release.
            Focus on the most important user-facing changes.
            Do NOT include headings, bullet points, code blocks, or expandable sections.
            Just a single plain paragraph of 2-3 sentences.
        """)
        result = _call_opencode(prompt, model)
        if result:
            return result.strip().strip('"').strip("'")
        print(
            "[whats-new] OpenCode summary call failed; using fallback.",
            file=sys.stderr,
        )

    # Fallback: count fixes and features from the raw notes
    fix_count = 0
    feat_count = 0
    for line in raw_notes.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if any(k in lower for k in ("fix", "bug", "resolv", "patch", "correct")):
            fix_count += 1
        elif any(k in lower for k in ("feat", "feature", "new", "add")):
            feat_count += 1

    parts = ["This release includes"]
    if feat_count > 0:
        parts.append(f"{feat_count} new feature{'s' if feat_count > 1 else ''}")
    if feat_count > 0 and fix_count > 0:
        parts.append("and")
    if fix_count > 0:
        parts.append(f"{fix_count} bug fix{'es' if fix_count > 1 else ''}")
    if len(parts) == 1:
        parts.append("several improvements")

    return " ".join(parts) + ". See the detailed sections below."


def main() -> None:
    """Generate or update the What's New documentation entry."""
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

    if not is_prerelease:
        # Stable release: try to convert existing pre-release section first.
        # This preserves the carefully structured content (zero-baseline
        # grouping, agentic docs, CI/CD, etc.) rather than discarding it
        # for an AI-generated replacement.
        if _convert_prerelease_to_stable(
            WHATS_NEW_PATH,
            tag,
            date_str,
            _generate_summary_block(raw_notes, tag),
        ):
            print("[whats-new] Existing pre-release section converted to stable.")
        else:
            # No pre-release section exists — remove any stale one for this
            # version and generate a full entry from scratch.
            print(
                "[whats-new] No pre-release section found; generating from scratch.",
                file=sys.stderr,
            )
            _remove_prerelease_for_version_in_file(WHATS_NEW_PATH, _base_version(tag))
            body = _build_section(tag, raw_notes, is_prerelease)
            heading = _make_heading(tag, is_prerelease, date_str)
            new_block = heading + "\n\n" + body
            _inject_section(WHATS_NEW_PATH, new_block)
    else:
        # Pre-release: generate or update existing stream section.
        body = _build_section(tag, raw_notes, is_prerelease)
        heading = _make_heading(tag, is_prerelease, date_str)
        new_block = heading + "\n\n" + body
        if not _upsert_prerelease_stream_section(WHATS_NEW_PATH, new_block, tag):
            _inject_section(WHATS_NEW_PATH, new_block)

    # Update the develop-branch note block (no-op when sentinels absent, e.g. main).
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
