---
name: cicd
description: Use when touching .github/workflows/, release config (.releaserc*), mkdocs.yml, docs deploy, or the whats-new generator. Covers branch-to-release-channel mapping, semantic-release, docs deploy gating, and the agentic whats-new flow.
---

# CI/CD in pytrendy

## Workflows

| File | Purpose |
|---|---|
| `test.yaml` | Core tests → non-core (`--cov-append`) → Codecov. Runs on push to main/develop + PRs. |
| `release.yaml` | Core tests → semantic-release → build & publish to PyPI. Runs on push to main/develop. |
| `docs.yaml` | Build + deploy docs to GitHub Pages (`gh-pages` branch, per-subdirectory). main/develop. |
| `docs-preview.yaml` | Per-PR docs preview at `russellsb.github.io/pytrendy/pr-<N>/`. |
| `whats-new.yaml` | Agentic: generates `docs/whats-new.md` entry via OpenCode CLI, opens PR back. |
| `check-base-branch.yml` | Fails PRs not targeting `develop` (with documented exceptions). |
| `lint-pr-title.yml` | Enforces Conventional Commits on PR titles. |
| `codeql.yml` | GitHub code scanning. |

## Branch → release channel

- `main` → stable. `.releaserc` config; `branches: ["main"]`; publishes to PyPI as `1.x.x`.
- `develop` → prerelease `dev` channel. Release workflow does `cp .releaserc.dev.json .releaserc` before running; `branches` includes `{ "name": "develop", "prerelease": "dev", "channel": "dev" }`; publishes as `1.x.x.devN` (the `prepareCmd` rewrites `-dev.` → `.dev` for PEP 440 compliance).

Version source of truth = semantic-release output, applied via `poetry version ${nextRelease.version}`. **Never hand-edit `pyproject.toml`'s version** — it desyncs the next release.

## Release flow (`release.yaml`)

1. `test-core` job: `pytest tests/ -m core`.
2. `semantic-release`: conventional-commits analyzer → bumps version → `poetry version` → commits `CHANGELOG.md` + `pyproject.toml` (main) or just `pyproject.toml` (develop) with `chore(release): <ver> [skip ci]`. Uses SSH deploy key (`RELEASE_SSH_KEY`).
3. `build-and-publish`: only if `released == 'true'` (or `force_publish` dispatch). `poetry build` → PyPI via OIDC trusted publishing (`id-token: write`, `environment: release`).

## Docs deploy gating (`docs.yaml`)

The `check-should-deploy` job decides whether to build:
- Skip if any commit message contains `[skip docs]`.
- Deploy if any commit matches `^(feat|fix|docs|refactor|perf)(\([^)]*\))?!?:` (semantic types).
- Deploy if any commit subject has `!` (breaking) or body has `BREAKING CHANGE`.
- Else diff files: deploy if changes touch `docs/`, `mkdocs.yml`, or `pytrendy/`.

Deploy writes to `gh-pages` under a per-branch subdirectory (`main/`, `develop/`) with `keep_files: true` so branches don't clobber each other. On `main`, also writes a root `index.html` redirecting to `main/`. The `offline` mkdocs plugin is stripped via `sed` for online deploys; `site_url` is injected per-branch.

GitHub Pages setup is a one-time manual step (Settings → Pages → gh-pages / root) — noted in `docs.yaml` header comment.

## Docs preview (`docs-preview.yaml`)

Triggers on PR open/sync/reopen/close touching `docs/**`, `mkdocs.yml`, or `pytrendy/**`. Skips fork PRs (no `gh-pages` write access). Bot comments URL with stable marker `<!-- docs-preview-bot -->` (updates existing comment rather than duplicating). On PR close, removes `pr-<N>/` from `gh-pages` and updates the comment.

## What's New generator (`whats-new.yaml`) — agentic

Triggers: on release publish, on Release workflow completion (main/develop), or manual dispatch.

1. Resolves release metadata (tag, name, body, prerelease flag, branch) from the event payload, falling back to the GitHub Releases API by tag, then `CHANGELOG.md`.
2. Checks out the release branch, installs the **OpenCode CLI** (see `whats-new.yaml` for the exact install step).
3. Runs `python scripts/generate_whats_new.py` with `OPENCODE_MODEL` (default `opencode-go/deepseek-v4-flash`) and a **deny-all permission block** (`bash/edit/webfetch/websearch/external_directory/task` all `deny`) — the agent can only read and write the whats-new file via the script.
4. The script prepends a user-friendly entry into `docs/whats-new.md` **between sentinel comments** `<!-- WHATS_NEW_CONTENT_START -->` / `<!-- WHATS_NEW_CONTENT_END -->`. Don't edit content inside those markers by hand — it gets regenerated.
5. Opens a PR (`docs/whats-new-<tag>`) back to the release branch via `peter-evans/create-pull-request@v7` using `DOCS_PREVIEW_PAT`. The PR's `base` is the release branch (develop or main) — these are the *only* PRs allowed to target `main` from a non-`develop` head (`check-base-branch.yml` whitelists `docs/whats-new-*`).
6. On stable releases (main, not prerelease), a `sync-to-develop` job re-generates the entry framed as stable and opens a follow-up PR to `develop` so both branches stay consistent.

Script-level agent instructions are in `scripts/generate_whats_new.py` docstring: verify CSV URLs resolve (use develop-branch raw URLs for pre-releases, main for stable), derive before/after plot scenarios from `tests/tests_crashes_edgecases/` not synthetic data, and keep figsize/grid/legend/colors identical between before/after images by routing through the same `detect_trends()` + `plot_pytrendy()` pipeline.

## Secrets the workflows expect

`CODECOV_TOKEN`, `RELEASE_SSH_KEY`, `OPENCODE_API_KEY`, `DOCS_PREVIEW_PAT`, plus `GITHUB_TOKEN`. `OPENCODE_MODEL` is a repo **variable** (vars context), not a secret. PyPI publish uses OIDC trusted publishing (no token).
