# AGENTS.md

PyTrendy: trend detection for time series. One entrypoint —
`pytrendy.detect_trends(df, value_col, date_col=None, ...)` → `PyTrendyResults`.
Python ≥3.10, CI on 3.12.

## Skills (load before ANY code work)

Before making changes, commits, or PRs, load these — they hold the conventions,
architecture and rules:

1. **`maintenance`** — commit format, branch model, deprecation policy, API surface, diff rules, remote session behaviour
2. **`pytrendy`** — 5-stage pipeline, module map, install/verify, datasets, `PyTrendyResults`, docs tooling

Other skills load on demand: `test` (pytest-mpl baselines), `cicd` (release pipeline),
`debug` (pipeline bisects), `pr-plots` (before/after evidence).

## Branch model (enforced by CI)

- Default: branch off `develop`, target `develop`. `check-base-branch.yml` fails PRs to `main` that don't qualify.
- CI/CD-only or agentic-config changes (`.github/`, `.opencode/`, `AGENTS.md`, `pyproject.toml`, `opencode.json`, `.releaserc`) may target `main` directly for urgent fixes; library code and docs go through `develop`.
- `main` = stable, `develop` = prerelease. `develop`→`main` release PRs and `docs/whats-new-*` bypass the check.

## Critical rules (details in skills)

- **Conventional Commits** on commits and PR titles — `^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?!?: .+`, lowercase, imperative, <72 chars, no trailing period.
- `feat`→minor, `fix`→patch, `!`→major. Deprecating a public param is `feat:`, never `refactor:`.
- **`chore`/`ci` vs `fix`**: `fix:` triggers a patch release — don't use it for maintenance-only work.
- Never hand-edit `pyproject.toml` version or `CHANGELOG.md`.
- Verify with `pytest` before pushing; plot baselines are pinned to `matplotlib==3.11.2`.
