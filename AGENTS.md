# AGENTS.md

PyTrendy: trend detection lib for time series. Python ≥3.10, CI on 3.12.

## Skills (load before ANY code work)

Before making code changes, commits, or PRs, load these skills — they contain
the conventions, architecture, and rules this project enforces:

1. **`maintenance`** — commit format, branch model, deprecation policy, API surface, diff rules, remote session behaviour
2. **`pytrendy`** — 5-stage pipeline, module map, install/verify, datasets, PyTrendyResults, docs tooling

Other skills (`test`, `cicd`, `debug`, `pr-plots`) load on demand for specific tasks.

## Critical rules (summary — details in skills)

- PRs target `develop`, not `main`
- **Commits must use Conventional Commits**: `fix:`, `feat:`, `chore:`, `ci:`, `docs:`, `refactor:`, `perf:`, `test:`, `build:`, `revert:`
  - Lowercase, imperative, <72 chars, no trailing period
  - This applies to all commit messages you generate, including auto-generated summaries
  - `lint-pr-title.yml` enforces this on PR titles
- Deprecating a public param = `feat:` (minor), NOT `refactor:`
- Never hand-edit `pyproject.toml` version or `CHANGELOG.md`
