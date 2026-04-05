# Contribution Guide

Contributions to PyTrendy are welcome! This guide covers the standards and processes we follow to keep the project consistent and easy to collaborate on.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/pytrendy.git
   cd pytrendy
   ```
3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```
   The `dev` group installs everything needed to run the project and its test suite — the core dependencies (`numpy`, `pandas`, `scipy`, `matplotlib`) are pulled in automatically, and `dev` adds `pytest`, `pytest-cov`, `pytest-mpl`, and `pytest-timeout` on top.

   If your PR touches documentation, also install the docs group:
   ```bash
   pip install -e ".[dev,docs]"
   ```

---

## Roadmap & Issues

The project roadmap lives in **[GitHub Issues](https://github.com/RussellSB/pytrendy/issues)**. Browse open issues to find tasks to pick up, report bugs, or propose improvements.

Before starting work, check whether a relevant issue already exists. If it does, comment on it to signal you are working on it. If it doesn't, open a new issue first so the approach can be discussed before code is written.

### Issue Title Conventions

Issue titles follow a `[Tag] Short description` format:

| Tag | Description |
|---|---|
| `[Bug]` | Something is broken or behaving incorrectly |
| `[Docs]` | Documentation additions or improvements |
| `[Enhancement]` | Improvement to an existing feature |
| `[Feature]` | A new capability or feature |
| `[CI]` | Changes to CI/CD pipelines or automation |
| `[Maintenance]` | Refactoring, tooling, dependency updates |

### Labels

Issues are tagged with one or more labels for filtering:

| Label | Meaning |
|---|---|
| `bug` | Confirmed bug |
| `documentation` | Relates to docs |
| `enhancement` | Improvement to existing behaviour |
| `feature` | New functionality |
| `test` | Requires test coverage |
| `maintenance` | Tooling, CI, or refactoring work |
| `good first issue` | A good starting point for new contributors |
| `priority` | Needs to be resolved soon |

---

## Branching

All work should branch off `develop`, **not** `main`. The `main` branch is reserved for stable releases.

```bash
git fetch origin develop
git checkout -b my-feature origin/develop
```

Branch names should be short and descriptive, e.g. `fix/trend-detection-edge-case` or `docs/add-contribution-guide`.

---

## Commit Messages

This project uses the [Conventional Commits](https://www.conventionalcommits.org/) specification. Following this format is important because the project uses **semantic-release** to automate versioning and changelog generation — the commit type determines what version bump is applied on release:

- `feat` → minor bump (0.**x**.0)
- `fix` → patch bump (0.0.**x**)
- breaking change (with `!` or `BREAKING CHANGE:` in footer) → major bump (**x**.0.0)
- All other types (`docs`, `ci`, `chore`, etc.) → no version bump

All commit messages **must** follow this format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | When to use |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Formatting changes with no logic impact |
| `refactor` | Code restructuring without feature or fix changes |
| `perf` | Performance improvements |
| `test` | Adding or correcting tests |
| `build` | Build system or dependency changes |
| `ci` | CI configuration changes |
| `chore` | Miscellaneous changes that don't modify src or tests |
| `revert` | Reverts a previous commit |

### Examples

```
feat: add trend detection for seasonal data
fix: correct calculation in SNR algorithm
docs: update quickstart guide with new examples
test: add unit tests for detect_trends function
```

### Guidelines

- Use the **imperative mood**: "add feature" not "added feature"
- Keep the subject line **under 72 characters**
- **Lowercase** first letter, no period at the end
- Use the body to explain **what and why**, not how
- Reference issues in the footer: `Closes #55`

### Breaking Changes

Append `!` to the type or add `BREAKING CHANGE:` in the footer:

```
feat!: remove deprecated API endpoint

BREAKING CHANGE: The /old-api endpoint has been removed. Use /new-api instead.
```

---

## Pull Requests

### PR Title

The PR title is what gets recorded in the changelog when a PR is **squash-merged into `main`**. It must therefore follow the same Conventional Commits format as individual commit messages (`feat: ...`, `fix: ...`, etc.), and should accurately reflect the overall intent of the change.

### Before Opening a PR

- Ensure your branch is based on `develop` and is up to date with it.
- Run the full test suite locally: `pytest`
- Keep the scope focused — one PR should address one issue or concern.

### PR Checklist

- [ ] Branch is based off `develop`
- [ ] PR title follows Conventional Commits format (e.g. `feat: add weekly data support`)
- [ ] PR is linked to a relevant GitHub issue via `Closes #<issue-number>` in the description
- [ ] Tests are added or updated as appropriate
- [ ] Documentation is updated if the change affects public behaviour

### Docs Preview

Every PR that touches `docs/`, `mkdocs.yml`, or `pytrendy/` automatically builds and deploys a temporary docs preview. A bot will comment on the PR with the URL (`https://russellsb.github.io/pytrendy/pr-<N>/`) so reviewers can browse the rendered site without running anything locally. The preview is removed when the PR is closed.

---

## Code Style

Follow existing code patterns in the module you are editing. Write docstrings in [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) and add type hints to all public functions. Do not leave debug prints or commented-out code in your PR.

Keep your diff clean. Only modify the lines directly relevant to your change — avoid reformatting code you are not touching, and do not introduce stray whitespace, blank-line, or indentation changes as side effects. These make reviews harder and can cause merge conflicts.

---

## Tests

Tests live in the `tests/` directory and use `pytest`. To run them:

```bash
pytest
```

Every new feature or bug fix should include appropriate test coverage. See existing tests for patterns to follow.

---

## Documentation

Docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). To preview locally, install the docs group first (see [Getting Started](#getting-started)), then run:

```bash
mkdocs serve
```

If you are adding a new page, register it under `nav:` in `mkdocs.yml`. For docs-only PRs, the automated preview (see [Docs Preview](#docs-preview)) is a convenient way to share the rendered output with reviewers.

---

## Asking for Help

If you are unsure about anything, feel free to open a [Discussion](https://github.com/RussellSB/pytrendy/discussions) or comment on the relevant issue. The maintainers are happy to guide you in the right direction.

