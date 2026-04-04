# GitHub Copilot Instructions

## Commit Message Standards

This repository follows the [Conventional Commits](https://www.conventionalcommits.org/) specification for all commit messages.

### Commit Message Format

All commit messages MUST follow this format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Where:
- **type** is REQUIRED and must be one of:
  - `feat`: A new feature
  - `fix`: A bug fix
  - `docs`: Documentation only changes
  - `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
  - `refactor`: A code change that neither fixes a bug nor adds a feature
  - `perf`: A code change that improves performance
  - `test`: Adding missing tests or correcting existing tests
  - `build`: Changes that affect the build system or external dependencies
  - `ci`: Changes to CI configuration files and scripts
  - `chore`: Other changes that don't modify src or test files
  - `revert`: Reverts a previous commit

- **scope** is OPTIONAL and represents the area of the codebase affected (e.g., `api`, `ui`, `docs`, `tests`)

- **subject** is REQUIRED and should be a short description (imperative, lowercase, no period at the end)

- **body** is OPTIONAL and provides additional context

- **footer** is OPTIONAL and can reference issues or include breaking changes

### Examples

Good commit messages:
```
feat: add trend detection for seasonal data
fix: correct calculation in SNR algorithm
docs: update quickstart guide with new examples
refactor: simplify window size calculation logic
test: add unit tests for detect_trends function
chore(release): 1.2.3 [skip ci]
```

Bad commit messages:
```
✗ Fixed bug
✗ Update code
✗ WIP
✗ changes
```

### Breaking Changes

When introducing breaking changes, add `BREAKING CHANGE:` in the footer or append `!` after the type/scope:

```
feat!: remove deprecated API endpoint

BREAKING CHANGE: The /old-api endpoint has been removed. Use /new-api instead.
```

### Automated Release

This repository uses semantic-release, which automatically:
- Determines the next version number based on commit messages
- Generates release notes from conventional commits
- Publishes releases to GitHub and PyPI

Commit types trigger the following version bumps:
- `feat`: Minor version bump (0.x.0)
- `fix`: Patch version bump (0.0.x)
- Breaking changes (with `!` or `BREAKING CHANGE:`): Major version bump (x.0.0)
- Other types: No version bump

### Commit Message Guidelines

1. **Use imperative mood**: "add feature" not "added feature" or "adds feature"
2. **Keep subject line under 72 characters**
3. **Start with lowercase**: "add feature" not "Add feature"
4. **No period at the end of the subject line**
5. **Separate subject from body with a blank line**
6. **Use the body to explain what and why, not how**
7. **Reference issues and pull requests when applicable**

## PR Title Standards

Pull request titles MUST also follow the Conventional Commits format, just like commit messages. This ensures consistency across the repository and helps with automated changelog generation.

### PR Title Format

PR titles should follow the same format as commit messages:

```
<type>(<scope>): <subject>
```

Where:
- **type** is REQUIRED (same types as commit messages: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`)
- **scope** is OPTIONAL (represents the area of the codebase affected)
- **subject** is REQUIRED (short description in imperative mood, lowercase, no period at the end)

### PR Title Examples

Good PR titles:
```
feat: add trend detection for seasonal data
fix: correct calculation in SNR algorithm
docs: update quickstart guide with new examples
refactor: simplify window size calculation logic
test: add unit tests for detect_trends function
feat(api)!: remove deprecated endpoint
```

Bad PR titles:
```
✗ Fixed a bug
✗ Update
✗ WIP changes
✗ Merge pull request #123
```

### PR Title Guidelines

1. **Use imperative mood**: "add feature" not "added feature" or "adds feature"
2. **Keep title under 72 characters**
3. **Start with lowercase**: "add feature" not "Add feature"
4. **No period at the end**
5. **Match the primary change**: If the PR contains multiple commits, the title should reflect the most significant change
6. **Include breaking change indicator**: Append `!` after type/scope for breaking changes

### Tools

When creating pull requests, ensure your PR titles follow these standards. When making commits, ensure your commit messages follow these standards. The CI/CD pipeline uses these messages to automate releases and generate changelogs.

## Default Base Branch

All pull requests **must** target the `develop` branch by default, not `main`. Never target `main` directly unless explicitly instructed otherwise.

This is because:
- `main` is the stable release branch — only `develop` gets merged into `main` via a release process
- PRs that accidentally target `main` will include unintended diffs (e.g. release version bumps already on `develop`) when the base is corrected

When creating or rebasing branches, always branch off `develop`:

```bash
git fetch origin develop
git checkout -b my-feature origin/develop
```
