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

### Tools

When making commits, ensure your commit messages follow these standards. The CI/CD pipeline uses these messages to automate releases and generate changelogs.
