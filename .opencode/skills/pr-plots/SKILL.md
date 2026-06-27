---
name: pr-plots
description: Use when preparing a fix or feature PR for review and adding before/after plot evidence to the PR body. Covers generating pre-fix and post-fix plots via detect_trends(plot=True), updating the PR body with image placeholders via gh CLI, the human-in-the-loop image upload to user-attachments, and cleanup. Never commits plot files or temp scripts.
---

# Before/After plots for PR body

When a fix or feature PR is ready for review, generate visual evidence showing the change's effect and add it to the PR description. This is a human-in-the-loop process: the agent generates plots and writes the PR body with `<img>` placeholders; the user pastes images into GitHub's editor to get `user-attachments/assets/` URLs.

## When to generate

| PR type | Plots | Notes |
|---|---|---|
| **Fix PR** | Before + After for the main bug | Show the bug behavior (pre-fix) and the corrected behavior (post-fix) |
| **Feature PR** | After only (usually) | If there's a prior behavior to compare, include Before; otherwise just show the new behavior |
| **Guard rail** | After only | When a fix introduces a workaround or edge-case suppression, add a complementary plot showing that other scenarios still work correctly |

### Guard rail plots

When a fix introduces a workaround (e.g. suppressing noise on a zero-baseline leading edge), add a guard rail plot to verify the fix doesn't break another scenario. Use a **single "after" plot** (not before/after) — the point is confirming no regression. Label the section "Guard rail" and explain in the text why it was added.

## Generating "before" plots (pre-fix state)

1. **Find the PR's first commit**: `gh pr view <N> --json commits -q '.commits[0].oid'` or GitHub MCP `get_commits`.
2. **Find its parent**: `git fetch origin <sha> && git rev-parse <sha>^`.
3. **Checkout parent commit** (detached HEAD): `git checkout <parent-sha>`.
4. **If test fixture data doesn't exist at that commit**, extract from the target branch: `git show origin/develop:path/to/fixture.csv > /tmp/fixture.csv`.
5. **Write a temp Python script** that:
   - Sets matplotlib to Agg backend: `matplotlib.use('Agg')`
   - Loads the fixture data (from `/tmp/` if extracted, or from the repo if it exists at that commit)
   - Runs `detect_trends(plot=True)` with the same params as the test
   - Saves the plot: `plt.savefig('before_<scenario>.png', dpi=150, bbox_inches='tight')`
   - Closes the figure: `plt.close()`
6. **Run the script**: `python _gen_before_plots.py`.

## Generating "after" plots (post-fix state)

1. **Checkout the working branch** (develop or feature branch).
2. **Run the same script** with output filenames changed to `after_<scenario>.png`.

## Updating PR body

1. **Write the PR body markdown** to `/tmp/pr_body.md` with `<img>` placeholder tags (empty `src=""`).
2. **Update via gh CLI REST API** (the GitHub MCP token lacks PR write permission):
   ```bash
   gh api repos/OWNER/REPO/pulls/N --method PATCH --field body="$(cat /tmp/pr_body.md)"
   ```
3. **Never `git add`, `git commit`, or `git push`** plot files or temp scripts. Plots stay local only.
4. **User pastes each image** into the GitHub PR editor (edit PR → paste image into body) to get `user-attachments/assets/` URLs.
5. **User provides URLs** → agent updates the PR body with real image URLs (same `gh api` PATCH call).

## PR body structure template

```markdown
## Before / After — `test_<name>`

The core fix: <one-line description>

**Before fix** — <description of the bug behavior>:

<img width="1990" height="478" alt="<description>" src="" />

**After fix** — <description of the fixed behavior>:

<img width="1990" height="478" alt="<description>" src="" />

## Guard rail — `test_<name>`

Added to verify the fix doesn't break <scenario>. Confirmed working post-fix:

<img width="1990" height="478" alt="<description>" src="" />
```

## Cleanup

After the user has pasted the images and confirmed the PR body looks good:

- Delete temp plot files (`*.png`) from the repo working directory
- Delete temp Python scripts (`_gen_*.py`)
- Delete temp CSV fixtures from `/tmp/`
- Delete temp branch if one was created (`git branch -D <temp-branch>`)
- Switch back to the original working branch
- Do NOT leave any untracked files in the repo

## Constraints

- **Zero commits**: Never `git add` or `git commit` plot files, temp scripts, or temp fixtures. These are ephemeral.
- **MCP token limitation**: The GitHub MCP token does not have PR write permission. Use `gh api` (REST) or `gh pr edit` (if it works) with the user's local PAT instead.
- **Image hosting**: GitHub has no API for uploading images to `user-attachments/assets/`. This is a web-editor-only feature. The human-in-the-loop step is mandatory.
- **matplotlib backend**: Always use `matplotlib.use('Agg')` in temp scripts to avoid interactive display warnings and ensure PNGs save correctly in non-interactive environments.
