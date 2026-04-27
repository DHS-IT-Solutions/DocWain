---
name: No Claude Attribution in Commits
description: Never include Co-Authored-By, Claude, or Anthropic references in any git commit messages or code
type: feedback
---

Never add `Co-Authored-By` lines, "Claude", or "Anthropic" references to git commit messages or anywhere in code/docs pushed to GitHub.

**Why:** User does not want any trace of AI assistance visible in their GitHub repository.

**How to apply:** When creating any git commit, omit the `Co-Authored-By: Claude ...` trailer entirely. Do not mention Claude, Anthropic, or AI assistance in commit messages, PR descriptions, code comments, or documentation. A commit-msg git hook is installed at `.git/hooks/commit-msg` that blocks these references as a safety net.
