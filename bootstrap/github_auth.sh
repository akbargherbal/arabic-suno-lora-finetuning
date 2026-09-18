#!/usr/bin/env bash
# Authenticate git pushes without exposing the token to the agent.
#
# If GH_TOKEN is already set in the environment, it's used directly.
# Otherwise the user is prompted (hidden input) for a token in their own
# terminal. It validates the token, points git at gh's credential helper,
# and prints the authenticated account. The token is never echoed, never
# stored in the repo, and never written to shell history.
#
# Token needs: classic PAT with `repo` (add `workflow` to push workflow files),
# or a fine-grained PAT with Contents: Read and write on this repository.
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install it (https://cli.github.com) and retry." >&2
  exit 1
fi

if [ -z "${GH_TOKEN:-}" ]; then
  read -rsp "GitHub token: " GH_TOKEN
  echo
fi

if [ -z "${GH_TOKEN:-}" ]; then
  echo "No token entered." >&2
  exit 1
fi

if printf '%s' "$GH_TOKEN" | gh auth login --hostname github.com --with-token; then
  gh auth setup-git
  echo "OK: authenticated as $(gh api user --jq .login 2>/dev/null || echo unknown)"
  gh auth status
else
  echo "FAILED: GitHub rejected the token (check for a typo or missing scope)." >&2
  unset GH_TOKEN
  exit 1
fi
unset GH_TOKEN
