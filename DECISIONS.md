# Decisions & insights

Major, durable decisions and non-obvious insights only — the things a fresh
session must not re-litigate or rediscover. One short entry per decision. This
is not a log: routine per-session state belongs in `agent_notes/current.md`.

## Terminal/API over the ComfyUI UI

- The user is new to ComfyUI and finds navigating the UI high-friction. Chosen
  workflow: copy-paste terminal commands that the agent can inspect and debug.
- Therefore queue training through ComfyUI's HTTP API (`/prompt`) instead of
  asking the user to load workflows in the UI. `queue_train.py` (repo root) is
  the client; it queues the same graph the UI would, so run state, previews,
  adapters, and GCS backups are unchanged.
- Keep giving exact commands, not UI click paths.
