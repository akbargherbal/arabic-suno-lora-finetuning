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

## Sequence budget: the text prefix counts, and truncation doesn't cover it

- `sequence_tokens` must exceed the longest `style + lyrics` prefix plus one;
  `allow_truncation` trims only the audio codec, never the text prefix. A budget
  smaller than the longest prefix fails **every** example with
  `Caption/lyrics exceed training sequence budget`, even with truncation on.
- Measured on the 267-track Arabic dataset (from `prepared/`): max text prefix
  **1617** tokens; max `prefix + codec + 1` = **10704**. So the AR full-run
  default `12288` fits everything without truncation, joint `24576` easily does,
  and a two-step smoke needs `>= ~1664` (our client uses `2048` with
  truncation on).
- The shipped `training_smoke.json` sets `sequence_tokens=512`, which is
  unusable for this dataset — don't copy its budget.
- Audio is exactly **25 codec tokens/second**. On this dataset songs run
  2.7–6.2 min, and the AR full-run default `12288` leaves ~7.1 min of audio
  room after the longest prompt, so **every song trains in full with no
  truncation**; joint `24576` gives ~15 min. A song longer than the room would
  fail the full run (truncation off) unless the budget is raised.

## Backup: live job logs must not gate the settle wait

- `backup_to_gcp.py` waits for writes to settle before syncing a folder. The
  worker writes `output/yue2_training/jobs/<id>.log` continuously, so that log
  is always fresh and the `runs/` sync would never run during an active job.
  `wait_for_settle` now ignores the `jobs` subtree. All real state files
  (`resume.pt`, `run.json`, adapters, acoustic targets) are written atomically
  (temp + `os.replace`), so this is safe.
- Without this, `loras/` adapters still sync at checkpoints, but `runs/`
  (`run.json`, `resume.pt`, caches) only syncs once a run stops — losing resume
  ability if the VM dies mid-run. The daemon must be restarted to load the fix.
