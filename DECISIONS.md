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

## Recipe vs dataset: run the score pilot before the joint recipe

- This dataset has **no `.abc.txt` scores by design**: `prepare_dataset.py`'s
  docstring calls transcription "a decision gate, not a build step" and only
  emits a pilot candidate list. `dataset_prep_spec.md` §6 requires a ~10–15
  track SheetSage2 pilot (checking Hijaz's augmented-second and Kurd's
  lowered-second) before committing to scored training; the pilot was never run.
- The joint recipe is score-conditioned: `abc_dropout` is inert with
  `score_planning=off`, and in direct mode the AR CE is already saturated
  (base model predicts the codec tokens), so a direct-mode joint run mostly
  trains the NAR branch and wastes its other half. Don't run joint without scores.
  Observed on `arabic_joint_v1` (direct mode, 643 steps): ar_ce ~0.015 from step
  1, nar_flow flat ~0.97, validation flat.
- Legacy AR is the spec's fallback, but as-shipped it needs `cursor_weight=0`
  (or `align_lyrics=true`): the default 0.08 requires per-song lyric cursors the
  prepared data does not have (`trainer.py` raises "Missing lyric alignment").
- Even to **reuse** cached scores, `transcribe_missing_scores` must stay on:
  `prepare.py` raises if a track has no ABC and the sheetsage asset is absent,
  before it looks for the cached score file. So the scored joint run needs
  `--score-planning full --transcribe-scores`.
- Chosen path: pilot first; if the interval check passes, full scored prepare,
  then requeue the joint recipe under a new run name.

## User override: run the scored joint recipe anyway

- The 16-track pilot (2026-09-18) showed the SheetSage2 ABCs do **not** carry the
  maqam — all four maqams come out diatonic natural minor, 0/16 with a lowered
  2nd, 1/16 with a major 3rd. By `dataset_prep_spec.md` §6.3 this is the "maqam
  transcribes badly" branch, and the recommendation was Legacy AR (no score).
- The user was told and chose to proceed with the **scored joint** recipe anyway
  (`arabic_joint_v2`, `--score-planning full --transcribe-scores`). Do not
  re-open this; treat the scores as a weak/again-to-be-validated conditioning
  signal, not as maqam ground truth. Any future "the scores don't encode the
  maqam" finding is already known.
- Measured precedent for capacity: `arabic_joint_v1` (rank 32, seq 24576, direct
  mode) peaked at **12.4 GB allocated** — fits a 15 GB T4 with ~2.6 GB headroom,
  tighter once the ABC tokens lengthen the text prefix. Fallback if it OOMs:
  switch to L4 and restore the scored prep cache from the run's GCS prefix
  (`<run>/runs/` → `output/yue2_training/`).
