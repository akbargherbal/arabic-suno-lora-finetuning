# AGENTS.md

## What you're for

- Get a training run ready: write or fix the training script, confirm the
  dataset and checkpoint are actually in place, sanity-check before it's
  launched.
- When a run crashes, find the cause and fix it.
- When asked how a run is going, check the actual log/checkpoint state and
  report that — don't guess or recall from an earlier turn.

Nothing else. Not a monitor, not a companion — a problem solver, called in
when needed.

## Output: chat is for talking, not for copying

Anything the user would need to copy-paste, or anything long enough to be
annoying to read in a terminal — a command, a fix, a set of steps, a status
report — goes in `agent_notes/current.md` instead. Overwrite that file each
time, don't append or make a new one. Chat reply stays to one line: what you
did and that it's there, e.g. "Fixed the crash — see `agent_notes/current.md`
for what changed and the command to re-run." Short one-off answers (yes/no,
a single short fact) can stay in chat.

## Never

- Start, stop, or resume the training process. The user runs it, always —
  but give the exact steps, every time: which workflow JSON to load in the
  ComfyUI UI, and the exact widget values to set on the Train Config /
  Joint Score Train Config node. There is no CLI training command in
  FL-YuE2 — training only happens by queuing a ComfyUI graph. Vague
  hand-waving forces the user to reconstruct settings you already know.
- Modify the dataset or the model checkpoint.
- Run anything GPU-heavy while a run might be active — check `nvidia-smi`
  first. One GPU, shared.

## Runtime reality — Colab is ephemeral, storage is cheap

We run on Google Colab: competitively priced and GPU-strong, but **stateless**.
Switching runtime (say L4→A100) or losing the VM wipes local disk (`/content`,
`/workspace`) — including the repo working tree and your own session context.
Only GitHub + GCS persist, so hours of GPU progress can vanish if the outputs
weren't mirrored. Storage costs far less than compute, so **bias hard toward
persisting** anything expensive to regenerate (checkpoints, prep caches,
renders) and toward docs a fresh, context-less session can restore from without
the user re-explaining. Restore path: `docs/architecture.md` → "VM / GPU
switch", `PLAN.md` §2.4.

## Backup responsibility

- Before any run that writes new checkpoint types (a new `--out` path, a new
  workflow variant), check whether `backup_to_gcp.py`'s `TARGETS` list
  actually covers the new output folder. If not, that's a bug to fix in the
  script, not something to work around by hand.
- Before the user starts a training run, confirm `backup_to_gcp.py` is
  actually running (check for the process, or freshness of
  `/content/logs/gcp_backup.log`). If it isn't, say so and give the exact
  command to start it — don't assume it's running because it usually is.
- When asked "is my progress backed up," check GCS object timestamps
  (`gs://akbar-december-2024-backup/YuE2-3B_Arabic_Suno_Finetuning/`)
  against local checkpoint timestamps and report the actual drift — don't
  assume the last known-good state is still current.

## Repo docs & checks

- Durable design/reference: `docs/architecture.md`. Open problems with
  statuses: `docs/known-issues.md`. Current state/next steps: `context.md`.
- **Canonical commands:** training in `PLAN.md`, generation in `MANUAL.md`.
  Don't re-paste them elsewhere — link.
- Checks (fast, no GPU): `pytest` and `ruff check .` (config in
  `pyproject.toml`). Run both after changing Python.

## Where to look before answering "what's going on"

- Training log: `/content/logs/train.log` (bootstrap per-job logs are
  `/content/logs/{setup,checkpoint,comfyui,dataset}.log`). Confirm this
  path once training actually starts — FL-YuE2 may log differently since
  it's a ComfyUI custom node, not a standalone script; update this line if so.
- Loss / checkpoint state: `/content/prepare_dataset/arabic-suno-lora-finetuning/ComfyUI/models/loras/`
  — confirm exact output filenames (`.safetensors`, resume state, etc.)
  once you've run Train Config once; FL-YuE2's naming may differ from the
  old trainer's `<out>.loss.json` / `<out>.resume` convention.
  Base model: `ComfyUI/models/checkpoints/` (check FL-YuE2 docs for exact
  YuE2-3B filename).
- Dataset: `/content/prepare_dataset/arabic-suno-lora-finetuning/dataset_comfyui/`
  — flat per-track sidecar files (`<uuid>.mp3`, `.caption.txt`, `.lyrics.txt`,
  optional `.abc.txt`), not the old maqam manifest tree.
- Training entrypoint: no CLI script. Training runs via the ComfyUI UI/API,
  queuing the Train Config or Joint Score Train Config node graph. Workflow
  JSON location: TBD once you've saved a working graph — update this line
  then.
