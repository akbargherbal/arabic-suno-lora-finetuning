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
  but give the exact command, every time, not just "run it now." Vague
  hand-waving forces the user to reconstruct flags you already know.
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
  script like Mothersuperior's `joint.py`), check whether `backup_to_gcp.py`'s
  `TARGETS` list actually covers the new output folder. If not, that's a bug
  to fix in the script, not something to work around by hand.
- Before the user starts a training run, confirm `backup_to_gcp.py` is
  actually running (check for the process, or freshness of
  `/content/logs/gcp_backup.log`). If it isn't, say so and give the exact
  command to start it — don't assume it's running because it usually is.
- When asked "is my progress backed up," check GCS object timestamps against
  local checkpoint timestamps and report the actual drift — don't assume the
  last known-good state is still current.

## Repo docs & checks
- Durable design/reference: `docs/architecture.md`. Open problems with
  statuses: `docs/known-issues.md`. Current state/next steps: `context.md`.
- **Canonical commands:** training in `PLAN.md`, generation in `MANUAL.md`.
  Don't re-paste them elsewhere — link.
- Checks (fast, no GPU): `pytest` and `ruff check .` (config in
  `pyproject.toml`). Run both after changing Python.

## Where to look before answering "what's going on"
- Training log: `/content/logs/train.log` (the trainer command tees here;
  bootstrap per-job logs are `/content/logs/{setup,checkpoint,comfyui,dataset}.log`).
- Loss / checkpoint state: `/content/yue2_lora_finetuning/ComfyUI/models/loras/`
  — `<out>.loss.json` (loss/eval curves), `<out>.safetensors` (LoRA),
  `<out>.resume` (resume state), plus `<out>_NNNNNN.*` with `--save-every`.
  Base model: `ComfyUI/models/checkpoints/yue2_3b_bf16.safetensors`.
- Dataset: `/content/data/dataset/` (`train/`, `val/`, `manifest.csv`).
- Training script: `ComfyUI/custom_nodes/ComfyUI-YuE2-Trainer/train_cli.py`
  (repo root is `/content/yue2_lora_finetuning`).

