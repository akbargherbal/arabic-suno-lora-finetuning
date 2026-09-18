# AGENTS.md

## What you're for

- Get a training run ready: confirm the dataset, base models, and training
  assets are actually in place, and sanity-check the ComfyUI graph before the
  user queues it.
- When a run crashes, find the cause and fix it.
- When asked how a run is going, check the actual run/checkpoint state and
  report that — don't guess or recall from an earlier turn.

Nothing else. Not a monitor, not a companion — a problem solver, called in
when needed.

## Durable decisions & progress

Read `DECISIONS.md` and `PROGRESS.md` (repo root) at the start of any task
involving training or tooling. `DECISIONS.md` records major, cross-session
decisions and non-obvious insights — the things a fresh session must not
re-litigate or rediscover. `PROGRESS.md` records the milestone trail: runs and
their outcomes, and what's next. Keep both short and add an entry only when
losing it would cost real work; neither is a log or a dump of routine findings.
Routine, per-session state goes in `agent_notes/current.md`.

## The stack, so a fresh session isn't guessing

ComfyUI + `filliptm/ComfyUI-FL-YuE2` (a custom node). **There is no training
CLI** — training happens only by queuing a ComfyUI graph: the `Train Config`
node (legacy AR) or `Joint Score Train Config` (joint AR/NAR). The pack's
Python lives in `ComfyUI/custom_nodes/ComfyUI-FL-YuE2/yue2/` (incl.
`yue2/training/`). One GPU, shared.

## Output: chat is for talking, not for copying

Anything the user would need to copy-paste, or anything long enough to be
annoying to read in a terminal — a command, a fix, a set of steps, a status
report — goes in `agent_notes/current.md` instead. Overwrite that file each
time, don't append or make a new one (it's gitignored; create the folder if
it's missing). Chat reply stays to one line: what you did and that it's there,
e.g. "Fixed the crash — see `agent_notes/current.md` for what changed and the
command to re-run." Short one-off answers (yes/no, a single short fact) can
stay in chat.

## Never

- Start, stop, or resume the training process. The user runs it, always — but
  give the exact steps every time: which workflow JSON to load, and the exact
  widget values to set on the Train Config / Joint Score Train Config node.
  Vague hand-waving forces the user to reconstruct settings you already know.
- Modify the dataset (`dataset_comfyui/`) or the model weights
  (`ComfyUI/models/yue2/`).
- Run anything GPU-heavy while a run might be active — check `nvidia-smi`
  first. One GPU, shared.

## Runtime reality — Colab is ephemeral, storage is cheap

We run on Google Colab: strong GPUs, but **stateless**. Switching runtime
(say L4→A100) or losing the VM wipes local disk (`/content`) — including the
repo working tree and your own session context. Only GitHub + GCS + Hugging
Face persist, so GPU progress can vanish if outputs weren't mirrored. Storage
costs far less than compute, so **bias hard toward persisting** anything
expensive to regenerate (adapters, run state, prep caches) and toward docs a
context-less session can restore from.

**Restore on a fresh VM:** re-clone this repo, then:

```bash
cd <repo>
bash bootstrap/setup.sh     # idempotent: re-clones ComfyUI, re-downloads models + dataset
```

`bootstrap/setup.sh` re-creates ComfyUI, every `ComfyUI/models/yue2/` weight,
and the dataset. The dataset's GCS path comes from `GCP_DATASET_PATH`, which
the launching notebook exports — it is deliberately not stored in this repo.
It does **not** restore trained adapters or run state — pull those from the
run's GCS prefix (base in `GCP_BACKUP_BASE`, exported by the notebook):
`<run>/loras/` → `ComfyUI/models/loras/`, `<run>/runs/` →
`ComfyUI/output/yue2_training/`.

## Backup responsibility

- `backup_to_gcp.py` mirrors `ComfyUI/models/loras` (exported adapters),
  `ComfyUI/output/yue2_training` (`run.json` / `resume.pt` / metrics / previews
  plus the `prepared/` and `acoustic_targets/` prep caches), `/content/logs`,
  and `agent_notes/`. Before a run that writes a new output folder, check
  `TARGETS` covers it — if not, fix the script, don't work around it by hand.
- Before the user starts a run, confirm the daemon is actually running
  (`pgrep -af backup_to_gcp.py`, or freshness of `/content/logs/gcp_backup.log`).
  If it isn't, give the exact command — `--run-name` is required, and the GCS
  base comes from `GCP_BACKUP_BASE`, which the launching notebook exports:

  ```bash
  cd /content/arabic-suno-lora-finetuning
  setsid nohup python backup_to_gcp.py --run-name <run> \
    > /content/logs/gcp_backup_stdout.log 2>&1 & disown
  ```

- "Is my progress backed up?" → compare GCS object timestamps under the run's
  configured prefix (`GCP_BACKUP_BASE` / `--base`) with the local timestamps
  and report the actual drift — don't assume the last known-good state is
  still current.

## GitHub pushes

- Pushing needs auth that the user supplies, never the agent. Ask them to run
  `bash bootstrap/github_auth.sh` in their terminal and paste a PAT at the hidden
  prompt; it validates the token, runs `gh auth setup-git`, and prints the
  authenticated account. After that, `git push` works for the rest of the
  session. Never ask for the token in chat.
- `/content` is ephemeral, so re-run it on every fresh VM.

## Repo docs & checks

- Dataset build and independent validation: `prepare_dataset.py`,
  `verify_dataset.py`; the spec is `dataset_prep_spec.md`; the VM bootstrap is
  `bootstrap/setup.sh`.
- Checks (fast, no GPU): `pytest` and `ruff check .` (config in
  `pyproject.toml`). Run both after changing Python. **Known baseline:**
  `ruff check .` currently reports 7 pre-existing findings in
  `prepare_dataset.py` / `verify_dataset.py`; clean those up when you touch
  those files, and don't let them hide new ones.

## Where to look before answering "what's going on"

- **Training progress:** no log file — FL-YuE2 runs inside ComfyUI, and the
  node + UI show step / loss / previews. If the server was started to a file
  it's `/content/logs/comfyui.log`. Bootstrap per-job logs are
  `/content/logs/{opencode,comfyui,dataset,yue2,vae,mert,assets}.log`.
- **Run state:** `ComfyUI/output/yue2_training/<run>/run.json` (status,
  checkpoints, previews) and `resume.pt`; prep caches in the same folder
  (`prepared/`, `acoustic_targets/`).
- **Adapters:** `ComfyUI/models/loras/YuE2/<run>/step-NNNNNN.safetensors`
  (plus `-nar.safetensors` on acoustic-enabled runs).
- **Base models:** `ComfyUI/models/yue2/{YuE2-3B,YuE2-Vae,MERT-v2-FullSong}/model.safetensors`
  and `ComfyUI/models/yue2/training_assets/` (`tokenizer_head_joint_v4.pt`,
  `nar_lora_joint_v4.pt`, `minted_regularizer_pack.pt`).
- **Dataset:** `/content/data/dataset/dataset_comfyui/` — flat `<stem>.mp3` +
  `.caption.txt` + `.lyrics.txt` + `.caption.json` + `.song.txt` (optional
  `.abc.txt`), verified against FL-YuE2's own loader. Re-check any time with
  `verify_dataset.py`.
- **Workflows:** `ComfyUI/custom_nodes/ComfyUI-FL-YuE2/example_workflows/` —
  `training_studio.json` contains the legacy `Train Config`; replace it with
  `Joint Score Train Config` for the joint recipe. `training_smoke.json` is a
  fast two-step pipeline check, not a quality preset.
- **Training entrypoint:** none via CLI. The user loads the workflow in the
  ComfyUI UI and queues it; there is no script for you to run.
