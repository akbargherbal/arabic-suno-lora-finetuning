# Inference: maqam × adapter comparison matrix

Goal: hear the same workspace song rendered by the **base model (no LoRA)** and
by the **trained `arabic_joint_v2` LoRA**, for each of the four maqams
(Hijaz, Kurd, Ajam, Nahawand), so the adapter's effect is directly comparable.

`run_inference_matrix.py` loads YuE2-3B + its decoder once and drives FL-YuE2's
own `runtime` (the same code its nodes use); it does **not** need the ComfyUI UI
or server. Base and LoRA variants of a track share the identical prompt and seed.

## 0. Where to run

A fresh Colab notebook with its **own BF16 GPU** (L4 / A100 — not the T4 that is
training, and not the training VM at all). Expect ~1–3 min of GPU per rendered
minute of audio per variant.

## 1. Get the repo + stack

```bash
git clone https://github.com/akbargherbal/arabic-suno-lora-finetuning.git
cd arabic-suno-lora-finetuning

# ComfyUI + pinned FL-YuE2 + YuE2-3B/Vae. Idempotent.
# Needs HF_TOKEN in /root/.secrets.env and GCP_DATASET_PATH exported by the
# launching notebook (same as your training VM). It also pulls the 267-track
# dataset, which inference does not need — harmless, just extra download.
bash bootstrap/setup.sh
```

Verify it finished with `[ok] YuE2-3B weights`, `[ok] ComfyUI-FL-YuE2`, and
`comfy_kitchen importable`.

## 2. Download the trained adapters from GCS

Each checkpoint is a pair: `step-NNNNNN.safetensors` (AR) and
`step-NNNNNN-nar.safetensors` (NAR); the AR file's metadata points at its NAR
companion, and the script passes both.

```bash
RUN=arabic_joint_v2
gsutil -m cp -r \
  "${GCP_BACKUP_BASE}/$RUN/loras/YuE2/$RUN" \
  ComfyUI/models/loras/YuE2/
ls ComfyUI/models/loras/YuE2/$RUN/
```

If `GCP_BACKUP_BASE` is not exported in this notebook, use the literal base
`gs://akbar-december-2024-backup/YuE2-3B_Arabic_Suno_Finetuning`.

More checkpoints appear as training runs on; re-run the `gsutil cp` to grab the
new ones (it only transfers new files).

## 3. Preview the matrix (no GPU, no model load)

```bash
python INFERENCE/run_inference_matrix.py --dry-run --all-adapters
```

Prints every selected track (maqam, clip id, seed) and the cleaned style/lyrics.

## 4. Render

Default = `base` + the latest checkpoint, one track per maqam, 90 s each:

```bash
python INFERENCE/run_inference_matrix.py \
  --manifest INFERENCE/workspace_manifest.json \
  --lora-dir ComfyUI/models/loras/YuE2/arabic_joint_v2 \
  --out INFERENCE/outputs/arabic_joint_v2 \
  --duration 90
```

Useful options:

| flag | meaning |
| --- | --- |
| `--adapters step-000400,step-000800` | exactly these checkpoints (+ base) |
| `--all-adapters` | base + every checkpoint found |
| `--all-tracks` | include the A/B pairs, not just one track per maqam |
| `--maqams Hijaz,Kurd` | subset of maqams |
| `--duration 180` | longer songs (default 90) |
| `--planning full` | compose a symbolic score first (slower); default `off` is direct |
| `--seed 1234` | base seed; each track uses `seed + track_index` so variants match |
| `--raw` | use the manifest style/lyrics verbatim instead of cleaning them |

## 5. Outputs

```
INFERENCE/outputs/arabic_joint_v2/
  Hijaz/     e70e522d_base.wav   e70e522d_step-000800.wav   ...
  Kurd/      84fd8f54_base.wav   ...
  Ajam/      2c52e220_base.wav   ...
  Nahawand/  c9331556_base.wav   ...
  matrix.json         # every file, seed, duration, truncation flag
```

Listen to `<clip>_base.wav` vs `<clip>_step-NNNNNN.wav` for the LoRA effect, and
compare the four maqam folders for the maqam character.

## 6. Back the outputs up to GCS

`INFERENCE/outputs` is a `backup_to_gcp.py` target (mirrored to `inference/`
inside the run prefix), so the inference notebook can stream results to GCS the
same way training does. Start it before or while rendering:

```bash
cd /content/arabic-suno-lora-finetuning
setsid nohup python backup_to_gcp.py --run-name inference_arabic_joint_v2 \
  > /content/logs/gcp_backup_stdout.log 2>&1 & disown
```

That mirrors into `gs://.../YuE2-3B_Arabic_Suno_Finetuning/inference_arabic_joint_v2/`:

```
inference/   <- INFERENCE/outputs/  (the WAVs + matrix.json)
loras/       <- the adapters that were tested
logs/        <- /content/logs
agent_notes/ <- agent_notes/
```

So the audio lands at
`.../inference_arabic_joint_v2/inference/<Maqam>/<clip8>_<variant>.wav`.

One-shot upload when the matrix is done:

```bash
python backup_to_gcp.py --run-name inference_arabic_joint_v2 --once
```

Notes:
- If `GCP_BACKUP_BASE` isn't exported in this notebook, add
  `--base gs://akbar-december-2024-backup/YuE2-3B_Arabic_Suno_Finetuning`.
- WAVs are written continuously; the daemon waits `--settle-seconds` (default
  60) before a pass so it never uploads a half-written file. Lower it (e.g.
  `--settle-seconds 20`) for more frequent passes.
- `rsync` is append/update-only — nothing is ever deleted in GCS.

## Notes

- **Prompt cleaning:** the manifest's `styles` is the pre-clean Suno block
  (`[Is_MAX_MODE]`, `genre: "..."`) and its `lyrics` carry stage directions
  (`[Intro | single clean guitar | ...]`). By default the script reduces these to
  the same shape as the training captions: the quoted genre/vocals/production/
  instrumentation/mood text, and `[Section]` tags. Use `--raw` to disable.
- **Planning:** default `off` (direct semantic generation) is the fastest and
  isolates the adapter. The joint recipe was trained with scores, so
  `--planning full` is a valid alternate comparison, at the cost of a second AR
  pass per song.
- The model loader requires an NVIDIA GPU with BF16 support; the T4 emulates it,
  so use L4/A100.
- Do not run this on the training VM — it competes for the same GPU.
