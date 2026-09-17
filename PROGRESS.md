# Progress

Durable, cross-session milestone record: what has actually been run, what it
produced, and what's next. Append an entry when a run finishes or a phase
changes; keep it short. Not a log — *why* things are the way they are lives in
`DECISIONS.md`, and transient per-session detail lives in
`agent_notes/current.md`.

## 2026-09-17 — Environment ready

- Fresh Colab VM verified: setup finished clean, repo current, 267-track dataset
  at `/content/data/dataset/dataset_comfyui`, all YuE2 weights and FL-YuE2
  present.
- Built `queue_train.py` (ComfyUI HTTP API) so runs are queued from the terminal.

## 2026-09-17 — AR smoke test passed

- `arabic_smoke`, legacy AR: rank 4, seq 2048, 2 steps, previews on.
- Outcome: `complete`; `step-000001/000002.safetensors`; loss 0.262 → 0.280.
- Validated the full pipeline (prep cache → train → checkpoint → preview → GCS
  backup). Fixed a too-small `sequence_tokens` inherited from
  `training_smoke.json` (see `DECISIONS.md`).

## 2026-09-17 — Joint AR/NAR `arabic_joint_v1` stopped (recipe/data mismatch)

- Joint recipe run in **direct mode** (no ABC scores): rank 32, seq 24576,
  steps 3000, nar_start base. Ran to step ~643, checkpoints 200/400/600.
- Analysis: NAR flow (~97% of the loss) essentially flat (~0.96–0.99), AR CE
  saturated from step 1 (~0.015), validation flat. The joint recipe's
  score-conditioned half never ran because the dataset has no `.abc` files.
  See `DECISIONS.md` for the root cause; stopped by decision, artifacts kept.
- Tooling added: `plot_training.py`; `queue_train.py --prepare-only` /
  `--cache-dir`.

## Next — score pilot, then scored joint (path A)

- 16-track pilot set already built at `/content/data/pilot16` (4 per maqam).
- Run `queue_train.py --prepare-only --dataset /content/data/pilot16
  --score-planning full --transcribe-scores --cache-dir pilot_scores`, inspect
  Hijaz/Kurd intervals; if it passes, full scored prepare then joint
  `arabic_joint_v2` with `--score-planning full --transcribe-scores`.
- Full runbook and stop/requeue commands: `agent_notes/current.md`.
