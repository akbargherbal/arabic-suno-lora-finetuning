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

## In flight — joint AR/NAR `arabic_joint_v1`

- Joint recipe: rank 32, seq 24576, steps 3000, save_every 200, accumulation 1,
  nar_start base, previews on (30 s).
- Acoustic-target encode finished (267/267); training started, step 5/3000 at
  ~12.6 s/step → **ETA ~10–11 h** on the L4, plus ~15 preview renders.
- Interruption/resume: `python queue_train.py --recipe joint --preset full
  --run-name arabic_joint_v1 --resume resume.pt`.
- Next: first checkpoint at step 200; review loss/previews, then judge whether
  3000 steps is the right budget.
