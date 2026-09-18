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

## 2026-09-18 — Score pilot ran on T4; scores do not capture the maqams

- Fresh VM (T4); `/content` was wiped, so rebuilt the 16-track `/content/data/pilot16`
  from the persisted GCS candidate CSV + `dataset_comfyui`.
- Ran `queue_train.py --prepare-only --dataset /content/data/pilot16
  --score-planning full --transcribe-scores --cache-dir pilot_scores`. Outcome:
  `complete`, 16/16 tracks, ~8.2 min, no OOM, no repair warnings.
- Decision-gate finding: all 16 ABCs are diatonic natural minor in both voices;
  0/16 show a lowered 2nd (Kurd) and only 1/16 a major 3rd (Hijaz). The
  augmented-second / lowered-second intervals did not survive. Spec §6.3 →
  don't commit to the scored joint run.
- Detail and by-ear verification files: `agent_notes/current.md`.

## Next — scored joint `arabic_joint_v2` (user committed, 2026-09-18)

- Despite the pilot gate saying the scores miss the maqam, the user chose to run
  the scored joint recipe on the full 267-track dataset. Recorded in
  `DECISIONS.md`; don't re-open.
- Run: `queue_train.py --recipe joint --preset full --run-name arabic_joint_v2
  --score-planning full --transcribe-scores --cache-dir arabic_joint_v2` (Prepare
  transcribes all 267 scores, then joint training; rank 32, seq 24576).
- T4 should fit (`arabic_joint_v1` peaked at 12.4 GB); if training OOMs, switch
  to L4 and restore the scored cache from GCS. Exact runbook: `agent_notes/current.md`.
