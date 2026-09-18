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

## 2026-09-18 — scored joint `arabic_joint_v2` running on L4; metrics flat

- Despite the pilot gate saying the scores miss the maqam, the user chose to run
  the scored joint recipe on the full 267-track dataset (recorded in
  `DECISIONS.md`; not re-opened). Queued on L4 at 07:33: prepare transcribed all
  267 scores (~34 min), acoustic targets encoded (~26 min), joint training began
  08:36 (rank 32, seq 24576, 3000 steps, `nar_start=base`).
- At step ~936/3000 (~16 s/step, ETA ~9 h): train loss flat ~1.0, `ar_ce`
  saturated ~0.02, `ar_kl` ~0.1, `nar_flow` oscillating ~0.9–1.0 with no
  downward trend. Validation at 200/400/600/800: `ar_validation` ~0.024 flat,
  `nar_validation` 0.963 → 0.956 (very slow), `artist_validation` ~1.01 flat.
- Same behaviour as `arabic_joint_v1` — the scored conditioning is not moving the
  loss. Not stopping (user decision). Checkpoints 200–800 and their previews are
  backed up to GCS. Curves + analysis: `TRAINING_ANALYSIS/arabic_joint_v2/`
  (regenerate with
  `python plot_training.py --run-name <run> --out-dir TRAINING_ANALYSIS/<run>`).
- Next: let it run to 3000, then judge the final adapter/previews; Legacy AR
  remains the spec's fallback if it is not worth keeping.

## 2026-09-18 — Inference matrix works; checkpoint comparison added

- Fixed two blockers in `INFERENCE/`: README step 2 `gsutil cp` needed a
  `mkdir -p` destination, and `run_inference_matrix.py` drove FL-YuE2's runtime
  without `torch.no_grad()`, so autograd retained the NAR ODE graph and OOM'd the
  L4 even on `base`. Wrapped `make_plan`/`render`/`decode` in `torch.no_grad()`;
  verified `Hijaz` base + `step-000200` render on L4 (~6.9 GB peak, exit 0).
- User auditioned checkpoints: the style is emerging (positive), but output
  sometimes babbles / chews words and does not end cleanly. Not yet a verdict.
- Added `--compare 400,800,1200` to render base plus selected checkpoint steps;
  README documents it.
- `backup_to_gcp.py` default interval lowered 25 → 15 min.
- Next: run the comparison matrix (base, 400, 800, 1200) and judge the
  checkpoints; the final 3000-step `arabic_joint_v2` adapter is still pending.

## 2026-09-18 — Bootstrap skips already-present downloads

- `bootstrap/setup.sh` no longer re-downloads on a same-VM re-run: the dataset is
  gated by `/content/data/dataset/.bootstrap_complete` and transferred with
  `gsutil rsync` (partial downloads resume), and each YuE2 weight/asset job skips
  when its file already exists in `ComfyUI/models/yue2/`. Fresh VMs are unchanged
  (nothing present, full download).
