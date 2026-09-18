# `arabic_joint_v2` — training metric analysis

Run: scored joint AR/NAR recipe, 267-track Arabic dataset, L4.
Config: rank 32, `sequence_tokens` 24576, 3000 steps, `nar_start=base`,
`abc_dropout` 0.5, `score_planning=full`.
Snapshot at step ~936/3000 (2026-09-18 13:00 UTC).

Graphs in this folder were produced with:

```bash
python plot_training.py --run-name arabic_joint_v2 \
  --out-dir TRAINING_ANALYSIS/arabic_joint_v2
```

`01_overview.png` is the 2×2 summary; `02_…`–`10_…` are one PNG per metric
(raw trace + rolling-mean trend).

## What the curves say

- **Total loss:** one fast drop (~1.25 → ~1.0) over the first ~50 steps, then
  **dead flat at ~1.0** from step 50 to 936. No further learning after warmup.
- **Loss terms (the important one):**
  - `ar_ce` ~**0.02, flat from step 1** — the AR branch is saturated; the base
    model already predicts the codec tokens from style+lyrics.
  - `ar_kl` ~0.05–0.2, noisy, no trend.
  - `nar_flow` is **~90% of the loss and oscillates 0.8–1.1 with no downward
    trend** across 900 steps. This is the only term that can actually learn, and
    it is not.
- **Gradient norm:** noisy, mostly <0.2 with occasional spikes to ~1.0 early;
  no systematic drift. Optimisation is stable, not diverging — the signal is
  just not moving the NAR objective.
- **Learning rate:** constant `1e-4` (the joint config exposes no decay
  schedule), so the plateau is not a schedule artifact.
- **Peak VRAM:** ~13.7–15.0 GB / 23 GB, comfortable on the L4.
- **Validation at 200/400/600/800:**
  - `ar_validation` ~0.024 → 0.028 → 0.024, **flat**;
  - `nar_validation` 0.963 → 0.960 → 0.956, a **~0.7% drift over 600 steps**;
  - `artist_validation` ~1.011, **flat**; `kl_validation` ~0.15, flat.

## Reading

The scored joint recipe is behaving exactly like `arabic_joint_v1` in direct
mode: AR saturated from step 1, NAR flow dominating but not descending, and an
essentially flat artist validation. Adding the SheetSage2 scores has not changed
the optimisation — consistent with the pilot finding that the transcribed ABCs
do not carry the maqam, so there is no score signal for the recipe's
score-conditioned half to use. At the current rate ([~16 s/step, ETA ~9 h]) the
remaining ~2000 steps cost ~9 GPU-hours for what the 200→800 validation trend
suggests will be another fraction of a percent of `nar_validation`.

This is the outcome the pilot gate predicted; it is reported, not re-litigated
(the user chose to run it). Legacy AR is the spec's documented fallback.

## Run status at snapshot

- Step ~936/3000 (~31%), ~16 s/step, ETA ~9 h (~22:00 UTC).
- Checkpoints 200/400/600/800 and previews `000000`–`000800` backed up to GCS;
  step 1000 next. No errors; GPU ~15 GB, 100% util.
