# `arabic_joint_v2` — training metric analysis

Run: scored joint AR/NAR recipe, 267-track Arabic dataset, L4.
Config: rank 32, `sequence_tokens` 24576, 3000 steps, `nar_start=base`,
`abc_dropout` 0.5, `score_planning=full`.
Snapshot: **stopped by the user at step 2400/3000**, 2026-09-18 20:02 UTC
(resumable from `resume.pt`; 600 steps / ~2.7 h remain if wanted).

Graphs in this folder were produced with:

```bash
python plot_training.py --run-name arabic_joint_v2 \
  --out-dir TRAINING_ANALYSIS/arabic_joint_v2
```

`01_overview.png` is the 2×2 summary; `02_…`–`10_…` are one PNG per metric
(raw trace + rolling-mean trend).

## What the curves say

- **Total loss:** one fast drop (~1.25 → ~1.0) over the first ~50 steps, then
  **flat at ~1.0 all the way to step 2400.** No further learning after warmup.
- **Loss terms (the important one):**
  - `ar_ce` ~**0.02, flat from step 1** — the AR branch is saturated; the base
    model already predicts the codec tokens from style+lyrics.
  - `ar_kl` ~0.05–0.22, noisy, no trend.
  - `nar_flow` is **~90% of the loss and oscillates 0.8–1.1 with no downward
    trend** across 2400 steps. This is the only term that can actually learn, and
    it is not.
- **Gradient norm:** noisy, mostly <0.2 with occasional spikes to ~1.0 early;
  no systematic drift. Optimisation is stable, not diverging — the signal is
  just not moving the NAR objective.
- **Learning rate:** constant `1e-4` (the joint config exposes no decay
  schedule), so the plateau is not a schedule artifact.
- **Peak VRAM:** ~13.7–15.0 GB / 23 GB, comfortable on the L4.

### Validation at checkpoints

| step | ar_val | nar_val | kl_val | artist_val |
| --- | --- | --- | --- | --- |
| 200 | 0.0233 | 0.9629 | 0.1603 | 1.0183 |
| 400 | 0.0278 | 0.9595 | 0.1388 | 1.0151 |
| 600 | 0.0250 | 0.9578 | 0.1494 | 1.0127 |
| 800 | 0.0240 | 0.9562 | 0.1538 | 1.0110 |
| 1000 | 0.0241 | 0.9543 | 0.1537 | 1.0091 |
| 1200 | 0.0258 | 0.9527 | 0.1468 | 1.0078 |
| 1400 | 0.0242 | 0.9524 | 0.1522 | 1.0070 |
| 1600 | 0.0248 | 0.9519 | 0.1495 | 1.0066 |
| 1800 | 0.0241 | 0.9508 | 0.1526 | 1.0054 |
| 2000 | 0.0391 | 0.9511 | 0.0993 | 1.0101 |
| 2200 | 0.0228 | 0.9502 | 0.1655 | 1.0061 |
| 2400 | 0.0303 | 0.9494 | 0.1283 | 1.0054 |

`ar_validation` is flat/noisy (~0.025). `nar_validation` falls **monotonically
but very slowly**: 0.9629 → 0.9494 over 2200 steps (−0.0135, ~0.0012 per 200
steps). `artist_validation` (AR + NAR) drifts 1.0183 → 1.0054.

## Reading

The scored joint recipe behaves like `arabic_joint_v1` in direct mode: AR
saturated from step 1, NAR flow dominating yet not descending, and artist
validation essentially flat apart from a tiny, slow `nar_validation` drift.
Adding the SheetSage2 scores did not change the optimisation — consistent with
the pilot finding that the transcribed ABCs do not carry the maqam, so the
recipe's score-conditioned half had no usable signal. Stopping at 2400 loses
almost nothing relative to the full 3000; the remaining 600 steps (~2.7 h) buy
at most another ~0.004 of `nar_validation` by the observed slope.

This is the outcome the pilot gate predicted; it is reported, not re-litigated
(the user chose to run it). Legacy AR is the spec's documented fallback.

## Run state at stop

- Step 2400/3000. In GCS under `<run>/{loras,runs}`: adapters **200–2400**
  (`step-NNNNNN.safetensors` + `-nar`), `resume.pt` (800 MB), `run.json`,
  previews, and the 267 prepared/acoustic caches.
- Resume later: restore `<run>/runs/<run>/` → `ComfyUI/output/yue2_training/`,
  start ComfyUI, then
  `queue_train.py --recipe joint --preset full --run-name arabic_joint_v2
  --score-planning full --transcribe-scores --cache-dir arabic_joint_v2
  --resume resume.pt`.
