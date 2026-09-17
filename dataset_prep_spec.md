# Dataset Prep Spec: Maqam Suno Corpus → YuE2 LoRA Training Set

## 1. Objective

Turn ~270 existing Suno-generated Arabic vocal tracks (classical poetry set to a
fixed "winning template" prompt, in four maqams) into a clean, verified,
deduplication-aware dataset directory that can feed either the **Legacy AR**
or **Joint AR/NAR** training path in `ComfyUI-FL-YuE2`, with the goal of a LoRA
that reproduces this corpus's style on new lyrics.

This spec covers **preparation only** — scanning, cleaning, deduplicating,
and laying out the data. It does not cover launching the actual training run.

## 2. Background / Constraints

- Source root: `min_4stars_ai_music/<maqam>/<workspace>/*.mp3` +
  `workspace_manifest.json` per workspace folder. Four maqam folders: `ajam`,
  `hijaz`, `kurd`, `nahawand`.
- Folder name (`min_4stars_ai_music`) implies tracks are already
  quality-filtered upstream — treat everything present as in-scope unless a
  specific check below says otherwise.
- Each `workspace_manifest.json` lists one entry per generated clip with:
  `clip_id` (UUID, globally unique), `original_title`, `assigned_filename`,
  `styles` (the full Suno prompt actually used), `exclude_styles`, `lyrics`,
  `created_at`, `status`.
- The four maqams were deliberately chosen (see `maqam_prompt_generator.py`
  docstring) because Hijaz, Nahawand, Ajam, and Kurd are all representable in
  standard 12-tone/Western notation — no quarter-tones. This matters later:
  it means SheetSage2's audio→ABC transcription is not fighting an
  unrepresentable scale for this corpus, unlike maqams such as Rast or Bayati.
- The `styles` field is Suno-syntax (control tokens + quoted key/value
  fields), not a natural-language caption. It needs cleanup before use as a
  training caption.
- YuE2 has no negative-prompt input, so `exclude_styles` cannot condition
  training directly — carry it through as metadata only.

## 3. Per-Track Data Model (fields to extract per clip)

| Field | Source | Notes |
|---|---|---|
| `clip_id` | manifest | Use as the **canonical stable filename stem** — sidesteps Arabic-filename Unicode/cross-platform issues entirely. |
| `maqam` | parent folder name | First-class field, not just implicit in path. |
| `workspace` | workspace folder name | Keep for traceability/debugging. |
| `original_title` | manifest | Used for human-readable grouping, not as an ID (not unique). |
| `styles_raw` | manifest | Preserved verbatim. |
| `styles_clean` | derived | See §5.4. |
| `lyrics_raw` | manifest | Preserved verbatim. |
| `lyrics_clean` | derived | See §5.5. |
| `exclude_styles` | manifest | Passed through unmodified, metadata only. |
| `created_at`, `status` | manifest | Passed through. |
| `duplicate_group_id`, `duplicate_group_size` | derived | See §5.1. |

**Never overwrite or discard the raw fields** — always keep `*_raw` alongside
any cleaned/derived version.

## 4. Known Data Quality Issues & Required Handling

### 4.1 Duplicates — group by lyrics, not by styles or title

Some poems were regenerated across multiple dated workspace folders (e.g.
`amro_bin_kalthoum_20082026`, `_23072026`, `_24072026`,
`amrro_bin_kalthoum_11082026` — note the typo variant — all under `hijaz`,
same poem). The prompt template drifted slightly over time (per the script's
own docstring, a `mood` field was once auto-varied and later removed), so
`styles_raw` is **not** a reliable duplicate key even for the same poem.

**Use this key instead:** `(maqam, normalized(lyrics_raw))`, where
normalization = strip the `///***///` divider, strip section-annotation
brackets, collapse whitespace, and Unicode-normalize (NFC).

Important distinction: the **same `original_title` appearing under different
top-level maqam folders is not a duplicate** — it's the same poem
deliberately rendered in a different maqam, and both are valid, distinct
training examples. Only match within the same maqam.

**Policy: tag, don't delete.** For each group, assign `duplicate_group_id`
and `duplicate_group_size`. Keep every audio file (each is still a distinct
recording/performance), but surface the group sizes in the QA report (§6) so
a later, reversible decision can be made about down-weighting or subsampling
overrepresented poems before training. Do not silently drop files at prep
time.

### 4.2 Orphan / mismatched files

For every `.mp3` found on disk, look up `assigned_filename` in that folder's
manifest:
- No match found → log as **orphan audio**, exclude from the dataset, do not
  guess a caption for it.
- Manifest lists a `clip_id` whose `assigned_filename` doesn't exist on disk
  → log as **missing audio**.
- A manifest's schema deviates from the expected fields (this is *known* to
  be a risk in the `منوعات` / "assorted" folders under `ajam` and `hijaz`,
  which may not follow the same convention) → validate per-file, don't
  assume.

Both cases go in the QA report, not into `_reports/build_log.txt` silently.

### 4.3 `status` field audit

Enumerate every distinct `status` value seen across all manifests before
deciding what to include. Confirm whether `"skipped_existing"` genuinely
means "audio already present, nothing wrong" (as it appears to), and check
for any failure-like status that should exclude a track even if an audio
file happens to exist.

### 4.4 Filename inconsistencies

`RETAKE_` vs `Retake_` prefixing is inconsistent. Don't parse retake-grouping
from the filename — the manifest's `original_title` is the source of truth
for grouping; filenames are just artifacts.

### 4.5 Caption cleanup (`styles_clean`)

Strip control tokens (`[Is_MAX_MODE: MAX](MAX)`, `[QUALITY: MAX](MAX)`,
`[REALISM: MAX](MAX)`, `[START_ON: TRUE]`, `[START_ON: "..."]`). Parse the
quoted `genre:`/`vocals:`/`production:`/`instrumentation:`/`mood:` fields out
of `styles_raw` and flatten into one natural-language caption paragraph
(most trainers/captioning conventions expect free text, not a Suno tag
block). Keep `styles_raw` untouched alongside it.

### 4.6 Lyrics cleanup (`lyrics_clean`)

Strip the `///***///` divider. Section headers like
`[Verse 1 | epic soaring vocals | heavy power chords and fast majestic strings]`
mix a structural tag with Suno-specific performance/production direction.
Normalize to a plain section tag (e.g. `[Verse]`) and drop the
descriptive-direction text for the "clean" version — but keep the full
original in `lyrics_raw` since it may be useful reference later.

## 5. Output Dataset Layout

```
dataset/
  <maqam>/
    <clip_id>.mp3                 # copied, not re-encoded (mp3/wav/flac are all accepted by existing YuE2 trainers)
    <clip_id>.caption.txt         # styles_clean
    <clip_id>.lyrics.txt          # lyrics_clean
    <clip_id>.meta.json           # everything else in the data model, incl. *_raw fields
  _reports/
    orphan_audio.csv
    missing_audio.csv
    schema_deviations.csv
    status_value_counts.csv
    duplicate_groups.csv
    build_log.txt
```

This is a **neutral intermediate layout**, not necessarily the exact sidecar
naming the ComfyUI training node expects. Before wiring it into
`ComfyUI-FL-YuE2`'s Prepare Dataset node, check that pack's own `docs/`
training guide in the cloned repo for its actual expected sidecar
filenames/extensions, and add a final rename/adapt step — I couldn't fetch
that page directly (GitHub blocked automated access) to confirm the exact
convention, so treat this as a verify-before-you-wire-it-up step rather than
an assumption baked into the pipeline.

## 6. Transcription (ABC) — decision gate, not a build step

Per your question: yes, transcription can happen as part of the training
prep, live inside ComfyUI, so this pipeline does **not** need to hand-produce
`.abc.txt` files for all 270 tracks. Recommended sequence:

1. **Pilot first.** Run SheetSage2 transcription on a small sample (~10–15
   tracks spanning all four maqams) before committing to a full run.
   Specifically check how Hijaz's augmented-second and Kurd's lowered-second
   intervals survive the transcription on real Suno-rendered (not
   studio-clean) audio — the maqam choice should make this tractable, but
   it's worth confirming on your actual files rather than assuming.
2. **If the pilot looks reasonable** → enable SheetSage2 transcription
   directly in the Joint Score Train Config / Prepare Dataset node for the
   full corpus. No pre-generated ABC needed.
3. **If a maqam transcribes badly** → fall back to the Legacy AR Config
   (caption + audio, no score) for that subset, or the whole corpus. Document
   whichever path was actually used.

## 7. Acceptance Criteria

- Every retained track has audio + clean caption + clean lyrics + maqam
  label + full raw metadata.
- No filename collisions (one file per `clip_id` per type).
- Every excluded or orphaned item is listed in `_reports/` with a reason —
  nothing silently dropped.
- Duplicate groups are tagged, not deleted.
- Source files are never modified in place.
- All filename/text comparisons are Unicode-NFC-normalized (the corpus was
  generated on Windows; prep likely runs elsewhere — don't let
  NFC/NFD mismatches cause silent misses).
- Final counts reconcile: `files in dataset/` = `total manifest tracks` −
  `(orphans + missing + schema-deviant + excluded-by-status)`, with every
  subtraction accounted for in the reports.


## More Info:
```
https://github.com/filliptm/ComfyUI-FL-YuE2.git
```