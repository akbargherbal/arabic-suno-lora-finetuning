#!/usr/bin/env python3
"""
prepare_dataset.py — Maqam Suno corpus -> YuE2 LoRA training set.

Implements dataset_prep_spec.md §2-§6 plus the "verify-before-you-wire-it-up"
step in §5, against the real expected format of filliptm/ComfyUI-FL-YuE2.

What it does
------------
1. Walks min_4stars_ai_music/<maqam>/<workspace>/{*.mp3,workspace_manifest.json}.
2. Treats a manifest entry as in-scope only when its assigned_filename is
   physically present in its own workspace folder (the 4+ star survivors).
   Missing audio, orphans and schema deviations are logged, never guessed.
3. Builds the neutral spec layout:

       dataset/<maqam>/<clip_id>.mp3
       dataset/<maqam>/<clip_id>.caption.txt   (styles_clean)
       dataset/<maqam>/<clip_id>.lyrics.txt     (lyrics_clean)
       dataset/<maqam>/<clip_id>.meta.json      (full model incl. *_raw)

4. Emits every report required by §5 under dataset/_reports/.
5. Emits an *adapted*, flat layout for ComfyUI-FL-YuE2's Dataset Maker node
   (dataset_comfyui/), whose loader (yue2/training/data.py) is single-folder
   and expects:

       <stem>.mp3 / .wav / .flac
       <stem>.caption.txt   style caption   (optional -> default_style)
       <stem>.lyrics.txt    lyrics          (required; empty == instrumental)
       <stem>.abc.txt       score           (optional; SheetSage2 can fill)
       <stem>.caption.json  metadata        (optional; audio_sha256 checked)
       <stem>.song.txt      split identity  (optional; keeps takes together)

   Source clips are copied byte-for-byte (mp3 is accepted, no re-encode).
   Source files are never modified.

Transcription (spec §6) is a decision gate, not a build step: no .abc.txt is
produced here. A pilot candidate list spanning all four maqams is written to
_reports/transcription_pilot_candidates.csv instead.

Usage
-----
    python prepare_dataset.py
    python prepare_dataset.py --src min_4stars_ai_music --out dataset \\
        --comfyui-out dataset_comfyui

    # Rebuild only: --overwrite removes previous outputs first.
    python prepare_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MAQAM_DIRS = {
    "ajam": "Ajam",
    "hijaz": "Hijaz",
    "kurd": "Kurd",
    "nahawand": "Nahawand",
}

AUDIO_EXTS = {".mp3", ".wav", ".flac"}

REQUIRED_TRACK_FIELDS = (
    "clip_id",
    "original_title",
    "assigned_filename",
    "styles",
    "exclude_styles",
    "lyrics",
    "created_at",
    "status",
)

STRING_TRACK_FIELDS = ("clip_id", "assigned_filename", "styles", "lyrics", "status")

# Statuses that mean "the audio is fine, keep it". Anything matching a
# failure-like pattern is excluded even if a file exists; anything else is
# kept but reported as unknown so a human can adjudicate.
GOOD_STATUSES = {"downloaded", "skipped_existing"}
FAILURE_HINT = re.compile(r"fail|error|cancel|reject|remov|delet|abort|missing", re.I)

CAPTION_FIELD_ORDER = ("genre", "vocals", "production", "instrumentation", "mood")

SECTION_WORDS = {
    "intro": "Intro",
    "verse": "Verse",
    "prechorus": "Pre-Chorus",
    "pre-chorus": "Pre-Chorus",
    "pre chorus": "Pre-Chorus",
    "chorus": "Chorus",
    "bridge": "Bridge",
    "outro": "Outro",
    "hook": "Hook",
    "refrain": "Refrain",
    "interlude": "Interlude",
    "breakdown": "Breakdown",
    "coda": "Coda",
    "tag": "Tag",
    "instrumental": "Instrumental",
}

DIVIDER_LINE = re.compile(r"^[\s/*]*$")
TAG_RE = re.compile(r"^\[(?P<body>[^\]]*)\]\s*$")
ALL_TAG_RE = re.compile(r"\[[^\]]*\]")

CONTROL_TOKEN_RES = (
    re.compile(r"\[(?:Is_MAX_MODE|QUALITY|REALISM)\s*:[^\]]*\]\s*\([^)]*\)", re.I),
    re.compile(r"\[START_ON:\s*(?:TRUE|\"[^\"]*\")\s*\]", re.I),
    re.compile(r"\[[^\]]*\]\s*\([^)]*\)"),  # any leftover generic control token
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def nfc(text: str) -> str:
    """Unicode NFC. The corpus was authored on Windows; prep may not be."""
    return unicodedata.normalize("NFC", text if isinstance(text, str) else str(text))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_control_tokens(styles: str) -> str:
    text = styles
    for regex in CONTROL_TOKEN_RES:
        text = regex.sub(" ", text)
    return text


def parse_caption_fields(styles: str) -> list[str]:
    """Pull genre/vocals/production/instrumentation/mood values, in the order
    they appear, from Suno-style `key: "value"` lines. Handles unquoted values
    and the plain-natural-language `منوعات` variant gracefully."""
    values: list[tuple[int, str]] = []
    for lineno, raw_line in enumerate(strip_control_tokens(styles).splitlines()):
        line = raw_line.strip()
        m = re.match(
            r"^(genre|vocals|production|instrumentation|mood)\s*:\s*(.+?)\s*$",
            line,
            re.I,
        )
        if not m:
            continue
        value = m.group(2).strip().strip('"').strip()
        if value:
            values.append((lineno, value))
    if values:
        return [v for _, v in values]
    # No structured fields: fall back to the control-token-free text itself.
    plain = re.sub(r"\s+", " ", strip_control_tokens(styles)).strip()
    return [plain] if plain else []


def clean_caption(styles: str) -> tuple[str, str]:
    """Return (caption, source) where source is 'parsed' or 'fallback'."""
    fields = parse_caption_fields(styles)
    if not fields:
        return "", "empty"
    # Flatten to one natural-language paragraph, one sentence each.
    sentences = []
    for value in fields:
        value = re.sub(r"\s+", " ", value).strip().rstrip(".").strip()
        if value:
            sentences.append(value + ".")
    caption = " ".join(sentences)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption, "parsed"


def normalize_lyrics_key(raw: str) -> str:
    """Spec §4.1 duplicate key normalization: strip the ///***/// divider,
    strip section-annotation brackets, collapse whitespace, NFC."""
    text = nfc(raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = ALL_TAG_RE.sub(" ", text)
    lines = [
        ln for ln in text.split("\n")
        if not (ln.strip() and DIVIDER_LINE.match(ln))
    ]
    text = "\n".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_section_label(body: str) -> str | None:
    """Map a bracketed tag body to a plain section tag, or None to drop it.

    '[Verse 1 | epic vocals | power chords]' -> 'Verse'
    '[Instrumental Build-up: guitars]'       -> 'Instrumental'
    '[orchestral strings swell]'             -> None (production cue)
    """
    head = re.split(r"[|:]", body, maxsplit=1)[0]
    head = head.strip()
    if not head:
        return None
    tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", head)
    for token in tokens:
        key = token.strip().lower()
        if key in SECTION_WORDS:
            return SECTION_WORDS[key]
        squashed = key.replace("-", "")
        if squashed in SECTION_WORDS:
            return SECTION_WORDS[squashed]
    return None


def clean_lyrics(raw: str) -> str:
    """Spec §4.6 lyrics_clean: drop the divider, reduce mixed
    structure+direction headers to a plain section tag, drop pure
    production-direction cues, keep every sung line verbatim."""
    if not raw:
        return ""
    out: list[str] = []
    for raw_line in nfc(raw).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            out.append("")
            continue
        if DIVIDER_LINE.match(line):
            continue  # ///***///
        m = TAG_RE.match(line)
        if m:
            label = canonical_section_label(m.group("body"))
            if label:
                out.append(f"[{label}]")
            continue  # non-structural cue -> drop
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
    return text


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Track:
    clip_id: str
    maqam_key: str
    maqam: str
    workspace: str
    original_title: str
    source_audio: Path
    styles_raw: str
    lyrics_raw: str
    exclude_styles: str
    created_at: str
    status: str
    downloaded_at: str = ""
    duplicate_group_id: str = ""
    duplicate_group_size: int = 0
    styles_clean: str = ""
    caption_source: str = ""
    lyrics_clean: str = ""
    source_sha256: str = ""
    source_bytes: int = 0
    audio_ext: str = ""

    def meta(self, src_root: Path) -> dict:
        data = {
            "clip_id": self.clip_id,
            "maqam": self.maqam,
            "workspace": self.workspace,
            "original_title": self.original_title,
            "styles_raw": self.styles_raw,
            "styles_clean": self.styles_clean,
            "caption_source": self.caption_source,
            "lyrics_raw": self.lyrics_raw,
            "lyrics_clean": self.lyrics_clean,
            "exclude_styles": self.exclude_styles,
            "created_at": self.created_at,
            "status": self.status,
            "duplicate_group_id": self.duplicate_group_id,
            "duplicate_group_size": self.duplicate_group_size,
            "audio_extension": self.audio_ext,
            "source_audio": str(self.source_audio.relative_to(src_root)),
            "source_audio_sha256": self.source_sha256,
            "source_audio_bytes": self.source_bytes,
        }
        if self.downloaded_at:
            data["downloaded_at"] = self.downloaded_at
        return data


@dataclass
class Report:
    orphans: list[dict] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)
    schema: list[dict] = field(default_factory=list)
    statuses: Counter = field(default_factory=Counter)
    status_retained: Counter = field(default_factory=Counter)
    status_excluded: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)

    def add_schema(self, maqam, workspace, clip_id, field_name, issue, action):
        self.schema.append({
            "maqam": maqam,
            "workspace": workspace,
            "clip_id": clip_id or "",
            "field": field_name,
            "issue": issue,
            "action": action,
        })


# --------------------------------------------------------------------------
# Corpus walk
# --------------------------------------------------------------------------

def find_audio(workspace_dir: Path, assigned_filename: str) -> Path | None:
    """NFC-tolerant lookup of assigned_filename inside its workspace folder."""
    direct = workspace_dir / nfc(assigned_filename)
    if direct.is_file():
        return direct
    target = nfc(assigned_filename)
    for candidate in workspace_dir.iterdir():
        if candidate.is_file() and nfc(candidate.name) == target:
            return candidate
    return None


def classify_status(status: str) -> str:
    s = (status or "").strip().lower()
    if s in GOOD_STATUSES:
        return "included"
    if FAILURE_HINT.search(s):
        return "excluded"
    return "unknown"


def walk_corpus(src_root: Path, report: Report) -> list[Track]:
    tracks: list[Track] = []
    total_manifest_tracks = 0

    if not src_root.is_dir():
        raise SystemExit(f"source root not found: {src_root}")

    known = {k.lower(): (k, v) for k, v in MAQAM_DIRS.items()}

    for maqam_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        key = maqam_dir.name.strip().lower()
        if key not in known:
            report.notes.append(
                f"[warn] top-level folder '{maqam_dir.name}' is not a known maqam; skipped"
            )
            continue
        maqam_key, maqam = known[key]

        for manifest_path in sorted(maqam_dir.rglob("workspace_manifest.json")):
            workspace_dir = manifest_path.parent
            workspace = nfc(workspace_dir.name)
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:  # malformed manifest
                report.add_schema(maqam, workspace, "", "<file>",
                                  f"unreadable manifest: {exc}", "workspace skipped")
                continue

            raw_tracks = data.get("tracks")
            if not isinstance(raw_tracks, list):
                report.add_schema(maqam, workspace, "", "tracks",
                                  f"'tracks' is {type(raw_tracks).__name__}, expected list",
                                  "workspace skipped")
                continue

            disk_files = [p for p in workspace_dir.iterdir() if p.is_file()]
            disk_norm = {nfc(p.name): p for p in disk_files}
            consumed: set[str] = set()

            for index, rt in enumerate(raw_tracks):
                total_manifest_tracks += 1
                if not isinstance(rt, dict):
                    report.add_schema(maqam, workspace, "", f"tracks[{index}]",
                                      "entry is not an object", "excluded")
                    continue

                missing_fields = [f for f in REQUIRED_TRACK_FIELDS if f not in rt]
                if missing_fields:
                    report.add_schema(maqam, workspace, rt.get("clip_id", ""),
                                      ",".join(missing_fields),
                                      "missing required manifest field(s)", "excluded")
                    continue

                bad_type = [f for f in STRING_TRACK_FIELDS
                            if not isinstance(rt.get(f), str)]
                if bad_type:
                    report.add_schema(maqam, workspace, rt.get("clip_id", ""),
                                      ",".join(bad_type),
                                      "field(s) are not strings", "excluded")
                    continue

                clip_id = nfc(rt["clip_id"])
                assigned = nfc(rt["assigned_filename"])
                status = rt.get("status", "")
                report.statuses[status] += 1

                if not clip_id.strip():
                    report.add_schema(maqam, workspace, "",
                                      "clip_id", "empty clip_id", "excluded")
                    continue

                if classify_status(status) == "excluded":
                    report.status_excluded[status] += 1
                    continue

                audio = find_audio(workspace_dir, assigned) or disk_norm.get(assigned)
                if audio is None:
                    report.missing.append({
                        "maqam": maqam,
                        "workspace": workspace,
                        "clip_id": clip_id,
                        "assigned_filename": assigned,
                        "status": status,
                        "reason": "no local audio file (below rating bar or not downloaded)",
                    })
                    continue

                consumed.add(nfc(audio.name))
                track = Track(
                    clip_id=clip_id,
                    maqam_key=maqam_key,
                    maqam=maqam,
                    workspace=workspace,
                    original_title=nfc(rt["original_title"]),
                    source_audio=audio,
                    styles_raw=rt["styles"],
                    lyrics_raw=rt["lyrics"],
                    exclude_styles=rt.get("exclude_styles", ""),
                    created_at=rt.get("created_at", ""),
                    status=status,
                    downloaded_at=rt.get("downloaded_at", ""),
                    audio_ext=audio.suffix.lower(),
                )
                report.status_retained[status] += 1
                tracks.append(track)

            for name, path in disk_norm.items():
                if path.suffix.lower() not in AUDIO_EXTS:
                    continue
                if name not in consumed:
                    report.orphans.append({
                        "maqam": maqam,
                        "workspace": workspace,
                        "filename": name,
                        "reason": "audio file not listed as assigned_filename in workspace_manifest.json",
                    })

    report.notes.append(f"manifest tracks scanned: {total_manifest_tracks}")
    return tracks


# --------------------------------------------------------------------------
# Derivation: captions, lyrics, duplicate grouping
# --------------------------------------------------------------------------

def derive_fields(tracks: list[Track], report: Report) -> None:
    for track in tracks:
        track.styles_clean, track.caption_source = clean_caption(track.styles_raw)
        if track.caption_source != "parsed":
            report.notes.append(
                f"[warn] {track.clip_id}: caption not parsed into fields "
                f"(source={track.caption_source}); used control-token-free text"
            )
        if not track.styles_clean:
            track.styles_clean = re.sub(
                r"\s+", " ", strip_control_tokens(track.styles_raw)
            ).strip()
            track.caption_source = "raw-fallback"
        track.lyrics_clean = clean_lyrics(track.lyrics_raw)
        track.source_sha256 = sha256_file(track.source_audio)
        track.source_bytes = track.source_audio.stat().st_size


def assign_duplicate_groups(tracks: list[Track]) -> None:
    groups: dict[str, list[Track]] = defaultdict(list)
    for track in tracks:
        key = nfc(track.lyrics_raw)
        digest = hashlib.sha1(normalize_lyrics_key(key).encode("utf-8")).hexdigest()[:12]
        track.duplicate_group_id = f"dup-{track.maqam_key}-{digest}"
        groups[track.duplicate_group_id].append(track)
    for group_id, members in groups.items():
        for track in members:
            track.duplicate_group_size = len(members)


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_neutral(track: Track, out_dir: Path, src_root: Path, dry_run: bool) -> dict:
    maqam_dir = out_dir / track.maqam_key
    stem = track.clip_id
    dest_audio = maqam_dir / f"{stem}{track.audio_ext}"
    dest_caption = maqam_dir / f"{stem}.caption.txt"
    dest_lyrics = maqam_dir / f"{stem}.lyrics.txt"
    dest_meta = maqam_dir / f"{stem}.meta.json"
    if not dry_run:
        maqam_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(track.source_audio, dest_audio)
        dest_caption.write_text(track.styles_clean + "\n", encoding="utf-8")
        dest_lyrics.write_text(track.lyrics_clean + "\n", encoding="utf-8")
        dest_meta.write_text(
            json.dumps(track.meta(src_root), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return {
        "clip_id": stem,
        "maqam": track.maqam,
        "workspace": track.workspace,
        "original_title": track.original_title,
        "duplicate_group_id": track.duplicate_group_id,
        "duplicate_group_size": track.duplicate_group_size,
        "status": track.status,
        "dest_audio": str(dest_audio),
    }


def write_comfyui(track: Track, out_dir: Path, used_stems: set[str],
                  dry_run: bool) -> tuple[dict, str]:
    """Flat, single-folder layout accepted by FL_YuE2_DatasetMaker."""
    stem = track.clip_id
    if stem.casefold() in used_stems:
        stem = f"{track.clip_id}_{track.maqam_key}"
    used_stems.add(stem.casefold())

    dest_audio = out_dir / f"{stem}{track.audio_ext}"
    dest_caption = out_dir / f"{stem}.caption.txt"
    dest_lyrics = out_dir / f"{stem}.lyrics.txt"
    dest_song = out_dir / f"{stem}.song.txt"       # split identity (lyric group)
    dest_meta = out_dir / f"{stem}.caption.json"   # optional, audio_sha256 checked

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(track.source_audio, dest_audio)
        dest_caption.write_text(track.styles_clean + "\n", encoding="utf-8")
        dest_lyrics.write_text(track.lyrics_clean + "\n", encoding="utf-8")
        dest_song.write_text(track.duplicate_group_id + "\n", encoding="utf-8")
        dest_meta.write_text(
            json.dumps({
                "audio_sha256": track.source_sha256,
                "clip_id": track.clip_id,
                "maqam": track.maqam,
                "workspace": track.workspace,
                "original_title": track.original_title,
                "duplicate_group_id": track.duplicate_group_id,
                "duplicate_group_size": track.duplicate_group_size,
                "status": track.status,
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return {"stem": stem, "audio": str(dest_audio)}, stem


def status_report_rows(report: Report) -> list[dict]:
    rows = []
    for status, total in sorted(report.statuses.items(), key=lambda kv: -kv[1]):
        classification = classify_status(status)
        rows.append({
            "status": status,
            "classification": classification,
            "total_count": total,
            "retained_count": report.status_retained.get(status, 0),
            "excluded_count": report.status_excluded.get(status, 0),
            "note": {
                "included": "audio matches the upstream 4+ star filter; included",
                "excluded": "failure-like status; excluded even if audio exists",
                "unknown": "unrecognized status; retained but flagged for review",
            }[classification],
        })
    return rows


def duplicate_report_rows(tracks: list[Track]) -> list[dict]:
    groups: dict[str, list[Track]] = defaultdict(list)
    for track in tracks:
        groups[track.duplicate_group_id].append(track)
    rows: list[dict] = []
    for group_id, members in sorted(groups.items(),
                                    key=lambda kv: (-len(kv[1]), kv[0])):
        rows.append({
            "duplicate_group_id": group_id,
            "maqam": members[0].maqam,
            "duplicate_group_size": len(members),
            "is_duplicate": len(members) > 1,
            "clip_ids": "|".join(t.clip_id for t in members),
            "original_titles": "|".join(t.original_title for t in members),
            "workspaces": "|".join(sorted({t.workspace for t in members})),
            "normalized_lyrics_sha1": hashlib.sha1(
                normalize_lyrics_key(members[0].lyrics_raw).encode("utf-8")
            ).hexdigest(),
        })
    return rows


def pilot_candidates(tracks: list[Track], per_maqam: int = 4) -> list[dict]:
    """~10-15 tracks spanning all four maqams for the §6 transcription pilot.
    Deterministic: shortest tracks first (quickest to transcribe and review)."""
    chosen: list[dict] = []
    for maqam_key in MAQAM_DIRS:
        pool = [t for t in tracks if t.maqam_key == maqam_key]
        pool.sort(key=lambda t: (t.source_bytes, t.clip_id))
        # Prefer one example from a duplicate group > 1, to exercise repeats.
        picks = [t for t in pool if t.duplicate_group_size > 1][:per_maqam]
        if len(picks) < per_maqam:
            picks += [t for t in pool if t not in picks][: per_maqam - len(picks)]
        for track in picks:
            chosen.append({
                "maqam": track.maqam,
                "clip_id": track.clip_id,
                "workspace": track.workspace,
                "original_title": track.original_title,
                "audio": f"dataset/{track.maqam_key}/{track.clip_id}{track.audio_ext}",
                "duplicate_group_size": track.duplicate_group_size,
                "note": "pilot: transcribe with SheetSage2 and inspect Hijaz/Kurd intervals",
            })
    return chosen


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, default=Path("min_4stars_ai_music"))
    parser.add_argument("--out", type=Path, default=Path("dataset"))
    parser.add_argument("--comfyui-out", type=Path, default=Path("dataset_comfyui"))
    parser.add_argument("--no-comfyui", action="store_true",
                        help="Skip the adapted flat ComfyUI-FL-YuE2 layout.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove previous --out / --comfyui-out before writing.")
    args = parser.parse_args()

    src_root = args.src.resolve()
    out_dir = args.out.resolve()
    comfy_dir = args.comfyui_out.resolve()
    reports_dir = out_dir / "_reports"

    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log: list[str] = [
        "Maqam Suno corpus -> YuE2 LoRA dataset build log",
        f"started (UTC): {started}",
        f"source root: {src_root}",
        f"neutral output: {out_dir}",
        f"comfyui adapted output: {'(disabled)' if args.no_comfyui else comfy_dir}",
        f"dry run: {bool(args.dry_run)}",
        "",
    ]

    if args.overwrite and not args.dry_run:
        for target in (out_dir, comfy_dir):
            if target.exists():
                shutil.rmtree(target)
                log.append(f"removed previous output: {target}")

    report = Report()
    tracks = walk_corpus(src_root, report)
    log.append(f"retained tracks (audio present): {len(tracks)}")
    log.append(f"missing audio (manifest entries with no local file): {len(report.missing)}")
    log.append(f"orphan audio (files not in any manifest): {len(report.orphans)}")
    log.append(f"schema deviations: {len(report.schema)}")
    log.append("")

    derive_fields(tracks, report)
    assign_duplicate_groups(tracks)

    # ---- neutral layout --------------------------------------------------
    manifest_rows: list[dict] = []
    for track in sorted(tracks, key=lambda t: (t.maqam_key, t.clip_id)):
        manifest_rows.append(write_neutral(track, out_dir, src_root, args.dry_run))
    manifest_rows.sort(key=lambda r: (r["maqam"], r["clip_id"]))

    # ---- ComfyUI-FL-YuE2 adapted layout ---------------------------------
    comfy_count = 0
    if not args.no_comfyui:
        used_stems: set[str] = set()
        for track in sorted(tracks, key=lambda t: (t.maqam_key, t.clip_id)):
            _, stem = write_comfyui(track, comfy_dir, used_stems, args.dry_run)
            if stem != track.clip_id:
                report.notes.append(
                    f"[warn] flat-layout stem collision: {track.clip_id} -> {stem}"
                )
            comfy_count += 1

    # ---- reports ---------------------------------------------------------
    if not args.dry_run:
        reports_dir.mkdir(parents=True, exist_ok=True)

        write_csv(reports_dir / "orphan_audio.csv", report.orphans,
                  ["maqam", "workspace", "filename", "reason"])
        write_csv(reports_dir / "missing_audio.csv", report.missing,
                  ["maqam", "workspace", "clip_id", "assigned_filename", "status", "reason"])
        write_csv(reports_dir / "schema_deviations.csv", report.schema,
                  ["maqam", "workspace", "clip_id", "field", "issue", "action"])
        write_csv(reports_dir / "status_value_counts.csv", status_report_rows(report),
                  ["status", "classification", "total_count", "retained_count",
                   "excluded_count", "note"])
        dup_rows = duplicate_report_rows(tracks)
        write_csv(reports_dir / "duplicate_groups.csv", dup_rows,
                  ["duplicate_group_id", "maqam", "duplicate_group_size", "is_duplicate",
                   "clip_ids", "original_titles", "workspaces", "normalized_lyrics_sha1"])
        write_csv(reports_dir / "dataset_manifest.csv", manifest_rows,
                  ["clip_id", "maqam", "workspace", "original_title",
                   "duplicate_group_id", "duplicate_group_size", "status", "dest_audio"])
        write_csv(reports_dir / "transcription_pilot_candidates.csv",
                  pilot_candidates(tracks),
                  ["maqam", "clip_id", "workspace", "original_title", "audio",
                   "duplicate_group_size", "note"])

    # ---- reconciliation --------------------------------------------------
    total_manifest = sum(report.statuses.values())
    excluded_by_status = sum(report.status_excluded.values())
    accounted = (len(tracks) + len(report.missing)
                 + len(report.schema) + excluded_by_status)
    reconciled = accounted == total_manifest

    duplicate_tracks = sum(1 for t in tracks if t.duplicate_group_size > 1)
    dup_groups = len({t.duplicate_group_id for t in tracks if t.duplicate_group_size > 1})

    log += [
        "decisions / policies",
        "  - dedup key: (maqam, NFC(normalize_lyrics_raw)); tag-only, nothing deleted (§4.1)",
        "  - status policy: downloaded + skipped_existing retained; failure-like statuses excluded",
        "  - caption: quoted genre/vocals/production/instrumentation/mood flattened to one paragraph;",
        "             styles_raw preserved verbatim in .meta.json / .caption.json",
        "  - lyrics: divider removed, headers reduced to plain section tags, direction cues dropped;",
        "             lyrics_raw preserved verbatim (§4.6)",
        "  - exclude_styles: metadata only, never used for conditioning (YuE2 has no negative prompt)",
        "  - ComfyUI-FL-YuE2 format verified from cloned repo docs/TRAINING.md and yue2/training/data.py:",
        "      <stem>.(mp3|wav|flac) + <stem>.caption.txt + <stem>.lyrics.txt [required]",
        "      + optional <stem>.abc.txt / <stem>.caption.json / <stem>.song.txt",
        "      loader is single-folder and non-recursive -> flat dataset_comfyui/ provided",
        "      .song.txt = duplicate_group_id so same-lyric takes stay in one split",
        "  - transcription (§6): not built here; pilot candidate list emitted for SheetSage2",
        "",
        "counts",
        f"  total manifest tracks: {total_manifest}",
        f"  retained (copied): {len(tracks)}",
        f"    ajam: {sum(1 for t in tracks if t.maqam_key == 'ajam')}",
        f"    hijaz: {sum(1 for t in tracks if t.maqam_key == 'hijaz')}",
        f"    kurd: {sum(1 for t in tracks if t.maqam_key == 'kurd')}",
        f"    nahawand: {sum(1 for t in tracks if t.maqam_key == 'nahawand')}",
        f"  missing audio: {len(report.missing)}",
        f"  schema-deviant: {len(report.schema)}",
        f"  excluded by status: {excluded_by_status}",
        f"  orphans (not counted against manifest total): {len(report.orphans)}",
        f"  duplicate groups (size>1): {dup_groups} covering {duplicate_tracks} tracks",
        f"  comfyui adapted tracks: {comfy_count if not args.no_comfyui else 0}",
        "",
        f"reconciliation: retained + missing + schema-deviant + excluded-by-status "
        f"= {accounted} == {total_manifest} -> {'OK' if reconciled else 'MISMATCH'}",
        "",
        "notes",
    ]
    log += [f"  {note}" for note in report.notes if note]

    if not args.dry_run:
        (reports_dir / "build_log.txt").write_text("\n".join(log) + "\n",
                                                   encoding="utf-8")

    print("\n".join(log))

    if not reconciled:
        print("\nERROR: reconciliation failed", file=sys.stderr)
        return 1
    if args.dry_run:
        print("\n[dry-run] no files written")
    else:
        print(f"\nneutral dataset: {out_dir}")
        if not args.no_comfyui:
            print(f"comfyui dataset: {comfy_dir}")
        print(f"reports:         {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
