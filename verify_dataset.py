#!/usr/bin/env python3
"""
verify_dataset.py — independent validation of the prepared dataset against
  (a) dataset_prep_spec.md §7 acceptance criteria, and
  (b) the real dataset contract of filliptm/ComfyUI-FL-YuE2.

Point (b) is checked two ways:
  1. a faithful re-implementation of the rules in
     yue2/training/data.py (audio ext, unique stems, required sidecars,
     audio_sha256 metadata, mono/stereo + nonzero frames), and
  2. if a ComfyUI-FL-YuE2 clone is available (--comfyui-repo), the *actual*
     `data.audio_files()` and `data.dataset()` functions are executed against
     a hard-linked temp copy of dataset_comfyui/, so the pack's own loader
     confirms it can consume the output.

Exit code: 0 if no FAIL, 1 otherwise. WARNs never fail the run.

Usage:
    python verify_dataset.py
    python verify_dataset.py --comfyui-repo /tmp/opencode/ComfyUI-FL-YuE2
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

MAQAM_DIRS = {"ajam", "hijaz", "kurd", "nahawand"}
AUDIO_EXTS = {".mp3", ".wav", ".flac"}
REQUIRED_TRACK_FIELDS = (
    "clip_id", "original_title", "assigned_filename", "styles",
    "exclude_styles", "lyrics", "created_at", "status",
)
META_FIELDS = (
    "clip_id", "maqam", "workspace", "original_title",
    "styles_raw", "styles_clean", "lyrics_raw", "lyrics_clean",
    "exclude_styles", "created_at", "status",
    "duplicate_group_id", "duplicate_group_size", "audio_extension",
)
DIVIDER_LINE = __import__("re").compile(r"^[\s/*]*$")
ALL_TAG_RE = __import__("re").compile(r"\[[^\]]*\]")


RESULTS: list[tuple[str, str, str]] = []  # (level, check, detail)


def record(level: str, check: str, detail: str) -> None:
    RESULTS.append((level, check, detail))
    print(f"[{level:4}] {check}: {detail}")


def ok(check: str, detail: str = "") -> None:
    record("PASS", check, detail)


def warn(check: str, detail: str) -> None:
    record("WARN", check, detail)


def fail(check: str, detail: str) -> None:
    record("FAIL", check, detail)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def is_nfc(text: str) -> bool:
    return unicodedata.is_normalized("NFC", text)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_lyrics_key(raw: str) -> str:
    import re as _re
    text = nfc(raw).replace("\r\n", "\n").replace("\r", "\n")
    text = ALL_TAG_RE.sub(" ", text)
    lines = [ln for ln in text.split("\n")
             if not (ln.strip() and DIVIDER_LINE.match(ln))]
    return _re.sub(r"\s+", " ", "\n".join(lines)).strip()


# ---------------------------------------------------------------------------
# Source re-scan (independent of the build script)
# ---------------------------------------------------------------------------

def rescan_source(src_root: Path) -> dict:
    stats = {
        "total": 0, "retained": 0, "missing": 0, "schema": 0,
        "excluded_status": 0, "orphans": 0,
        "statuses": Counter(), "retained_by_maqam": Counter(),
        "present_pairs": [], "orphan_names": [],
    }
    failure = __import__("re").compile(r"fail|error|cancel|reject|remov|delet|abort|missing", __import__("re").I)
    for maqam_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        key = maqam_dir.name.lower()
        if key not in MAQAM_DIRS:
            continue
        for manifest in sorted(maqam_dir.rglob("workspace_manifest.json")):
            wd = manifest.parent
            data = json.loads(manifest.read_text(encoding="utf-8"))
            disk = {nfc(p.name): p for p in wd.iterdir() if p.is_file()}
            consumed = set()
            for rt in data.get("tracks", []):
                stats["total"] += 1
                if not isinstance(rt, dict) or any(f not in rt for f in REQUIRED_TRACK_FIELDS):
                    stats["schema"] += 1
                    continue
                clip_id = nfc(rt["clip_id"])
                status = rt.get("status", "")
                stats["statuses"][status] += 1
                if status.strip().lower() not in {"downloaded", "skipped_existing"} \
                        and failure.search(status.strip().lower()):
                    stats["excluded_status"] += 1
                    continue
                assigned = nfc(rt["assigned_filename"])
                audio = disk.get(assigned)
                if audio is None:
                    stats["missing"] += 1
                    continue
                consumed.add(nfc(audio.name))
                stats["retained"] += 1
                stats["retained_by_maqam"][key] += 1
                stats["present_pairs"].append(
                    (key, clip_id, wd.name, rt["original_title"], rt["lyrics"], str(audio))
                )
            for name, path in disk.items():
                if path.suffix.lower() in AUDIO_EXTS and name not in consumed:
                    stats["orphans"] += 1
                    stats["orphan_names"].append(name)
    return stats


# ---------------------------------------------------------------------------
# Checks against the spec
# ---------------------------------------------------------------------------

def check_reconciliation(src: dict, metas: dict, out: Path, report_dir: Path) -> None:
    total = src["total"]
    accounted = (src["retained"] + src["missing"] + src["schema"]
                 + src["excluded_status"])
    if accounted == total:
        ok("reconcile.source-counts",
           f"retained({src['retained']}) + missing({src['missing']}) + "
           f"schema({src['schema']}) + excluded-by-status({src['excluded_status']}) "
           f"= {total} manifest tracks")
    else:
        fail("reconcile.source-counts",
             f"accounted {accounted} != manifest total {total}")

    files_ok = True
    for maqam in MAQAM_DIRS:
        d = out / maqam
        if not d.is_dir():
            fail("reconcile.layout", f"missing maqam folder {d}")
            files_ok = False
            continue
        audio: list[Path] = []
        for ext in (".mp3", ".wav", ".flac"):
            audio = list(d.glob(f"*{ext}"))
            if audio:
                break
        n_audio = len(audio)
        n_cap = len(list(d.glob("*.caption.txt")))
        n_lyr = len(list(d.glob("*.lyrics.txt")))
        n_meta = len(list(d.glob("*.meta.json")))
        expected = src["retained_by_maqam"][maqam]
        if {n_audio, n_cap, n_lyr, n_meta} != {expected}:
            fail("reconcile.maqam-files",
                 f"{maqam}: audio={n_audio} caption={n_cap} lyrics={n_lyr} "
                 f"meta={n_meta}, expected {expected} each")
            files_ok = False
    if files_ok:
        ok("reconcile.maqam-files",
           f"every maqam has audio==caption==lyrics==meta==retained count")

    if len(metas) == src["retained"]:
        ok("reconcile.meta-count", f"{len(metas)} meta.json == {src['retained']} retained")
    else:
        fail("reconcile.meta-count",
             f"{len(metas)} meta.json != {src['retained']} retained")

    # status_value_counts.csv must sum to the manifest total.
    try:
        with (report_dir / "status_value_counts.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        s = sum(int(r["total_count"]) for r in rows)
        if s == total:
            ok("reconcile.status-report", f"status_value_counts sums to {total}")
        else:
            fail("reconcile.status-report", f"status counts sum {s} != {total}")
        for r in rows:
            if int(r["total_count"]) != src["statuses"].get(r["status"], 0):
                fail("reconcile.status-report",
                     f"{r['status']}: report={r['total_count']} "
                     f"actual={src['statuses'].get(r['status'], 0)}")
    except FileNotFoundError:
        fail("reconcile.status-report", "status_value_counts.csv missing")


def check_completeness(metas: dict, out: Path) -> None:
    missing_audio = missing_cap = missing_lyr = missing_meta = 0
    empty_cap = empty_lyr = 0
    bad_maqam = bad_clip = 0
    non_nfc = 0
    for stem, (path, meta) in metas.items():
        maqam = path.parent.name
        d = path.parent
        matches = list(d.glob(f"{stem}.*"))
        audio = [p for p in matches if p.suffix.lower() in AUDIO_EXTS]
        if not audio:
            missing_audio += 1
            continue
        cap = d / f"{stem}.caption.txt"
        lyr = d / f"{stem}.lyrics.txt"
        mt = d / f"{stem}.meta.json"
        if not cap.exists():
            missing_cap += 1
        elif not cap.read_text(encoding="utf-8").strip():
            empty_cap += 1
        if not lyr.exists():
            missing_lyr += 1
        elif not lyr.read_text(encoding="utf-8").strip():
            empty_lyr += 1
        if not mt.exists():
            missing_meta += 1
        if meta.get("maqam", "").lower() != maqam:
            bad_maqam += 1
        if meta.get("clip_id") != stem:
            bad_clip += 1
        for key in ("clip_id", "maqam", "workspace", "original_title",
                    "styles_clean", "lyrics_clean", "duplicate_group_id"):
            value = meta.get(key, "")
            if isinstance(value, str) and not is_nfc(value):
                non_nfc += 1
                break
        for key in META_FIELDS:
            if key not in meta:
                fail("dataset.meta-fields", f"{stem}: missing meta field {key}")
                break

    checks = [
        (missing_audio, "audio file present for every meta.json"),
        (missing_cap, "caption.txt present for every track"),
        (missing_lyr, "lyrics.txt present for every track"),
        (missing_meta, "meta.json present for every track"),
        (bad_maqam, "meta maqam matches parent folder"),
        (bad_clip, "meta clip_id matches filename stem"),
        (non_nfc, "meta string fields are NFC"),
    ]
    for count, label in checks:
        if count:
            fail("dataset.completeness", f"{count} x {label}")
        else:
            ok("dataset.completeness", label)
    if empty_cap:
        fail("dataset.caption-nonempty", f"{empty_cap} empty caption files")
    else:
        ok("dataset.caption-nonempty", "all captions non-empty")
    if empty_lyr:
        warn("dataset.lyrics-nonempty", f"{empty_lyr} empty lyrics files")
    else:
        ok("dataset.lyrics-nonempty", "all lyrics non-empty")


def check_collisions(metas: dict, comfy_dir: Path | None) -> None:
    per_maqam: dict[str, Counter] = defaultdict(Counter)
    for stem, (path, _meta) in metas.items():
        per_maqam[path.parent.name][stem] += 1
    collisions = {m: [k for k, v in c.items() if v > 1] for m, c in per_maqam.items()}
    collisions = {m: v for m, v in collisions.items() if v}
    if collisions:
        fail("dataset.collisions", f"duplicate clip_id stems within maqam: {collisions}")
    else:
        ok("dataset.collisions", "one file per clip_id per type, per maqam")

    if comfy_dir and comfy_dir.is_dir():
        audio = [p for p in comfy_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
        stems = [p.stem for p in audio]
        folded = [s.casefold() for s in stems]
        if len(set(folded)) != len(folded):
            dup = [s for s, c in Counter(folded).items() if c > 1]
            fail("comfyui.stems", f"non-unique stems (casefold): {dup}")
        else:
            ok("comfyui.stems", f"{len(stems)} unique flat stems (casefold)")


def check_source_integrity(metas: dict, src_root: Path) -> None:
    missing_src = mismatch = non_nfc_name = 0
    for stem, (_path, meta) in metas.items():
        rel = meta.get("source_audio", "")
        src = src_root / rel
        if not src.is_file():
            missing_src += 1
            continue
        if not is_nfc(src.name):
            non_nfc_name += 1
        if sha256_file(src) != meta.get("source_audio_sha256"):
            mismatch += 1
    if missing_src:
        fail("integrity.source-present", f"{missing_src} source files missing")
    else:
        ok("integrity.source-present", "every referenced source audio still present")
    if mismatch:
        fail("integrity.copy-identical",
             f"{mismatch} copies differ from source bytes (re-encode?)")
    else:
        ok("integrity.copy-identical",
           "every dataset copy is byte-identical to its source (no re-encode)")
    if non_nfc_name:
        warn("integrity.source-nfc",
             f"{non_nfc_name} source filenames are not NFC (lookup used NFC matching)")
    else:
        ok("integrity.source-nfc", "source filenames are NFC")


def check_duplicates(src: dict, metas: dict, report_dir: Path) -> None:
    # Recompute lyric-group membership independently from source.
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, clip_id, _ws, _title, lyrics, _audio in src["present_pairs"]:
        digest = hashlib.sha1(normalize_lyrics_key(lyrics).encode()).hexdigest()[:12]
        groups[(key, digest)].append(clip_id)

    meta_by_clip = {stem: meta for stem, (_p, meta) in metas.items()}
    size_mismatch = group_mismatch = untagged = 0
    actual_sizes: Counter = Counter()
    for stem, meta in meta_by_clip.items():
        gid = meta.get("duplicate_group_id", "")
        size = meta.get("duplicate_group_size", 0)
        if not gid or size < 1:
            untagged += 1
        actual_sizes[gid] += 1
    for (_key, digest), clips in groups.items():
        gid = next((meta_by_clip[c]["duplicate_group_id"]
                    for c in clips if c in meta_by_clip), None)
        if gid is None:
            continue
        report_size = len(clips)
        for c in clips:
            if c in meta_by_clip and meta_by_clip[c]["duplicate_group_size"] != report_size:
                size_mismatch += 1
            if c in meta_by_clip and meta_by_clip[c]["duplicate_group_id"] != gid:
                group_mismatch += 1

    if untagged:
        fail("duplicates.tagged", f"{untagged} tracks lack duplicate_group_id/size")
    else:
        ok("duplicates.tagged", "every track has duplicate_group_id + size")

    # Report/actual agreement.
    try:
        with (report_dir / "duplicate_groups.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        report_sizes = {r["duplicate_group_id"]: int(r["duplicate_group_size"])
                        for r in rows}
        bad = [g for g, n in actual_sizes.items()
               if report_sizes.get(g) != n]
        if bad:
            fail("duplicates.report", f"{len(bad)} group sizes disagree with report")
        else:
            ok("duplicates.report",
               f"{len(rows)} groups reported; sizes match dataset")
        dup_group_count = sum(1 for r in rows if r["is_duplicate"] == "True")
        dup_tracks = sum(int(r["duplicate_group_size"]) for r in rows
                         if r["is_duplicate"] == "True")
        ok("duplicates.retained-not-deleted",
           f"{dup_group_count} duplicate groups covering {dup_tracks} tracks, "
           f"all present in dataset (tag, don't delete)")
    except FileNotFoundError:
        fail("duplicates.report", "duplicate_groups.csv missing")

    if size_mismatch or group_mismatch:
        fail("duplicates.key",
             f"independent re-score: {size_mismatch} size / {group_mismatch} id mismatches")
    else:
        ok("duplicates.key",
           "grouping reproduces from (maqam, normalized lyrics): tag, don't delete")

    # Cross-maqam same-title must NOT be grouped.
    title_maqams: dict[str, set[str]] = defaultdict(set)
    for key, _clip, _ws, title, _lyrics, _audio in src["present_pairs"]:
        title_maqams[title].add(key)
    cross = {t: ms for t, ms in title_maqams.items() if len(ms) > 1}
    if cross:
        for title, ms in list(cross.items())[:5]:
            gids = set()
            for key, clip, _ws, t, _l, _a in src["present_pairs"]:
                if t == title:
                    gids.add(meta_by_clip.get(clip, {}).get("duplicate_group_id"))
            if len(gids) > 1:
                ok("duplicates.cross-maqam",
                   f"same title across {sorted(ms)} kept as distinct groups")
            else:
                warn("duplicates.cross-maqam",
                     f"same title across {sorted(ms)} shares one group (check lyrics)")
    else:
        ok("duplicates.cross-maqam", "no identical titles span multiple maqams among retained")


# ---------------------------------------------------------------------------
# ComfyUI-FL-YuE2 contract
# ---------------------------------------------------------------------------

def check_comfyui_contract(comfy_dir: Path) -> list[dict]:
    songs = []
    if not comfy_dir.is_dir():
        fail("comfyui.dir", f"{comfy_dir} not found")
        return songs
    audio = sorted(p for p in comfy_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    bad_ext = [p.name for p in comfy_dir.iterdir()
               if p.is_file() and p.suffix and p.suffix.lower() not in AUDIO_EXTS
               and p.suffix.lower() not in {".txt", ".json"}]
    if not audio:
        fail("comfyui.audio", "no wav/flac/mp3 files")
        return songs
    ok("comfyui.audio", f"{len(audio)} audio files in a single flat folder")
    if bad_ext:
        warn("comfyui.ext", f"unexpected file types: {bad_ext[:5]}")
    else:
        ok("comfyui.ext", "audio extensions within {mp3,wav,flac}")

    try:
        import soundfile as sf
    except ImportError:
        sf = None
    missing_lyr = missing_cap = missing_json = missing_song = sha_bad = 0
    frames_bad = 0
    durations = []
    for a in audio:
        lyr = a.with_suffix(".lyrics.txt")
        cap = a.with_suffix(".caption.txt")
        meta = a.with_suffix(".caption.json")
        song = a.with_suffix(".song.txt")
        if not lyr.is_file() or not lyr.read_text(encoding="utf-8").strip():
            missing_lyr += 1
        if not cap.is_file() or not cap.read_text(encoding="utf-8").strip():
            missing_cap += 1
        if not meta.is_file():
            missing_json += 1
        else:
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                if m.get("audio_sha256") != sha256_file(a):
                    sha_bad += 1
            except Exception:
                sha_bad += 1
        if not song.is_file() or not song.read_text(encoding="utf-8").strip():
            missing_song += 1
        if sf is not None:
            try:
                info = sf.info(str(a))
                if info.frames <= 0 or info.channels not in (1, 2):
                    frames_bad += 1
                else:
                    durations.append(info.duration)
            except Exception:
                frames_bad += 1
        songs.append({"name": a.stem, "audio": str(a)})

    (ok if missing_lyr == 0 else fail)(
        "comfyui.lyrics", f"missing/empty .lyrics.txt: {missing_lyr}")
    (ok if missing_cap == 0 else fail)(
        "comfyui.caption", f"missing/empty .caption.txt: {missing_cap}")
    (ok if missing_json == 0 and sha_bad == 0 else fail)(
        "comfyui.metadata", f".caption.json missing={missing_json} sha mismatch={sha_bad}")
    (ok if missing_song == 0 else fail)(
        "comfyui.song-identity", f"missing/empty .song.txt: {missing_song}")
    (ok if frames_bad == 0 else fail)(
        "comfyui.readable-audio",
        f"nonempty mono/stereo audio: {frames_bad} bad; "
        f"duration min={min(durations):.1f}s max={max(durations):.1f}s"
        if durations else f"{frames_bad} bad")
    return songs


def real_loader_test(comfy_dir: Path, repo: Path, seed: int = 42) -> None:
    """Execute the pack's actual data.audio_files()/data.dataset()."""
    data_py = repo / "yue2" / "training" / "data.py"
    if not data_py.is_file():
        warn("comfyui.real-loader", f"data.py not found under {repo}; skipped")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stub = tmp_path / "stub"
        (stub / "yue2" / "training").mkdir(parents=True)
        (stub / "yue2" / "__init__.py").write_text("", encoding="utf-8")
        (stub / "yue2" / "training" / "__init__.py").write_text("", encoding="utf-8")
        # Minimal downloads stub providing digest().
        (stub / "yue2" / "downloads.py").write_text(
            "import hashlib\n"
            "from pathlib import Path\n"
            "def digest(path):\n"
            "    h = hashlib.sha256()\n"
            "    with Path(path).open('rb') as fh:\n"
            "        for b in iter(lambda: fh.read(1 << 20), b''):\n"
            "            h.update(b)\n"
            "    return h.hexdigest()\n",
            encoding="utf-8",
        )
        shutil.copyfile(data_py, stub / "yue2" / "training" / "data.py")
        # Minimal torch stub (only serialization.safe_globals is referenced).
        torch = types.ModuleType("torch")
        torch.serialization = types.SimpleNamespace(
            safe_globals=lambda *a, **k: contextlib.nullcontext()
        )
        sys.modules["torch"] = torch

        # Hard-link a copy of the dataset so dataset() can write its manifest.
        work = tmp_path / "comfy"
        work.mkdir()
        for p in comfy_dir.iterdir():
            if p.is_file():
                os.link(p, work / p.name)

        sys.path.insert(0, str(stub))
        try:
            import yue2.training.data as fl_data  # type: ignore
            files = fl_data.audio_files(work)
            manifest_path = fl_data.dataset(
                work, trigger="", default_style="instrumental",
                validation_fraction=0.1, seed=seed,
            )
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - reported, not raised
            fail("comfyui.real-loader", f"pack loader rejected the dataset: {exc}")
            return
        finally:
            sys.path.remove(str(stub))
            sys.modules.pop("torch", None)
            for name in [n for n in sys.modules if n.startswith("yue2")]:
                sys.modules.pop(name, None)

        songs = manifest["songs"]
        ok("comfyui.real-loader",
           f"data.audio_files() found {len(files)}; data.dataset() accepted "
           f"{len(songs)} songs and wrote {Path(manifest_path).name}")

        trained = [s for s in songs if s["split"] == "train"]
        val = [s for s in songs if s["split"] == "validation"]
        sets = {s["song"] for s in songs}
        if len({s["song"] for s in trained}) + len({s["song"] for s in val}) == len(sets):
            ok("comfyui.split-leak", f"no lyric-group leaks: {len(trained)} train / "
                                     f"{len(val)} val songs, {len(sets)} groups")
        else:
            fail("comfyui.split-leak", "a .song.txt identity appears in both splits")

        empty = [s["name"] for s in songs if s["instrumental"]]
        if empty:
            warn("comfyui.instrumental", f"{len(empty)} songs had empty lyrics")
        else:
            ok("comfyui.instrumental", "all songs have lyrics (none instrumental)")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("min_4stars_ai_music"))
    parser.add_argument("--out", type=Path, default=Path("dataset"))
    parser.add_argument("--comfyui-out", type=Path, default=Path("dataset_comfyui"))
    parser.add_argument("--comfyui-repo", type=Path,
                        default=Path("/tmp/opencode/ComfyUI-FL-YuE2"))
    parser.add_argument("--skip-real-loader", action="store_true")
    args = parser.parse_args()

    src_root = args.src.resolve()
    out = args.out.resolve()
    report_dir = out / "_reports"
    comfy_dir = args.comfyui_out.resolve()

    print("=" * 78)
    print("ComfyUI-FL-YuE2 dataset verification")
    print(f"  source:  {src_root}")
    print(f"  dataset: {out}")
    print(f"  comfyui: {comfy_dir}")
    print("=" * 78)

    if not out.is_dir():
        fail("dataset.dir", f"{out} not found")
        return 1

    # Load neutral metas.
    metas: dict[str, tuple[Path, dict]] = {}
    for meta_path in out.glob("*/*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail("dataset.meta-json", f"{meta_path}: {exc}")
            continue
        metas[meta_path.name[: -len(".meta.json")]] = (meta_path, meta)
    ok("dataset.meta-json", f"loaded {len(metas)} meta.json sidecars")

    # Reports exist.
    for name in ("orphan_audio.csv", "missing_audio.csv", "schema_deviations.csv",
                 "status_value_counts.csv", "duplicate_groups.csv", "build_log.txt"):
        p = report_dir / name
        (ok if p.is_file() else fail)("reports.present", f"{name}: {'ok' if p.is_file() else 'MISSING'}")

    src = rescan_source(src_root)
    check_reconciliation(src, metas, out, report_dir)
    check_completeness(metas, out)
    check_collisions(metas, comfy_dir if comfy_dir.is_dir() else None)
    check_source_integrity(metas, src_root)
    check_duplicates(src, metas, report_dir)

    # Orphan/missing report agreement.
    try:
        with (report_dir / "missing_audio.csv").open(encoding="utf-8") as fh:
            missing_rows = list(csv.DictReader(fh))
        (ok if len(missing_rows) == src["missing"] else fail)(
            "reports.missing-count",
            f"missing_audio.csv rows={len(missing_rows)} == recomputed {src['missing']}")
        with (report_dir / "orphan_audio.csv").open(encoding="utf-8") as fh:
            orphan_rows = list(csv.DictReader(fh))
        (ok if len(orphan_rows) == src["orphans"] else fail)(
            "reports.orphan-count",
            f"orphan_audio.csv rows={len(orphan_rows)} == recomputed {src['orphans']}")
    except FileNotFoundError as exc:
        fail("reports.missing-count", str(exc))

    # ComfyUI contract.
    real_loader = not args.skip_real_loader
    check_comfyui_contract(comfy_dir)
    if real_loader:
        real_loader_test(comfy_dir, args.comfyui_repo.resolve())

    # Summary.
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    passes = [r for r in RESULTS if r[0] == "PASS"]
    print("\n" + "=" * 78)
    print(f"SUMMARY: {len(passes)} PASS  {len(warns)} WARN  {len(fails)} FAIL")
    if fails:
        print("\nFAILURES:")
        for _lvl, check, detail in fails:
            print(f"  - {check}: {detail}")
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
