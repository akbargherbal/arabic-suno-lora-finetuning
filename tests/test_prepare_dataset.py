"""Tests for prepare_dataset.py: pure helpers, corpus walk, and a real build."""

from __future__ import annotations

import sys
from pathlib import Path

from conftest import DEFAULT_LYRICS, build_corpus, four_maqam_corpus, track, write_wav

import prepare_dataset as pd

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def test_nfc_normalizes_nfd():
    assert pd.nfc("e\u0301") == "é"


def test_strip_control_tokens_removes_suno_header():
    styles = (
        '[Is_MAX_MODE: MAX](MAX)\n'
        '[QUALITY: MAX](MAX)\n'
        '[REALISM: MAX](MAX)\n'
        '[START_ON: TRUE]\n'
        'genre: "A"'
    )
    out = pd.strip_control_tokens(styles)
    assert "MAX" not in out and "START_ON" not in out
    assert 'genre: "A"' in out


def test_parse_caption_fields_structured_order():
    styles = (
        'genre: "A"\nvocals: "B"\nproduction: "C"\n'
        'instrumentation: "D"\nmood: "E"'
    )
    assert pd.parse_caption_fields(styles) == ["A", "B", "C", "D", "E"]


def test_parse_caption_fields_unquoted_values():
    assert pd.parse_caption_fields("genre: Rock\nvocals: Tenor") == ["Rock", "Tenor"]


def test_parse_caption_fields_falls_back_to_plain_text():
    assert pd.parse_caption_fields("just a plain description") == ["just a plain description"]


def test_clean_caption_parsed_flattens_to_paragraph():
    caption, source = pd.clean_caption('genre: "A"\nvocals: "B"')
    assert caption == "A. B."
    assert source == "parsed"


def test_clean_caption_empty():
    assert pd.clean_caption("") == ("", "empty")


def test_normalize_lyrics_key_drops_divider_tags_and_whitespace():
    raw = "///***///\n[Verse 1 | loud]\nكلمة   كلمة\n"
    assert pd.normalize_lyrics_key(raw) == "كلمة كلمة"


def test_canonical_section_label():
    assert pd.canonical_section_label("Verse 1 | epic soaring vocals") == "Verse"
    assert pd.canonical_section_label("Instrumental Build-up: guitars") == "Instrumental"
    assert pd.canonical_section_label("orchestral strings swell") is None


def test_clean_lyrics_simplifies_headers_and_drops_cues():
    out = pd.clean_lyrics(DEFAULT_LYRICS)
    assert "///***///" not in out
    assert "[Verse]" in out and "[Chorus]" in out
    assert "|" not in out and "power chords" not in out
    assert "آذنتنا ببينها" in out


def test_clean_lyrics_empty():
    assert pd.clean_lyrics("") == ""


def test_classify_status():
    assert pd.classify_status("downloaded") == "included"
    assert pd.classify_status("skipped_existing") == "included"
    assert pd.classify_status("failed") == "excluded"
    assert pd.classify_status("something_else") == "unknown"


def test_find_audio_is_nfc_tolerant(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "e\u0301.wav").write_bytes(b"x")  # NFD on disk
    assert pd.find_audio(workspace, "é.wav") is not None  # NFC lookup


# ---------------------------------------------------------------------------
# Corpus walk
# ---------------------------------------------------------------------------

def test_walk_corpus_retains_missing_and_reports_orphans(tmp_path):
    root = tmp_path / "corpus"
    build_corpus(root, [
        track(clip_id="keep", filename="keep.wav"),
        track(clip_id="gone", filename="gone.wav", present=False),
    ])
    write_wav(root / "hijaz" / "ws_a" / "orphan.wav")

    report = pd.Report()
    kept = pd.walk_corpus(root, report)

    assert [t.clip_id for t in kept] == ["keep"]
    assert [m["clip_id"] for m in report.missing] == ["gone"]
    assert [o["filename"] for o in report.orphans] == ["orphan.wav"]


def _track(clip_id: str, maqam_key: str, lyrics: str) -> pd.Track:
    return pd.Track(
        clip_id=clip_id,
        maqam_key=maqam_key,
        maqam=maqam_key.title(),
        workspace="ws",
        original_title="T",
        source_audio=Path(f"{clip_id}.wav"),
        styles_raw="",
        lyrics_raw=lyrics,
        exclude_styles="",
        created_at="",
        status="downloaded",
    )


def test_duplicate_groups_are_per_maqam_and_per_lyrics():
    a = _track("a", "hijaz", "one two")
    b = _track("b", "hijaz", "one two")
    other_maqam = _track("c", "kurd", "one two")
    pd.assign_duplicate_groups([a, b, other_maqam])

    assert a.duplicate_group_id == b.duplicate_group_id
    assert a.duplicate_group_size == 2
    assert other_maqam.duplicate_group_id != a.duplicate_group_id
    assert other_maqam.duplicate_group_size == 1


def test_write_comfyui_disambiguates_colliding_stems(tmp_path):
    used: set[str] = set()
    first = _track("dup", "hijaz", "x")
    second = _track("dup", "kurd", "y")
    _, stem_a = pd.write_comfyui(first, tmp_path, used, dry_run=True)
    _, stem_b = pd.write_comfyui(second, tmp_path, used, dry_run=True)

    assert stem_a == "dup"
    assert stem_b == "dup_kurd"


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------

def test_main_builds_neutral_and_comfyui_layouts(tmp_path, monkeypatch):
    src = tmp_path / "src"
    four_maqam_corpus(src)
    out = tmp_path / "dataset"
    comfy = tmp_path / "dataset_comfyui"
    monkeypatch.setattr(sys, "argv", [
        "prepare_dataset.py", "--src", str(src),
        "--out", str(out), "--comfyui-out", str(comfy),
    ])

    assert pd.main() == 0

    for maqam in ("ajam", "hijaz", "kurd", "nahawand"):
        stem = f"{maqam}-0001"
        assert (out / maqam / f"{stem}.wav").is_file()
        assert (out / maqam / f"{stem}.caption.txt").read_text(encoding="utf-8").strip()
        assert (out / maqam / f"{stem}.lyrics.txt").read_text(encoding="utf-8").strip()
        assert (out / maqam / f"{stem}.meta.json").is_file()

    assert (comfy / "hijaz-0001.wav").is_file()
    assert (comfy / "hijaz-0001.caption.json").is_file()
    assert (comfy / "hijaz-0001.song.txt").read_text(encoding="utf-8").strip()

    for name in (
        "orphan_audio.csv", "missing_audio.csv", "schema_deviations.csv",
        "status_value_counts.csv", "duplicate_groups.csv",
        "dataset_manifest.csv", "build_log.txt",
    ):
        assert (out / "_reports" / name).is_file()
