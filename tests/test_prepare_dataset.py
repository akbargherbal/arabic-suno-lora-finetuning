"""Tests for prepare_dataset.py's pure functions and a real build."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import prepare_dataset as pd

# ---------------------------------------------------------------------------
# clean_lyrics
# ---------------------------------------------------------------------------

RAW = (
    "///***///\n"
    "[Verse 1 | epic soaring vocals | heavy power chords]\n"
    "آذَ نَتْنا بِبَينِها\n"
    "[guitars surge — Ajam]\n"
    "[Chorus]\n"
    "ثُمَّ نَأَتْ"
)


def test_clean_lyrics_simplify():
    out = pd.clean_lyrics(RAW, "simplify")
    assert "///***///" not in out
    assert "[Verse 1]" in out
    assert "[Verse 1 |" not in out
    assert "guitars surge" not in out
    assert "[Chorus]" in out
    assert "آذَ نَتْنا بِبَينِها" in out
    assert "ثُمَّ نَأَتْ" in out


def test_clean_lyrics_full_keeps_tags_but_drops_marker():
    out = pd.clean_lyrics(RAW, "full")
    assert "///***///" not in out
    assert "[Verse 1 | epic soaring vocals | heavy power chords]" in out
    assert "[guitars surge — Ajam]" in out


def test_clean_lyrics_strip_removes_all_tags():
    out = pd.clean_lyrics(RAW, "strip")
    assert "[" not in out and "]" not in out
    assert "آذَ نَتْنا بِبَينِها" in out


def test_clean_lyrics_collapses_blank_runs():
    assert pd.clean_lyrics("a\n\n\n\nb", "simplify") == "a\n\nb"


def test_clean_lyrics_empty():
    assert pd.clean_lyrics("", "simplify") == ""


# ---------------------------------------------------------------------------
# ascii_safe_slug
# ---------------------------------------------------------------------------


def test_ascii_safe_slug_strips_arabic():
    # The collision that motivated poem_id: two unrelated Arabic titles collapse
    # onto the same slug because all Arabic characters are stripped.
    assert pd.ascii_safe_slug("01-الوداع-الطويل") == "01"
    assert pd.ascii_safe_slug("01-عزة-الفارس") == "01"


def test_ascii_safe_slug_fallback_and_max_len():
    assert pd.ascii_safe_slug("الوداع") == "track"
    assert pd.ascii_safe_slug("a" * 100) == "a" * 40


# ---------------------------------------------------------------------------
# caption / header parsing
# ---------------------------------------------------------------------------


def test_extract_caption_maqam():
    assert pd.extract_caption_maqam("... Maqam Hijaz ...") == "Hijaz"
    assert pd.extract_caption_maqam("Maqam Nahawand") == "Nahawand"
    assert pd.extract_caption_maqam("Maqam Rast") is None  # not one of the 4
    assert pd.extract_caption_maqam("no maqam here") is None


def test_extract_start_phrase_and_mood():
    assert pd.extract_start_phrase('[START_ON: "آذنتنا"]') == "آذنتنا"
    assert pd.extract_start_phrase("nothing") is None
    assert pd.extract_mood('mood: "wistful"') == "wistful"
    assert pd.extract_mood("nothing") is None


# ---------------------------------------------------------------------------
# split_train_val
# ---------------------------------------------------------------------------


def _tracks(spec):
    """spec: list of (group_key, maqam, clip_id)."""
    return [
        pd.Track(
            clip_id=c,
            workspace="ws",
            maqam=m,
            caption_maqam=m,
            original_title=g,
            audio_path=Path(f"{c}.mp3"),
            styles="",
            lyrics="",
            start_phrase=None,
            mood=None,
            group_key=g,
        )
        for g, m, c in spec
    ]


SPEC = [
    ("hijaz::a", "Hijaz", "c1"),
    ("hijaz::a", "Hijaz", "c2"),
    ("hijaz::b", "Hijaz", "c3"),
    ("hijaz::c", "Hijaz", "c4"),
    ("kurd::d", "Kurd", "c5"),
    ("kurd::e", "Kurd", "c6"),
    ("kurd::f", "Kurd", "c7"),
]


def test_split_train_val_no_poem_leaks():
    train, val = pd.split_train_val(_tracks(SPEC), 0.34, seed=13)
    train_groups = {t.group_key for t in train}
    val_groups = {t.group_key for t in val}
    assert train_groups & val_groups == set()
    assert val_groups  # non-empty
    assert "hijaz" in {t.maqam.lower() for t in val}
    assert "kurd" in {t.maqam.lower() for t in val}


def test_split_train_val_deterministic():
    a = pd.split_train_val(_tracks(SPEC), 0.34, seed=13)
    b = pd.split_train_val(_tracks(SPEC), 0.34, seed=13)
    assert [t.clip_id for t in a[0]] == [t.clip_id for t in b[0]]
    assert [t.clip_id for t in a[1]] == [t.clip_id for t in b[1]]


def test_cap_per_song():
    tracks = _tracks(SPEC)
    assert len(pd.cap_per_song(tracks, None)) == len(tracks)
    capped = pd.cap_per_song(tracks, 1)
    assert len(capped) == 6  # the hijaz::a pair collapses to one
    assert len(pd.cap_per_song(tracks, 2)) == 7


# ---------------------------------------------------------------------------
# integration: build from a fixture corpus
# ---------------------------------------------------------------------------


def _run_prepare(corpus_root: Path, out_dir: Path, monkeypatch, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dataset.py",
            "--dataset-root",
            str(corpus_root),
            "--out-dir",
            str(out_dir),
            *extra,
        ],
    )
    pd.main()
    with (out_dir / "manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_build_clean_corpus(corpus, tmp_path, monkeypatch):
    out = tmp_path / "dataset"
    rows = _run_prepare(corpus("manifest_clean.json"), out, monkeypatch)

    # 3 on-disk tracks (the below-rating entry has no file) across 2 poems.
    assert len(rows) == 3
    assert len({r["dest_audio"] for r in rows}) == 3
    assert len({r["clip_id"] for r in rows}) == 3

    # Sidecars exist and the lyrics were cleaned.
    for r in rows:
        assert Path(r["dest_audio"]).exists()
        assert Path(r["dest_style"]).exists()
        lyrics = Path(r["dest_lyrics"]).read_text(encoding="utf-8")
        assert "///***///" not in lyrics
        assert "|" not in lyrics  # simplify collapsed the pipe tag
        style = Path(r["dest_style"]).read_text(encoding="utf-8")
        assert "Maqam Hijaz" in style


def test_build_colliding_slugs_get_unique_destinations(corpus, tmp_path, monkeypatch):
    out = tmp_path / "dataset"
    rows = _run_prepare(corpus("manifest_naming_collision.json"), out, monkeypatch)

    assert len(rows) == 3
    # Every destination is distinct even though all three slugs are "01".
    assert len({r["dest_audio"] for r in rows}) == 3
    train = [r for r in rows if r["split"] == "train"]
    assert len(train) == 2
    assert len({Path(r["dest_audio"]).name for r in train}) == 2
    for r in rows:
        assert Path(r["dest_audio"]).exists()


def test_clean_flag_wipes_previous_build(corpus, tmp_path, monkeypatch):
    out = tmp_path / "dataset"
    _run_prepare(corpus("manifest_clean.json"), out, monkeypatch)
    stale = out / "train" / "stale-does-not-belong.mp3"
    stale.write_bytes(b"ID3stale")
    _run_prepare(corpus("manifest_clean.json"), out, monkeypatch, "--clean")
    assert not stale.exists()
