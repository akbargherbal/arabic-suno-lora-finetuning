"""Tests for scripts/export_mothersuperior_format.py.

Covers the pure helpers and the export loop with ffmpeg stubbed out, so the
suite does not depend on ffmpeg being installed.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from collections import Counter
from pathlib import Path

import export_mothersuperior_format as ems


def test_compose_style_prepends_trigger_and_keeps_body():
    out = ems.compose_style("maqamverse, in the style of maqamverse.", '  genre: "x"  \n')
    assert out == 'maqamverse, in the style of maqamverse. genre: "x"\n'


def test_audit_tags_counts_and_unsimple_flags():
    lyrics = "[Intro]\nline\n[Verse 1]\nline\n[Chorus]\nline\n[chorus]\nline\n"
    counts = ems.audit_tags(lyrics)
    assert counts == Counter({"Intro": 1, "Verse 1": 1, "Chorus": 1, "chorus": 1})
    assert ems.unsimple_tags(counts) == ["Chorus", "Intro", "Verse 1"]


def test_unsimple_tags_empty_when_all_simple():
    assert ems.unsimple_tags(Counter({"verse": 2, "pre-chorus": 1})) == []


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        dataset_root=tmp_path / "in",
        out_root=tmp_path / "out",
        trigger_phrase="maqamverse, in the style of maqamverse.",
        splits="train,val",
        ffmpeg="ffmpeg",
        overwrite=False,
        dry_run=False,
    )


def _write_track(root: Path, split: str, base: str, with_sidecars: bool = True) -> None:
    d = root / split
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{base}.mp3").write_bytes(b"ID3fake-audio-bytes")
    if with_sidecars:
        (d / f"{base}.lyrics.txt").write_text("[Verse 1]\nنص\n", encoding="utf-8")
        (d / f"{base}.style.txt").write_text('genre: "x"\n', encoding="utf-8")


def test_export_writes_flat_triplet_and_counts_failures(tmp_path, monkeypatch):
    args = _args(tmp_path)
    _write_track(args.dataset_root, "train", "hijaz_0001_01_take01")
    _write_track(args.dataset_root, "val", "kurd_0002_01_take01")
    _write_track(args.dataset_root, "train", "broken_0003_01_take01", with_sidecars=False)

    def fake_encode(src, dst, ffmpeg, logger):
        dst.write_bytes(b"FLACfake")
        return True

    monkeypatch.setattr(ems, "encode_flac", fake_encode)
    logger = logging.getLogger("test_export")
    code = ems.export(args, logger)

    assert code == 1
    out = args.out_root
    assert (out / "hijaz_0001_01_take01.flac").read_bytes() == b"FLACfake"
    assert (out / "kurd_0002_01_take01.flac").is_file()
    assert (out / "hijaz_0001_01_take01.lyrics.txt").read_text(encoding="utf-8") == "[Verse 1]\nنص\n"
    assert (out / "hijaz_0001_01_take01.txt").read_text(encoding="utf-8") == (
        "maqamverse, in the style of maqamverse. genre: \"x\"\n"
    )
    assert not (out / "broken_0003_01_take01.flac").exists()


def test_export_skips_up_to_date_flac(tmp_path, monkeypatch):
    args = _args(tmp_path)
    _write_track(args.dataset_root, "train", "ajam_0001_01_take01")
    args.out_root.mkdir(parents=True)
    (args.out_root / "ajam_0001_01_take01.flac").write_bytes(b"existing")

    calls = []
    monkeypatch.setattr(ems, "encode_flac", lambda *a: calls.append(a) or True)

    assert ems.export(args, logging.getLogger("test_export")) == 0
    assert calls == []
