"""Tests for verify_dataset.py: a clean build passes, each corruption trips it."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from conftest import default_style, write_dataset

import verify_dataset as vd


def _run_verify(out_dir: Path, monkeypatch, *extra) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_dataset.py",
            "--out-dir",
            str(out_dir),
            *extra,
        ],
    )
    return vd.main()


def _clean_entries() -> list[dict]:
    return [
        {
            "split": "train",
            "maqam": "hijaz",
            "poem": "0001",
            "slug": "poem-a",
            "take": "01",
            "title": "Poem A",
            "style": default_style("Hijaz"),
            "lyrics": "[Verse 1]\nآذنتنا",
        },
        {
            "split": "train",
            "maqam": "hijaz",
            "poem": "0002",
            "slug": "poem-b",
            "take": "01",
            "title": "Poem B",
            "style": default_style("Hijaz"),
            "lyrics": "[Verse 1]\nيا عزة",
        },
        {
            "split": "val",
            "maqam": "kurd",
            "poem": "0003",
            "slug": "poem-c",
            "take": "01",
            "title": "Poem C",
            "style": default_style("Kurd"),
            "lyrics": "[Verse 1]\nقفا نبك",
        },
    ]


def test_clean_dataset_passes(tmp_path, monkeypatch, capsys):
    out = tmp_path / "dataset"
    write_dataset(out, _clean_entries())
    rc = _run_verify(out, monkeypatch)
    capsys.readouterr()
    assert rc == 0


def test_missing_sidecar_fails(tmp_path, monkeypatch, capsys):
    out = tmp_path / "dataset"
    write_dataset(out, _clean_entries())
    (out / "train" / "hijaz_0001_poem-a_take01.lyrics.txt").unlink()
    rc = _run_verify(out, monkeypatch)
    output = capsys.readouterr().out
    assert rc == 1
    assert "no .lyrics.txt" in output


def test_leaked_suno_marker_fails(tmp_path, monkeypatch, capsys):
    out = tmp_path / "dataset"
    write_dataset(out, _clean_entries())
    (out / "train" / "hijaz_0001_poem-a_take01.lyrics.txt").write_text("///***///\n[Verse 1]\nآذنتنا", encoding="utf-8")
    rc = _run_verify(out, monkeypatch)
    output = capsys.readouterr().out
    assert rc == 1
    assert "Suno UI" in output


def test_duplicate_destination_fails(tmp_path, monkeypatch, capsys):
    out = tmp_path / "dataset"
    rows = write_dataset(out, _clean_entries())
    rows[1]["dest_audio"] = rows[0]["dest_audio"]
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    rc = _run_verify(out, monkeypatch)
    output = capsys.readouterr().out
    assert rc == 1
    assert "more than once" in output


def test_train_val_leakage_fails(tmp_path, monkeypatch, capsys):
    out = tmp_path / "dataset"
    entries = _clean_entries()
    entries[1]["split"] = "val"
    entries[1]["title"] = "Poem A"  # same poem title, opposite split
    write_dataset(out, entries)
    rc = _run_verify(out, monkeypatch)
    output = capsys.readouterr().out
    assert rc == 1
    assert "BOTH train and val" in output


def test_zero_byte_audio_fails(tmp_path, monkeypatch, capsys):
    out = tmp_path / "dataset"
    write_dataset(out, _clean_entries())
    (out / "train" / "hijaz_0001_poem-a_take01.mp3").write_bytes(b"")
    rc = _run_verify(out, monkeypatch)
    output = capsys.readouterr().out
    assert rc == 1
    assert "zero-byte" in output


def test_old_naming_scheme_fails(tmp_path, monkeypatch, capsys):
    out = tmp_path / "dataset"
    write_dataset(out, _clean_entries())
    d = out / "train"
    new = "hijaz_0001_poem-a_take01"
    old = "hijaz_poem-a_take01"  # pre-poem_id scheme
    for ext in (".mp3", ".style.txt", ".lyrics.txt"):
        d.joinpath(new + ext).rename(d / (old + ext))
    with (out / "manifest.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["dest_audio"] == str(d / f"{new}.mp3"):
            r["dest_audio"] = str(d / f"{old}.mp3")
            r["dest_style"] = str(d / f"{old}.style.txt")
            r["dest_lyrics"] = str(d / f"{old}.lyrics.txt")
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    rc = _run_verify(out, monkeypatch)
    output = capsys.readouterr().out
    assert rc == 1
    assert "OLD naming scheme" in output


def test_verify_after_real_build(corpus, tmp_path, monkeypatch, capsys):
    """End-to-end: prepare_dataset.py builds, verify_dataset.py signs off."""
    import prepare_dataset as pd

    src = corpus("manifest_clean.json")
    out = tmp_path / "dataset"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dataset.py",
            "--dataset-root",
            str(src),
            "--out-dir",
            str(out),
        ],
    )
    pd.main()
    capsys.readouterr()
    rc = _run_verify(out, monkeypatch)
    capsys.readouterr()
    assert rc == 0
