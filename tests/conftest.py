"""Shared pytest fixtures/helpers for the YuE2 dataset tests.

The JSON files in ``tests/fixtures/`` mirror the real corpus's
``workspace_manifest.json`` field shape (``clip_id, original_title,
assigned_filename, styles, exclude_styles, lyrics, created_at, status``) plus
two test-only top-level keys, ``maqam`` and ``workspace``, that say where the
manifest should be materialized.

Fixture files and the scenario each exercises:

  manifest_clean.json            a well-formed workspace (2 poems, one with two
                                 takes, plus one below-rating entry with no file)
  manifest_naming_collision.json two different poems whose ASCII slug collides
  manifest_missing_sidecar.json  a valid single-track build (delete a sidecar
                                 to reproduce the missing-sidecar check)
  manifest_leaked_marker.json    lyrics carrying ``///***///`` and a pipe tag
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIXTURES_DIR = TESTS_DIR / "fixtures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Fixture name -> assigned_filenames deliberately absent from disk (the
# below-4-star entries that prepare_dataset.py's file-existence filter drops).
ABSENT_FILES = {
    "manifest_clean.json": {"03-حذفت-لأنها-ضعيفة.mp3"},
}


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def materialize_corpus(root: Path, fixture_name: str) -> Path:
    """Write a raw corpus tree (``<root>/<maqam>/<workspace>/...``) from a fixture."""
    fixture = load_fixture(fixture_name)
    workspace_dir = root / fixture["maqam"] / fixture["workspace"]
    workspace_dir.mkdir(parents=True, exist_ok=True)
    absent = ABSENT_FILES.get(fixture_name, set())
    for track in fixture["tracks"]:
        if track["assigned_filename"] in absent:
            continue
        (workspace_dir / track["assigned_filename"]).write_bytes(b"ID3fake-audio-bytes")
    (workspace_dir / "workspace_manifest.json").write_text(
        json.dumps({"tracks": fixture["tracks"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def corpus(tmp_path: Path):
    """Return a factory ``build(fixture_name) -> Path`` rooted in tmp_path."""

    def build(fixture_name: str) -> Path:
        root = tmp_path / f"corpus_{fixture_name.replace('.json', '')}"
        return materialize_corpus(root, fixture_name)

    return build


MANIFEST_FIELDS = [
    "clip_id",
    "workspace",
    "original_title",
    "maqam",
    "mood",
    "take_no_in_group",
    "source_audio",
    "dest_audio",
    "dest_style",
    "dest_lyrics",
    "split",
]


def write_dataset(out_dir: Path, entries: list[dict]) -> list[dict]:
    """Materialize a minimal built dataset directly (no prepare_dataset run).

    Each entry needs: split, maqam, poem, slug, take, title, style, lyrics.
    ``clip_id`` defaults to a unique value. Returns the manifest rows.
    """
    rows = []
    for i, e in enumerate(entries):
        split = e["split"]
        d = out_dir / split
        d.mkdir(parents=True, exist_ok=True)
        base = f"{e['maqam']}_{e['poem']}_{e['slug']}_take{e['take']}"
        audio = d / f"{base}.mp3"
        style = d / f"{base}.style.txt"
        lyrics = d / f"{base}.lyrics.txt"
        audio.write_bytes(e.get("audio_bytes", b"ID3fake-audio-bytes"))
        style.write_text(e["style"], encoding="utf-8")
        lyrics.write_text(e.get("lyrics", ""), encoding="utf-8")
        rows.append(
            {
                "clip_id": e.get("clip_id", f"clip-{i:04d}"),
                "workspace": "ws",
                "original_title": e["title"],
                "maqam": e["maqam"],
                "mood": "",
                "take_no_in_group": 1,
                "source_audio": str(audio),
                "dest_audio": str(audio),
                "dest_style": str(style),
                "dest_lyrics": str(lyrics),
                "split": split,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def default_style(maqam: str, title: str = "") -> str:
    return (
        f'genre: "Symphonic cinematic orchestral ballad, Maqam {maqam}"\n'
        f'vocals: "deep male vocals"\n'
        f'production: "Audiophile recording"\n'
        f'instrumentation: "Distorted electric guitars"'
    )
