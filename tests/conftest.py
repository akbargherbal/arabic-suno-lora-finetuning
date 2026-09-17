"""Shared helpers for the prepare/verify tests.

A corpus is built programmatically rather than from fixture JSON: the tests
pin the *live* workspace_manifest.json contract (REQUIRED_TRACK_FIELDS, unique
stems, real audio) instead of a snapshot that can silently drift.
"""

from __future__ import annotations

import json
import sys
import wave
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAQAM_KEYS = ("ajam", "hijaz", "kurd", "nahawand")

DEFAULT_STYLES = (
    'genre: "Symphonic cinematic orchestral ballad, Maqam {maqam}"\n'
    'vocals: "deep male vocals"\n'
    'production: "Audiophile recording"\n'
    'instrumentation: "Distorted electric guitars"'
)

DEFAULT_LYRICS = (
    "///***///\n"
    "[Verse 1 | epic soaring vocals | heavy power chords]\n"
    "آذنتنا ببينها\n"
    "[Chorus]\n"
    "ثم نأت"
)


def write_wav(path: Path, seconds: float = 0.05, sample_rate: int = 8000) -> None:
    """Write a real, nonempty mono WAV so soundfile/sf.info can read it.

    Content is derived from the filename: FL-YuE2's loader rejects any two
    recordings with identical sha256, so every track's audio must differ.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = zlib.crc32(path.name.encode("utf-8"))
    frames = max(1, int(seconds * sample_rate)) + (seed % 100)
    payload = bytearray(frames * 2)
    for i in range(frames):
        payload[2 * i] = (seed >> (8 * (i % 4))) & 0xFF
        payload[2 * i + 1] = i & 0xFF
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(payload))


def track(
    maqam: str = "hijaz",
    workspace: str = "ws_a",
    clip_id: str = "clip-0001",
    filename: str = "clip-0001.wav",
    title: str = "Poem A",
    status: str = "downloaded",
    styles: str | None = None,
    lyrics: str | None = None,
    exclude_styles: str = "",
    present: bool = True,
) -> dict:
    """One workspace_manifest.json entry, plus test-only location hints."""
    return {
        "maqam": maqam,
        "workspace": workspace,
        "present": present,
        "clip_id": clip_id,
        "original_title": title,
        "assigned_filename": filename,
        "styles": styles if styles is not None else DEFAULT_STYLES.format(maqam=maqam.title()),
        "exclude_styles": exclude_styles,
        "lyrics": lyrics if lyrics is not None else DEFAULT_LYRICS,
        "created_at": "2026-01-01T00:00:00Z",
        "status": status,
    }


def build_corpus(root: Path, tracks: list[dict]) -> Path:
    """Materialize <root>/<maqam>/<workspace>/{audio,workspace_manifest.json}.

    An entry with ``present=False`` gets no audio file on disk, which models a
    manifest row whose recording wasn't downloaded (i.e. a "missing audio").
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for item in tracks:
        grouped.setdefault((item["maqam"], item["workspace"]), []).append(item)

    for (maqam, workspace), entries in grouped.items():
        workspace_dir = root / maqam / workspace
        workspace_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for item in entries:
            manifest.append({
                key: value
                for key, value in item.items()
                if key not in ("maqam", "workspace", "present")
            })
            if item["present"]:
                write_wav(workspace_dir / item["assigned_filename"])
        (workspace_dir / "workspace_manifest.json").write_text(
            json.dumps({"tracks": manifest}, ensure_ascii=False), encoding="utf-8"
        )
    return root


def four_maqam_corpus(root: Path) -> Path:
    """One distinct lyric group per maqam, so every neutral maqam dir exists."""
    return build_corpus(root, [
        track(
            maqam=maqam,
            clip_id=f"{maqam}-0001",
            filename=f"{maqam}-0001.wav",
            title=f"Poem {maqam}",
            lyrics=f"///***///\n[Verse]\nكلمات {maqam}",
        )
        for maqam in MAQAM_KEYS
    ])
