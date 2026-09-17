"""End-to-end tests for verify_dataset.py: a clean build passes, and each
corruption trips the corresponding FAIL check.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from conftest import four_maqam_corpus

import prepare_dataset as pd
import verify_dataset as vd


@pytest.fixture(autouse=True)
def _isolate_results():
    vd.RESULTS.clear()
    yield
    vd.RESULTS.clear()


def _build(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    src = tmp_path / "src"
    four_maqam_corpus(src)
    out = tmp_path / "dataset"
    comfy = tmp_path / "dataset_comfyui"
    monkeypatch.setattr(sys, "argv", [
        "prepare_dataset.py", "--src", str(src),
        "--out", str(out), "--comfyui-out", str(comfy),
    ])
    assert pd.main() == 0
    return src, out, comfy


def _verify(monkeypatch, src: Path, out: Path, comfy: Path, *extra, skip_real: bool = True) -> int:
    argv = [
        "verify_dataset.py", "--src", str(src), "--out", str(out),
        "--comfyui-out", str(comfy), *extra,
    ]
    if skip_real:
        argv.append("--skip-real-loader")
    monkeypatch.setattr(sys, "argv", argv)
    return vd.main()


def _failed() -> list[str]:
    return [check for level, check, _detail in vd.RESULTS if level == "FAIL"]


def test_clean_dataset_passes(tmp_path, monkeypatch):
    src, out, comfy = _build(tmp_path, monkeypatch)
    assert _verify(monkeypatch, src, out, comfy) == 0
    assert _failed() == []


def test_missing_comfyui_lyrics_fails(tmp_path, monkeypatch):
    src, out, comfy = _build(tmp_path, monkeypatch)
    (comfy / "hijaz-0001.lyrics.txt").unlink()
    assert _verify(monkeypatch, src, out, comfy) == 1
    assert "comfyui.lyrics" in _failed()


def test_removed_caption_json_fails(tmp_path, monkeypatch):
    src, out, comfy = _build(tmp_path, monkeypatch)
    (comfy / "hijaz-0001.caption.json").unlink()
    assert _verify(monkeypatch, src, out, comfy) == 1
    assert "comfyui.metadata" in _failed()


def test_zero_byte_comfyui_audio_fails(tmp_path, monkeypatch):
    src, out, comfy = _build(tmp_path, monkeypatch)
    (comfy / "hijaz-0001.wav").write_bytes(b"")
    assert _verify(monkeypatch, src, out, comfy) == 1
    assert "comfyui.readable-audio" in _failed()


def test_duplicate_comfyui_stem_fails(tmp_path, monkeypatch):
    src, out, comfy = _build(tmp_path, monkeypatch)
    # Same stem, different extension -> data.audio_files() would reject it.
    (comfy / "hijaz-0001.mp3").write_bytes((comfy / "hijaz-0001.wav").read_bytes())
    assert _verify(monkeypatch, src, out, comfy) == 1
    assert "comfyui.stems" in _failed()


def test_empty_neutral_caption_fails(tmp_path, monkeypatch):
    src, out, comfy = _build(tmp_path, monkeypatch)
    (out / "hijaz" / "hijaz-0001.caption.txt").write_text("\n", encoding="utf-8")
    assert _verify(monkeypatch, src, out, comfy) == 1
    assert "dataset.caption-nonempty" in _failed()


def _fl_yue2_repo() -> Path | None:
    candidates = [
        os.environ.get("FL_YUE2_REPO"),
        Path(__file__).resolve().parent.parent / "ComfyUI" / "custom_nodes" / "ComfyUI-FL-YuE2",
        Path(__file__).resolve().parent.parent.parent / "ComfyUI-FL-YuE2",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "yue2" / "training" / "data.py").is_file():
            return Path(candidate)
    return None


def test_real_loader_accepts_the_dataset(tmp_path, monkeypatch):
    repo = _fl_yue2_repo()
    if repo is None:
        pytest.skip("ComfyUI-FL-YuE2 clone not found (set FL_YUE2_REPO)")
    src, out, comfy = _build(tmp_path, monkeypatch)
    assert _verify(monkeypatch, src, out, comfy, "--comfyui-repo", str(repo),
                   skip_real=False) == 0
    assert "comfyui.real-loader" in [
        check for level, check, _detail in vd.RESULTS if level == "PASS"
    ]
