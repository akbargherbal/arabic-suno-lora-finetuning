#!/usr/bin/env python3
"""Render a maqam x adapter comparison matrix with FL-YuE2.

Loads YuE2-3B + its decoder once, then for every requested workspace track and
every selected adapter (`base` = no LoRA, plus each trained checkpoint) renders
a song. Base and LoRA variants of the same track share the exact prompt and
seed, so the outputs are directly comparable by ear.

Run from the repo root:

    python INFERENCE/run_inference_matrix.py \
        --manifest INFERENCE/workspace_manifest.json \
        --lora-dir ComfyUI/models/loras/YuE2/arabic_joint_v2 \
        --out INFERENCE/outputs/arabic_joint_v2

    # print the matrix and the resolved prompts without loading the model:
    python INFERENCE/run_inference_matrix.py --dry-run

By default it uses one track per maqam (Hijaz, Kurd, Ajam, Nahawand) and the
`base` + latest checkpoint variants. `--all-tracks` includes the A/B pairs;
`--all-adapters` or `--adapters step-000400,step-000800` select checkpoints.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMFY_DIR = REPO_ROOT / "ComfyUI"
CUSTOM_DIR = COMFY_DIR / "custom_nodes" / "ComfyUI-FL-YuE2"
for _path in (str(CUSTOM_DIR), str(COMFY_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DEFAULT_MAQAMS = "Hijaz,Kurd,Ajam,Nahawand"
MAQAM_RE = re.compile(r"Maqam\s+([A-Za-z]+)", re.IGNORECASE)
STYLE_FIELD_RE = re.compile(
    r'^\s*(genre|vocals|production|instrumentation|mood)\s*:\s*"(.*)"\s*$', re.IGNORECASE)
SECTION_RE = re.compile(r"\[([^\]|]*?)\s*\|[^\]]*\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "INFERENCE" / "workspace_manifest.json")
    parser.add_argument("--lora-dir", type=Path, default=COMFY_DIR / "models" / "loras" / "YuE2" / "arabic_joint_v2")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "INFERENCE" / "outputs")
    parser.add_argument("--adapters", default="", help="Comma list of checkpoint labels, e.g. step-000400,step-000800")
    parser.add_argument("--all-adapters", action="store_true", help="Use base plus every checkpoint found")
    parser.add_argument("--maqams", default=DEFAULT_MAQAMS)
    parser.add_argument("--all-tracks", action="store_true", help="Include every track, not one per maqam")
    parser.add_argument("--planning", choices=("off", "melody", "full"), default="off")
    parser.add_argument("--duration", type=int, default=90, help="Maximum seconds per song")
    parser.add_argument("--acoustic-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--tile-frames", type=int, default=1024)
    parser.add_argument("--ar-strength", type=float, default=1.0)
    parser.add_argument("--nar-strength", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--raw", action="store_true", help="Use manifest style/lyrics verbatim instead of cleaning them")
    parser.add_argument("--dry-run", action="store_true", help="Print the matrix and prompts, load no model")
    return parser.parse_args()


def maqam_of(track: dict) -> str:
    match = MAQAM_RE.search(track.get("styles", ""))
    return match.group(1).capitalize() if match else "Unknown"


def clean_style(text: str) -> str:
    fields = [match.group(2).strip() for line in text.splitlines() if (match := STYLE_FIELD_RE.match(line))]
    return " ".join(fields).strip() or text.strip()


def clean_lyrics(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip() not in ("///***///", "")]
    return "\n".join(SECTION_RE.sub(lambda match: "[" + match.group(1).strip() + "]", line) for line in lines)


def load_tracks(args: argparse.Namespace) -> list[dict]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    wanted = {name.strip().lower() for name in args.maqams.split(",") if name.strip()}
    tracks = []
    seen = set()
    for index, track in enumerate(manifest["tracks"]):
        maqam = maqam_of(track)
        if wanted and maqam.lower() not in wanted:
            continue
        if not args.all_tracks and maqam.lower() in seen:
            continue
        seen.add(maqam.lower())
        tracks.append({
            "index": index,
            "clip_id": track["clip_id"],
            "maqam": maqam,
            "style": track["styles"] if args.raw else clean_style(track["styles"]),
            "lyrics": track["lyrics"] if args.raw else clean_lyrics(track.get("lyrics", "")),
        })
    return tracks


def find_adapters(lora_dir: Path) -> dict[str, tuple[Path, Path | None]]:
    found = {}
    for ar_path in sorted(lora_dir.glob("step-*.safetensors")):
        if ar_path.name.endswith("-nar.safetensors"):
            continue
        nar_path = ar_path.with_name(ar_path.stem + "-nar.safetensors")
        found[ar_path.stem] = (ar_path, nar_path if nar_path.is_file() else None)
    return found


def load_variants(args: argparse.Namespace) -> list[tuple[str, list[Path] | None]]:
    available = find_adapters(args.lora_dir)
    if args.adapters:
        labels = [label.strip() for label in args.adapters.split(",") if label.strip()]
    elif args.all_adapters:
        labels = sorted(available)
    else:
        labels = sorted(available)[-1:]
    variants: list[tuple[str, list[Path] | None]] = [("base", None)]
    for label in labels:
        if label not in available:
            raise SystemExit(f"No adapter {label!r} under {args.lora_dir} (have {sorted(available)})")
        ar_path, nar_path = available[label]
        paths = [ar_path] + ([nar_path] if nar_path else [])
        variants.append((label, paths))
    return variants


def save_audio(audio: dict, path: Path) -> None:
    import soundfile as sf

    waveform = audio["waveform"][0].detach().cpu().transpose(0, 1).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.numpy(), audio["sample_rate"])


def main() -> int:
    args = parse_args()
    tracks = load_tracks(args)
    if not tracks:
        raise SystemExit("No tracks selected; check --maqams / --all-tracks")

    if args.dry_run:
        variants = load_variants(args)
        print(f"manifest: {args.manifest}")
        print(f"lora dir: {args.lora_dir}")
        print(f"variants: {[label for label, _ in variants]}")
        for track in tracks:
            print(f"\n=== {track['maqam']} | {track['clip_id'][:8]} (seed {args.seed + track['index']}) ===")
            print("style:  " + track["style"])
            print("lyrics: " + track["lyrics"].replace("\n", " / ")[:200])
        return 0

    import torch
    from yue2 import adapters, runtime

    variants = load_variants(args)
    print(f"loading YuE2-3B + decoder (variants: {[label for label, _ in variants]})")
    music, vae = runtime.load_models(download_missing=True)

    out_root = args.out
    matrix = {"manifest": str(args.manifest), "lora_dir": str(args.lora_dir),
              "planning": args.planning, "duration": args.duration, "seed": args.seed, "files": []}
    for label, paths in variants:
        patched = music if paths is None else adapters.patch_music(music, [str(p) for p in paths], args.ar_strength, args.nar_strength)
        print(f"\n--- adapter: {label} ---")
        for track in tracks:
            seed = args.seed + track["index"]
            started = time.time()
            plan = runtime.make_plan(patched, track["style"], track["lyrics"], seed, args.planning, "", args.max_tokens)
            latent, truncated, _ = runtime.render(
                patched, plan, args.duration, args.temperature, args.top_p, args.top_k,
                args.repetition_penalty, args.guidance, args.acoustic_steps)
            audio = runtime.decode(vae, latent, args.tile_frames)
            path = out_root / track["maqam"] / f"{track['clip_id'][:8]}_{label}.wav"
            save_audio(audio, path)
            seconds = audio["waveform"].shape[-1] / audio["sample_rate"]
            print(f"{track['maqam']:9s} {track['clip_id'][:8]}  {seconds:5.1f}s audio  "
                  f"{time.time() - started:5.1f}s wall  truncated={truncated}  -> {path}")
            matrix["files"].append({"clip_id": track["clip_id"], "maqam": track["maqam"],
                                    "adapter": label, "seed": seed, "path": str(path),
                                    "seconds": round(seconds, 2), "truncated": truncated})
        if patched is not music:
            del patched
            torch.cuda.empty_cache()

    (out_root / "matrix.json").parent.mkdir(parents=True, exist_ok=True)
    (out_root / "matrix.json").write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(matrix['files'])} files under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
