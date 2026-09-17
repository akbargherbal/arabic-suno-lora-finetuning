#!/usr/bin/env python3
"""Queue a FL-YuE2 training graph through ComfyUI's HTTP API.

This is the terminal equivalent of loading a workflow in ComfyUI and pressing
Queue: the same graph runs inside ComfyUI, so run state, previews, adapters and
GCS backups are identical. This script never trains anything itself; it only
submits a prompt and can optionally watch it.

Start ComfyUI first, then:

    python queue_train.py --preset smoke --run-name arabic_smoke
    python queue_train.py --preset smoke --run-name arabic_smoke --dry-run
    python queue_train.py --recipe joint --run-name arabic_joint --steps 3000 --wait
    python queue_train.py --monitor --run-name arabic_smoke
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER = "http://127.0.0.1:8188"
DEFAULT_DATASET = "/content/data/dataset/dataset_comfyui"
RUN_ROOT = REPO_ROOT / "ComfyUI" / "output" / "yue2_training"
JOBS_ROOT = RUN_ROOT / "jobs"
RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")

AR_DEFAULTS = {
    "rank": 64,
    "learning_rate": 1e-4,
    "generated_fraction": 0.5,
    "cursor_weight": 0.08,
    "sequence_tokens": 12288,
    "allow_truncation": False,
    "steps": 1600,
    "save_every": 200,
    "schedule_steps": 3000,
    "warmup_steps": 50,
    "accumulation": 2,
    "seed": 42,
}

JOINT_DEFAULTS = {
    "rank": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "ar_kl_weight": 0.2,
    "abc_dropout": 0.5,
    "ar_lr_multiplier": 1.0,
    "acoustic_window_frames": 1500,
    "nar_start": "base",
    "sequence_tokens": 24576,
    "allow_truncation": False,
    "steps": 3000,
    "save_every": 200,
    "accumulation": 1,
    "seed": 42,
}

SMOKE_OVERRIDES = {
    "rank": 4,
    "steps": 2,
    "save_every": 1,
    "accumulation": 1,
    "sequence_tokens": 2048,
    "allow_truncation": True,
    "schedule_steps": 3,
    "warmup_steps": 0,
    "cursor_weight": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--server", default=DEFAULT_SERVER, help="ComfyUI base URL")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Audio folder with sidecars")
    parser.add_argument("--run-name", default="arabic_smoke", help="Run folder under output/yue2_training")
    parser.add_argument("--recipe", choices=("ar", "joint"), default="ar", help="Legacy AR or joint AR/NAR recipe")
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke", help="Two-step check or real training")
    parser.add_argument("--trigger", default="", help="Text prepended to each style caption")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--align-lyrics", action="store_true")
    parser.add_argument("--score-planning", choices=("off", "melody", "full"), default="off")
    parser.add_argument("--transcribe-scores", action="store_true")
    parser.add_argument("--train-acoustic", action="store_true", help="AR recipe: also train the NAR companion")
    parser.add_argument("--resume", default="", help="Checkpoint relative to the run folder (e.g. resume.pt); config must match the original run")
    parser.add_argument("--nar-start", choices=("base", "community_v4"), default=None)
    parser.add_argument("--preview-style", default="instrumental acoustic guitar, warm folk")
    parser.add_argument("--preview-lyrics", default="")
    parser.add_argument("--preview-seconds", type=int, default=None)
    parser.add_argument("--no-previews", dest="render_previews", action="store_false")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--sequence-tokens", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--accumulation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-truncation", dest="allow_truncation", action="store_true", default=None)
    parser.add_argument("--no-allow-truncation", dest="allow_truncation", action="store_false")
    parser.add_argument("--wait", action="store_true", help="Poll until the queued prompt finishes")
    parser.add_argument("--monitor", action="store_true", help="Watch an existing run only; never queues")
    parser.add_argument("--timeout", type=float, default=0.0, help="Seconds to wait with --wait/--monitor (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate the prompt, do not submit")
    return parser.parse_args()


def request_json(server: str, path: str, payload: dict | None = None, timeout: float = 15.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(server + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            detail = body
        raise SystemExit(f"ComfyUI rejected the request ({error.code}): {json.dumps(detail, indent=2)}")
    except urllib.error.URLError as error:
        raise SystemExit(f"Cannot reach ComfyUI at {server}: {error.reason}\nStart it first, see agent_notes/current.md")


def combo_options(definition):
    if isinstance(definition, list) and definition and isinstance(definition[0], list):
        return definition[0]
    if isinstance(definition, list) and len(definition) > 1 and isinstance(definition[1], dict):
        return definition[1].get("options")
    return None


def validate_prompt(prompt: dict, info: dict) -> list[str]:
    errors: list[str] = []
    for node_id, node in prompt.items():
        class_type = node["class_type"]
        if class_type not in info:
            errors.append(f"{node_id}: ComfyUI has no node named {class_type}")
            continue
        schema = info[class_type]["input"]
        required = schema.get("required", {})
        optional = schema.get("optional", {})
        supplied = node["inputs"]
        for name, definition in required.items():
            if name not in supplied:
                errors.append(f"{node_id} ({class_type}): missing required input '{name}'")
                continue
            value = supplied[name]
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                if value[0] not in prompt:
                    errors.append(f"{node_id} ({class_type}): input '{name}' links to unknown node {value[0]}")
                continue
            options = combo_options(definition)
            if options and value not in options:
                errors.append(f"{node_id} ({class_type}): '{name}'={value!r} not in {options}")
        for name in supplied:
            if name not in required and name not in optional:
                errors.append(f"{node_id} ({class_type}): unknown input '{name}'")
    return errors


def build_config(args: argparse.Namespace) -> tuple[str, dict]:
    config = dict(JOINT_DEFAULTS if args.recipe == "joint" else AR_DEFAULTS)
    if args.preset == "smoke":
        config.update({key: value for key, value in SMOKE_OVERRIDES.items() if key in config})
    overrides = {
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "sequence_tokens": args.sequence_tokens,
        "steps": args.steps,
        "save_every": args.save_every,
        "accumulation": args.accumulation,
        "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None and key in config:
            config[key] = value
    if args.allow_truncation is not None and "allow_truncation" in config:
        config["allow_truncation"] = args.allow_truncation
    if args.nar_start is not None and "nar_start" in config:
        config["nar_start"] = args.nar_start
    if args.train_acoustic and args.recipe == "ar":
        config["train_acoustic"] = True
    class_type = "FL_YuE2_JointTrainConfig" if args.recipe == "joint" else "FL_YuE2_TrainConfig"
    return class_type, config


def build_prompt(args: argparse.Namespace) -> dict:
    config_type, config = build_config(args)
    preview_seconds = args.preview_seconds
    if preview_seconds is None:
        preview_seconds = 8 if args.preset == "smoke" else 30
    return {
        "1": {
            "class_type": "FL_YuE2_TrainingModels",
            "inputs": {
                "tokenizer_head": "tokenizer_head_joint_v4.pt",
                "regularizer": "minted_regularizer_pack.pt",
                "download_missing": True,
            },
        },
        "2": {"class_type": config_type, "inputs": config},
        "3": {
            "class_type": "FL_YuE2_DatasetMaker",
            "inputs": {
                "audio_directory": args.dataset,
                "trigger": args.trigger,
                "default_style": "",
                "validation_fraction": args.validation_fraction,
                "seed": args.seed if args.seed is not None else 42,
            },
        },
        "4": {
            "class_type": "FL_YuE2_PrepareDataset",
            "inputs": {
                "dataset": ["3", 0],
                "assets": ["1", 0],
                "align_lyrics": args.align_lyrics,
                "cache_directory": "prepared",
                "score_planning": args.score_planning,
                "transcribe_missing_scores": args.transcribe_scores,
            },
        },
        "5": {
            "class_type": "FL_YuE2_LoRATrainer",
            "inputs": {
                "action": "train",
                "output_name": args.run_name,
                "resume": args.resume,
                "selected_step": 0,
                "render_previews": args.render_previews,
                "preview_style": args.preview_style,
                "preview_lyrics": args.preview_lyrics,
                "preview_seed": 42,
                "preview_seconds": preview_seconds,
                "assets": ["1", 0],
                "dataset": ["4", 0],
                "config": ["2", 0],
            },
        },
    }


def latest_job_event() -> str:
    logs = sorted(JOBS_ROOT.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not logs:
        return ""
    try:
        lines = logs[0].read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        if not line.startswith("YUE2_EVENT "):
            continue
        try:
            event = json.loads(line[len("YUE2_EVENT "):])
        except json.JSONDecodeError:
            continue
        if event.get("message"):
            return event["message"]
        if event.get("type") == "progress":
            return f"progress {event.get('step')}/{event.get('max_steps')}"
    return ""


def newest_job_start() -> float:
    jobs = list(JOBS_ROOT.glob("*.json"))
    return max((path.stat().st_mtime for path in jobs), default=0.0)


def run_record(run_name: str) -> dict | None:
    path = RUN_ROOT / run_name / "run.json"
    if not path.is_file():
        return None
    if path.stat().st_mtime < newest_job_start():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_progress(run_name: str) -> str:
    record = run_record(run_name)
    if record is None:
        return latest_job_event() or "waiting for run.json"
    return f"status={record.get('status')} step={record.get('step')}/{record.get('config', {}).get('steps', '?')}"


def monitor_run(run_name: str, timeout: float) -> int:
    started = time.time()
    last = ""
    while True:
        record = run_record(run_name)
        if record is not None:
            metrics = record.get("metrics") or []
            steps = record.get("config", {}).get("steps", "?")
            line = f"status={record.get('status')} step={record.get('step')}/{steps}"
            if metrics and metrics[-1].get("loss") is not None:
                line += f" loss={metrics[-1]['loss']:.4f}"
            if line != last:
                print(line, flush=True)
                last = line
            if record.get("status") in ("complete", "failed", "cancelled", "paused"):
                return 0 if record["status"] == "complete" else 1
        else:
            line = latest_job_event() or f"No run or worker log yet for {run_name!r}"
            if line != last:
                print(line, flush=True)
                last = line
        if timeout and time.time() - started > timeout:
            return 1
        time.sleep(5)


def wait_for(server: str, prompt_id: str, run_name: str, timeout: float) -> int:
    started = time.time()
    while True:
        history = request_json(server, f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            print(f"\nFinished: status={status.get('status_str', 'unknown')} completed={status.get('completed')}")
            if not status.get("completed"):
                print(json.dumps(status.get("messages", []), indent=2))
                return 1
            return 0
        print(f"\r{read_progress(run_name)}", end="", flush=True)
        if timeout and time.time() - started > timeout:
            print(f"\nStill running after {timeout:.0f}s; prompt {prompt_id} continues in ComfyUI")
            return 1
        time.sleep(5)


def main() -> int:
    args = parse_args()
    if not RUN_NAME_RE.fullmatch(args.run_name):
        raise SystemExit(f"Invalid --run-name {args.run_name!r}: use 1-80 letters, digits, underscores or hyphens")
    if args.monitor:
        return monitor_run(args.run_name, args.timeout)
    prompt = build_prompt(args)
    class_types = sorted({node["class_type"] for node in prompt.values()})
    info = {}
    for class_type in class_types:
        data = request_json(args.server, f"/object_info/{class_type}")
        if class_type not in data:
            raise SystemExit(f"ComfyUI does not expose {class_type}; is ComfyUI-FL-YuE2 installed and loaded?")
        info[class_type] = data[class_type]
    errors = validate_prompt(prompt, info)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(prompt, indent=2))
        print(f"\nPrompt valid for {args.dataset!r}; run name {args.run_name!r}.")
        return 0
    response = request_json(args.server, "/prompt", {"prompt": prompt, "client_id": uuid.uuid4().hex})
    prompt_id = response["prompt_id"]
    print(f"Queued prompt {prompt_id} (queue #{response.get('number')}) for run {args.run_name!r}")
    print(f"Progress: /content/logs/comfyui.log; state: {RUN_ROOT / args.run_name / 'run.json'}")
    if args.wait:
        return wait_for(args.server, prompt_id, args.run_name, args.timeout)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
