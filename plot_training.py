#!/usr/bin/env python3
"""Plot FL-YuE2 training metrics for a run.

Reads every `YUE2_EVENT` progress line from the worker logs that belong to the
run (the jobs directory is shared, so logs are matched via their request JSON),
keeps the last value per step, and draws loss / AR-CE / KL / NAR-flow plus
gradient norm, learning rate and checkpoint validation. Writes a PNG.

    python plot_training.py --run-name arabic_joint_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
RUN_ROOT = REPO_ROOT / "ComfyUI" / "output" / "yue2_training"
JOBS_ROOT = RUN_ROOT / "jobs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="arabic_joint_v1")
    parser.add_argument("--out", default=None, help="PNG path (default: <run folder>/training_curve.png)")
    return parser.parse_args()


def job_matches_run(stem: str, run_name: str) -> bool:
    request = JOBS_ROOT / f"{stem}.json"
    if not request.is_file():
        return False
    try:
        data = json.loads(request.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    path = str(data.get("run_directory") or data.get("run") or "").replace("\\", "/")
    return run_name in Path(path).parts


def progress_events(run_name: str) -> list[tuple[Path, list[dict]]]:
    logs = []
    for log in JOBS_ROOT.glob("*.log"):
        if job_matches_run(log.stem, run_name):
            logs.append(log)
    logs.sort(key=lambda path: path.stat().st_mtime)
    series = []
    for log in logs:
        events = []
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("YUE2_EVENT "):
                continue
            try:
                event = json.loads(line[len("YUE2_EVENT "):])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "progress" and isinstance(event.get("step"), int):
                events.append(event)
        if events:
            series.append((log, events))
    return series


def main() -> int:
    args = parse_args()
    out = Path(args.out) if args.out else RUN_ROOT / args.run_name / "training_curve.png"
    series = progress_events(args.run_name)
    by_step: dict[int, dict] = {}
    for _, events in series:
        for event in events:
            by_step[event["step"]] = event
    if not by_step:
        raise SystemExit(f"No progress events found for run {args.run_name!r}")

    steps = sorted(by_step)
    data = [by_step[step] for step in steps]
    last = data[-1]
    total = last.get("max_steps", steps[-1])

    rate = 0.0
    if series:
        events = series[-1][1]
        span = events[-1].get("seconds", 0.0) - events[0].get("seconds", 0.0)
        count = events[-1]["step"] - events[0]["step"]
        rate = span / count if count else 0.0

    validation = {}
    run_json = RUN_ROOT / args.run_name / "run.json"
    if run_json.is_file():
        record = json.loads(run_json.read_text(encoding="utf-8"))
        for metric in record.get("metrics", []):
            if "ar_validation" in metric:
                validation[metric["step"]] = metric

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(steps, [m.get("loss") for m in data], lw=1, color="#1f77b4")
    if validation:
        axes[0, 0].plot(list(validation), [m["artist_validation"] for m in validation.values()],
                        "o-", color="#d62728", ms=3, lw=1, label="validation")
        axes[0, 0].legend()
    axes[0, 0].set_title("total loss")
    axes[0, 1].plot(steps, [m.get("ar_ce") for m in data], lw=1, label="ar_ce")
    axes[0, 1].plot(steps, [m.get("ar_kl") for m in data], lw=1, label="ar_kl")
    axes[0, 1].plot(steps, [m.get("nar_flow") for m in data], lw=1, label="nar_flow")
    axes[0, 1].legend()
    axes[0, 1].set_title("loss terms")
    axes[1, 0].plot(steps, [m.get("gradient_norm") for m in data], lw=1, color="#2ca02c")
    axes[1, 0].set_title("gradient norm")
    if validation:
        axes[1, 1].plot(list(validation), [m["ar_validation"] for m in validation.values()], "o-", ms=3, label="ar")
        axes[1, 1].plot(list(validation), [m["nar_validation"] for m in validation.values()], "o-", ms=3, label="nar")
        axes[1, 1].legend()
    axes[1, 1].set_title("validation loss at checkpoints")
    for row in axes:
        for axis in row:
            axis.set_xlabel("step")
            axis.grid(alpha=0.3)
    eta = (total - steps[-1]) * rate
    figure.suptitle(f"{args.run_name}: step {steps[-1]}/{total}"
                    + (f", {rate:.1f} s/step, ETA {eta / 3600:.1f} h" if rate else ""))
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=120)
    print(f"wrote {out}")
    print(f"step {steps[-1]}/{total} | loss {last.get('loss'):.4f} "
          f"ar_ce {last.get('ar_ce'):.4f} ar_kl {last.get('ar_kl'):.4f} nar_flow {last.get('nar_flow'):.4f}")
    if rate:
        print(f"{rate:.1f} s/step | ETA {eta / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
