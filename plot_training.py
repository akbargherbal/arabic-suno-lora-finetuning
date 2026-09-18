#!/usr/bin/env python3
"""Plot FL-YuE2 training metrics for a run.

Reads every `YUE2_EVENT` progress line from the worker logs that belong to the
run (the jobs directory is shared, so logs are matched via their request JSON),
keeps the last value per step, and draws loss / AR-CE / KL / NAR-flow plus
gradient norm, learning rate and checkpoint validation. Writes a PNG.

    python plot_training.py --run-name arabic_joint_v1

Pass `--out-dir` to also write one graph per metric (plus smoothed trend lines)
into a folder, which is what gets committed under `TRAINING_ANALYSIS/<run>/`:

    python plot_training.py --run-name arabic_joint_v2 --out-dir TRAINING_ANALYSIS/arabic_joint_v2
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
SMOOTH_WINDOW = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="arabic_joint_v1")
    parser.add_argument("--out", default=None, help="PNG path (default: <run folder>/training_curve.png)")
    parser.add_argument("--out-dir", default=None, help="Also write one PNG per metric into this folder")
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW, help="Rolling-mean window for trend lines (0 disables)")
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


def rolling_mean(values: list, window: int) -> list:
    if window <= 1 or len(values) < 2:
        return list(values)
    window = min(window, len(values))
    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        chunk = [v for v in values[lo:hi] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def validate(record: dict) -> dict:
    return {m["step"]: m for m in record.get("metrics", []) if "ar_validation" in m}


def load_run_json(run_name: str) -> dict:
    path = RUN_ROOT / run_name / "run.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def plot_overview(steps: list, data: list, validation: dict, total: int, rate: float, run_name: str, out: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(steps, [m.get("loss") for m in data], lw=0.8, alpha=0.35, color="#1f77b4")
    axes[0, 0].plot(steps, rolling_mean([m.get("loss") for m in data], SMOOTH_WINDOW), lw=1.6, color="#1f77b4")
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
    figure.suptitle(f"{run_name}: step {steps[-1]}/{total}"
                    + (f", {rate:.1f} s/step, ETA {eta / 3600:.1f} h" if rate else ""))
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=120)
    plt.close(figure)
    print(f"wrote {out}")


def plot_metric(steps: list, data: list, keys: list[str], title: str, out: Path, smooth_window: int, logy: bool = False) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    for key in keys:
        values = [m.get(key) for m in data]
        axis.plot(steps, values, lw=0.8, alpha=0.3)
        axis.plot(steps, rolling_mean(values, smooth_window), lw=1.8, label=f"{key} (smooth {smooth_window})")
    axis.set_title(title)
    axis.set_xlabel("step")
    axis.grid(alpha=0.3)
    axis.legend()
    if logy:
        axis.set_yscale("log")
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=120)
    plt.close(figure)
    print(f"wrote {out}")


def plot_validation(validation: dict, out: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    for key, label in (("ar_validation", "ar (artist)"), ("nar_validation", "nar (artist)"),
                       ("kl_validation", "ar_kl"), ("artist_validation", "artist total")):
        xs = [step for step, m in validation.items() if m.get(key) is not None]
        ys = [m[key] for m in validation.values() if m.get(key) is not None]
        if xs:
            axis.plot(xs, ys, "o-", ms=4, label=label)
    axis.set_title("validation loss at checkpoints")
    axis.set_xlabel("step")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=120)
    plt.close(figure)
    print(f"wrote {out}")


def write_metric_suite(steps: list, data: list, validation: dict, out_dir: Path, smooth_window: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(steps, data, ["loss"], "total loss", out_dir / "02_total_loss.png", smooth_window)
    plot_metric(steps, data, ["ar_ce"],
                "AR cross-entropy", out_dir / "03_ar_ce.png", smooth_window)
    plot_metric(steps, data, ["ar_kl"],
                "AR KL to base", out_dir / "04_ar_kl.png", smooth_window)
    plot_metric(steps, data, ["nar_flow"],
                "NAR flow loss", out_dir / "05_nar_flow.png", smooth_window)
    plot_metric(steps, data, ["ar_ce", "ar_kl", "nar_flow"],
                "loss terms", out_dir / "06_loss_terms.png", smooth_window)
    plot_metric(steps, data, ["gradient_norm"],
                "gradient norm", out_dir / "07_gradient_norm.png", smooth_window, logy=True)
    plot_metric(steps, data, ["lr"],
                "learning rate", out_dir / "08_learning_rate.png", smooth_window)
    plot_metric(steps, data, ["peak_gb"],
                "peak GPU memory", out_dir / "09_peak_vram.png", smooth_window)
    if validation:
        plot_validation(validation, out_dir / "10_validation.png")


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

    validation = validate(load_run_json(args.run_name))

    plot_overview(steps, data, validation, total, rate, args.run_name, out)
    if args.out_dir:
        overview = Path(args.out_dir) / "01_overview.png"
        plot_overview(steps, data, validation, total, rate, args.run_name, overview)
        write_metric_suite(steps, data, validation, Path(args.out_dir), args.smooth)

    last_val = data[-1]
    print(f"step {last_val['step']}/{total} | loss {last_val.get('loss'):.4f} "
          f"ar_ce {last_val.get('ar_ce'):.4f} ar_kl {last_val.get('ar_kl'):.4f} nar_flow {last_val.get('nar_flow'):.4f}")
    if validation:
        final = validation[max(validation)]
        print(f"validation { {k: round(v, 4) for k, v in final.items() if k.endswith('_validation') and v is not None} }")
    if rate:
        eta = (total - steps[-1]) * rate
        print(f"{rate:.1f} s/step | ETA {eta / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
