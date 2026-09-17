#!/usr/bin/env python3
"""Mirror the high-stake run artifacts to GCS on a timer.

FL-YuE2 only checkpoints every `save_every` optimizer steps (there is no
time-based save), so a Colab/VM interruption can lose GPU work. Run this in a
second terminal next to training: it mirrors only the folders needed to resume
or debug the run up to a GCS prefix with `gsutil rsync`.

rsync is append/update-only here (no `-d`): nothing is ever deleted on the
remote side. If a pass catches a checkpoint mid-write, the next pass re-uploads
it once the local file stops changing, so a torn upload self-heals.

Every run gets its own subfolder -- `<base>/<run-name>/` -- under one generic
root (`.../YuE2-3B_Arabic_Suno_Finetuning/`), so a new run can never overwrite
a previous one and the bucket root stays fixed. `--run-name` is required and a
`run_manifest.json` is written at the run folder's root so it identifies
itself. `dataset/` already lives under the root and is reserved.

Usage:
    python backup_to_gcp.py --run-name my_song_lora
    python backup_to_gcp.py --run-name my_song_lora --interval-minutes 20
    python backup_to_gcp.py --run-name my_song_lora --once
    python backup_to_gcp.py --run-name my_song_lora --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BUCKET = "gs://akbar-december-2024-backup"
# One generic root for the whole project; each run gets a subfolder under it.
DEFAULT_BASE = f"{BUCKET}/YuE2-3B_Arabic_Suno_Finetuning"
# Non-run folders that already live under DEFAULT_BASE; never use as a run name.
RESERVED_SUBFOLDERS = {"dataset"}
DEFAULT_LOG = Path("/content/logs/gcp_backup.log")

# FL-YuE2 output roots (its own code: yue2/training/service.py + nodes.py):
#   adapters   = ComfyUI/models/loras/YuE2/<run>/step-NNNNNN.safetensors (+ -nar)
#   run state  = ComfyUI/output/yue2_training/<run>/{run.json,resume.pt,previews}
#   prep cache = ComfyUI/output/yue2_training/prepared (MERT feats + semantic
#                tokens) and .../acoustic_targets (VAE targets)
LORA_ROOT = REPO_ROOT / "ComfyUI" / "models" / "loras"
RUN_ROOT = REPO_ROOT / "ComfyUI" / "output" / "yue2_training"

# (source folder, remote subfolder, wait for writes to settle before syncing)
TARGETS = [
    (LORA_ROOT, "loras", True),
    (RUN_ROOT, "runs", True),
    (Path("/content/logs"), "logs", False),
    (REPO_ROOT / "agent_notes", "agent_notes", False),
]
DEFAULT_EXCLUDES = [r".*\.tmp$", r".*put_loras_here$", r".*put_checkpoints_here$"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-name",
        required=True,
        help="This run's subfolder under <base> (e.g. maqamverse_calib_v1).",
    )
    p.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"GCS root to mirror into (default: {DEFAULT_BASE}).",
    )
    p.add_argument(
        "--interval-minutes",
        type=float,
        default=25.0,
        help="Minutes between passes (default: 25)",
    )
    p.add_argument(
        "--once", action="store_true", help="Run one pass and exit (for cron)."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the sync commands but upload nothing.",
    )
    p.add_argument(
        "--include-cache",
        action="store_true",
        help="Deprecated no-op: the prep cache lives under the mirrored runs/ target.",
    )
    p.add_argument(
        "--settle-seconds",
        type=float,
        default=60.0,
        help="For checkpoint folders, wait until the newest file is this old before syncing (default: 60).",
    )
    p.add_argument("--gsutil", default="gsutil", help="gsutil executable to use.")
    p.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG),
        help=f"Log file (default: {DEFAULT_LOG})",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Extra gsutil -x regex to exclude (repeatable).",
    )
    return p.parse_args()


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("gcp_backup")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def wait_for_settle(folder: Path, seconds: float, logger: logging.Logger) -> None:
    """Block until the newest file in `folder` has been untouched for `seconds`."""
    if seconds <= 0:
        return
    while True:
        newest = max(
            (p.stat().st_mtime for p in folder.rglob("*") if p.is_file()), default=0.0
        )
        age = time.time() - newest
        if age >= seconds:
            return
        wait = min(seconds - age, 5.0)
        logger.info(
            "%s: newest file is %.0fs old, waiting %.0fs for the write to finish",
            folder.name,
            age,
            wait,
        )
        time.sleep(wait)


def sync(src: Path, dst: str, args: argparse.Namespace, logger: logging.Logger) -> bool:
    # gsutil honours only the last -x flag, so combine every pattern into one alternation.
    patterns = DEFAULT_EXCLUDES + args.exclude
    cmd = [
        args.gsutil,
        "-m",
        "rsync",
        "-r",
        "-x",
        "(" + "|".join(patterns) + ")",
        str(src),
        dst.rstrip("/") + "/",
    ]
    logger.info("syncing %s -> %s", src, dst)
    if args.dry_run:
        logger.info("dry-run: %s", " ".join(cmd))
        return True
    started = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "sync failed (exit %d): %s",
            result.returncode,
            (result.stderr or result.stdout).strip()[-2000:],
        )
        return False
    logger.info("ok: %s in %.1fs", src, time.perf_counter() - started)
    return True


def read_remote_json(
    remote: str, args: argparse.Namespace, logger: logging.Logger
) -> dict | None:
    """Return the parsed JSON at `remote`, or None if it isn't there / isn't JSON."""
    result = subprocess.run(
        [args.gsutil, "cat", remote], capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("could not parse %s; treating it as absent", remote)
        return None


def ensure_manifest(
    prefix: str,
    run_name: str,
    targets,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> bool:
    """Write the run marker, refusing to write into a prefix owned by another run."""
    remote = f"{prefix.rstrip('/')}/run_manifest.json"
    existing = read_remote_json(remote, args, logger)
    if existing is not None and existing.get("run_name") != run_name:
        logger.error(
            "run prefix %s already belongs to run %r; refusing to mix.",
            prefix,
            existing.get("run_name"),
        )
        return False
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    manifest = {
        "run_name": run_name,
        "created": (existing or {}).get("created", stamp),
        "updated": stamp,
        "base": args.base,
        "prefix": prefix,
        "targets": [{"source": str(src), "subfolder": sub} for src, sub, _ in targets],
    }
    payload = json.dumps(manifest, indent=2) + "\n"
    if args.dry_run:
        logger.info("dry-run: would write manifest %s", remote)
        return True
    result = subprocess.run(
        [args.gsutil, "cp", "-", remote], input=payload, capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(
            "could not write manifest %s: %s",
            remote,
            (result.stderr or result.stdout).strip()[-2000:],
        )
        return False
    logger.info("run manifest: %s", remote)
    return True


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)

    gsutil = shutil.which(args.gsutil)
    if not gsutil:
        logger.error("could not find '%s' on PATH", args.gsutil)
        return 2
    args.gsutil = gsutil

    if args.run_name in RESERVED_SUBFOLDERS:
        logger.error(
            "%r is a reserved non-run folder under %s; pick another run name.",
            args.run_name,
            DEFAULT_BASE,
        )
        return 3

    prefix = f"{args.base.rstrip('/')}/{args.run_name}"

    targets = list(TARGETS)

    logger.info(
        "backup run %r to %s/%s every %.0f min (once=%s, dry-run=%s)",
        args.run_name,
        args.base.rstrip("/"),
        args.run_name,
        args.interval_minutes,
        args.once,
        args.dry_run,
    )
    for src, sub, _ in targets:
        logger.info("  watching %s -> %s/%s", src, prefix, sub)

    if not ensure_manifest(prefix, args.run_name, targets, args, logger):
        return 3

    try:
        while True:
            failures = 0
            for src, sub, settle in targets:
                if not src.is_dir():
                    logger.warning("skipping missing folder %s", src)
                    continue
                if settle:
                    wait_for_settle(src, args.settle_seconds, logger)
                if not sync(src, f"{prefix}/{sub}", args, logger):
                    failures += 1
            logger.info(
                "pass complete: %d/%d folders synced",
                len(targets) - failures,
                len(targets),
            )
            if args.once:
                break
            time.sleep(max(1.0, args.interval_minutes * 60.0))
    except KeyboardInterrupt:
        logger.info("interrupted; last completed pass is what is in GCS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
