#!/usr/bin/env python3
"""Generate a read-only progress snapshot for the final Lab 2 suite."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil


VARIANTS = (
    ("01_official_sharp_baseline", "Official SHARP baseline"),
    ("02_qknorm_uncertainty_geometry", "SHARP + QKNorm + uncertainty + geometry"),
    (
        "03_qknorm_uncertainty_geometry_temporal_mamba",
        "Run 2 + residual temporal-agent Mamba",
    ),
)
METRICS = ("MR", "b-minFDE6", "minADE1", "minADE6", "minFDE1", "minFDE6")
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
PROGRESS = re.compile(r"Epoch\s+(\d+):\s*(\d+)%\|([^\n]*)")
IMPORTANT = re.compile(
    r"FINAL_|SHARP_|RUN_COMPLETE|SUITE_|TRAINING_ATTEMPT|Epoch\s+\d+:\s+100%|"
    r"Validate metric|Trainer\.fit stopped|LOCAL_RANK|CUDA_VISIBLE_DEVICES|"
    r"NCCL version|Trainable params|Total params|checkpoint|Traceback|"
    r"Error executing|ERROR|NCCL WARN|UserWarning|RuntimeWarning|FutureWarning"
)
WARNING = re.compile(
    r"Traceback|Error executing|illegal memory|out of memory|NCCL WARN|"
    r"UserWarning|RuntimeWarning|FutureWarning|DeprecationWarning",
    re.IGNORECASE,
)


def process_commands() -> list[str]:
    commands = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            command = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ")
            text = command.decode(errors="replace").strip()
        except OSError:
            continue
        if "run_final_3run_suite" in text or "train.py" in text:
            commands.append(f"{proc.name}  {text}")
    return sorted(commands, key=lambda value: int(value.split()[0]))


def copy_complete_terminal(source: Path, destination: Path) -> tuple[int, list[str]]:
    warnings: deque[str] = deque(maxlen=200)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", errors="replace", newline=None) as reader:
        with destination.open("w", encoding="utf-8", newline="\n") as writer:
            for raw in reader:
                line = ANSI.sub("", raw.rstrip("\n"))
                writer.write(line + "\n")
                if WARNING.search(line):
                    warnings.append(line)
    return destination.stat().st_size, list(warnings)


def compact_terminal(source: Path, destination: Path) -> str:
    if source.stat().st_size <= 8 * 1024 * 1024:
        shutil.copy2(source, destination)
        return "complete transcript"

    first: list[str] = []
    important: list[str] = []
    last: deque[str] = deque(maxlen=300)
    latest_epoch: dict[int, str] = {}
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for index, raw in enumerate(handle):
            line = raw.rstrip("\n")
            if index < 150:
                first.append(line)
            last.append(line)
            match = PROGRESS.search(line)
            if match:
                latest_epoch[int(match.group(1))] = line
            if IMPORTANT.search(line):
                important.append(line)

    selected = first + important + [latest_epoch[key] for key in sorted(latest_epoch)] + list(last)
    normalized = []
    seen = set()
    for line in selected:
        if line not in seen:
            normalized.append(line)
            seen.add(line)
    destination.write_text("\n".join(normalized) + "\n", encoding="utf-8")
    return "compact transcript; complete transcript retained locally"


def tail_text(path: Path, limit: int = 8 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace").replace("\r", "\n")


def history(path: Path) -> list[dict]:
    records = {}
    if not path.is_file():
        return []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
            records[int(record["epoch"])] = record
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return [records[key] for key in sorted(records)]


def number(value) -> str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if not math.isfinite(result) else f"{result:.4f}"


def checkpoint_summary(root: Path) -> tuple[int, str]:
    checkpoints = list(root.glob("run/checkpoints/*.ckpt"))
    candidates = []
    for path in checkpoints:
        match = re.search(r"minADE6[_=]([0-9]+(?:\.[0-9]+)?)", path.name)
        if match:
            candidates.append((float(match.group(1)), path.name))
    best = min(candidates)[1] if candidates else "N/A"
    return len(checkpoints), best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    args = parser.parse_args()

    results = args.results.resolve()
    suite_log = results / "suite.log"
    if not suite_log.is_file():
        raise SystemExit(f"Suite log was not found: {suite_log}")

    args.stage.mkdir(parents=True, exist_ok=True)
    local_terminal = args.local_dir / "Terminal.txt"
    full_size, warning_lines = copy_complete_terminal(suite_log, local_terminal)
    terminal_mode = compact_terminal(local_terminal, args.stage / "Terminal.txt")

    processes = process_commands()
    process_text = "\n".join(processes) + ("\n" if processes else "No final-suite process found.\n")
    (args.stage / "PROCESS_STATUS.txt").write_text(process_text, encoding="utf-8")

    rows = []
    completed = 0
    active = []
    checkpoint_lines = []
    for slug, title in VARIANTS:
        root = results / slug
        done = (root / "COMPLETED").is_file()
        completed += int(done)
        is_active = any(slug in command for command in processes)
        if is_active:
            active.append(slug)
        records = history(root / "run/metrics_history.jsonl")
        best = min(records, key=lambda item: item.get("minADE6", math.inf)) if records else {}
        latest = records[-1] if records else {}
        progress_matches = PROGRESS.findall(tail_text(root / "full_run.log"))
        progress = f"epoch {progress_matches[-1][0]}, {progress_matches[-1][1]}%" if progress_matches else "not started"
        if done:
            status = "completed"
        elif is_active:
            status = "running"
        elif (root / "full_run.log").is_file():
            status = "saved / awaiting runner"
        else:
            status = "pending"
        count, best_checkpoint = checkpoint_summary(root)
        checkpoint_lines.append(f"{slug}: {count} checkpoint(s); best filename: {best_checkpoint}")
        rows.append(
            f"| {title} | {status} | {progress} | {latest.get('epoch', 'N/A')} | "
            + " | ".join(number(best.get(metric)) for metric in METRICS)
            + f" | {count} |"
        )

    if not active and processes:
        unfinished = [slug for slug, _ in VARIANTS if not (results / slug / "COMPLETED").is_file()]
        if unfinished:
            active.append(unfinished[0] + " (inferred)")

    captured = datetime.now().astimezone().isoformat(timespec="seconds")
    summary = [
        "# Lab 2 Final SHARP Three-Run Progress",
        "",
        f"- Captured: {captured}",
        f"- Results root: `{results}`",
        f"- Completed variants: {completed}/3",
        f"- Active variant: `{', '.join(active) if active else 'none detected'}`",
        f"- Suite complete: {(results / 'SUITE_COMPLETED').is_file()}",
        f"- Complete local transcript: `{local_terminal}` ({full_size} bytes)",
        f"- GitHub Terminal.txt mode: {terminal_mode}",
        "",
        "## Progress And Best Validation Metrics",
        "",
        "All metrics are lower-is-better. Best values are selected by the lowest recorded minADE6 epoch.",
        "",
        "| Run | Status | Current progress | Latest validated epoch | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 | Checkpoints |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Runtime Diagnostics",
        "",
        f"- Matched warning/error lines in the complete transcript: {len(warning_lines)} retained (latest 200 maximum).",
        "- Snapshot generation only read logs and `/proc`; it did not send signals to training.",
    ]
    if warning_lines:
        summary.extend(("", "```text", *warning_lines[-40:], "```"))
    else:
        summary.extend(("", "No matched runtime warnings or errors were found."))
    (args.stage / "Summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (args.stage / "CHECKPOINTS.txt").write_text("\n".join(checkpoint_lines) + "\n", encoding="utf-8")
    (args.stage / "RUN_STATUS.txt").write_text(
        f"captured={captured}\nresults={results}\ncompleted={completed}/3\n"
        f"active={','.join(active) if active else 'none'}\nlocal_terminal={local_terminal}\n",
        encoding="utf-8",
    )
    (args.stage / ".gitignore").write_text(
        "*.ckpt\n*.tar.gz\nfull_run.log\n*Token.txt\n", encoding="utf-8"
    )
    print(f"LOCAL_COMPLETE_TERMINAL={local_terminal}")
    print(f"SNAPSHOT_STAGE={args.stage}")
    print(f"COMPLETED_VARIANTS={completed}/3")
    print(f"ACTIVE_VARIANT={','.join(active) if active else 'none'}")


if __name__ == "__main__":
    main()
