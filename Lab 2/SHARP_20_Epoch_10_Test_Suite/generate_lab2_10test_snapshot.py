#!/usr/bin/env python3
"""Generate a compact, read-only progress snapshot for the Lab 2 suite."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


METRICS = ("MR", "b-minFDE6", "minADE1", "minADE6", "minFDE1", "minFDE6")
VARIANTS = (
    ("01_baseline", "Baseline SHARP"),
    ("02_confidence_gated_memory", "Confidence-gated memory"),
    ("03_cross_window_consistency", "Cross-window consistency"),
    ("04_learned_temporal_pool", "Learned temporal pooling"),
    ("05_uncertainty_target_context", "Uncertainty-aware target context"),
    ("06_relative_geometry_bias", "Relative geometry attention bias"),
    ("07_kinematic_motion_stem", "Kinematic motion stem"),
    ("08_endpoint_refinement_decoder", "Endpoint refinement decoder"),
    ("09_lane_topology_graph", "Lane topology graph"),
    ("10_agent_temporal_mamba", "Agent temporal Mamba"),
)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
CHECKPOINT = re.compile(
    r"epoch[_=](\d+).*?minADE6[_=]([0-9]+(?:\.[0-9]+)?)"
)
PROGRESS = re.compile(
    r"Epoch\s+(\d+):\s*(\d+)%.*?(\d+)/(\d+)(?:.*?([0-9]+(?:\.[0-9]+)?)it/s)?"
)


def tail_text(path: Path, limit: int = 64 << 20) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", "replace").replace("\r", "\n")
    except OSError:
        return ""


def iter_records(path: Path):
    buffer = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            buffer += chunk
            records = re.split(br"[\r\n]+", buffer)
            buffer = records.pop()
            for record in records:
                yield ANSI.sub("", record.decode("utf-8", "replace")).strip()
    if buffer:
        yield ANSI.sub("", buffer.decode("utf-8", "replace")).strip()


def process_commands() -> str:
    commands = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            commands.append(
                process.joinpath("cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", "ignore")
            )
        except OSError:
            pass
    return "\n".join(commands)


def variant_logs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        set(root.rglob("*.log")),
        key=lambda path: path.stat().st_mtime,
    )


def best_checkpoint(root: Path):
    candidates = []
    if root.is_dir():
        for path in root.rglob("*.ckpt"):
            if path.name == "last.ckpt":
                continue
            match = CHECKPOINT.search(path.name)
            if match:
                candidates.append((float(match.group(2)), int(match.group(1)), path))
    return min(candidates, default=(None, None, None))


def checkpoint_count(root: Path) -> int:
    return len(list(root.rglob("*.ckpt"))) if root.is_dir() else 0


def metric_pattern(metric: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z0-9_-]){re.escape(metric)}"
        rf"(?![A-Za-z0-9_])[^0-9\n+-]*([0-9]+(?:\.[0-9]+)?)"
    )


def line_metrics(line: str) -> dict[str, float]:
    values = {}
    for metric in METRICS:
        match = metric_pattern(metric).search(line)
        if match:
            values[metric] = float(match.group(1))
    return values


def markdown_metrics(root: Path) -> dict[str, float]:
    values = {}
    for path in (root / "SUMMARY.md", root / "Summary.md"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for metric in METRICS:
            match = re.search(
                rf"^\|\s*{re.escape(metric)}\s*\|\s*([0-9.eE+-]+)\s*\|",
                text,
                flags=re.MULTILINE,
            )
            if match:
                try:
                    values[metric] = float(match.group(1))
                except ValueError:
                    pass
    return values


def json_metrics(root: Path) -> dict[str, float]:
    def visit(value, found):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in METRICS and isinstance(item, (int, float)):
                    found[key] = float(item)
                visit(item, found)
        elif isinstance(value, list):
            for item in value:
                visit(item, found)

    found = {}
    candidates = list(root.rglob("metrics.json")) if root.is_dir() else []
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime):
        try:
            visit(json.loads(path.read_text(encoding="utf-8")), found)
        except (OSError, ValueError):
            pass
    return found


def log_state(logs: list[Path], best_score: float | None):
    latest_epoch = None
    latest_percent = None
    latest_step = None
    latest_total = None
    latest_speed = None
    metric_records = []
    text_parts = []
    table_values = {}

    for path in logs[-8:]:
        text = tail_text(path)
        text_parts.append(text)
        for line in text.splitlines():
            progress = PROGRESS.search(line)
            if progress:
                epoch = int(progress.group(1))
                if latest_epoch is None or epoch >= latest_epoch:
                    latest_epoch = epoch
                    latest_percent = int(progress.group(2))
                    latest_step = int(progress.group(3))
                    latest_total = int(progress.group(4))
                    latest_speed = progress.group(5)

            values = line_metrics(line)
            if len(values) >= 4:
                metric_records.append(values)

            table = re.match(
                r"^\s*[|\u2502].*?"
                r"(MR|b-minFDE6|minADE1|minADE6|minFDE1|minFDE6)"
                r"\s*[|\u2502].*?([0-9]+(?:\.[0-9]+)?)",
                line,
            )
            if table:
                table_values[table.group(1)] = float(table.group(2))
                if len(table_values) == len(METRICS):
                    metric_records.append(dict(table_values))
                    table_values.clear()

    if metric_records:
        if best_score is None:
            values = metric_records[-1]
        else:
            values = min(
                metric_records,
                key=lambda record: abs(record.get("minADE6", 1e9) - best_score),
            )
    else:
        values = {}

    progress = None
    if latest_epoch is not None:
        progress = {
            "epoch": latest_epoch,
            "percent": latest_percent,
            "step": latest_step,
            "total": latest_total,
            "speed": latest_speed,
        }
    return values, progress, "\n".join(text_parts)


def is_active(commands: str, root: Path) -> bool:
    if not root.is_dir():
        return False
    pattern = re.compile(re.escape(str(root.resolve())) + r"(?=[/\s]|$)")
    return any(pattern.search(line) for line in commands.splitlines())


def status_for(root: Path, active: bool, text: str, progress) -> str:
    if (root / "COMPLETED").is_file():
        return "Completed"
    if active:
        if progress:
            return (
                f"Running (epoch index {progress['epoch']}/19, "
                f"{progress['percent']}%)"
            )
        return "Running"
    if re.search(
        r"Traceback|CUDA error|illegal memory access|OutOfMemoryError|CUDA out of memory|VARIANT_STOPPED",
        text,
        flags=re.IGNORECASE,
    ):
        if (root / "run" / "checkpoints" / "last.ckpt").is_file():
            return "Stopped/error; resumable from last checkpoint"
        return "Stopped/error before checkpoint; restart required"
    if (root / "run" / "checkpoints" / "last.ckpt").is_file():
        return "Saved; resumable"
    if root.is_dir() and any(root.iterdir()):
        return "Started/inactive"
    return "Pending"


def metric_text(value) -> str:
    return "-" if value is None else f"{value:.6f}"


def progress_text(progress) -> str:
    if not progress:
        return "-"
    detail = f"epoch index {progress['epoch']}/19"
    if progress["percent"] is not None:
        detail += f", {progress['percent']}%"
    if progress["step"] is not None and progress["total"] is not None:
        detail += f" ({progress['step']}/{progress['total']})"
    return detail


def write_terminal(source: Path, destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as output:
        output.write(f"===== COMPACTED SUITE LOG: {source} =====\n")
        progress_seen = set()
        validation_seen = set()
        for line in iter_records(source):
            if not line:
                continue
            progress = re.search(r"Epoch\s+(\d+):\s*(\d+)%", line)
            validation = re.search(r"Validation DataLoader.*?(\d+)%", line)
            if progress:
                epoch, percent = map(int, progress.groups())
                key = (epoch, percent)
                if percent % 10 or key in progress_seen:
                    continue
                progress_seen.add(key)
            elif validation:
                percent = int(validation.group(1))
                key = (line[:40], percent)
                if percent not in (0, 25, 50, 75, 100) or key in validation_seen:
                    continue
                validation_seen.add(key)
            output.write(line[:8000] + "\n")


def write_errors(source: Path, destination: Path) -> None:
    pattern = re.compile(
        r"Traceback|Error executing|CUDA error|illegal memory access|"
        r"OutOfMemoryError|CUDA out of memory|NCCL WARN|VARIANT_STOPPED|"
        r"UserWarning|WARNING",
        flags=re.IGNORECASE,
    )
    matches = []
    for line in iter_records(source):
        if pattern.search(line):
            matches.append(line[:8000])
    destination.write_text("\n".join(matches[-1000:]) + ("\n" if matches else ""), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: generate_lab2_10test_snapshot.py RESULTS SUITE_LOG STAGE"
        )
    results = Path(sys.argv[1]).resolve()
    suite_log = Path(sys.argv[2]).resolve()
    stage = Path(sys.argv[3]).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    commands = process_commands()
    rows = []
    completed = 0
    running = []

    for key, label in VARIANTS:
        root = results / key
        best_score, best_epoch, _ = best_checkpoint(root)
        logs = variant_logs(root)
        log_values, progress, text = log_state(logs, best_score)
        values = dict(log_values)
        values.update(markdown_metrics(root))
        values.update(json_metrics(root))
        if best_score is not None:
            values["minADE6"] = best_score
        active = is_active(commands, root)
        status = status_for(root, active, text, progress)
        completed += status == "Completed"
        if active:
            running.append(label)
        rows.append(
            {
                "label": label,
                "status": status,
                "progress": progress_text(progress),
                "best_epoch": best_epoch,
                "checkpoints": checkpoint_count(root),
                "values": values,
            }
        )

    baseline = rows[0]["values"].get("minADE6")
    with (stage / "Summary.md").open("w", encoding="utf-8") as output:
        output.write("# Lab 2 SHARP 20-Epoch Ten-Test Summary\n\n")
        output.write(
            f"Generated: {datetime.now(timezone.utc).astimezone().isoformat()}\n\n"
        )
        output.write(
            "All variants use AV2, seed 2333, 20 epochs, global batch 32 "
            "(8 per GPU across four RTX 2080 Ti GPUs), AdamW, LR 1e-4 to "
            "1e-5, 13 warm-up epochs, and SyncBatchNorm. Lower is better for "
            "all metrics.\n\n"
        )
        output.write(
            "| Test | Status | Live progress | Best epoch | Checkpoints | MR | "
            "b-minFDE6 | minADE1 | minADE6 | Delta minADE6 vs baseline | "
            "minFDE1 | minFDE6 |\n"
        )
        output.write(
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for row in rows:
            score = row["values"].get("minADE6")
            delta = "-"
            if baseline is not None and score is not None:
                delta = f"{score - baseline:+.6f}"
            fields = [
                row["label"],
                row["status"],
                row["progress"],
                str(row["best_epoch"]) if row["best_epoch"] is not None else "-",
                str(row["checkpoints"]),
                metric_text(row["values"].get("MR")),
                metric_text(row["values"].get("b-minFDE6")),
                metric_text(row["values"].get("minADE1")),
                metric_text(score),
                delta,
                metric_text(row["values"].get("minFDE1")),
                metric_text(row["values"].get("minFDE6")),
            ]
            output.write("| " + " | ".join(fields) + " |\n")
        output.write(
            f"\nCompleted variants: {completed}/10. "
            f"Currently active: {', '.join(running) if running else 'none detected'}.\n\n"
        )
        output.write(
            "`Delta minADE6 vs baseline` is variant minADE6 minus baseline "
            "minADE6; a negative value is better. Best minADE6 is read from "
            "checkpoint filenames. The other metrics use a saved variant summary "
            "when available, otherwise the closest matching or latest validation "
            "record in the logs. Lightning labels the 20 epochs from 0 through 19. "
            "Running results are provisional. This is a "
            "20-epoch screening suite and should not be presented as equivalent to "
            "SHARP's full 80-epoch result.\n"
        )

    with (stage / "VARIANT_STATUS.txt").open("w", encoding="utf-8") as output:
        output.write(f"Results root: {results}\n")
        output.write(f"Completed: {completed}/10\n")
        output.write(f"Active: {', '.join(running) if running else 'none detected'}\n\n")
        for row in rows:
            output.write(
                f"{row['label']}: {row['status']}; {row['progress']}; "
                f"best_epoch={row['best_epoch']}; "
                f"best_minADE6={metric_text(row['values'].get('minADE6'))}; "
                f"checkpoints={row['checkpoints']}\n"
            )

    write_terminal(suite_log, stage / "Terminal.txt")
    write_errors(suite_log, stage / "CURRENT_WARNINGS_AND_ERRORS.txt")


if __name__ == "__main__":
    main()
