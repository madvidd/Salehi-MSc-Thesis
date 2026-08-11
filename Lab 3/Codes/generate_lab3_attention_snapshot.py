#!/usr/bin/env python3

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


METRICS = ("MR", "b-minFDE6", "minADE1", "minADE6", "minFDE1", "minFDE6")
VARIANTS = (
    ("baseline_mha", "Baseline MHA"),
    ("qknorm", "QK-Norm"),
    ("talking_heads", "Talking-Heads"),
    ("qknorm_talking_heads", "QK-Norm + Talking-Heads"),
)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def tail_text(path: Path, limit: int = 32 << 20) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", "replace").replace("\r", "\n")
    except OSError:
        return ""


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


def roots_for(results: Path, key: str) -> list[Path]:
    if key == "baseline_mha":
        roots = [results / key]
        roots.extend(sorted(results.glob("baseline_mha_resume_*")))
        return [root for root in roots if root.is_dir()]
    root = results / key
    return [root] if root.is_dir() else []


def find_logs(roots: list[Path]) -> list[Path]:
    logs = []
    for root in roots:
        logs.extend(root.rglob("*.log"))
    return sorted(set(logs), key=lambda path: path.stat().st_mtime)


def json_metrics(roots: list[Path]) -> dict[str, float]:
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
    candidates = []
    for root in roots:
        candidates.extend(root.rglob("metrics.json"))
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime):
        try:
            visit(json.loads(path.read_text(encoding="utf-8")), found)
        except (OSError, ValueError):
            pass
    return found


def log_state(logs: list[Path]) -> tuple[dict[str, float], int | None, str]:
    values = {}
    latest_epoch = None
    combined = []
    for path in logs[-12:]:
        text = tail_text(path)
        combined.append(text)
        for record in text.splitlines():
            epoch = re.search(r"Epoch\s+(\d+):", record)
            if epoch:
                latest_epoch = max(latest_epoch or 0, int(epoch.group(1)))
            for metric in METRICS:
                pattern = (
                    rf"(?<![A-Za-z0-9_-]){re.escape(metric)}"
                    rf"(?![A-Za-z0-9_])[^0-9\n+-]*([0-9]+(?:\.[0-9]+)?)"
                )
                match = re.search(pattern, record)
                if match:
                    values[metric] = float(match.group(1))
    return values, latest_epoch, "\n".join(combined)


def best_checkpoint(roots: list[Path]):
    candidates = []
    pattern = re.compile(
        r"epoch[_=](\d+).*?minADE6[_=]([0-9]+(?:\.[0-9]+)?)"
    )
    for root in roots:
        for path in root.rglob("*.ckpt"):
            if path.name == "last.ckpt":
                continue
            match = pattern.search(path.name)
            if match:
                candidates.append((float(match.group(2)), int(match.group(1)), path))
    return min(candidates, default=(None, None, None))


def exact_active(commands: str, roots: list[Path]) -> str:
    for root in roots:
        pattern = re.compile(re.escape(str(root)) + r"(?=[/\s]|$)")
        matches = [line for line in commands.splitlines() if pattern.search(line)]
        if matches:
            return "\n".join(matches)
    return ""


def metric_text(value) -> str:
    return "-" if value is None else f"{value:.6f}"


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


def write_terminal(results: Path, destination: Path) -> None:
    suite_logs = sorted(
        results.glob("remaining_attention_suite_*.log"),
        key=lambda path: path.stat().st_mtime,
    )
    with destination.open("w", encoding="utf-8") as output:
        for run_number, path in enumerate(suite_logs, start=1):
            output.write(f"===== SUITE LOG {run_number}: {path} =====\n")
            seen = set()
            for line in iter_records(path):
                if not line:
                    continue
                progress = re.search(r"Epoch\s+(\d+):\s*(\d+)%", line)
                validation = re.search(r"Validation DataLoader.*?(\d+)%", line)
                if progress:
                    epoch, percent = map(int, progress.groups())
                    key = (epoch, percent)
                    if percent % 10 or key in seen:
                        continue
                    seen.add(key)
                elif validation:
                    percent = int(validation.group(1))
                    if percent not in (0, 25, 50, 75, 100):
                        continue
                output.write(line[:8000] + "\n")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: generate_lab3_attention_snapshot.py RESULTS STAGE")
    results = Path(sys.argv[1]).resolve()
    stage = Path(sys.argv[2]).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    commands = process_commands()
    rows = []

    for key, label in VARIANTS:
        roots = roots_for(results, key)
        logs = find_logs(roots)
        log_values, latest_epoch, text = log_state(logs)
        values = log_values
        values.update(json_metrics(roots))
        best_score, best_epoch, _ = best_checkpoint(roots)
        active = exact_active(commands, roots)
        complete = any(root.joinpath("COMPLETE").is_file() for root in roots)
        if key == "baseline_mha" and latest_epoch is not None and latest_epoch >= 79:
            complete = True

        if complete:
            status = "Completed"
        elif active:
            batch = re.search(r"batch_size=(\d+)", active)
            accumulation = re.search(r"accumulate_grad_batches=(\d+)", active)
            details = []
            if batch:
                details.append(f"batch {batch.group(1)}")
            if accumulation:
                details.append(f"accumulation {accumulation.group(1)}")
            status = "Running" + (f" ({', '.join(details)})" if details else "")
        elif "OutOfMemoryError" in text or "CUDA out of memory" in text:
            status = "Stopped after OOM; checkpoint preserved"
        else:
            status = "Saved/inactive"

        rows.append(
            [
                label,
                status,
                str(best_epoch) if best_epoch is not None else "-",
                metric_text(best_score),
                str(latest_epoch) if latest_epoch is not None else "-",
                *[metric_text(values.get(metric)) for metric in METRICS],
            ]
        )

    summary = stage / "Summary.md"
    with summary.open("w", encoding="utf-8") as output:
        output.write("# Lab 3 Attention Experiment Summary\n\n")
        output.write(f"Generated: {datetime.now(timezone.utc).astimezone().isoformat()}\n\n")
        output.write("Lower values are better for every listed metric.\n\n")
        output.write(
            "| Attention | Status | Best epoch | Best minADE6 | Latest epoch | "
            "MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |\n"
        )
        output.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            output.write("| " + " | ".join(row) + " |\n")
        output.write(
            "\nBest minADE6 is read from checkpoint filenames. Remaining metrics use "
            "validated metrics.json values when available, otherwise the latest values "
            "recoverable from logs; they may not belong to the best-minADE6 checkpoint.\n"
        )

    write_terminal(results, stage / "Terminal.txt")


if __name__ == "__main__":
    main()
