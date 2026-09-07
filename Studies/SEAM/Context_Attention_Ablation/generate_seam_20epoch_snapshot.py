#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
VARIANTS = (
    "01_baseline",
    "02_uncertainty_target_context",
    "03_relative_geometry_bias",
    "04_qknorm",
)


def normalise_stream(source: Path, destination: Path) -> None:
    with source.open("r", encoding="utf-8", errors="replace", newline=None) as reader:
        with destination.open("w", encoding="utf-8", newline="\n") as writer:
            for line in reader:
                writer.write(ANSI.sub("", line))


def is_prefix(prefix: Path, full: Path) -> bool:
    if prefix.stat().st_size > full.stat().st_size:
        return False
    with prefix.open("rb") as left, full.open("rb") as right:
        while True:
            block = left.read(1024 * 1024)
            if not block:
                return True
            if right.read(len(block)) != block:
                return False


def merge_terminal(previous: Path, current: Path, label: str) -> None:
    previous.parent.mkdir(parents=True, exist_ok=True)
    if not previous.is_file() or is_prefix(previous, current):
        os.replace(current, previous)
        return
    with previous.open("ab") as writer, current.open("rb") as reader:
        if previous.stat().st_size:
            writer.write(b"\n")
        writer.write(f"===== APPENDED SNAPSHOT {label} =====\n".encode())
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    current.unlink()


def copy_if_small(source: Path, destination: Path, limit=10 * 1024 * 1024):
    if source.is_file() and source.stat().st_size <= limit:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def count_processes(results: Path) -> int:
    proc = Path("/proc")
    if not proc.is_dir():
        return 0
    needle = str(results).encode()
    count = 0
    for entry in proc.glob("[0-9]*"):
        try:
            command = entry.joinpath("cmdline").read_bytes()
        except OSError:
            continue
        if needle in command and (b"train.py" in command or b"run_seam_20epoch" in command):
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--stage-dir", required=True)
    args = parser.parse_args()

    results = Path(args.results_root).resolve()
    experiment = Path(args.experiment_root).resolve()
    local = Path(args.local_dir).resolve()
    stage = Path(args.stage_dir).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().isoformat()

    suite_log = results / "suite.log"
    if suite_log.is_file():
        with tempfile.NamedTemporaryFile(
            dir=local, prefix="Terminal.current.", suffix=".txt", delete=False
        ) as handle:
            current_terminal = Path(handle.name)
        normalise_stream(suite_log, current_terminal)
        merge_terminal(local / "Terminal.txt", current_terminal, stamp)

    top_level = (
        "Summary.md",
        "Comparison.csv",
        "Timing.csv",
        "minADE6_comparison.svg",
        "RUN_STATUS.txt",
        "CHECKPOINTS.txt",
        "CURRENT_DIAGNOSTICS.txt",
        "RUN_MANIFEST.json",
        "DATA_ROOT.txt",
        "EXPERIMENT_ROOT.txt",
        "SUITE_COMPLETE",
    )
    for name in top_level:
        copy_if_small(results / name, stage / name)

    copy_if_small(experiment / "EXPERIMENT_MANIFEST.md", stage / "EXPERIMENT_MANIFEST.md")
    copy_if_small(experiment / "SOURCE_PROVENANCE.md", stage / "SOURCE_PROVENANCE.md")
    for variant_dir in VARIANTS:
        source = results / variant_dir
        destination = stage / "Variants" / variant_dir
        for name in (
            "Summary.md",
            "STATUS.txt",
            "CONTROLLED_CONFIGURATION.txt",
            "LOG_TAIL.txt",
            "ACTIVE_SECONDS.txt",
            "ATTEMPTS.tsv",
            "TRAINING_COMPLETE",
        ):
            copy_if_small(source / name, destination / name)

    terminal = local / "Terminal.txt"
    max_terminal = 9 * 1024 * 1024
    if terminal.is_file() and terminal.stat().st_size <= max_terminal:
        shutil.copy2(terminal, stage / "Terminal.txt")
    elif terminal.is_file():
        data = terminal.read_bytes()
        compact = (
            b"COMPACT_GITHUB_TRANSCRIPT=True\n"
            + f"Full local path: {terminal}\nFull bytes: {len(data)}\n".encode()
            + b"The beginning and latest log output follow.\n\n"
            + data[: 512 * 1024]
            + b"\n\n===== OMITTED MIDDLE OF FULL LOCAL TRANSCRIPT =====\n\n"
            + data[-8 * 1024 * 1024 :]
        )
        (stage / "Terminal.txt").write_bytes(compact)

    (stage / "PROCESS_STATUS.txt").write_text(
        "\n".join(
            [
                f"generated={stamp}",
                f"matching_suite_processes={count_processes(results)}",
                f"results_root={results}",
                f"experiment_root={experiment}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (stage / ".gitignore").write_text(
        "*.ckpt\n*.pt\n*.pth\n*.tar.gz\ntrain.log\nToken.txt\n",
        encoding="utf-8",
    )
    print(f"LOCAL_TERMINAL={terminal}")
    print(f"SNAPSHOT_STAGE={stage}")


if __name__ == "__main__":
    main()
