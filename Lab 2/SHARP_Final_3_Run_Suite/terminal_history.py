#!/usr/bin/env python3
"""Maintain append-only local and GitHub terminal transcripts."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re


ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
PREFIX_BYTES = 64 * 1024


def source_identity(path: Path, prefix_length: int | None = None) -> dict[str, object]:
    stat = path.stat()
    length = min(stat.st_size, PREFIX_BYTES if prefix_length is None else prefix_length)
    with path.open("rb") as handle:
        prefix = handle.read(length)
    return {
        "source": str(path.resolve()),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "prefix_length": length,
        "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        "size": stat.st_size,
    }


def normalized_text(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace").replace("\r", "\n")
    return "\n".join(ANSI.sub("", line) for line in text.splitlines())


def append_source(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    destination = args.destination.resolve()
    state_path = args.state.resolve()
    if not source.is_file():
        raise SystemExit(f"Terminal source was not found: {source}")

    previous = {}
    if state_path.is_file():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}

    identity = source_identity(source)
    previous_prefix_length = int(previous.get("prefix_length", 0))
    comparable = source_identity(source, previous_prefix_length)
    same_source = (
        all(previous.get(key) == comparable[key] for key in ("source", "device", "inode"))
        and previous.get("prefix_length") == comparable["prefix_length"]
        and previous.get("prefix_sha256") == comparable["prefix_sha256"]
    )
    previous_offset = int(previous.get("offset", 0)) if same_source else 0
    if previous_offset < 0 or previous_offset > int(identity["size"]):
        previous_offset = 0
        same_source = False

    with source.open("rb") as handle:
        handle.seek(previous_offset)
        addition = normalized_text(handle.read())

    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.is_file() else 0
    if existing_size and addition.startswith("\n"):
        addition = addition[1:]
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        if existing_size and previous_offset == 0:
            handle.write(
                f"\n===== SOURCE RESTART OR FIRST TRACKED SNAPSHOT: {args.label} =====\n"
            )
        if addition:
            handle.write(addition)
            handle.write("\n")

    state = {
        **identity,
        "offset": identity["size"],
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"LOCAL_TERMINAL={destination}")
    print(f"SOURCE_BYTES_APPENDED={int(identity['size']) - previous_offset}")
    print(f"LOCAL_TERMINAL_BYTES={destination.stat().st_size}")


def merge_unique(args: argparse.Namespace) -> None:
    previous = args.previous.resolve()
    current = args.current.resolve()
    output = args.output.resolve()
    if not current.is_file():
        raise SystemExit(f"Current terminal snapshot was not found: {current}")

    previous_bytes = previous.read_bytes() if previous.is_file() else b""
    previous_text = previous_bytes.decode("utf-8", errors="replace").replace("\r", "\n")
    current_text = current.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    seen = set(previous_text.splitlines())
    additions = []
    for line in current_text.splitlines():
        clean = ANSI.sub("", line)
        if clean not in seen:
            additions.append(clean)
            seen.add(clean)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(previous_bytes)
        if previous_bytes and not previous_bytes.endswith((b"\n", b"\r")):
            handle.write(b"\n")
        if additions:
            marker = f"===== APPENDED SNAPSHOT: {args.label} =====\n"
            handle.write(marker.encode("utf-8"))
            handle.write(("\n".join(additions) + "\n").encode("utf-8"))

    print(f"PREVIOUS_LINES={len(previous_text.splitlines())}")
    print(f"NEW_UNIQUE_LINES={len(additions)}")
    print(f"MERGED_TERMINAL_BYTES={output.stat().st_size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    append_parser = subparsers.add_parser("append-source")
    append_parser.add_argument("--source", required=True, type=Path)
    append_parser.add_argument("--destination", required=True, type=Path)
    append_parser.add_argument("--state", required=True, type=Path)
    append_parser.add_argument("--label", required=True)
    append_parser.set_defaults(function=append_source)

    merge_parser = subparsers.add_parser("merge-unique")
    merge_parser.add_argument("--previous", required=True, type=Path)
    merge_parser.add_argument("--current", required=True, type=Path)
    merge_parser.add_argument("--output", required=True, type=Path)
    merge_parser.add_argument("--label", required=True)
    merge_parser.set_defaults(function=merge_unique)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
