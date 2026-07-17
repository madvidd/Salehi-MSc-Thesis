#!/usr/bin/env python3
import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: find_best_checkpoint.py CHECKPOINT_DIRECTORY", file=sys.stderr)
        return 2
    directory = Path(sys.argv[1])
    candidates = []
    for path in directory.glob("*.ckpt"):
        match = re.search(r"minADE6_([0-9]+(?:\.[0-9]+)?)", path.name)
        if match:
            candidates.append((float(match.group(1)), path.resolve()))
    if not candidates:
        print(f"No scored checkpoint found in {directory}", file=sys.stderr)
        return 1
    print(min(candidates)[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
