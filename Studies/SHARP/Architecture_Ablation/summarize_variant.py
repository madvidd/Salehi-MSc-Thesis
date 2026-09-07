#!/usr/bin/env python3
"""Write a compact result summary after one variant completes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


METRICS = ("MR", "b-minFDE6", "minADE1", "minADE6", "minFDE1", "minFDE6")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant_root", type=Path)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    checkpoints = sorted(args.variant_root.rglob("*.ckpt"))
    records = []
    for checkpoint in checkpoints:
        match = re.search(r"epoch_(\d+)-minADE6_([0-9]+(?:\.[0-9]+)?)", checkpoint.name)
        if match:
            records.append((float(match.group(2)), int(match.group(1)), checkpoint))
    if not records:
        raise SystemExit("No metric checkpoint was found")
    best_value, best_epoch, best_path = min(records)

    log_path = args.variant_root / "full_run.log"
    text = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    table_values = {}
    for metric in METRICS:
        matches = re.findall(
            rf"[│|]\s*{re.escape(metric)}\s*[│|]\s*([0-9.eE+-]+)", text
        )
        if matches:
            table_values[metric] = matches[-1]

    lines = [
        f"# {args.variant}",
        "",
        f"- Best checkpoint epoch: {best_epoch}",
        f"- Best checkpoint minADE6: {best_value:.9f}",
        f"- Best checkpoint: `{best_path}`",
        f"- Saved checkpoint files: {len(checkpoints)}",
        "",
        "## Final best-checkpoint validation",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {metric} | {table_values.get(metric, 'not parsed')} |" for metric in METRICS)
    lines.extend(
        [
            "",
            "All checkpoints and the full log remain in the local variant result directory.",
        ]
    )
    (args.variant_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.variant_root / "CHECKPOINTS.txt").write_text(
        "\n".join(str(path) for path in checkpoints) + "\n", encoding="utf-8"
    )
    print(f"BEST_CHECKPOINT={best_path}")
    print(f"BEST_MINADE6={best_value:.9f}")


if __name__ == "__main__":
    main()
