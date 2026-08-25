#!/usr/bin/env python3
import argparse
import re
from datetime import datetime
from pathlib import Path

from generate_seam_summary import (
    METRICS,
    checkpoint_rows,
    current_epoch,
    parse_validation,
    tail_text,
)


VARIANTS = ("baseline", "mamba_agent_add", "mamba_future_replace")
NAMES = {
    "baseline": "SEAM baseline",
    "mamba_agent_add": "SEAM + agent-history Mamba",
    "mamba_future_replace": "SEAM with future-head Mamba replacement",
}
ERROR_PATTERN = re.compile(
    r"Traceback|CUDA error|out of memory|NCCL WARN|Error executing job|RuntimeError|FATAL",
    re.IGNORECASE,
)


def process_count(results_root: Path) -> int:
    count = 0
    proc = Path("/proc")
    if not proc.is_dir():
        return 0
    needle = str(results_root).encode()
    for entry in proc.glob("[0-9]*"):
        try:
            command = entry.joinpath("cmdline").read_bytes()
        except OSError:
            continue
        if needle in command and b"train.py" in command:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results = Path(args.results_root).resolve()
    experiment = Path(args.experiment_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    checkpoint_lines = []
    error_lines = []

    for variant in VARIANTS:
        run = results / variant
        text = tail_text(run / "train.log")
        epoch, percent = current_epoch(text)
        validations = parse_validation(text)
        latest = validations[-1] if validations else (None, {})
        checkpoints = checkpoint_rows(run) if run.is_dir() else []
        scored = [item for item in checkpoints if item["minADE6"] is not None]
        best = min(scored, key=lambda item: item["minADE6"]) if scored else None
        complete = run.joinpath("TRAINING_COMPLETE").is_file()

        if complete:
            status = "complete"
        elif text or checkpoints:
            status = "running/resumable"
        else:
            status = "pending"

        if epoch is not None:
            progress = f"epoch {epoch} ({percent}%)"
        elif latest[0] is not None:
            progress = f"validated epoch {latest[0]}"
        else:
            progress = "not started"

        rows.append(
            {
                "variant": variant,
                "status": status,
                "epoch": epoch,
                "percent": percent,
                "progress": progress,
                "validation_epoch": latest[0],
                "metrics": latest[1],
                "best": best,
                "checkpoint_count": len(checkpoints),
            }
        )

        for item in checkpoints:
            checkpoint_lines.append(
                f"{variant} | {item['size']} bytes | epoch={item['epoch']} | "
                f"minADE6={item['minADE6']} | {item['path']}"
            )

        matches = [line for line in text.splitlines() if ERROR_PATTERN.search(line)]
        for line in matches[-100:]:
            error_lines.append(f"{variant}: {line}")

    active = process_count(results)
    complete_count = sum(row["status"] == "complete" for row in rows)
    generated = datetime.now().astimezone().isoformat()

    summary = [
        "# SEAM AV2 Three-Run Progress",
        "",
        f"- Generated: `{generated}`",
        f"- Results root: `{results}`",
        f"- Experiment root: `{experiment}`",
        f"- Matching active suite processes detected: **{active}**",
        f"- Completed variants: **{complete_count}/3**",
        "",
        "## Progress",
        "",
        "| Run | Status | Current progress | Checkpoints | Best minADE6 |",
        "|---|---|---:|---:|---:|",
    ]

    for row in rows:
        best = (
            f"{row['best']['minADE6']:.6f} (epoch {row['best']['epoch']})"
            if row["best"]
            else "not available"
        )
        summary.append(
            f"| {NAMES[row['variant']]} | {row['status']} | {row['progress']} | "
            f"{row['checkpoint_count']} | {best} |"
        )

    summary.extend(
        [
            "",
            "## Latest Completed Validation",
            "",
            "| Run | Validation epoch | " + " | ".join(METRICS) + " |",
            "|---|---:|" + "---:|" * len(METRICS),
        ]
    )
    for row in rows:
        values = row["metrics"]
        metric_cells = [
            f"{values[metric]:.3f}" if metric in values else "-" for metric in METRICS
        ]
        summary.append(
            f"| {NAMES[row['variant']]} | {row['validation_epoch'] or '-'} | "
            + " | ".join(metric_cells)
            + " |"
        )

    summary.extend(
        [
            "",
            "## Controlled Setup",
            "",
            "All three runs use AV2, 80 epochs, seed 2333, AdamW, peak/minimum LR "
            "1e-3/1e-5, 13 warm-up epochs, weight decay 1e-2, gradient clipping 5, "
            "and effective global batch 32. The Lab 3 hardware adaptation is two GPUs, "
            "microbatch 8 per GPU, and two-step gradient accumulation.",
            "",
            "## Error Scan",
            "",
        ]
    )
    if error_lines:
        summary.append(
            "Error-pattern lines exist in the saved logs. They may belong to an earlier "
            "failed attempt; inspect `CURRENT_ERRORS.txt` and current process activity."
        )
    else:
        summary.append("No fatal error pattern was found in the retained log tails.")

    output.joinpath("Summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    output.joinpath("CHECKPOINTS.txt").write_text(
        "\n".join(checkpoint_lines) + ("\n" if checkpoint_lines else ""),
        encoding="utf-8",
    )
    output.joinpath("CURRENT_ERRORS.txt").write_text(
        "\n".join(error_lines) + ("\n" if error_lines else ""),
        encoding="utf-8",
    )
    output.joinpath("RUN_STATUS.txt").write_text(
        "\n".join(
            [
                f"generated={generated}",
                f"results_root={results}",
                f"experiment_root={experiment}",
                f"active_suite_processes={active}",
                f"completed_variants={complete_count}",
                f"suite_complete={results.joinpath('SUITE_COMPLETE').is_file()}",
            ]
            + [
                f"{row['variant']}={row['status']};progress={row['progress']};"
                f"last_validation_epoch={row['validation_epoch']}"
                for row in rows
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"SUITE_SUMMARY_WRITTEN={output / 'Summary.md'}")


if __name__ == "__main__":
    main()
