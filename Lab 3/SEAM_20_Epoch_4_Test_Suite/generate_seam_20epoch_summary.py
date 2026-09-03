#!/usr/bin/env python3
import argparse
import csv
import html
import re
import time
from datetime import datetime
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRICS = ("MR", "b-minFDE6", "minADE1", "minADE6", "minFDE1", "minFDE6")
VARIANTS = (
    "baseline",
    "uncertainty_target_context",
    "relative_geometry_bias",
    "qknorm",
)
DISPLAY_NAMES = {
    "baseline": "SEAM baseline",
    "uncertainty_target_context": "Uncertainty-aware target context",
    "relative_geometry_bias": "Relative-geometry attention bias",
    "qknorm": "QKNorm attention",
}
DESCRIPTIONS = {
    "baseline": (
        "The verified SEAM AV2 architecture is retained unchanged and trained for "
        "the common 20-epoch budget."
    ),
    "uncertainty_target_context": (
        "Previous-window modal probabilities control the radius and feature gain of "
        "SEAM's existing endpoint-centred target-context branch."
    ),
    "relative_geometry_bias": (
        "Relative displacement, distance, and heading difference provide a learned "
        "per-head bias in the current-window scene self-attention blocks."
    ),
    "qknorm": (
        "Per-head L2-normalised queries and keys with learned logit scales replace "
        "the attention kernels while preserving SEAM's topology and residual paths."
    ),
}
ERROR_PATTERN = re.compile(
    r"Traceback|CUDA error|out of memory|NCCL WARN|Error executing job|"
    r"RuntimeError|AssertionError|FATAL",
    re.IGNORECASE,
)
WARNING_PATTERN = re.compile(r"UserWarning|FutureWarning|DeprecationWarning")


def read_log(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        data = stream.read()
    return ANSI.sub("", data.decode(errors="replace").replace("\r", "\n"))


def parse_progress(text: str):
    matches = re.findall(r"Epoch\s+(\d+):\s*([0-9]+)%", text)
    if matches:
        return int(matches[-1][0]), int(matches[-1][1])
    return None, None


def parse_metrics(text: str):
    values = {}
    for metric in METRICS:
        table_values = re.findall(
            rf"│\s*{re.escape(metric)}\s*│\s*([-+0-9.eE]+)\s*│", text
        )
        if table_values:
            values[metric] = float(table_values[-1])

    if len(values) < len(METRICS):
        progress_lines = [
            line for line in text.splitlines() if "minADE6=" in line and "Epoch " in line
        ]
        if progress_lines:
            line = progress_lines[-1]
            for metric in METRICS:
                matches = re.findall(rf"{re.escape(metric)}=([0-9]+(?:\.[0-9]+)?)", line)
                if matches and metric not in values:
                    values[metric] = float(matches[-1])
    return values


def checkpoints(run: Path):
    rows = []
    for path in run.rglob("*.ckpt") if run.is_dir() else ():
        epoch_match = re.search(r"epoch[_=](\d+)", path.name)
        metric_match = re.search(r"minADE6[_=]([0-9]+(?:\.[0-9]+)?)", path.name)
        rows.append(
            {
                "path": path,
                "epoch": int(epoch_match.group(1)) if epoch_match else None,
                "minADE6": float(metric_match.group(1)) if metric_match else None,
                "bytes": path.stat().st_size,
            }
        )
    return sorted(rows, key=lambda row: str(row["path"]))


def format_duration(seconds):
    if seconds is None:
        return "-"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    prefix = f"{days}d " if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def read_int(path: Path):
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def collect(results: Path):
    rows = []
    checkpoint_lines = []
    diagnostic_lines = []
    for index, variant in enumerate(VARIANTS, start=1):
        run = results / f"{index:02d}_{variant}"
        text = read_log(run / "train.log")
        epoch, percent = parse_progress(text)
        metrics = parse_metrics(text)
        saved = checkpoints(run)
        scored = [row for row in saved if row["minADE6"] is not None]
        best = min(scored, key=lambda row: row["minADE6"]) if scored else None
        complete = (run / "TRAINING_COMPLETE").is_file()
        if complete:
            status = "complete"
        elif text or saved:
            status = "running/resumable"
        else:
            status = "pending"

        active_seconds = read_int(run / "ACTIVE_SECONDS.txt")
        current_start = read_int(run / "CURRENT_ATTEMPT_START_SECONDS")
        if current_start is not None and (run / "train.log").is_file():
            observed_end = min(int(time.time()), int((run / "train.log").stat().st_mtime))
            active_seconds = (active_seconds or 0) + max(0, observed_end - current_start)
        rows.append(
            {
                "index": index,
                "variant": variant,
                "run": run,
                "status": status,
                "epoch": epoch,
                "percent": percent,
                "metrics": metrics,
                "checkpoints": saved,
                "best": best,
                "active_seconds": active_seconds,
            }
        )
        for saved_row in saved:
            checkpoint_lines.append(
                f"{variant} | {saved_row['bytes']} bytes | epoch={saved_row['epoch']} | "
                f"minADE6={saved_row['minADE6']} | {saved_row['path']}"
            )
        for line in text.splitlines():
            if ERROR_PATTERN.search(line) or WARNING_PATTERN.search(line):
                diagnostic_lines.append(f"{variant}: {line}")
    return rows, checkpoint_lines, diagnostic_lines


def write_variant_summary(row):
    run = row["run"]
    run.mkdir(parents=True, exist_ok=True)
    progress = (
        f"epoch {row['epoch']}, {row['percent']}%"
        if row["epoch"] is not None
        else "not started"
    )
    lines = [
        f"# {DISPLAY_NAMES[row['variant']]}",
        "",
        DESCRIPTIONS[row["variant"]],
        "",
        "## Status",
        "",
        f"- State: **{row['status']}**",
        f"- Progress: {progress}",
        f"- Saved checkpoints: {len(row['checkpoints'])}",
        f"- Active training time: {format_duration(row['active_seconds'])}",
        "",
        "## Metrics",
        "",
        "| MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |",
        "|---:|---:|---:|---:|---:|---:|",
        "| " + " | ".join(
            f"{row['metrics'][metric]:.3f}" if metric in row["metrics"] else "-"
            for metric in METRICS
        ) + " |",
        "",
        "## Best Saved Checkpoint",
        "",
    ]
    if row["best"]:
        lines.extend(
            [
                f"- Epoch: {row['best']['epoch']}",
                f"- minADE6: {row['best']['minADE6']:.6f}",
                f"- Local path: `{row['best']['path']}`",
            ]
        )
    else:
        lines.append("No metric-named checkpoint is available yet.")
    lines.extend(
        [
            "",
            "## Controlled Setup",
            "",
            "All four variants use AV2, 20 epochs, seed 2333, AdamW, peak/minimum "
            "learning rates 1e-3/1e-5, warm-up ratio 0.167, weight decay 1e-2, "
            "gradient clipping 5, and effective global batch 32.",
        ]
    )
    (run / "Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run / "STATUS.txt").write_text(
        "\n".join(
            [
                f"variant={row['variant']}",
                f"status={row['status']}",
                f"epoch={row['epoch'] if row['epoch'] is not None else 'unknown'}",
                f"percent={row['percent'] if row['percent'] is not None else 'unknown'}",
                f"checkpoint_count={len(row['checkpoints'])}",
                f"active_seconds={row['active_seconds'] if row['active_seconds'] is not None else 'unknown'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    text = read_log(run / "train.log")
    (run / "LOG_TAIL.txt").write_text(
        "\n".join(text.splitlines()[-5000:]) + "\n", encoding="utf-8"
    )


def write_svg(rows, output: Path):
    complete = [row for row in rows if "minADE6" in row["metrics"]]
    width, height = 1000, 520
    left, right, top, bottom = 110, 40, 70, 110
    chart_w, chart_h = width - left - right, height - top - bottom
    values = [row["metrics"]["minADE6"] for row in complete]
    max_value = max(values, default=1.0) * 1.12
    bar_w = chart_w / max(1, len(rows)) * 0.55
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111} .axis{stroke:#111;stroke-width:2} .grid{stroke:#bbb;stroke-width:1} .bar{fill:white;stroke:#111;stroke-width:2}</style>',
        '<text x="500" y="34" text-anchor="middle" font-size="24" font-weight="bold">SEAM 20-Epoch Controlled Study: minADE6</text>',
    ]
    for tick in range(6):
        value = max_value * tick / 5
        y = top + chart_h - chart_h * tick / 5
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="16">{value:.2f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}"/>')
    for position, row in enumerate(rows):
        x_center = left + chart_w * (position + 0.5) / len(rows)
        if "minADE6" in row["metrics"]:
            value = row["metrics"]["minADE6"]
            bar_h = chart_h * value / max_value
            parts.append(f'<rect class="bar" x="{x_center-bar_w/2:.1f}" y="{top+chart_h-bar_h:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}"/>')
            parts.append(f'<text x="{x_center:.1f}" y="{top+chart_h-bar_h-10:.1f}" text-anchor="middle" font-size="17">{value:.3f}</text>')
        label = DISPLAY_NAMES[row["variant"]]
        parts.append(f'<text x="{x_center:.1f}" y="{top+chart_h+28}" text-anchor="middle" font-size="14">{html.escape(label)}</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()
    results = Path(args.results_root).resolve()
    rows, checkpoint_lines, diagnostic_lines = collect(results)
    for row in rows:
        write_variant_summary(row)

    complete_count = sum(row["status"] == "complete" for row in rows)
    active = next((row for row in rows if row["status"] == "running/resumable"), None)
    summary = [
        "# SEAM AV2 20-Epoch Controlled Four-Test Study",
        "",
        f"- Updated: `{datetime.now().astimezone().isoformat()}`",
        f"- Completed variants: **{complete_count}/4**",
        f"- Active/resumable variant: **{DISPLAY_NAMES[active['variant']] if active else 'none'}**",
        "",
        "## Results",
        "",
        "| Run | State | Progress | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 | Best saved minADE6 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        progress = f"epoch {row['epoch']} ({row['percent']}%)" if row["epoch"] is not None else "-"
        metrics = [
            f"{row['metrics'][metric]:.3f}" if metric in row["metrics"] else "-"
            for metric in METRICS
        ]
        best = f"{row['best']['minADE6']:.6f}" if row["best"] else "-"
        summary.append(
            f"| {DISPLAY_NAMES[row['variant']]} | {row['status']} | {progress} | "
            + " | ".join(metrics)
            + f" | {best} |"
        )
    summary.extend(
        [
            "",
            "All reported displacement and miss metrics are lower-is-better. Only "
            "within-study comparisons are valid because the four runs share one data "
            "pipeline, schedule, seed, and 20-epoch budget.",
            "",
            "## Training Time",
            "",
            "| Run | Active training time |",
            "|---|---:|",
        ]
    )
    for row in rows:
        summary.append(
            f"| {DISPLAY_NAMES[row['variant']]} | {format_duration(row['active_seconds'])} |"
        )
    summary.extend(
        [
            "",
            "## Interventions",
            "",
        ]
    )
    for row in rows:
        summary.extend(
            [f"### {row['index']:02d}. {DISPLAY_NAMES[row['variant']]}", "", DESCRIPTIONS[row["variant"]], ""]
        )
    summary.extend(
        [
            "## Diagnostics",
            "",
            (
                "No warning or fatal-error signature is present in the retained logs."
                if not diagnostic_lines
                else "Diagnostic signatures are listed in `CURRENT_DIAGNOSTICS.txt`."
            ),
        ]
    )
    (results / "Summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (results / "CHECKPOINTS.txt").write_text(
        "\n".join(checkpoint_lines) + ("\n" if checkpoint_lines else ""), encoding="utf-8"
    )
    (results / "CURRENT_DIAGNOSTICS.txt").write_text(
        "\n".join(diagnostic_lines) + ("\n" if diagnostic_lines else ""), encoding="utf-8"
    )
    (results / "RUN_STATUS.txt").write_text(
        "\n".join(
            [
                f"updated={datetime.now().astimezone().isoformat()}",
                f"completed_variants={complete_count}",
                f"suite_complete={(results / 'SUITE_COMPLETE').is_file()}",
            ]
            + [f"{row['variant']}={row['status']}" for row in rows]
        )
        + "\n",
        encoding="utf-8",
    )

    with (results / "Comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variant", "status", *METRICS, "best_saved_minADE6"))
        for row in rows:
            writer.writerow(
                (
                    row["variant"],
                    row["status"],
                    *(row["metrics"].get(metric, "") for metric in METRICS),
                    row["best"]["minADE6"] if row["best"] else "",
                )
            )
    with (results / "Timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variant", "active_training_seconds", "active_training_time"))
        for row in rows:
            writer.writerow(
                (row["variant"], row["active_seconds"] or "", format_duration(row["active_seconds"]))
            )
    write_svg(rows, results / "minADE6_comparison.svg")
    print(f"SUMMARY_WRITTEN={results / 'Summary.md'}")


if __name__ == "__main__":
    main()
