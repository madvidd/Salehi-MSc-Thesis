#!/usr/bin/env python3
"""Generate dissertation-ready tables, plots, summaries, and compact logs."""

from __future__ import annotations

import argparse
import binascii
import csv
import html
import json
import math
import re
import shutil
import struct
import zlib
from pathlib import Path


METRICS = ("MR", "b-minFDE6", "minADE1", "minADE6", "minFDE1", "minFDE6")
ARTICLE = {
    "MR": 0.140,
    "b-minFDE6": 1.822,
    "minADE1": 1.569,
    "minADE6": 0.639,
    "minFDE1": 3.850,
    "minFDE6": 1.197,
}
VARIANT_DETAILS = {
    "01_official_sharp_baseline": (
        "Official SHARP baseline",
        "Pinned official architecture with the paper's AV2 optimization schedule.",
    ),
    "02_qknorm_uncertainty_geometry": (
        "SHARP + QKNorm + uncertainty context + geometry bias",
        "All SHARP MHA constructors use QKNorm; streamed target context adapts to mode uncertainty; scene attention receives a learned relative-pose bias.",
    ),
    "03_qknorm_uncertainty_geometry_temporal_mamba": (
        "Run 2 + residual temporal-agent Mamba",
        "Run 2 plus a small bidirectional residual fused-CUDA Mamba branch between temporal-agent encoder blocks 2 and 3.",
    ),
}

FONT = {
    " ": ("00000",) * 7,
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
}


class RasterCanvas:
    """Minimal dependency-free RGB canvas and PNG writer."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray([255]) * (width * height * 3)

    def pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset : offset + 3] = bytes(color)

    def line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        width: int = 1,
        dashed: bool = False,
    ) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        step = 0
        while True:
            if not dashed or (step // 6) % 2 == 0:
                radius = max(0, width // 2)
                for yy in range(y0 - radius, y0 + radius + 1):
                    for xx in range(x0 - radius, x0 + radius + 1):
                        self.pixel(xx, yy, color)
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy
            step += 1

    def rectangle(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[int, int, int],
    ) -> None:
        for yy in range(max(0, y), min(self.height, y + height)):
            start = (yy * self.width + max(0, x)) * 3
            end_x = min(self.width, x + width)
            if end_x > max(0, x):
                self.pixels[start : (yy * self.width + end_x) * 3] = bytes(color) * (
                    end_x - max(0, x)
                )

    def text(
        self,
        x: int,
        y: int,
        value: str,
        color: tuple[int, int, int] = (35, 43, 48),
        scale: int = 2,
    ) -> None:
        cursor = x
        for character in value.upper():
            glyph = FONT.get(character, FONT["?"])
            for row, bits in enumerate(glyph):
                for column, bit in enumerate(bits):
                    if bit == "1":
                        self.rectangle(
                            cursor + column * scale,
                            y + row * scale,
                            scale,
                            scale,
                            color,
                        )
            cursor += 6 * scale

    @staticmethod
    def _chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
        )

    def save(self, path: Path) -> None:
        rows = []
        stride = self.width * 3
        for y in range(self.height):
            rows.append(b"\x00" + bytes(self.pixels[y * stride : (y + 1) * stride]))
        payload = b"".join(rows)
        png = b"\x89PNG\r\n\x1a\n"
        png += self._chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),
        )
        png += self._chunk(b"IDAT", zlib.compress(payload, 9))
        png += self._chunk(b"IEND", b"")
        path.write_bytes(png)


def finite_range(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    low, high = min(finite), max(finite)
    if low == high:
        padding = max(abs(low) * 0.1, 0.1)
    else:
        padding = (high - low) * 0.12
    return max(0.0, low - padding), high + padding


def chart_layout(count: int) -> tuple[int, int, int, int]:
    columns = 3 if count > 1 else 1
    rows = math.ceil(count / columns)
    width = 1320 if columns == 3 else 900
    height = 70 + rows * 340
    return columns, rows, width, height


def history_charts(
    history: list[dict],
    metrics: tuple[str, ...],
    png_path: Path,
    svg_path: Path,
    title: str,
) -> None:
    columns, _, width, height = chart_layout(len(metrics))
    canvas = RasterCanvas(width, height)
    canvas.text(24, 20, title, scale=2)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="32" font-family="sans-serif" font-size="20" font-weight="600">{html.escape(title)}</text>',
    ]
    epochs = [int(record["epoch"]) + 1 for record in history]
    for index, metric in enumerate(metrics):
        column = index % columns
        row = index // columns
        panel_x = 20 + column * 430
        panel_y = 65 + row * 340
        left, top = panel_x + 58, panel_y + 42
        plot_width, plot_height = 340 if columns == 3 else 790, 235
        values = [float(record.get(metric, math.nan)) for record in history]
        low, high = finite_range(values + [ARTICLE[metric]])
        x_min, x_max = min(epochs), max(epochs)

        def px(epoch: int) -> int:
            return left + int((epoch - x_min) / max(1, x_max - x_min) * plot_width)

        def py(value: float) -> int:
            return top + plot_height - int((value - low) / (high - low) * plot_height)

        canvas.text(panel_x + 8, panel_y + 7, metric, scale=2)
        canvas.line(left, top, left, top + plot_height, (75, 82, 87))
        canvas.line(left, top + plot_height, left + plot_width, top + plot_height, (75, 82, 87))
        canvas.text(panel_x + 2, top - 5, f"{high:.3f}", scale=1)
        canvas.text(panel_x + 2, top + plot_height - 6, f"{low:.3f}", scale=1)
        article_y = py(ARTICLE[metric])
        canvas.line(left, article_y, left + plot_width, article_y, (192, 57, 43), dashed=True)
        svg.extend(
            (
                f'<text x="{panel_x + 8}" y="{panel_y + 24}" font-family="sans-serif" font-size="16" font-weight="600">{html.escape(metric)}</text>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#4B5257"/>',
                f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#4B5257"/>',
                f'<line x1="{left}" y1="{article_y}" x2="{left + plot_width}" y2="{article_y}" stroke="#C0392B" stroke-dasharray="6 5"/>',
                f'<text x="{panel_x + 3}" y="{top}" font-family="monospace" font-size="10">{high:.3f}</text>',
                f'<text x="{panel_x + 3}" y="{top + plot_height}" font-family="monospace" font-size="10">{low:.3f}</text>',
            )
        )
        points = []
        previous = None
        for epoch, value in zip(epochs, values):
            if not math.isfinite(value):
                previous = None
                continue
            point = (px(epoch), py(value))
            points.append(point)
            if previous is not None:
                canvas.line(*previous, *point, (23, 107, 135), width=2)
            previous = point
        if points:
            svg.append(
                '<polyline fill="none" stroke="#176B87" stroke-width="2" points="'
                + " ".join(f"{x},{y}" for x, y in points)
                + '"/>'
            )
        canvas.text(left, top + plot_height + 12, f"EPOCH {x_min}", scale=1)
        canvas.text(left + plot_width - 70, top + plot_height + 12, f"{x_max}", scale=1)
        svg.append(
            f'<text x="{left}" y="{top + plot_height + 22}" font-family="sans-serif" font-size="11">Epoch {x_min} to {x_max}</text>'
        )
    svg.append("</svg>")
    canvas.save(png_path)
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def comparison_charts(
    records: list[tuple[str, str, dict]],
    png_path: Path,
    svg_path: Path,
) -> None:
    columns, _, width, height = chart_layout(len(METRICS))
    canvas = RasterCanvas(width, height)
    title = "FINAL SHARP AV2 COMPARISON - LOWER IS BETTER"
    canvas.text(24, 20, title, scale=2)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="32" font-family="sans-serif" font-size="20" font-weight="600">{title}</text>',
    ]
    colors = ((127, 140, 141), (23, 107, 135), (46, 139, 87), (180, 95, 6))
    hex_colors = ("#7F8C8D", "#176B87", "#2E8B57", "#B45F06")
    for index, metric in enumerate(METRICS):
        column, row = index % columns, index // columns
        panel_x, panel_y = 20 + column * 430, 65 + row * 340
        left, top, plot_width, plot_height = panel_x + 58, panel_y + 42, 340, 235
        values = [ARTICLE[metric]] + [float(data.get(metric, math.nan)) for _, _, data in records]
        _, high = finite_range(values)
        canvas.text(panel_x + 8, panel_y + 7, metric, scale=2)
        canvas.line(left, top, left, top + plot_height, (75, 82, 87))
        canvas.line(left, top + plot_height, left + plot_width, top + plot_height, (75, 82, 87))
        svg.extend(
            (
                f'<text x="{panel_x + 8}" y="{panel_y + 24}" font-family="sans-serif" font-size="16" font-weight="600">{html.escape(metric)}</text>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#4B5257"/>',
                f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#4B5257"/>',
            )
        )
        slot = plot_width / len(values)
        for item_index, value in enumerate(values):
            if not math.isfinite(value):
                continue
            bar_width = max(16, int(slot * 0.58))
            x = int(left + item_index * slot + (slot - bar_width) / 2)
            bar_height = int(value / high * plot_height)
            y = top + plot_height - bar_height
            canvas.rectangle(x, y, bar_width, bar_height, colors[item_index])
            label = "A" if item_index == 0 else str(item_index)
            canvas.text(x + max(0, bar_width // 2 - 3), top + plot_height + 12, label, scale=1)
            svg.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{hex_colors[item_index]}"/>'
            )
            svg.append(
                f'<text x="{x + bar_width / 2}" y="{top + plot_height + 20}" text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>'
            )
        canvas.text(panel_x + 2, top - 5, f"{high:.3f}", scale=1)
        svg.append(
            f'<text x="{panel_x + 3}" y="{top}" font-family="monospace" font-size="10">{high:.3f}</text>'
        )
    svg.append("</svg>")
    canvas.save(png_path)
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(path: Path) -> list[dict]:
    records: dict[int, dict] = {}
    if not path.is_file():
        return []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
            records[int(record["epoch"])] = record
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return [records[key] for key in sorted(records)]


def format_value(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def write_history(history: list[dict], artifacts: Path) -> None:
    csv_path = artifacts / "metrics_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "global_step", *METRICS))
        writer.writeheader()
        for record in history:
            writer.writerow({name: record.get(name, "") for name in writer.fieldnames})

    lines = [
        "| Epoch | " + " | ".join(METRICS) + " |",
        "|---:|" + "---:|" * len(METRICS),
    ]
    for record in history:
        lines.append(
            f"| {record['epoch']} | "
            + " | ".join(format_value(record.get(name)) for name in METRICS)
            + " |"
        )
    (artifacts / "metrics_history.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_history(history: list[dict], artifacts: Path) -> None:
    if not history:
        return
    history_charts(
        history,
        METRICS,
        artifacts / "training_metrics.png",
        artifacts / "training_metrics.svg",
        "SHARP AV2 VALIDATION METRICS - DASHED LINE IS ARTICLE",
    )
    history_charts(
        history,
        ("minADE6",),
        artifacts / "minADE6_curve.png",
        artifacts / "minADE6_curve.svg",
        "AV2 MINADE6 TRAINING CURVE",
    )


def compact_terminal(source: Path, destination: Path) -> None:
    if not source.is_file():
        destination.write_text("Training log was not found.\n", encoding="utf-8")
        return
    text = source.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    lines = text.splitlines()
    pattern = re.compile(
        r"(FINAL_|SHARP_|Validate metric|"
        r"\bMR\b|b-minFDE6|minADE1|minADE6|minFDE1|minFDE6|"
        r"Traceback|Error executing|ERROR|WARN|Warning|checkpoint|"
        r"LOCAL_RANK|CUDA_VISIBLE_DEVICES|NCCL version|Trainer.fit stopped|"
        r"Trainable params|Total params|Experiments are stored)"
    )
    selected = lines[:120]
    epoch_summaries: dict[int, str] = {}
    validation_summary = ""
    for line in lines[120:-200]:
        clean = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", line)
        epoch = re.match(r"Epoch\s+(\d+):", clean)
        if epoch:
            if "100%" in clean:
                epoch_summaries[int(epoch.group(1))] = clean
            continue
        if clean.startswith("Validation DataLoader"):
            if "100%" in clean:
                validation_summary = clean
            continue
        if pattern.search(clean):
            selected.append(clean)
    selected.extend(epoch_summaries[epoch] for epoch in sorted(epoch_summaries))
    if validation_summary:
        selected.append(validation_summary)
    selected.extend(lines[-200:])
    normalized = []
    seen = set()
    for line in selected:
        line = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", line)
        if line not in seen:
            normalized.append(line)
            seen.add(line)

    max_bytes = 8 * 1024 * 1024
    payload = "\n".join(normalized) + "\n"
    if len(payload.encode("utf-8")) > max_bytes:
        marker = "[middle of compact transcript omitted to satisfy the 8 MiB GitHub limit]"
        head = []
        used = len((marker + "\n").encode("utf-8"))
        head_budget = max_bytes // 3
        for line in normalized:
            encoded = len((line + "\n").encode("utf-8"))
            if used + encoded > head_budget:
                break
            head.append(line)
            used += encoded
        tail = []
        for line in reversed(normalized[len(head):]):
            encoded = len((line + "\n").encode("utf-8"))
            if used + encoded > max_bytes:
                break
            tail.append(line)
            used += encoded
        payload = "\n".join(head + [marker] + list(reversed(tail))) + "\n"
    destination.write_text(payload, encoding="utf-8")


def warning_report(source: Path, destination: Path) -> None:
    if not source.is_file():
        destination.write_text("Log unavailable.\n", encoding="utf-8")
        return
    pattern = re.compile(
        r"Traceback|Error executing|illegal memory|out of memory|NCCL WARN|"
        r"UserWarning|RuntimeWarning|FutureWarning|DeprecationWarning",
        re.IGNORECASE,
    )
    text = source.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    matches = [line for line in text.splitlines() if pattern.search(line)]
    destination.write_text(
        ("\n".join(matches[-1000:]) + "\n") if matches else "No matched warnings or errors.\n",
        encoding="utf-8",
    )


def checkpoint_inventory(checkpoint_dir: Path, destination: Path) -> None:
    lines = []
    for path in sorted(checkpoint_dir.glob("*.ckpt")):
        stat = path.stat()
        lines.append(f"{stat.st_size} bytes  {path}")
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_provenance(results: Path, slug: str, run: Path, artifacts: Path) -> None:
    sources = (
        (run / ".hydra/config.yaml", artifacts / "resolved_config.yaml"),
        (run / ".hydra/overrides.yaml", artifacts / "hydra_overrides.yaml"),
        (run / "ARCHITECTURE_AUDIT.txt", artifacts / "ARCHITECTURE_AUDIT.txt"),
        (run / "BEST_CHECKPOINT.txt", artifacts / "BEST_CHECKPOINT.txt"),
        (results / slug / "RUN_TIMING.json", artifacts / "RUN_TIMING.json"),
        (results / slug / "ATTEMPTS.tsv", artifacts / "ATTEMPTS.tsv"),
        (results / slug / "TRAIN_COMMAND.txt", artifacts / "TRAIN_COMMAND.txt"),
        (results / slug / "evaluation.log", artifacts / "evaluation.log"),
    )
    for source, destination in sources:
        if source.is_file() and source.stat().st_size <= 10 * 1024 * 1024:
            shutil.copy2(source, destination)


def run_summary(results: Path, slug: str) -> None:
    run = results / slug / "run"
    artifacts = results / slug / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts / "metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(f"Final metrics are missing: {metrics_path}")
    metrics = load_json(metrics_path)
    history = load_history(run / "metrics_history.jsonl")
    write_history(history, artifacts)
    plot_history(history, artifacts)
    compact_terminal(results / slug / "full_run.log", artifacts / "Terminal.txt")
    warning_report(results / slug / "full_run.log", artifacts / "warnings_and_errors.txt")
    checkpoint_inventory(run / "checkpoints", artifacts / "CHECKPOINTS.txt")
    copy_provenance(results, slug, run, artifacts)

    title, description = VARIANT_DETAILS[slug]
    best_history = min(history, key=lambda item: item.get("minADE6", math.inf)) if history else None
    timing_path = results / slug / "RUN_TIMING.json"
    timing = load_json(timing_path) if timing_path.is_file() else {}
    rows = [
        "| Metric | Final validation | SHARP article | Delta (run - article) |",
        "|---|---:|---:|---:|",
    ]
    for name in METRICS:
        value = metrics.get(name)
        delta = None if value is None else float(value) - ARTICLE[name]
        rows.append(
            f"| {name} | {format_value(value)} | {ARTICLE[name]:.6f} | {format_value(delta)} |"
        )
    summary = [
        f"# {title}",
        "",
        description,
        "",
        "## Status",
        "",
        f"- Completed: {(results / slug / 'COMPLETED').is_file()}",
        f"- Official source commit: `f6bf2fc0109f9838cdc24bfb763b5c3e6847c2ae`",
        f"- Best checkpoint: `{metrics.get('checkpoint', 'N/A')}`",
        f"- Training duration seconds: {timing.get('training_duration_seconds', 'N/A')}",
        f"- Final evaluation: one GPU, batch 32, full AV2 validation split",
        f"- Best recorded training epoch by minADE6: {best_history.get('epoch') + 1 if best_history else 'N/A'}",
        "",
        "## Controlled Training Setup",
        "",
        "- AV2 processed train/validation data; 10 historical steps; 100 future steps; split points 10/20/30/40/50; radius 150 m.",
        "- 80 epochs; 13 warm-up epochs; AdamW; LR 1e-4 to 1e-5 cosine decay; weight decay 1e-2.",
        "- Seed 2333; global batch 32 (8 per rank); four GPUs; FP32; gradient clipping norm 5; SyncBatchNorm.",
        "- No data augmentation and no cross-dataset training.",
        "",
        "## Final Metrics",
        "",
        *rows,
        "",
        "Delta means run metric minus article metric; negative is better for every listed error metric.",
        "",
        "Full checkpoints and the unfiltered log remain on Lab 2 and are intentionally excluded from GitHub.",
    ]
    (artifacts / "Summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    with (artifacts / "final_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "run", "sharp_article", "delta"))
        for name in METRICS:
            value = metrics.get(name)
            writer.writerow((name, value, ARTICLE[name], None if value is None else float(value) - ARTICLE[name]))


def comparison(results: Path) -> None:
    destination = results / "dissertation_artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for slug, (title, _) in VARIANT_DETAILS.items():
        path = results / slug / "artifacts/metrics.json"
        if path.is_file():
            records.append((slug, title, load_json(path)))
    if not records:
        return

    table = [
        "| Run | " + " | ".join(METRICS) + " |",
        "|---|" + "---:|" * len(METRICS),
        "| SHARP article | " + " | ".join(f"{ARTICLE[name]:.6f}" for name in METRICS) + " |",
    ]
    for _, title, metrics in records:
        table.append(
            f"| {title} | "
            + " | ".join(format_value(metrics.get(name)) for name in METRICS)
            + " |"
        )
    destination.joinpath("Comparison.md").write_text(
        "# Final SHARP AV2 Three-Run Comparison\n\n"
        + f"All three training/evaluation runs completed: {(results / 'SUITE_RUNS_COMPLETED').is_file()}\n\n"
        + "\n".join(table)
        + "\n\nAll values are lower-is-better. Final run values use single-GPU full validation to avoid DDP sampler padding.\n",
        encoding="utf-8",
    )
    with destination.joinpath("comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", *METRICS))
        writer.writerow(("SHARP article", *(ARTICLE[name] for name in METRICS)))
        for _, title, metrics in records:
            writer.writerow((title, *(metrics.get(name) for name in METRICS)))

    comparison_charts(
        records,
        destination / "comparison_metrics.png",
        destination / "comparison_metrics.svg",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--slug")
    parser.add_argument("--comparison", action="store_true")
    args = parser.parse_args()
    results = args.results.resolve()
    if args.slug:
        run_summary(results, args.slug)
    if args.comparison:
        comparison(results)
    print("FINAL_SUITE_ARTIFACTS_GENERATED")


if __name__ == "__main__":
    main()
