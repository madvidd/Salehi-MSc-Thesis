#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


VARIANTS = ("baseline_mha", "qknorm", "talking_heads", "qknorm_talking_heads")
METRICS = ("MR", "minADE1", "minADE6", "minFDE1", "minFDE6", "b-minFDE6")

# SHARP paper, AV2 validation, standard 5 s context (Table 4).
ARTICLE_VALIDATION = {"minADE6": 0.64, "minFDE6": 1.20, "b-minFDE6": 1.82}

# User's previous single-A4000 validation result.
PREVIOUS_A4000 = {
    "MR": 0.1492477655,
    "minADE1": 1.5778011084,
    "minADE6": 0.6502320766,
    "minFDE1": 3.8929622173,
    "minFDE6": 1.2232348919,
    "b-minFDE6": 1.8406755924,
}


def load_metrics(root: Path, variant: str) -> dict[str, float] | None:
    path = root / variant / "metrics.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return {key: float(raw[key]) for key in METRICS if key in raw}


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: compare_results.py RESULTS_ROOT", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    rows = [(variant, load_metrics(root, variant)) for variant in VARIANTS]
    available = [(name, data) for name, data in rows if data]
    if not available:
        print(f"No metrics.json files found below {root}", file=sys.stderr)
        return 1

    baseline = dict(rows).get("baseline_mha")
    csv_path = root / "attention_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        fields = ["variant", *METRICS, "delta_minADE6_vs_baseline", "delta_minFDE6_vs_baseline"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, data in available:
            row = {"variant": name, **{metric: fmt(data.get(metric)) for metric in METRICS}}
            if baseline:
                row["delta_minADE6_vs_baseline"] = fmt(data.get("minADE6", math.nan) - baseline.get("minADE6", math.nan))
                row["delta_minFDE6_vs_baseline"] = fmt(data.get("minFDE6", math.nan) - baseline.get("minFDE6", math.nan))
            writer.writerow(row)

    md_path = root / "attention_comparison.md"
    lines = [
        "# SHARP attention ablation comparison",
        "",
        "Lower values are better. Deltas are variant minus baseline, so negative is an improvement.",
        "",
        "| Variant | MR | minADE1 | minADE6 | minFDE1 | minFDE6 | b-minFDE6 | Delta minADE6 | Delta minFDE6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in available:
        d_ade = None if not baseline else data.get("minADE6", math.nan) - baseline.get("minADE6", math.nan)
        d_fde = None if not baseline else data.get("minFDE6", math.nan) - baseline.get("minFDE6", math.nan)
        values = [fmt(data.get(metric)) for metric in METRICS]
        lines.append(f"| {name} | {' | '.join(values)} | {fmt(d_ade)} | {fmt(d_fde)} |")

    lines.extend([
        "",
        "## References",
        "",
        f"- SHARP article AV2 validation: minADE6={ARTICLE_VALIDATION['minADE6']:.2f}, minFDE6={ARTICLE_VALIDATION['minFDE6']:.2f}, b-minFDE6={ARTICLE_VALIDATION['b-minFDE6']:.2f}.",
        f"- Previous A4000 run: MR={PREVIOUS_A4000['MR']:.6f}, minADE6={PREVIOUS_A4000['minADE6']:.6f}, minFDE6={PREVIOUS_A4000['minFDE6']:.6f}, b-minFDE6={PREVIOUS_A4000['b-minFDE6']:.6f}.",
        "- Article training used one RTX 8000 and global batch 32. This Lab 3 run uses three GPUs and global batch 24, so absolute reproduction is not mathematically identical; the within-run attention comparison remains controlled.",
        "",
    ])
    md_path.write_text("\n".join(lines))

    best = min(available, key=lambda item: item[1].get("minADE6", math.inf))
    print(f"CSV={csv_path}")
    print(f"MARKDOWN={md_path}")
    print(f"BEST_MINADE6={best[0]}:{best[1].get('minADE6')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
