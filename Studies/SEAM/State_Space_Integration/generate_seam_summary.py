#!/usr/bin/env python3
import argparse
import re
from datetime import datetime
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRICS = ("MR", "minADE1", "minADE6", "minFDE1", "minFDE6", "b-minFDE6")


def tail_text(path: Path, limit=50_000_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        data = stream.read()
    return ANSI.sub("", data.decode(errors="replace").replace("\r", "\n"))


def parse_validation(text: str):
    records = []
    pattern = re.compile(r"\[\s*(\d+)\]\s+Avg over .*? - (.*)")
    for match in pattern.finditer(text):
        values = {}
        for metric in METRICS:
            found = re.search(rf"{re.escape(metric)}:\s*([0-9.]+)", match.group(2))
            if found:
                values[metric] = float(found.group(1))
        if values:
            records.append((int(match.group(1)), values))
    return records


def parse_consolidated_validation(text: str):
    values = {}
    for metric in METRICS:
        matches = re.findall(
            rf"│\s*{re.escape(metric)}\s*│\s*([-+0-9.eE]+)\s*│",
            text,
        )
        if matches:
            values[metric] = float(matches[-1])
    return values


def current_epoch(text: str):
    matches = re.findall(r"Epoch\s+(\d+):\s*([0-9]+)%", text)
    return matches[-1] if matches else (None, None)


def checkpoint_rows(run: Path):
    rows = []
    for checkpoint in run.rglob("*.ckpt"):
        metric = re.search(r"minADE6[_=]([0-9]+(?:\.[0-9]+)?)", checkpoint.name)
        epoch = re.search(r"epoch[_=](\d+)", checkpoint.name)
        rows.append(
            {
                "path": checkpoint,
                "epoch": int(epoch.group(1)) if epoch else None,
                "minADE6": float(metric.group(1)) if metric else None,
                "size": checkpoint.stat().st_size,
            }
        )
    return sorted(rows, key=lambda item: str(item["path"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-dir", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    run = Path(args.variant_dir).resolve()
    experiment = Path(args.experiment).resolve()
    log = run / "train.log"
    text = tail_text(log)
    validations = parse_validation(text)
    epoch, percent = current_epoch(text)
    checkpoints = checkpoint_rows(run)
    scored = [item for item in checkpoints if item["minADE6"] is not None]
    best = min(scored, key=lambda item: item["minADE6"]) if scored else None
    complete = (run / "TRAINING_COMPLETE").is_file()

    status_lines = [
        f"variant={args.variant}",
        f"status={'complete' if complete else 'incomplete'}",
        f"current_epoch={epoch if epoch is not None else 'unknown'}",
        f"current_percent={percent if percent is not None else 'unknown'}",
        f"checkpoint_count={len(checkpoints)}",
        f"updated={datetime.now().astimezone().isoformat()}",
    ]
    (run / "STATUS.txt").write_text("\n".join(status_lines) + "\n", encoding="utf-8")

    checkpoint_lines = []
    for item in checkpoints:
        checkpoint_lines.append(
            f"{item['size']} bytes | epoch={item['epoch']} | "
            f"minADE6={item['minADE6']} | {item['path']}"
        )
    (run / "CHECKPOINTS.txt").write_text(
        "\n".join(checkpoint_lines) + ("\n" if checkpoint_lines else ""),
        encoding="utf-8",
    )

    descriptions = {
        "baseline": (
            "Official streaming SEAM architecture and released AV2 optimizer schedule; "
            "no Mamba module is instantiated."
        ),
        "mamba_agent_add": (
            "Adds one gated residual Mamba block after the four temporal MHA blocks "
            "and before agent-history max pooling."
        ),
        "mamba_future_replace": (
            "Replaces only the decoder trajectory-coordinate MLP with a two-block "
            "Mamba head over the ordered 80-step future sequence."
        ),
    }

    lines = [
        f"# SEAM AV2: {args.variant}",
        "",
        descriptions[args.variant],
        "",
        "## Run Status",
        "",
        f"- Status: **{'complete' if complete else 'incomplete/resumable'}**",
        f"- Current progress: epoch {epoch or 'unknown'}, {percent or 'unknown'}%",
        f"- Checkpoints saved: {len(checkpoints)}",
        f"- Run directory: `{run}`",
        "",
        "## Controlled Training Setup",
        "",
        "| Setting | Value |",
        "|---|---:|",
        "| Dataset | Argoverse 2 motion forecasting |",
        "| Observation / prediction | 3 s / 6 s |",
        "| Streaming passes | t = 3, 4, 5 s |",
        "| Epochs | 80 |",
        "| Effective global training batch | 32 |",
        "| Optimizer | AdamW |",
        "| Peak / minimum LR | 1e-3 / 1e-5 |",
        "| Warm-up | 13 epochs |",
        "| Weight decay | 1e-2 |",
        "| Gradient clipping | norm 5 |",
        "| Seed | 2333 |",
        "| GPUs | 2 x RTX 2080 Ti |",
        "| Microbatch / accumulation | 8 per GPU / 2 steps |",
        "| SyncBatchNorm activation batch | 16 (hardware adaptation) |",
        "",
        "## Latest Validation",
        "",
    ]

    if validations:
        val_epoch, values = validations[-1]
        consolidated = parse_consolidated_validation(text)
        if complete and consolidated:
            values = consolidated
        lines.extend(["| Epoch | " + " | ".join(METRICS) + " |", "|---:" * (len(METRICS) + 1) + "|"])
        row = [str(val_epoch)] + [f"{values.get(metric, float('nan')):.3f}" for metric in METRICS]
        lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("No completed validation record is present yet.")

    lines.extend(["", "## Best Saved Checkpoint", ""])
    if best:
        lines.extend(
            [
                f"- Epoch: {best['epoch']}",
                f"- minADE6: {best['minADE6']:.6f}",
                f"- Local path: `{best['path']}`",
            ]
        )
    else:
        lines.append("No metric-named checkpoint is present yet.")

    lines.extend(
        [
            "",
            "## Reference",
            "",
            "Released SEAM AV2 checkpoint metrics: MR 0.153, minADE1 1.598, "
            "minADE6 0.663, minFDE1 3.962, minFDE6 1.249, b-minFDE6 1.848.",
            "",
            "The paper prints peak LR 1e-2, but the released configuration and "
            "checkpoint scheduler state both use 1e-3. These runs use 1e-3 for "
            "reproducibility with the released implementation.",
        ]
    )
    (run / "Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    normalized_tail = "\n".join(text.splitlines()[-4000:]) + "\n"
    (run / "LOG_TAIL.txt").write_text(normalized_tail, encoding="utf-8")

    print(f"SUMMARY_WRITTEN={run / 'Summary.md'}")


if __name__ == "__main__":
    main()
