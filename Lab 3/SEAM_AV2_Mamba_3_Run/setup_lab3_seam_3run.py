#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


VARIANTS = (
    "baseline",
    "mamba_agent_add",
    "mamba_future_replace",
)


def has_processed_split(root: Path, split: str) -> bool:
    directory = root / split
    if not directory.is_dir():
        return False
    try:
        return any(entry.name.endswith(".pt") for entry in os.scandir(directory))
    except PermissionError:
        return False


def resolve_data_root(base: Path, explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            base / "Datasets/AV2/sharp_processed",
            base / "Datasets/AV2/seam_processed",
            base / "Datasets/AV2/processed",
        ]
    )
    for candidate in candidates:
        if has_processed_split(candidate, "train") and has_processed_split(
            candidate, "val"
        ):
            return candidate.resolve()
    joined = "\n".join(f"  - {item}" for item in candidates)
    raise FileNotFoundError(
        "No compatible processed AV2 train/val dataset was found. Checked:\n"
        + joined
    )


def copy_runtime_files(package: Path, experiment: Path) -> None:
    names = (
        "run_seam_3run_suite.sh",
        "preflight_seam_variants.py",
        "generate_seam_summary.py",
        "publish_seam_result.sh",
        "requirements-lab3.txt",
        "ARTICLE_AND_CODE_AUDIT.md",
        "SOURCE_PROVENANCE.md",
    )
    for name in names:
        shutil.copy2(package / name, experiment / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/server01/M")
    parser.add_argument("--data-root")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    package = Path(__file__).resolve().parent
    upstream = package / "upstream/seam-main"
    if not (upstream / "train.py").is_file():
        raise FileNotFoundError(f"Bundled SEAM source is incomplete: {upstream}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment = base / "Codes" / f"SEAM_AV2_MAMBA_3RUN_{stamp}"
    results = base / "Results" / f"SEAM_AV2_MAMBA_3RUN_{stamp}"
    code = experiment / "Code"
    data_root = resolve_data_root(base, args.data_root)

    experiment.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    shutil.copytree(upstream, code)
    copy_runtime_files(package, experiment)

    for variant in VARIANTS:
        (results / variant).mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": datetime.now().astimezone().isoformat(),
        "experiment_root": str(experiment),
        "results_root": str(results),
        "data_root": str(data_root),
        "variants": list(VARIANTS),
        "source": "Official SEAM release supplied as seam-main.zip",
        "dataset_reuse": "Processed AV2 tensors are read-only and schema-compatible.",
    }
    (experiment / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (results / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    pointers = {
        base / "Codes/LATEST_SEAM_AV2_MAMBA_3RUN_CODE.txt": experiment,
        base / "Results/LATEST_SEAM_AV2_MAMBA_3RUN.txt": results,
        results / "EXPERIMENT_ROOT.txt": experiment,
        results / "DATA_ROOT.txt": data_root,
    }
    for path, value in pointers.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value) + "\n", encoding="utf-8")

    for script in experiment.glob("*.sh"):
        script.chmod(0o755)

    print("SETUP_COMPLETE")
    print(f"EXPERIMENT_ROOT={experiment}")
    print(f"RESULTS_ROOT={results}")
    print(f"DATA_ROOT={data_root}")
    print("Previous Lab 3 code and results were not changed or deleted.")


if __name__ == "__main__":
    main()
