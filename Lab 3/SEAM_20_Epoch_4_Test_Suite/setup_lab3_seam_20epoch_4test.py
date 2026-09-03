#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


VARIANTS = (
    "baseline",
    "uncertainty_target_context",
    "relative_geometry_bias",
    "qknorm",
)


def has_processed_split(root: Path, split: str) -> bool:
    directory = root / split
    if not directory.is_dir():
        return False
    try:
        return any(entry.name.endswith(".pt") for entry in os.scandir(directory))
    except PermissionError:
        return False


def resolve_data_root(base: Path, explicit: Optional[str]) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))

    previous_results_pointer = base / "Results/LATEST_SEAM_AV2_MAMBA_3RUN.txt"
    if previous_results_pointer.is_file():
        previous_results = Path(
            previous_results_pointer.read_text(encoding="utf-8").strip()
        )
        previous_data_pointer = previous_results / "DATA_ROOT.txt"
        if previous_data_pointer.is_file():
            candidates.append(
                Path(previous_data_pointer.read_text(encoding="utf-8").strip())
            )

    candidates.extend(
        [
            base / "Datasets/AV2/seam_processed",
            base / "Datasets/AV2/sharp_processed",
            base / "Datasets/AV2/processed",
        ]
    )
    for candidate in candidates:
        if has_processed_split(candidate, "train") and has_processed_split(
            candidate, "val"
        ):
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "No compatible processed AV2 train/val dataset was found. Checked:\n"
        + checked
    )


def copy_runtime_files(package: Path, experiment: Path) -> None:
    names = (
        "run_seam_20epoch_4test_suite.sh",
        "preflight_seam_20epoch_4test.py",
        "real_data_smoke_test.py",
        "generate_seam_20epoch_summary.py",
        "generate_seam_20epoch_snapshot.py",
        "publish_seam_20epoch_snapshot.sh",
        "requirements-lab3.txt",
        "EXPERIMENT_MANIFEST.md",
        "SOURCE_PROVENANCE.md",
        "README.md",
        "SUITE_VERSION",
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
    required = (
        upstream / "train.py",
        upstream / "src/model/seam.py",
        upstream / "src/model/layers/controlled_ablation.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Bundled SEAM source is incomplete: " + ", ".join(missing))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment = base / "Codes" / f"SEAM_AV2_20EPOCH_4TEST_{stamp}"
    results = base / "Results" / f"SEAM_AV2_20EPOCH_4TEST_{stamp}"
    code = experiment / "Code"
    data_root = resolve_data_root(base, args.data_root)

    experiment.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    shutil.copytree(upstream, code)
    copy_runtime_files(package, experiment)

    for index, variant in enumerate(VARIANTS, start=1):
        (results / f"{index:02d}_{variant}").mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": datetime.now().astimezone().isoformat(),
        "experiment_root": str(experiment),
        "results_root": str(results),
        "data_root": str(data_root),
        "variants": list(VARIANTS),
        "epochs": 20,
        "seed": 2333,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "minimum_learning_rate": 0.00001,
        "warmup_ratio": 0.167,
        "weight_decay": 0.01,
        "gradient_clip_norm": 5.0,
        "gpu_count": 2,
        "per_gpu_batch": 8,
        "gradient_accumulation": 2,
        "effective_global_batch": 32,
        "source": "Isolated copy of the verified previous SEAM AV2 baseline source",
        "dataset_reuse": "Processed AV2 tensors are read-only.",
        "suite_version": (package / "SUITE_VERSION").read_text(
            encoding="utf-8"
        ).strip(),
    }
    for root in (experiment, results):
        (root / "RUN_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    pointers = {
        base / "Codes/LATEST_SEAM_AV2_20EPOCH_4TEST_CODE.txt": experiment,
        base / "Results/LATEST_SEAM_AV2_20EPOCH_4TEST.txt": results,
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
    print("Previous SEAM code and results were not changed or deleted.")


if __name__ == "__main__":
    main()
