#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from setup_lab3_seam_20epoch_4test import VARIANTS, resolve_data_root


RUNTIME_FILES = {
    "run_seam_20epoch_4test_max_resources.sh": "run_seam_20epoch_4test_suite.sh",
    "preflight_seam_20epoch_4test.py": "preflight_seam_20epoch_4test.py",
    "real_data_smoke_test.py": "real_data_smoke_test.py",
    "generate_seam_20epoch_summary.py": "generate_seam_20epoch_summary.py",
    "generate_seam_20epoch_snapshot.py": "generate_seam_20epoch_snapshot.py",
    "publish_seam_20epoch_snapshot.sh": "publish_seam_20epoch_snapshot.sh",
    "requirements-lab3.txt": "requirements-lab3.txt",
    "EXPERIMENT_MANIFEST_MAX_RESOURCES.md": "EXPERIMENT_MANIFEST.md",
    "SOURCE_PROVENANCE.md": "SOURCE_PROVENANCE.md",
    "README_MAX_RESOURCES.md": "README.md",
    "MAX_RESOURCE_SUITE_VERSION": "SUITE_VERSION",
}


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
        raise FileNotFoundError(
            "Bundled SEAM source is incomplete: " + ", ".join(missing)
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"SEAM_AV2_20EPOCH_4TEST_MAXRES_{stamp}"
    experiment = base / "Codes" / run_name
    results = base / "Results" / run_name
    code = experiment / "Code"
    data_root = resolve_data_root(base, args.data_root)

    experiment.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    shutil.copytree(upstream, code)
    for source_name, destination_name in RUNTIME_FILES.items():
        shutil.copy2(package / source_name, experiment / destination_name)

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
        "precision": 32,
        "resource_profile": "max_resources_three_gpu",
        "gpu_count": 3,
        "gpu_ids": [0, 1, 2],
        "per_gpu_batch": 8,
        "gradient_accumulation": 2,
        "effective_global_batch": 48,
        "data_worker_policy": "min(12, max(4, floor(nproc / 3) - 2)) per rank",
        "source": "Isolated copy of the verified SEAM AV2 baseline source",
        "dataset_reuse": "Processed AV2 tensors are read-only.",
        "suite_version": (package / "MAX_RESOURCE_SUITE_VERSION")
        .read_text(encoding="utf-8")
        .strip(),
    }
    for root in (experiment, results):
        (root / "RUN_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    pointers = {
        base / "Codes/LATEST_SEAM_AV2_20EPOCH_4TEST_MAXRES_CODE.txt": experiment,
        base / "Results/LATEST_SEAM_AV2_20EPOCH_4TEST_MAXRES.txt": results,
        results / "EXPERIMENT_ROOT.txt": experiment,
        results / "DATA_ROOT.txt": data_root,
    }
    for path, value in pointers.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value) + "\n", encoding="utf-8")

    for script in experiment.glob("*.sh"):
        script.chmod(0o755)

    print("MAX_RESOURCE_SETUP_COMPLETE")
    print(f"EXPERIMENT_ROOT={experiment}")
    print(f"RESULTS_ROOT={results}")
    print(f"DATA_ROOT={data_root}")
    print("GPU_COUNT=3")
    print("EFFECTIVE_GLOBAL_BATCH=48")
    print("Previous SEAM code and results were not changed or deleted.")


if __name__ == "__main__":
    main()
