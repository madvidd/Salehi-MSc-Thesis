import argparse
import ast
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from common import atomic_json, read_json, source_sha256, source_inventory


class EnableCombination(ast.NodeTransformer):
    def __init__(self):
        self.counts = Counter()

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            value = node.comparators[0]
            if isinstance(value, ast.Constant) and value.value in {
                "qknorm", "relative_geometry_bias", "uncertainty_target_context"
            }:
                self.counts[value.value] += 1
                node.ops = [ast.In()]
                node.comparators = [ast.Tuple(elts=[value, ast.Constant("combined")], ctx=ast.Load())]
        return node

    def visit_Set(self, node):
        if {getattr(x, "value", None) for x in node.elts} == {
            "baseline", "qknorm", "relative_geometry_bias", "uncertainty_target_context"
        }:
            node.elts.append(ast.Constant("combined"))
        return node


def combine_source(code, mamba_source, package):
    for name, expected in (
        ("src/model/seam.py", {"qknorm": 1, "relative_geometry_bias": 3, "uncertainty_target_context": 4}),
        ("src/model/layers/multimodal_decoder_attn.py", {"qknorm": 1}),
    ):
        path = code / name
        transform = EnableCombination()
        tree = transform.visit(ast.parse(path.read_text(encoding="utf-8")))
        if dict(transform.counts) != expected:
            raise RuntimeError(f"Source contract changed in {name}: {transform.counts}")
        path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")
    shutil.copy2(mamba_source, code / "src/model/layers/mamba_layers.py")
    shutil.copy2(package / "combined_model.py", code / "src/model/combined.py")

    # A completely padded attention row must have zero attention mass, including
    # a finite backward pass. Valid rows keep the previous QKNorm calculation.
    path = code / "src/model/layers/controlled_ablation.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("torch.finfo(logits.dtype).min", "float('-inf')")
    anchor = "        weights = torch.softmax(logits.float(), dim=-1).to(logits.dtype)"
    if text.count(anchor) != 1:
        raise RuntimeError("QKNorm softmax source contract changed")
    text = text.replace(anchor, "\n".join([
        "        empty = (~torch.isfinite(logits) | (logits <= torch.finfo(logits.dtype).min / 2)).all(-1, keepdim=True)",
        "        logits = logits.masked_fill(empty, 0.0)",
        anchor,
        "        weights = weights.masked_fill(empty, 0.0)",
    ]))
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/server01/M")
    args = parser.parse_args()
    base = Path(args.base).resolve()
    package = Path(__file__).resolve().parent
    pointer = base / "Codes/LATEST_SEAM_AV2_80EPOCH_COMBINED_CODE.txt"
    if pointer.is_file():
        experiment = Path(pointer.read_text().strip())
        from common import assert_sources
        assert_sources(experiment)
        print(f"RESUME_EXISTING_EXPERIMENT={experiment}")
        return

    old = package.parent / "SEAM_AV2_Mamba_3_Run"
    controls = package.parent / "SEAM_20_Epoch_4_Test_Suite"
    reference = old / "Results/SEAM_AV2_MAMBA_3RUN_20260824-202936/baseline/config"
    previous = base / "Results/SEAM_AV2_MAMBA_3RUN_20260824-202936"
    data_pointer = previous / "DATA_ROOT.txt"
    if not data_pointer.is_file():
        raise RuntimeError(f"Exact previous run's DATA_ROOT.txt missing: {data_pointer}")
    data = Path(data_pointer.read_text().strip()).resolve()
    for split in ("train", "val"):
        if not (data / split).is_dir():
            raise RuntimeError(f"Previous run's processed {split} split is missing: {data}")
        with os.scandir(data / split) as entries:
            if not any(p.name.endswith(".pt") for p in entries):
                raise RuntimeError(f"Previous run's processed {split} split is empty: {data}")
    if shutil.disk_usage(base).free < 25 * 1024**3:
        raise RuntimeError("At least 25 GiB free space is required for epoch checkpoints and logs")
    # Sources are pinned to the files tested with this package, not whichever
    # revision of a sibling happens to exist on a later invocation.
    for relative, expected in read_json(package / "INPUT_HASHES.json").items():
        if source_sha256(package.parent / relative) != expected:
            raise RuntimeError(f"Pinned source changed: {relative}")

    name = "SEAM_AV2_80EPOCH_COMBINED_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment, results = base / "Codes" / name, base / "Results" / name
    experiment.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    code = experiment / "Code"
    shutil.copytree(controls / "upstream/seam-main", code,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    for path in package.iterdir():
        if path.is_file() and path.suffix in {".py", ".json", ".sh", ".md"}:
            shutil.copy2(path, experiment / path.name)
    combine_source(code, old / "upstream/seam-main/src/model/layers/mamba_layers.py", package)
    shutil.copytree(reference, experiment / "reference_config")
    cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 8)
    workers = max(2, min(12, (cpu - 8) // 2))
    manifest = {
        "experiment": str(experiment), "results": str(results), "base": str(base),
        "data_root": str(data), "protocol": read_json(package / "protocol.json"),
        "workers_per_rank": workers, "cpu_threads_per_rank": max(1, min(4, (cpu - workers * 2) // 2)),
        "created": datetime.now().astimezone().isoformat(),
        "training_gpus": "0,1", "evaluation_gpu": "2",
    }
    for root in (experiment, results):
        atomic_json(root / "RUN_MANIFEST.json", manifest)
    atomic_json(experiment / "SOURCE_HASHES.json", source_inventory(experiment))
    for path, value in ((pointer, experiment),
                        (base / "Results/LATEST_SEAM_AV2_80EPOCH_COMBINED.txt", results),
                        (results / "EXPERIMENT_ROOT.txt", experiment)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value) + "\n", encoding="utf-8")
    print(f"EXPERIMENT={experiment}\nRESULTS={results}\nDATA_ROOT={data}")


if __name__ == "__main__":
    main()
