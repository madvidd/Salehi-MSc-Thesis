"""Verify published research figures using only the Python standard library."""

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
FORMATS = {"svg", "pdf", "png"}


def safe_path(root, value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"Non-local figure path: {value}")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()) or not result.is_file():
        raise ValueError(f"Missing or escaping figure path: {value}")
    return result


def validate(root=ROOT):
    collection = root / "Results/Figures"
    manifest = json.loads((collection / "Manifest.json").read_text(encoding="utf-8"))
    rows = manifest["figures"]
    if manifest["schema_version"] != 1 or manifest["figure_count"] != 24 or len(rows) != 24:
        raise ValueError("Expected the reviewed 24-figure collection")
    if len({r["figure"] for r in rows}) != 24 or len({r["source_stem"] for r in rows}) != 24:
        raise ValueError("Duplicate figure number or source")
    paths, sums, index_rows = set(), [], []
    for row in rows:
        if set(row["files"]) != FORMATS or not row["title"].strip():
            raise ValueError(f"Missing format or title: {row['figure']}")
        if not row["related_records"]:
            raise ValueError("Figure has no supporting records")
        for evidence in row["related_records"]:
            safe_path(root, evidence)
        index = {key: row[key] for key in ("figure", "title", "group")}
        for suffix, item in row["files"].items():
            path = safe_path(collection, item["path"])
            if item["path"] in paths or path.suffix != "." + suffix:
                raise ValueError(f"Duplicate path or incorrect extension: {path}")
            paths.add(item["path"])
            if PurePosixPath(item["path"]).parent.as_posix() != row["group"]:
                raise ValueError(f"Figure is outside its scientific group: {path}")
            data = path.read_bytes()
            if not 0 < len(data) < 10 * 1024**2 or len(data) != item["bytes"]:
                raise ValueError(f"Figure size mismatch: {path}")
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Figure checksum mismatch: {path}")
            if suffix == "svg" and ET.fromstring(data).tag.rsplit("}", 1)[-1] != "svg":
                raise ValueError(f"Invalid SVG: {path}")
            if suffix == "pdf" and not data.startswith(b"%PDF-"):
                raise ValueError(f"Invalid PDF: {path}")
            if suffix == "png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Invalid PNG: {path}")
            sums.append(f"{item['sha256']}  {item['path']}")
            index[suffix] = item["path"]
        index_rows.append(index)
    if manifest["asset_count"] != len(paths) or len(paths) != 72:
        raise ValueError("Expected three formats for each figure")
    actual = {p.relative_to(collection).as_posix() for p in collection.rglob("*")
              if p.is_file() and p.suffix.lstrip(".") in FORMATS}
    if actual != paths:
        raise ValueError("Unindexed, obsolete or missing figure asset")
    if (collection / "SHA256SUMS.txt").read_text().splitlines() != sums:
        raise ValueError("Checksum listing differs from the manifest")
    with (collection / "Figure_Index.csv").open(encoding="utf-8", newline="") as stream:
        if list(csv.DictReader(stream)) != index_rows:
            raise ValueError("CSV index differs from the manifest")
    return {"figures": len(rows), "assets": len(paths), "checksums_verified": len(paths)}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
