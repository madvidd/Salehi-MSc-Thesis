"""Offline repository checks; never imports a training model or starts a job."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

from build_results_catalog import ROOT, STUDIES, outputs


def historical(path):
    relative = path.relative_to(ROOT)
    return (relative.parts[:2] == ("Documentation", "Archive") or
            (relative.parts[0] == "Studies" and
             any(p in relative.parts[3:] for p in
                 ("Results", "Runs", "Main_Results", "upstream", "sharp_original"))))


def main():
    failures = []
    manifest = json.loads((ROOT / "Documentation/Repository_Audit/Original_File_Inventory.json").read_text())
    preserved = 0
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append("Missing original file: " + row["path"])
        elif row["preserve_bytes"]:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                failures.append("Recorded evidence changed: " + row["path"])
            preserved += 1

    sources, links = 0, 0
    paths = [ROOT / "README.md"]
    for directory in ("Studies", "Results", "Documentation", "Tools"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix == ".py":
            sources += 1
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
            except (SyntaxError, UnicodeError) as exc:
                failures.append(f"Python syntax: {relative}: {exc}")
        if path.suffix == ".md" and not historical(path):
            text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8-sig"), flags=re.S)
            columns = None
            for line in text.splitlines():
                if line.startswith("|") and line.endswith("|"):
                    count = len(re.split(r"(?<!\\)\|", line)) - 2
                    if columns is not None and columns != count:
                        failures.append(f"Inconsistent table columns: {relative}")
                    columns = count
                else:
                    columns = None
            for match in re.finditer(r"\[[^\]]*\]\(([^\s)]+)\)", text):
                target = unquote(match.group(1))
                if target.startswith(("#", "mailto:")) or urlsplit(target).scheme:
                    continue
                target = target.split("#", 1)[0]
                links += 1
                if not (path.parent / target).exists():
                    failures.append(f"Broken Markdown link: {relative} -> {target}")

    for relative, content in outputs().items():
        path = ROOT / relative
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            failures.append("Stale generated summary: " + relative)

    combined = ROOT / "Studies/SEAM/Combined_Extension"
    for relative, expected in json.loads((combined / "INPUT_HASHES.json").read_text()).items():
        path = combined.parent / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append("Broken SEAM source contract: " + relative)

    # In a staged or committed checkout, root tracked files have the same rule.
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in tracked:
        if name and "/" not in name and (ROOT / name).exists() and name not in ("README.md", "LICENSE"):
            failures.append("Unexpected root file: " + name)

    report = {"original_files": len(manifest["files"]), "preserved_records": preserved,
              "python_sources_parsed": sources, "maintained_local_links": links,
              "study_packages": len(STUDIES), "failures": failures}
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
