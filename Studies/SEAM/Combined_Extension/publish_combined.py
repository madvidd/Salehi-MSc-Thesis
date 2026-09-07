import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from common import METRICS, PACKAGE_PATH, TOKEN_PATTERN, atomic_json, read_json, sha256, token_from_file


def authenticate(token_file):
    token = token_from_file(token_file)
    def api(path):
        request = urllib.request.Request("https://api.github.com/" + path, headers={
            "Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
            "User-Agent": "SEAM-Results-Publisher",
        })
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"GitHub {path}: HTTP {exc.code}; check token access") from None
    user, repo = api("user"), api("repos/madvidd/Thesis")
    login = user.get("login")
    if login != "madviddd" or not repo.get("permissions", {}).get("push"):
        raise RuntimeError(f"Token account is {login}; a writable madviddd token is required")
    print(f"GitHub account: {login}\nPush permission: true", flush=True)
    return token


def preserve_terminal(source, destination):
    """Append only new bytes; if a log was replaced, retain the previous copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset_file = destination.with_name("Terminal_offset.json")
    stat = source.stat()
    previous = read_json(offset_file) if offset_file.exists() else {}
    offset = previous.get("offset", 0)
    with source.open("rb") as inp:
        prefix = inp.read(min(4096, previous.get("offset", 0)))
        import hashlib
        same = (previous.get("inode") == stat.st_ino and stat.st_size >= offset
                and previous.get("prefix") == hashlib.sha256(prefix).hexdigest()
                and destination.exists())
        if not same:
            offset = 0
        inp.seek(offset)
        with destination.open("ab") as out:
            if not same and out.tell():
                out.write(b"\n===== Additional preserved source transcript =====\n")
            remaining = stat.st_size - offset
            while remaining:
                block = inp.read(min(1024 * 1024, remaining))
                if not block:
                    raise RuntimeError("Source log changed while being saved; previous transcript retained")
                out.write(block)
                remaining -= len(block)
            out.flush()
            os.fsync(out.fileno())
        inp.seek(0)
        prefix_hash = hashlib.sha256(inp.read(min(4096, stat.st_size))).hexdigest()
    atomic_json(offset_file, {"inode": stat.st_ino, "offset": stat.st_size, "prefix": prefix_hash})


def create_stage(experiment, stage):
    manifest = read_json(experiment / "RUN_MANIFEST.json")
    results = Path(manifest["results"])
    run = results / "combined"
    local = Path(manifest["base"]) / "Terminal/SEAM_80_Epoch_Combined_Run" / results.name / "Terminal.txt"
    source = results / "suite.log"
    if source.exists():
        preserve_terminal(source, local)
    stage.mkdir(parents=True, exist_ok=False)
    terminal = stage / "Terminal.txt"
    if local.exists():
        with local.open("rb") as stream:
            if local.stat().st_size <= 7 * 1024**2:
                data = stream.read()
            else:
                data = stream.read(512 * 1024)
                data += (f"\n[GitHub excerpt; complete append-preserved transcript: {local}]\n").encode()
                stream.seek(-6 * 1024**2, os.SEEK_END)
                data += stream.read()
        terminal.write_bytes(TOKEN_PATTERN.sub(b"[REDACTED_PAT]", data).replace(b"\r", b"\n"))
    else:
        terminal.write_text("No training transcript yet.\n", encoding="utf-8")
    for root, names in ((results, ("RUN_MANIFEST.json", "ARCHITECTURE_AUDIT.json", "ENVIRONMENT.json", "ATTEMPTS.json", "RUN_STATUS.json")),
                        (run, ("epoch_metrics.csv", "HEARTBEAT.json", "RESUME.json", "TRAINING_COMPLETE.json", "FINAL_METRICS.json")),
                        (experiment, ("SOURCE_HASHES.json", "INPUT_HASHES.json", "protocol.json", "README.md"))):
        for name in names:
            if (root / name).is_file():
                shutil.copy2(root / name, stage / name)
    code_stage = stage / "Configuration_and_Source"
    code_stage.mkdir()
    for path in (experiment / "reference_config").glob("*.yaml"):
        shutil.copy2(path, code_stage / ("reference_" + path.name))
    for relative in ("src/model/combined.py", "src/model/seam.py", "src/model/layers/controlled_ablation.py",
                     "src/model/layers/mamba_layers.py", "src/model/layers/multimodal_decoder_attn.py",
                     "src/model/layers/transformer_blocks.py", "src/model/layers/custom_transformer_blocks.py"):
        target = code_stage / Path(relative).name
        shutil.copy2(experiment / "Code" / relative, target)
    rows = []
    if (run / "epoch_metrics.csv").exists():
        with (run / "epoch_metrics.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    complete = (run / "TRAINING_COMPLETE.json").exists()
    evaluated = (run / "FINAL_METRICS.json").exists()
    summary = ["# SEAM 80-Epoch Combined Experiment", "",
        "SEAM + uncertainty-aware target context + relative-geometry bias + QKNorm + future-head Mamba.", "",
        f"Captured: {datetime.now().astimezone().isoformat()}",
        f"Training: {'80 epochs completed' if complete else 'not yet confirmed complete'}.",
        f"Final selected-checkpoint evaluation: {'complete' if evaluated else 'not yet complete'}.", "",
        "Protocol: 80 epochs; seed 2333; global batch 32 (2 GPUs x 8 x accumulation 2); FP32; "
        "AdamW; LR 0.001 to 0.00001; warm-up ratio 0.167; weight decay 0.01; gradient clipping norm 5.", "",
        "| Measurement | " + " | ".join(METRICS) + " |",
        "|---|" + "---:|" * len(METRICS)]
    if rows:
        for label, row in (("Best validation epoch", min(rows, key=lambda r: float(r['minADE6']))), ("Latest validation epoch", rows[-1])):
            summary.append(f"| {label} {row['epoch']} | " + " | ".join(f"{float(row[m]):.6f}" for m in METRICS) + " |")
    if evaluated:
        metrics = read_json(run / "FINAL_METRICS.json")["metrics"]
        summary.append("| Selected-checkpoint evaluation | " + " | ".join(f"{metrics[m]:.6f}" for m in METRICS) + " |")
    if not rows:
        summary.append("| No validation result yet | " + " | ".join("-" for _ in METRICS) + " |")
    if (run / "HEARTBEAT.json").exists():
        heartbeat = read_json(run / "HEARTBEAT.json")
        summary.extend(["", f"Last recorded progress: epoch {heartbeat['epoch']}, batch {heartbeat['batch']}/{heartbeat['batches']}, step {heartbeat['global_step']}.",
                        "This is a saved observation, not confirmation of live progress."])
    attempts = read_json(results / "ATTEMPTS.json") if (results / "ATTEMPTS.json").exists() else []
    seconds = sum(r.get("seconds", 0) for r in attempts if r.get("phase") == "train")
    summary.extend(["", "## Training Time", "", "| Measurement | Hours |", "|---|---:|",
                    f"| Recorded training attempts | {seconds / 3600:.3f} |", "",
                    "Attempt duration includes training and epoch validation; failed attempts are retained separately in ATTEMPTS.json. Offline gaps are not added.",
                    "", f"Full logs and model checkpoints remain on the PC in `{results}`. Checkpoints are not uploaded to GitHub.",
                    "The combined model is an experiment, not a guaranteed improvement over its individual components."])
    (stage / "Summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    shutil.copy2(stage / "Summary.md", results / "Summary.md")
    checkpoints = []
    for path in sorted((run / "checkpoints").glob("*.ckpt")):
        checkpoints.append({"file": path.name, "bytes": path.stat().st_size,
                            "sha256": sha256(path) if complete else None})
    atomic_json(stage / "CHECKPOINTS.json", checkpoints)
    (stage / ".gitignore").write_text("*.ckpt\n*.pt\n*.pth\n*.log\nToken.txt\n", encoding="utf-8")
    return results, local


def validate_stage(stage):
    files = []
    for path in sorted(stage.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Publication staging must not contain symlinks")
        if not path.is_file():
            continue
        if path.stat().st_size >= 10 * 1024**2 or path.suffix in {".ckpt", ".pt", ".pth"} or path.name.lower() == "token.txt":
            raise RuntimeError(f"Publication size/type gate: {path.name}")
        if TOKEN_PATTERN.search(path.read_bytes()):
            raise RuntimeError(f"Credential text detected in {path.name}; nothing was pushed")
        files.append(path.relative_to(stage).as_posix())
    if sum((stage / f).stat().st_size for f in files) > 25 * 1024**2:
        raise RuntimeError("Snapshot exceeds the 25 MiB total budget")
    return files


def publish_locked(experiment):
    manifest = read_json(experiment / "RUN_MANIFEST.json")
    base = Path(manifest["base"])
    token = authenticate(base / "Token/Token.txt")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{os.getpid()}"
    stage = base / "Results/SEAM_80_Combined_Snapshots" / Path(manifest["results"]).name / stamp
    results, local = create_stage(experiment, stage)
    files = validate_stage(stage)
    clone = base / "Codes" / f"Thesis_SEAM80_Combined_Publish_{stamp}"
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.pop("GH_TOKEN", None)
    env.pop("GITHUB_TOKEN", None)
    with tempfile.TemporaryDirectory(prefix="seam80_auth_") as temporary:
        askpass = Path(temporary) / "askpass.sh"
        askpass.write_text('#!/usr/bin/env sh\ncase "$1" in\n*Username*) printf "%s\\n" "madviddd";;\n*Password*) printf "%s\\n" "$SEAM_GITHUB_TOKEN";;\nesac\n', encoding="utf-8")
        askpass.chmod(0o700)
        env.update(SEAM_GITHUB_TOKEN=token, GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT="0", LC_ALL="C")
        def git(*args, capture=False, root=None):
            command = ["/usr/bin/git", "-c", "credential.helper="]
            if root:
                command += ["-C", str(root)]
            return subprocess.run(command + list(args), env=env, check=True, stdout=subprocess.PIPE if capture else None,
                                  timeout=600).stdout
        git("clone", "--depth", "1", "--single-branch", "--branch", "main", "https://github.com/madvidd/Thesis.git", str(clone))
        rel = PACKAGE_PATH + "/Results/" + results.name
        destination = clone / rel
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, destination, dirs_exist_ok=True)
        validate_stage(destination)
        git("config", "user.name", "madviddd", root=clone)
        git("config", "user.email", "madviddd@users.noreply.github.com", root=clone)
        # Force only validated files, never a whole ignored Results hierarchy.
        git("add", "-f", "--", *[rel + "/" + name for name in files], root=clone)
        changed = git("diff", "--cached", "--name-only", capture=True, root=clone).strip()
        if changed:
            git("commit", "-m", "Update SEAM combined 80-epoch results", root=clone)
        for attempt in range(3):
            git("pull", "--no-rebase", "origin", "main", root=clone)
            try:
                git("push", "origin", "HEAD:main", root=clone)
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                time.sleep(10)
        git("fetch", "origin", "main", root=clone)
        for name in files:
            blob = git("show", "FETCH_HEAD:" + rel + "/" + name, capture=True, root=clone)
            if blob != (stage / name).read_bytes():
                raise RuntimeError(f"Remote verification failed: {name}")
        atomic_json(results / "PUBLICATION_OK.json", {"path": rel, "verified_files": len(files), "time": time.time()})
        print(f"REMOTE_SNAPSHOT_VERIFIED=True\nGitHub path: {rel}\nFull local transcript: {local}", flush=True)


def publish(experiment):
    import fcntl
    results = Path(read_json(experiment / "RUN_MANIFEST.json")["results"])
    with (results / ".publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        publish_locked(experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment")
    parser.add_argument("--auth-only", metavar="TOKEN_FILE")
    args = parser.parse_args()
    if args.auth_only:
        authenticate(Path(args.auth_only))
    elif args.experiment:
        publish(Path(args.experiment).resolve())
    else:
        parser.error("--experiment or --auth-only is required")
