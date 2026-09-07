import hashlib
import json
import os
import re
from pathlib import Path


METRICS = ("MR", "minADE1", "minADE6", "minFDE1", "minFDE6", "b-minFDE6")
PACKAGE_PATH = "Studies/SEAM/Combined_Extension"
TOKEN_PATTERN = re.compile(rb"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(root):
    return {
        path.relative_to(root).as_posix(): source_sha256(path)
        for path in sorted(Path(root).rglob("*"))
        if path.is_file() and path.suffix in {".py", ".yaml", ".json", ".sh"}
        and "__pycache__" not in path.parts
        and path.name not in {"SOURCE_HASHES.json", "RUN_MANIFEST.json", "AUTH_CHECK.json"}
    }


def source_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def assert_sources(experiment):
    expected = read_json(Path(experiment) / "SOURCE_HASHES.json")
    observed = source_inventory(experiment)
    if expected != observed:
        changed = sorted(set(expected) ^ set(observed) | {
            key for key in expected.keys() & observed.keys() if expected[key] != observed[key]
        })
        raise RuntimeError("Immutable experiment source changed: " + ", ".join(changed))


def token_from_file(path):
    data = Path(path).read_bytes()
    candidates = set()
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            text = data.decode(encoding)
        except UnicodeError:
            continue
        candidates.update(x.decode() for x in TOKEN_PATTERN.findall(text.encode()))
    if len(candidates) != 1:
        raise RuntimeError("Token.txt must contain exactly one GitHub PAT")
    return candidates.pop()


def checkpoint_metric(path):
    match = re.search(r"minADE6_([0-9]+[.][0-9]+)", Path(path).name)
    if not match:
        raise ValueError("Checkpoint filename has no minADE6 value")
    return float(match[1])


def choose_epoch_checkpoint(directory, load):
    """Only completed-epoch checkpoints are eligible; never use another run."""
    directory = Path(directory)
    candidates = list(directory.glob("epoch_*.ckpt"))
    candidates.sort(key=lambda p: (int(re.search(r"epoch_(\d+)", p.name)[1]), p.stat().st_mtime), reverse=True)
    if (directory / "last.ckpt").is_file():
        candidates.insert(0, directory / "last.ckpt")
    failures = []
    for path in candidates:
        try:
            state = load(path)
            if not state.get("state_dict") or not state.get("optimizer_states") or not state.get("lr_schedulers"):
                raise ValueError("Not a full training checkpoint")
            if state.get("combined_protocol") != "seam80_combined_v1":
                raise ValueError("Checkpoint does not belong to this combined protocol")
            return path, state, failures
        except Exception as exc:
            failures.append({"path": str(path), "error": type(exc).__name__})
    if candidates:
        raise RuntimeError("No readable complete checkpoint; existing files were retained")
    if any(directory.glob("*.ckpt")):
        raise RuntimeError("Unrecognised checkpoints exist; refusing a silent new training run")
    return None, None, failures
