"""Experiment manifest: config, seeds, code revision, environment, inputs, outputs, status. Stdlib only."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, atomic_write_text, sha256_file

CORE_SUBS = ("src/p2", "src/p2x")
PKGS = ["torch", "stable-baselines3", "gymnasium", "numpy", "pandas", "scipy", "networkx", "matplotlib", "pytest"]


def env_versions() -> dict:
    from importlib import metadata
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in PKGS:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def code_tree_sha256(root: Path = ROOT, subs=("src/p2", "src/p2x", "scripts")) -> str:
    """Hash of all .py files in the given sub-directories (works when the project is not a git repo).
    The freeze uses subs=CORE_SUBS (simulator + library only) so later edits to analysis scripts do not void a freeze."""
    h = hashlib.sha256()
    for sub in subs:
        for f in sorted((root / sub).glob("*.py")):
            h.update(f.relative_to(root).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def core_code_sha256() -> str:
    """Hash the simulator and overlay implementation together, independent of cwd."""
    import os

    project_root = Path(os.environ.get("P2X_PROJECT_ROOT", ROOT)).resolve()
    overlay_root = ROOT.resolve()
    roots = (
        ("project", project_root, ("src/p2",)),
        ("overlay", overlay_root, ("src/p2x",)),
    )
    h = hashlib.sha256()
    files = []
    for label, base, subs in roots:
        for sub in subs:
            folder = base / sub
            if not folder.is_dir():
                raise FileNotFoundError(f"core source directory missing: {folder}")
            for file in folder.rglob("*.py"):
                if "__pycache__" in file.parts:
                    continue
                files.append((label, base, file))
    for label, base, file in sorted(files, key=lambda item: (item[0], item[2].relative_to(item[1]).as_posix())):
        rel = f"{label}/{file.relative_to(base).as_posix()}"
        h.update(rel.encode("utf-8"))
        h.update(b"\\0")
        h.update(file.read_bytes())
        h.update(b"\\0")
    return h.hexdigest()


def git_revision(root: Path = ROOT):
    try:
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return None
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        return {"commit": rev.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return None


class RunManifest:
    """with RunManifest(path, name, cfg, args=..., inputs=[...], seeds={...}) as m: ...; m.add_output(p)"""

    def __init__(self, path, name, cfg, args=None, inputs=(), seeds=None, extra=None):
        self.path = Path(path)
        self.d = {
            "name": name, "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_path": cfg.get("_path"), "config_sha256": cfg.get("_sha256"),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in (args or {}).items()},
            "seeds": seeds or {}, "git": git_revision(), "code_tree_sha256": core_code_sha256(),
            "environment": env_versions(),
            "inputs": {}, "outputs": {}, "extra": extra or {},
        }
        for p in inputs:
            self.add_input(p)
        self._t0 = time.time()

    def add_input(self, p):
        p = Path(p)
        self.d["inputs"][str(p)] = sha256_file(p) if p.is_file() else "MISSING"

    def add_output(self, p):
        p = Path(p)
        self.d["outputs"][str(p)] = sha256_file(p) if p.is_file() else "MISSING"

    def set(self, key, value):
        self.d["extra"][key] = value

    def _write(self):
        atomic_write_text(self.path, json.dumps(self.d, indent=2, default=str))

    def __enter__(self):
        self._write()
        return self

    def __exit__(self, et, ev, tb):
        self.d["wall_s"] = round(time.time() - self._t0, 1)
        self.d["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if et is None:
            self.d["status"] = "completed"
            for p in list(self.d["outputs"]):
                self.add_output(p)
        else:
            self.d["status"] = "failed"
            self.d["error"] = "".join(traceback.format_exception_only(et, ev)).strip()
        self._write()
        return False
