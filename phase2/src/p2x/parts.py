"""Resumable part files: one CSV per (method, model seed[, ...]); a part is skipped if it already exists; written atomically."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .config import atomic_write_text, sha256_file


def part_name(*bits) -> str:
    return "__".join(str(b).replace("/", "-").replace("@", "-at-") for b in bits) + ".csv"


def write_part(path: Path, rows: list) -> None:
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
    atomic_write_text(path, buf.getvalue())


def merge_parts(parts_dir: Path, out_csv: Path) -> int:
    files = sorted(Path(parts_dir).glob("*.csv"))
    rows, keys = [], []
    for f in files:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
                for k in r:
                    if k not in keys:
                        keys.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, restval="")
    w.writeheader()
    w.writerows(rows)
    atomic_write_text(out_csv, buf.getvalue())
    return len(rows)


def verify_freeze(cfg, freeze_file: Path, ned_sha: str, root: Path):
    """Return the freeze dict or raise SystemExit-style ValueError describing the first mismatch."""
    from .manifest import core_code_sha256
    if not Path(freeze_file).is_file():
        raise ValueError(f"no freeze at {freeze_file}; run scripts/x_05_freeze.py --yes first")
    fr = json.loads(Path(freeze_file).read_text())
    if fr["config_sha256"] != cfg["_sha256"]:
        raise ValueError("configs/x_experiments.json changed since the freeze")
    if fr["topology_sha256"] != ned_sha:
        raise ValueError("topology differs from the frozen one")
    if fr["core_code_sha256"] != core_code_sha256():
        raise ValueError("src/p2 or src/p2x changed since the freeze")
    for aid, entry in fr["models"].items():
        for s, m in entry.items():
            p = root / m["path"]
            if not p.is_file() or sha256_file(p) != m["sha256"]:
                raise ValueError(f"checkpoint {aid} seed {s} changed or missing: {p}")
    return fr
