"""Tiny shared CLI helpers (logging, failure, config loading) so scripts stay thin."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import json

from .config import ROOT, ConfigError, load_config, require_real_topology

log = logging.getLogger("p2x")


def setup(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def die(msg: str, code: int = 2):
    log.error(msg)
    sys.exit(code)


def get_cfg(path):
    try:
        return load_config(path)
    except (ConfigError, OSError, KeyError, ValueError) as e:
        die(f"invalid experiment config: {e}")


def need_file(p, what):
    p = Path(p)
    if not p.is_file():
        die(f"missing {what}: {p}")
    return p


def need_dir(p, what):
    p = Path(p)
    if not p.is_dir():
        die(f"missing {what}: {p}")
    return p


def add_common(ap):
    ap.add_argument("--config", type=Path, default=ROOT / "configs/x_experiments.json", help="experiment config (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <results_root>/<stage>)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing a completed run in --out")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def prepare_out(out: Path, overwrite: bool) -> Path:
    """Never silently overwrite a completed run: refuse unless --overwrite is given."""
    out = Path(out)
    mf = out / "manifest.json"
    if mf.is_file() and not overwrite:
        try:
            status = json.loads(mf.read_text()).get("status")
        except ValueError:
            status = "unreadable"
        if status == "completed":
            die(f"{out} already holds a completed run (manifest.json). Use a different --out or pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def real_network(cfg):
    """Topology guard + Network(). Returns (net, ned_sha256)."""
    try:
        sha = require_real_topology(cfg)
    except ConfigError as e:
        die(str(e))
    from p2.sim import Network
    return Network(), sha
