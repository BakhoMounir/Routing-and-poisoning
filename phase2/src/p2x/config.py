"""Central configuration, seed registry and output-path guards. Stdlib only (no numpy/torch at import)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("P2X_PROJECT_ROOT", ROOT)).resolve()
DEFAULT_CONFIG = ROOT / "configs/x_experiments.json"


class ConfigError(ValueError):
    """Raised for invalid or unsafe experiment configuration."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_file():
        raise ConfigError(f"experiment config not found: {path}")
    cfg = json.loads(path.read_text())
    cfg["_path"] = str(path)
    cfg["_sha256"] = sha256_file(path)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    """Fail loudly on anything that could silently invalidate a comparison."""
    ranges = {k: tuple(v) for k, v in cfg["seeds"]["ranges"].items()}
    for k, (a, b) in ranges.items():
        if not (isinstance(a, int) and isinstance(b, int) and a < b):
            raise ConfigError(f"seed range {k} must be integer [start, stop) with start < stop")
    names = sorted(ranges, key=lambda k: ranges[k][0])
    for x, y in zip(names, names[1:]):
        if ranges[x][1] > ranges[y][0]:
            raise ConfigError(f"seed ranges overlap: {x}={ranges[x]} and {y}={ranges[y]}")
    for role, rs in cfg["seeds"]["allowed_roles"].items():
        for r in rs:
            if r not in ranges:
                raise ConfigError(f"role {role} references unknown range {r}")
    for role in ("development", "confirmatory", "poison"):
        bad = set(cfg["seeds"]["allowed_roles"][role]) & set(cfg["seeds"]["legacy_ranges"])
        if bad:
            raise ConfigError(f"role {role} must not use legacy ranges: {sorted(bad)}")
    sizes = (("diagnostics", "n_per_load", cfg["diagnostics"]["range"]),
             ("train_agent_a", "n_val_per_load", cfg["train_agent_a"]["val_range"]),
             ("confirmatory", "n_per_load", cfg["confirmatory"]["range"]))
    for sec, key, rng in sizes:
        if cfg[sec][key] > ranges[rng][1] - ranges[rng][0]:
            raise ConfigError(f"{sec}.{key}={cfg[sec][key]} exceeds size of seed range {rng}")
    for name, p in cfg["poisoning"]["presets"].items():
        a, b = ranges[p["range"]]
        if p["n_per_load"] > b - a:
            raise ConfigError(f"poison preset {name}: n_per_load exceeds range {p['range']}")
        if p["range"] not in cfg["seeds"]["allowed_roles"][p["role"]]:
            raise ConfigError(f"poison preset {name}: range {p['range']} not allowed for role {p['role']}")
    t = cfg["train_agent_a"]
    if t["max_steps"] % t["eval_every"] != 0 or t["eval_every"] % cfg["ppo_hp"]["n_steps"] != 0:
        raise ConfigError("train_agent_a: max_steps must be a multiple of eval_every, and eval_every of ppo_hp.n_steps")
    s = t["stopping"]
    if not (s["smooth_evals"] >= 1 and s["window_evals"] >= 1 and s["min_evals"] >= s["smooth_evals"] + s["window_evals"]):
        raise ConfigError("train_agent_a.stopping: min_evals must be >= smooth_evals + window_evals")
    ids = {a["id"] for a in cfg["confirmatory"]["agents"]} | set(cfg["confirmatory"]["baselines"])
    for c in cfg["confirmatory"]["primary_family"] + cfg["confirmatory"]["secondary"]:
        for k in ("test", "ref"):
            if c[k] not in ids:
                raise ConfigError(f"comparison references unknown method id: {c[k]}")
    protected = [Path(p).as_posix().rstrip("/") for p in cfg["legacy"]["protected_dirs"]]
    rr = Path(cfg["results_root"]).as_posix().rstrip("/")
    if any(rr == p or rr.startswith(p + "/") for p in protected):
        raise ConfigError(f"results_root {rr} lies inside a protected legacy directory")


def seed_range(cfg, name):
    a, b = cfg["seeds"]["ranges"][name]
    return a, b


def require_role(cfg, role: str, range_name: str) -> None:
    allowed = cfg["seeds"]["allowed_roles"].get(role)
    if allowed is None or range_name not in allowed:
        raise ConfigError(f"role '{role}' may not use seed range '{range_name}' (allowed: {allowed}). "
                          "This guard prevents development decisions on confirmatory/test seeds.")


def scenarios(cfg, range_name: str, loads, n_per_load: int, role: str):
    """[(scenario_idx, load, Scenario)] with seed = range.start + idx (same pairs at every load)."""
    from p2.sim import make_scenario  # lazy: keeps this module stdlib-only
    require_role(cfg, role, range_name)
    a, b = seed_range(cfg, range_name)
    if n_per_load > b - a:
        raise ConfigError(f"n_per_load={n_per_load} exceeds range {range_name} size {b - a}")
    return [(i, float(L), make_scenario(a + i, float(L))) for L in loads for i in range(n_per_load)]


def results_dir(cfg, *parts) -> Path:
    p = PROJECT_ROOT / cfg["results_root"]
    for x in parts:
        p = p / x
    return p


def assert_not_protected(cfg, path) -> None:
    path = Path(path).resolve()
    for d in cfg["legacy"]["protected_dirs"]:
        prot = (PROJECT_ROOT / d).resolve()
        if path == prot or prot in path.parents:
            raise ConfigError(f"refusing to write inside protected legacy path: {path}")


def lock_path(cfg, run_id: str) -> Path:
    return results_dir(cfg, run_id, "LOCK.json")


def freeze_path(cfg) -> Path:
    return results_dir(cfg, "freeze.json")


def require_real_topology(cfg) -> str:
    """Refuse to continue unless the NED on disk is byte-identical to the one recorded in the provenance files."""
    ned = PROJECT_ROOT / cfg["topology"]["ned"]
    if not ned.is_file():
        raise ConfigError(f"topology file missing: {ned}")
    got, want = sha256_file(ned), cfg["topology"]["sha256"]
    if got != want:
        raise ConfigError(f"topology SHA-256 mismatch for {ned}: got {got}, expected {want}. "
                          "Experiments must run on the recorded NSFNET-bw file; a stand-in is never acceptable as evidence.")
    return got


def agent_models_dir(ag: dict) -> Path:
    """Checkpoint directory of a confirmatory agent entry. Legacy layout is results/models/<agent>/seed<N>/, new runs use <models_dir>/seed<N>/."""
    d = PROJECT_ROOT / ag["models_dir"]
    return d / ag["agent"] if Path(ag["models_dir"]).as_posix().rstrip("/") == "results/models" else d
