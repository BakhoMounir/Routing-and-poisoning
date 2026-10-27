import os
"""Shared helpers for the p2x tests. The `net` fixture uses whatever topology is on disk: with the real NED these are real-topology tests;
in a sandbox without the NED they are LOGIC tests only (nothing they print is scientific evidence). The scripts' main() refuse a non-recorded topology."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("P2X_PROJECT_ROOT", ROOT)).resolve()


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class StubModel:
    """Duck-typed policy (predict/diag hooks). Depends on the observation, so poisoning can change its routes."""
    def __init__(self, agent, seed=0):
        self.agent, self.seed = agent, seed; self.w = np.random.default_rng(seed).normal(size=8)   # stand-in "parameters"
    def predict(self, obs, deterministic=True):
        if self.agent == "replace":
            return np.int64(int(abs(obs[:42].sum()) * 7) % 4), None
        return np.clip(0.8 * np.sin(obs[:42] * 5 + self.seed), -1, 1).astype(np.float32), None
    def diag_probe(self, obs):
        if self.agent == "replace":
            p = np.exp(obs[:4]); return p / p.sum()
        return 1.3 * self.predict(obs)[0]
    def diag_std(self): return np.full(42, 0.37)


def stub_hash(m): return "%016x" % abs(hash(m.w.tobytes()))
