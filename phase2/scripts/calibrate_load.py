"""Pre-registered load calibration using ONLY baselines on VALIDATION scenarios (no PPO involved)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from p2.sim import Network, Episode, make_scenario, VAL_SEED_BASE
from p2.baselines import BASELINES
net = Network(); rows = {}
for L in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    row = {}
    for name, fn in BASELINES.items():
        bl, mu, hp = [], [], []
        for i in range(100):
            ep = Episode(net, make_scenario(VAL_SEED_BASE + i, L))
            while not ep.done: ep.route(fn(net, ep, ep.true_util()))
            m = ep.metrics(); bl.append(m["blocking_ratio"]); mu.append(m["max_utilization"]); hp.append(m["mean_hops_accepted"])
        row[name] = dict(blocking=round(float(np.mean(bl)), 4), max_util=round(float(np.mean(mu)), 3), hops=round(float(np.mean(hp)), 3))
    rows[L] = row; print(L, row, flush=True)
(ROOT / "results/load_calibration_validation.json").write_text(json.dumps(rows, indent=2))
