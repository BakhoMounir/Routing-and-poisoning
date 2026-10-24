"""Train PPO agents on CLEAN observations with the original hyperparameters.

Hyperparameters are copied from originals/routing_poisoning.py:
    PPO("MlpPolicy", env, seed=S, n_steps=256, batch_size=64,
        learning_rate=3e-4), total_timesteps=20_000
Deviations (documented): device is fixed to "cpu" and torch uses 1 thread so
that training is bit-for-bit repeatable (verified: two same-seed runs gave
identical parameters on the authoring machine).

Usage:
    python scripts/train.py --seeds 42 43 44 --out results/models
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
import stable_baselines3  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from src.routing_env import RoutingEnv  # noqa: E402

HYPERPARAMS = dict(
    policy="MlpPolicy", n_steps=256, batch_size=64, learning_rate=3e-4,
    total_timesteps=20_000,
)


def param_hash(model):
    flat = torch.cat([p.flatten() for p in model.policy.parameters()]).detach().cpu().numpy()
    return hashlib.sha256(flat.tobytes()).hexdigest()


def train_one(seed, out_dir):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = RoutingEnv(poison=False)
    model = PPO(
        HYPERPARAMS["policy"], env, seed=seed,
        n_steps=HYPERPARAMS["n_steps"], batch_size=HYPERPARAMS["batch_size"],
        learning_rate=HYPERPARAMS["learning_rate"], verbose=0, device="cpu",
    )
    t0 = time.time()
    model.learn(total_timesteps=HYPERPARAMS["total_timesteps"])
    wall = time.time() - t0
    env.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ppo_seed{seed}"
    model.save(str(path))
    meta = {
        "train_seed": seed, **HYPERPARAMS, "device": "cpu", "torch_threads": 1,
        "train_env": "RoutingEnv(poison=False)  # clean observations only",
        "param_sha256": param_hash(model), "train_wall_seconds": round(wall, 1),
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__, "numpy": np.__version__,
        },
    }
    (out_dir / f"ppo_seed{seed}.json").write_text(json.dumps(meta, indent=2))
    print(f"trained seed={seed} in {wall:.1f}s -> {path}.zip", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "models")
    args = ap.parse_args()
    for s in args.seeds:
        train_one(s, args.out)


if __name__ == "__main__":
    main()
