
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from routing_poisoning import RoutingEnv


SEEDS = range(100, 120)
DELTAS = [0.0, 0.10, 0.25, 0.50]


def evaluate_condition(model, delta, seeds=SEEDS):
    """Evaluate one fixed PPO model under a given telemetry perturbation."""
    env = RoutingEnv(poison=(delta > 0), delta=delta)
    rows = []

    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            done = False
            final_info = {}
            actions = []

            while not done:
                # Predict the action using the same trained model.
                action, _ = model.predict(obs, deterministic=True)

                # Record the selected action for later comparison.
                actions.append(int(np.asarray(action).item()))

                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if done:
                    final_info = info

            rows.append({
                "seed": seed,
                "method": "ppo",
                "delta": delta,
                "condition": (
                    "clean" if delta == 0
                    else f"delta_{delta:.2f}"
                ),
                "actions": json.dumps(actions),
                **final_info,
            })

    finally:
        env.close()

    return rows


def main():
    model_path = Path("ppo_routing.zip")

    if not model_path.exists():
        raise FileNotFoundError(
            "ppo_routing.zip not found. "
            "Run routing_poisoning.py first."
        )

    # Load the existing trained model; do not retrain it.
    model = PPO.load(str(model_path))

    all_rows = []

    for delta in DELTAS:
        print(f"Evaluating PPO with poisoning delta={delta:.2f}...")
        all_rows.extend(evaluate_condition(model, delta))

    if not all_rows:
        raise RuntimeError("No evaluation results were produced.")

    output = Path("poisoning_strength_results.csv")

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} episode results to {output}")

    print("\nMean metrics by poisoning strength:")

    metrics = [
        "max_utilization",
        "mean_utilization",
        "squared_utilization",
    ]

    for delta in DELTAS:
        subset = [row for row in all_rows if row["delta"] == delta]

        print(
            f"delta={delta:.2f} | "
            f"max util={np.mean([r['max_utilization'] for r in subset]):.5f} | "
            f"mean util={np.mean([r['mean_utilization'] for r in subset]):.5f} | "
            f"squared util={np.mean([r['squared_utilization'] for r in subset]):.5f}"
        )

    print("\nAction-change analysis relative to clean telemetry:")

    clean_by_seed = {
        row["seed"]: json.loads(row["actions"])
        for row in all_rows
        if row["delta"] == 0.0
    }

    for delta in DELTAS:
        if delta == 0.0:
            continue

        attacked_rows = [
            row for row in all_rows if row["delta"] == delta
        ]

        changed_decisions = 0
        total_decisions = 0
        affected_episodes = 0

        for row in attacked_rows:
            seed = row["seed"]
            clean_actions = clean_by_seed[seed]
            attacked_actions = json.loads(row["actions"])

            if len(clean_actions) != len(attacked_actions):
                raise ValueError(
                    f"Action sequence length differs for seed {seed}, "
                    f"delta={delta:.2f}: clean={len(clean_actions)}, "
                    f"attacked={len(attacked_actions)}"
                )

            episode_changes = sum(
                clean_action != attacked_action
                for clean_action, attacked_action
                in zip(clean_actions, attacked_actions)
            )

            changed_decisions += episode_changes
            total_decisions += len(clean_actions)

            if episode_changes > 0:
                affected_episodes += 1

        percentage = (
            100.0 * changed_decisions / total_decisions
            if total_decisions else 0.0
        )

        print(
            f"delta={delta:.2f}: "
            f"{changed_decisions}/{total_decisions} decisions changed "
            f"({percentage:.2f}%), "
            f"{affected_episodes}/{len(attacked_rows)} episodes affected"
        )


if __name__ == "__main__":
    main()
