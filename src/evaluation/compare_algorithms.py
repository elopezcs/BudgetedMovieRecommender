from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.agents.baselines import ask_once_then_recommend_policy
from src.evaluation.runner import run_policy_evaluation
from src.training.common import build_env
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json, timestamp_tag


def _collect_latest_rl_summaries() -> list[dict]:
    rows = []
    for algo in ["q_learning", "dqn", "ppo"]:
        root = Path("results") / algo
        if not root.exists():
            continue
        summary_paths = [run_dir / "summary.json" for run_dir in root.iterdir() if run_dir.is_dir()]
        summary_paths = [path for path in summary_paths if path.exists()]
        if not summary_paths:
            continue
        summary_path = max(summary_paths, key=lambda path: path.stat().st_mtime)
        rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
    return rows


def compare_algorithms(config_path: str, episodes: int | None = None) -> Path:
    config = load_config(config_path)
    seed = int(config["seed"])
    eval_episodes = episodes or int(config["evaluation"]["episodes"])
    env_factory = lambda: build_env(config, seed)

    rows = _collect_latest_rl_summaries()
    summary, _ = run_policy_evaluation(
        env_factory=env_factory,
        policy_fn=ask_once_then_recommend_policy(),
        episodes=eval_episodes,
        seed=seed,
        algorithm_name="ask_once_then_recommend",
    )
    rows.append(summary.to_dict())
    rows = sorted(rows, key=lambda x: x["average_cumulative_reward"], reverse=True)
    out_dir = ensure_dir(Path("results/comparisons") / timestamp_tag())
    if rows:
        with (out_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        (out_dir / "comparison_summary.csv").write_text("", encoding="utf-8")
    save_json(rows, out_dir / "comparison_summary.json")

    plt.figure(figsize=(10, 5))
    plt.bar(
        [row["algorithm"] for row in rows],
        [row["average_cumulative_reward"] for row in rows],
    )
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Avg Cumulative Reward")
    plt.title("Algorithm Comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_reward_chart.png")
    plt.close()
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the latest RL algorithms plus ask_once_then_recommend.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    out_dir = compare_algorithms(args.config, args.episodes)
    print(f"Saved comparison outputs in {out_dir}")


if __name__ == "__main__":
    main()

