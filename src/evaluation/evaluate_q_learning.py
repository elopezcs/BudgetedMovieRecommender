from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.q_learning_agent import QLearningAgent
from src.evaluation.helpers import latest_run_dir
from src.evaluation.runner import run_policy_evaluation
from src.training.common import build_env
from src.utils.config import load_config
from src.utils.io import save_dataframe, save_json


def evaluate_q_learning(config_path: str, model_dir: str | None, episodes: int | None) -> None:
    config = load_config(config_path)
    seed = int(config["seed"])
    run_dir = latest_run_dir("models/q_learning") if model_dir is None else Path(model_dir)
    agent = QLearningAgent.load(run_dir)

    def policy(obs, _step, _info):
        state = agent.discretize_state(obs)
        return agent.select_action(state, explore=False)

    eval_eps = episodes or int(config["evaluation"]["episodes"])
    summary, detail_df = run_policy_evaluation(
        env_factory=lambda: build_env(config, seed),
        policy_fn=policy,
        episodes=eval_eps,
        seed=seed,
        algorithm_name="q_learning",
    )
    result_dir = Path("results/q_learning") / run_dir.name
    result_dir.mkdir(parents=True, exist_ok=True)
    save_dataframe(detail_df, result_dir / "evaluation_episodes.csv")
    save_json(summary.to_dict(), result_dir / "summary.json")
    print(f"Saved Q-learning evaluation in {result_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Q-learning model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    evaluate_q_learning(args.config, args.model_dir, args.episodes)


if __name__ == "__main__":
    main()

