from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.evaluation.runner import run_policy_evaluation
from src.training.common import action_map, build_env, build_run_dirs
from src.training.step_logging import SB3TrainingStepLogger
from src.utils.config import load_config
from src.utils.io import make_run_id, save_dataframe, save_json
from src.utils.seeding import set_global_seed


def train_ppo(config_path: str, seed: int | None = None) -> str:
    config = load_config(config_path)
    seed = config["seed"] if seed is None else seed
    set_global_seed(seed)
    ppo_cfg = config["ppo"]
    total_timesteps = int(ppo_cfg["total_timesteps"])
    run_id = make_run_id("ppo", seed, total_timesteps)
    model_dir, result_dir = build_run_dirs("ppo", run_id)

    env = Monitor(build_env(config, seed))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        learning_rate=float(ppo_cfg["learning_rate"]),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        ent_coef=float(ppo_cfg["ent_coef"]),
    )
    step_logger = SB3TrainingStepLogger(
        algorithm="ppo",
        seed=seed,
        output_path=result_dir / "training_steps.csv",
    )
    model.learn(total_timesteps=total_timesteps, callback=step_logger)
    model.save(model_dir / "model")

    def policy(obs, _step, _info):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    summary, detail_df = run_policy_evaluation(
        env_factory=lambda: build_env(config, seed),
        policy_fn=policy,
        episodes=int(ppo_cfg["eval_episodes"]),
        seed=seed,
        algorithm_name="ppo",
    )

    metadata = {
        "algorithm": "ppo",
        "seed": seed,
        "timesteps": total_timesteps,
        "action_map": action_map(),
        "hyperparameters": ppo_cfg,
    }
    save_dataframe(detail_df, result_dir / "evaluation_episodes.csv")
    save_json(summary.to_dict(), result_dir / "summary.json")
    save_json(metadata, model_dir / "metadata.json")
    save_json(config, result_dir / "config_snapshot.json")
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on movie environment.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_id = train_ppo(args.config, args.seed)
    print(f"PPO complete. Run ID: {run_id}")


if __name__ == "__main__":
    main()

