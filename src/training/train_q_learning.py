from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.q_learning_agent import QLearningAgent
from src.evaluation.runner import run_policy_evaluation
from src.training.common import action_map, build_env, build_run_dirs
from src.training.step_logging import make_step_log_row
from src.utils.config import load_config
from src.utils.io import make_run_id, save_dataframe, save_json
from src.utils.seeding import set_global_seed


def train_q_learning(config_path: str, seed: int | None = None) -> str:
    config = load_config(config_path)
    seed = config["seed"] if seed is None else seed
    set_global_seed(seed)

    qcfg = config["q_learning"]
    env = build_env(config, seed)
    agent = QLearningAgent(
        action_size=env.action_space.n,
        learning_rate=qcfg["learning_rate"],
        discount_factor=qcfg["discount_factor"],
        epsilon_start=qcfg["epsilon_start"],
        epsilon_min=qcfg["epsilon_min"],
        epsilon_decay=qcfg["epsilon_decay"],
        bins=qcfg["discretization_bins"],
    )

    episodes = int(qcfg["episodes"])
    run_id = make_run_id("q_learning", seed, episodes)
    model_dir, result_dir = build_run_dirs("q_learning", run_id)

    train_rows = []
    step_rows = []
    global_timestep = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, truncated = False, False
        cumulative_reward = 0.0
        episode_step = 0
        while not done and not truncated:
            state = agent.discretize_state(obs)
            action = agent.select_action(state, explore=True)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_obs)
            agent.update(state, action, reward, next_state, done or truncated)
            obs = next_obs
            cumulative_reward += float(reward)
            global_timestep += 1
            episode_step += 1
            step_rows.append(
                make_step_log_row(
                    algorithm="q_learning",
                    seed=seed,
                    episode=ep,
                    global_timestep=global_timestep,
                    episode_step=episode_step,
                    action_id=action,
                    reward=float(reward),
                    done=done or truncated,
                    truncated=truncated,
                    info=info,
                    obs=next_obs,
                    extra={
                        "epsilon": float(agent.epsilon),
                        "cumulative_reward": cumulative_reward,
                    },
                )
            )
        agent.decay_epsilon()
        train_rows.append({"episode": ep, "reward": cumulative_reward, "epsilon": agent.epsilon})

    metadata = {
        "algorithm": "q_learning",
        "seed": seed,
        "action_size": env.action_space.n,
        "action_map": action_map(),
        "episodes": episodes,
        "learning_rate": qcfg["learning_rate"],
        "discount_factor": qcfg["discount_factor"],
        "epsilon_min": qcfg["epsilon_min"],
        "epsilon_decay": qcfg["epsilon_decay"],
        "state_features_version": 2,
        "greedy_tie_break": "deterministic_state_hash",
        "config_path": str(Path(config_path)),
    }
    agent.save(model_dir, metadata)

    def eval_policy(obs, _step, _info):
        state = agent.discretize_state(obs)
        return agent.select_action(state, explore=False)

    summary, detail_df = run_policy_evaluation(
        env_factory=lambda: build_env(config, seed),
        policy_fn=eval_policy,
        episodes=int(qcfg["eval_episodes"]),
        seed=seed,
        algorithm_name="q_learning",
    )

    save_dataframe(detail_df, result_dir / "evaluation_episodes.csv")
    save_dataframe(train_rows, result_dir / "training_curve.csv")
    save_dataframe(step_rows, result_dir / "training_steps.csv")
    save_json(summary.to_dict(), result_dir / "summary.json")
    save_json(config, result_dir / "config_snapshot.json")
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular Q-learning agent.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_id = train_q_learning(args.config, args.seed)
    print(f"Q-learning complete. Run ID: {run_id}")


if __name__ == "__main__":
    main()

