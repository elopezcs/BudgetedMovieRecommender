from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all algorithms sequentially.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    try:
        from src.training.train_q_learning import train_q_learning
    except Exception as exc:
        raise RuntimeError(
            "Q-learning trainer could not be imported. "
            "Check your environment and project dependencies."
        ) from exc

    train_dqn = None
    try:
        from src.training.train_dqn import train_dqn as _train_dqn
        train_dqn = _train_dqn
    except Exception:
        train_dqn = None

    train_ppo = None
    try:
        from src.training.train_ppo import train_ppo as _train_ppo
        train_ppo = _train_ppo
    except Exception:
        train_ppo = None

    q_run = train_q_learning(args.config, args.seed)
    dqn_run = None
    ppo_run = None

    if train_dqn is not None:
        dqn_run = train_dqn(args.config, args.seed)

    if train_ppo is not None:
        ppo_run = train_ppo(args.config, args.seed)
    print("Training complete")
    print(f"q_learning={q_run}")
    print(f"dqn={dqn_run if dqn_run is not None else 'SKIPPED (unavailable in this environment)'}")
    print(f"ppo={ppo_run if ppo_run is not None else 'SKIPPED (unavailable in this environment)'}")


if __name__ == "__main__":
    main()

