from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from stable_baselines3 import DQN, PPO

from src.agents.q_learning_agent import QLearningAgent
from src.env.movie_recommender_env import ACTIONS


def _latest_run_dir(base_dir: str | Path) -> Path:
    root = Path(base_dir)
    if not root.exists():
        raise FileNotFoundError(f"Missing model root: {root}")
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories under {root}")
    return run_dirs[-1]


class PolicyAdapter:
    """Unified inference interface for q_learning, dqn, and ppo."""

    def __init__(self, algo: str, model_dir: str | None = None):
        self.algo = algo
        self.model_dir = Path(model_dir) if model_dir else self._default_model_dir(algo)
        self.model: Any = None
        self._load()

    def _default_model_dir(self, algo: str) -> Path:
        return _latest_run_dir(Path("models") / algo)

    def _load(self) -> None:
        if self.algo == "q_learning":
            self.model = QLearningAgent.load(self.model_dir)
        elif self.algo == "dqn":
            self.model = DQN.load(self.model_dir / "model")
        elif self.algo == "ppo":
            self.model = PPO.load(self.model_dir / "model")
        else:
            raise ValueError(f"Unsupported algorithm: {self.algo}")

    def predict_action(self, observation: list[float]) -> Dict[str, Any]:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (12,):
            raise ValueError("observation must be a float list of length 12")

        if self.algo == "q_learning":
            state = self.model.discretize_state(obs)
            action_id = int(self.model.select_action(state, explore=False))
        else:
            action_id, _ = self.model.predict(obs, deterministic=True)
            action_id = int(action_id)

        return {
            "algorithm": self.algo,
            "model_dir": str(self.model_dir),
            "action_id": action_id,
            "action_name": ACTIONS[action_id],
        }


def _parse_session_payload(payload: Dict[str, Any]) -> list[float]:
    if "observation" not in payload:
        raise ValueError("Input JSON must include 'observation'")
    observation = payload["observation"]
    if not isinstance(observation, list):
        raise ValueError("'observation' must be a list")
    return [float(x) for x in observation]


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified policy inference adapter.")
    parser.add_argument("--algo", choices=["q_learning", "dqn", "ppo"], required=True)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="Optional path to request JSON. If omitted, read from STDIN.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write response JSON. If omitted, print to STDOUT.",
    )
    args = parser.parse_args()

    request: Dict[str, Any]
    if args.input_json:
        request = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    else:
        request = json.loads(input())

    adapter = PolicyAdapter(args.algo, args.model_dir)
    observation = _parse_session_payload(request)
    response = adapter.predict_action(observation)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(response, indent=2), encoding="utf-8")
    else:
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()

