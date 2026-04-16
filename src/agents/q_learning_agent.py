from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class QLearningAgent:
    def __init__(
        self,
        action_size: int,
        learning_rate: float,
        discount_factor: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        bins: Dict[str, int],
    ):
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.bins = bins
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def discretize_state(self, obs: np.ndarray) -> Tuple[int, ...]:
        belief = obs[:5]
        uncertainty = obs[5]
        asked_ratio = obs[6]
        recommended_ratio = obs[7]
        repeats_ratio = obs[8]
        budget_ratio = obs[9]
        engagement = obs[10]
        step_ratio = obs[11]

        def b(value: float, n_bins: int) -> int:
            return min(n_bins - 1, int(value * n_bins))

        state = (
            int(np.argmax(belief)),
            b(float(uncertainty), self.bins["uncertainty"]),
            b(float(engagement), self.bins["engagement"]),
            b(float(budget_ratio), self.bins["budget"]),
            b(float(step_ratio), self.bins["step"]),
            b(float(asked_ratio), self.bins["asked"]),
            b(float(recommended_ratio), self.bins["recommended"]),
            b(float(repeats_ratio), self.bins["recommended"]),
        )
        return state

    def _ensure_state(self, state: Tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[state]

    def select_action(self, state: Tuple[int, ...], explore: bool = True) -> int:
        q_values = self._ensure_state(state)
        if explore and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.action_size))
        return int(np.argmax(q_values))

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool,
    ) -> None:
        q_values = self._ensure_state(state)
        next_q_values = self._ensure_state(next_state)
        td_target = reward + (0.0 if done else self.gamma * float(np.max(next_q_values)))
        q_values[action] = q_values[action] + self.lr * (td_target - q_values[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, output_dir: str | Path, metadata: Dict[str, Any]) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "q_table.npy", self.q_table, allow_pickle=True)
        payload = _to_builtin({"epsilon": float(self.epsilon), "bins": self.bins, **metadata})
        with (out / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, model_dir: str | Path) -> "QLearningAgent":
        model_path = Path(model_dir)
        with (model_path / "metadata.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        agent = cls(
            action_size=meta["action_size"],
            learning_rate=meta["learning_rate"],
            discount_factor=meta["discount_factor"],
            epsilon_start=meta.get("epsilon", 0.0),
            epsilon_min=meta.get("epsilon_min", 0.0),
            epsilon_decay=meta.get("epsilon_decay", 1.0),
            bins=meta["bins"],
        )
        table = np.load(model_path / "q_table.npy", allow_pickle=True).item()
        agent.q_table = {tuple(map(int, k)): v for k, v in table.items()}
        return agent

