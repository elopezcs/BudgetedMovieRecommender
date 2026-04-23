from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.env.movie_recommender_env import ACTIONS
from src.utils.io import save_dataframe


def _to_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _extract_obs_fields(obs: Any) -> dict[str, float | None]:
    if obs is None:
        return {
            "obs_uncertainty": None,
            "obs_engagement": None,
            "obs_budget_ratio": None,
            "obs_step_ratio": None,
        }

    flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    if flat.size < 12:
        raise ValueError("Expected observation vector with at least 12 values.")

    return {
        "obs_uncertainty": float(flat[5]),
        "obs_engagement": float(flat[10]),
        "obs_budget_ratio": float(flat[9]),
        "obs_step_ratio": float(flat[11]),
    }


def make_step_log_row(
    *,
    algorithm: str,
    seed: int,
    episode: int | None,
    global_timestep: int,
    episode_step: int,
    action_id: int,
    reward: float,
    done: bool,
    truncated: bool,
    info: dict[str, Any] | None,
    obs: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = info or {}
    row = {
        "algorithm": algorithm,
        "seed": seed,
        "episode": episode,
        "global_timestep": int(global_timestep),
        "episode_step": int(episode_step),
        "action_id": int(action_id),
        "action_name": ACTIONS[int(action_id)],
        "reward": float(reward),
        "done": bool(done),
        "truncated": bool(truncated),
        "accepted": _to_bool(info.get("accepted")),
        "abandoned": _to_bool(info.get("abandoned")),
        "question_asked": _to_bool(info.get("question_asked")),
        "over_budget": _to_bool(info.get("over_budget")),
        "recommended": _to_bool(info.get("recommended")),
        "genre_recommended": info.get("genre_recommended"),
        "user_type": info.get("user_type"),
        "step_count": info.get("step_count"),
        **_extract_obs_fields(obs),
    }
    if extra:
        row.update(extra)
    return row


class SB3TrainingStepLogger(BaseCallback):
    def __init__(
        self,
        *,
        algorithm: str,
        seed: int,
        output_path: str | Path,
    ) -> None:
        super().__init__(verbose=0)
        self.algorithm = algorithm
        self.seed = int(seed)
        self.output_path = Path(output_path)
        self.rows: list[dict[str, Any]] = []
        self._episode_by_env: list[int] = []
        self._episode_step_by_env: list[int] = []
        self._num_envs = 1

    def _on_training_start(self) -> None:
        self._num_envs = int(getattr(self.training_env, "num_envs", 1))
        self._episode_by_env = [0 for _ in range(self._num_envs)]
        self._episode_step_by_env = [0 for _ in range(self._num_envs)]

    def _on_step(self) -> bool:
        infos = list(self.locals.get("infos") or [])
        rewards = np.asarray(self.locals.get("rewards"))
        dones = np.asarray(self.locals.get("dones"))
        actions = np.asarray(self.locals.get("actions"))
        next_obs = self.locals.get("new_obs")
        next_obs_array = None if next_obs is None else np.asarray(next_obs)

        for env_idx, info in enumerate(infos):
            self._episode_step_by_env[env_idx] += 1
            done = _to_bool(dones[env_idx]) if dones.size else False
            truncated = _to_bool(info.get("TimeLimit.truncated"))
            action_value = actions[env_idx] if actions.ndim else actions.item()
            action_id = int(np.asarray(action_value).reshape(-1)[0])
            obs_value = None
            if next_obs_array is not None:
                obs_value = next_obs_array[env_idx] if next_obs_array.ndim > 1 else next_obs_array

            global_timestep = self.num_timesteps - (self._num_envs - env_idx - 1)
            self.rows.append(
                make_step_log_row(
                    algorithm=self.algorithm,
                    seed=self.seed,
                    episode=self._episode_by_env[env_idx],
                    global_timestep=global_timestep,
                    episode_step=self._episode_step_by_env[env_idx],
                    action_id=action_id,
                    reward=float(rewards[env_idx]),
                    done=done,
                    truncated=truncated,
                    info=info,
                    obs=obs_value,
                )
            )

            if done:
                self._episode_by_env[env_idx] += 1
                self._episode_step_by_env[env_idx] = 0

        return True

    def _on_training_end(self) -> None:
        save_dataframe(self.rows, self.output_path)
