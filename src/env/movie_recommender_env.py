from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.reward import compute_step_reward
from src.env.user_simulator import GENRES, QUESTION_TYPES, UserSimulator

QUESTION_ACTIONS = [f"q_{q}" for q in QUESTION_TYPES]
RECOMMEND_ACTIONS = [f"rec_{g}" for g in GENRES]
ACTIONS = QUESTION_ACTIONS + RECOMMEND_ACTIONS


class BudgetedMovieRecommenderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Dict[str, Any], seed: int | None = None):
        super().__init__()
        self.config = config
        self.max_steps = int(config["environment"]["max_steps"])
        self.question_budget = int(config["environment"]["question_budget"])
        self.initial_uncertainty = float(config["environment"]["initial_uncertainty"])
        self.initial_engagement = float(config["environment"]["initial_engagement"])
        self.reward_cfg = dict(config["reward"])

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self.np_random = np.random.default_rng(seed)
        self.user = UserSimulator(self.np_random)
        self._last_action: int | None = None
        self._last_recommendation: int | None = None
        self._asked = 0
        self._recommended = 0
        self._repeats = 0
        self._step = 0
        self._user_type = "balanced_viewer"
        self._uncertainty = self.initial_uncertainty
        self._engagement = self.initial_engagement
        self._belief = np.ones(len(GENRES), dtype=np.float32) / len(GENRES)

    def _get_obs(self) -> np.ndarray:
        history = np.array(
            [
                self._asked / max(1, self.max_steps),
                self._recommended / max(1, self.max_steps),
                self._repeats / max(1, self.max_steps),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate(
            [
                self._belief,
                np.array([self._uncertainty], dtype=np.float32),
                history,
                np.array(
                    [
                        max(0.0, (self.question_budget - self._asked) / max(1, self.question_budget)),
                        self._engagement,
                        self._step / max(1, self.max_steps),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def _is_question(self, action: int) -> bool:
        return action < len(QUESTION_ACTIONS)

    def _recommendation_genre_index(self, action: int) -> int:
        return action - len(QUESTION_ACTIONS)

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.user = UserSimulator(self.np_random)
        options = options or {}
        self._user_type = self.user.reset(seed=seed, user_type=options.get("user_type"))
        self._step = 0
        self._asked = 0
        self._recommended = 0
        self._repeats = 0
        self._last_action = None
        self._last_recommendation = None
        self._uncertainty = self.initial_uncertainty
        self._engagement = self.initial_engagement
        self._belief = np.ones(len(GENRES), dtype=np.float32) / len(GENRES)
        return self._get_obs(), {"user_type": self._user_type}

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._step += 1
        repeated_action = self._last_action == action
        if repeated_action:
            self._repeats += 1

        accepted = False
        skipped = False
        abandoned = False
        question_asked = False
        recommended = False
        genre_recommended = None
        over_budget = False

        if self._is_question(action):
            question_asked = True
            self._asked += 1
            if self._asked > self.question_budget:
                over_budget = True

            qtype = QUESTION_TYPES[action]
            outcome = self.user.respond_to_question(qtype, repeated_action)
            self._engagement = float(np.clip(self._engagement + outcome["engagement_delta"], 0.0, 1.0))
            self._uncertainty = float(np.clip(self._uncertainty + outcome["uncertainty_delta"], 0.0, 1.0))

            hinted_idx = int(outcome["hinted_genre"])
            hint = np.zeros(len(GENRES), dtype=np.float32)
            hint[hinted_idx] = 1.0
            self._belief = 0.78 * self._belief + 0.22 * hint
            self._belief = self._belief / self._belief.sum()
            abandoned = bool(self.np_random.random() < float(outcome["abandon_prob"]))
        else:
            recommended = True
            self._recommended += 1
            genre_recommended = self._recommendation_genre_index(action)
            repeated_rec = self._last_recommendation == genre_recommended
            outcome = self.user.respond_to_recommendation(genre_recommended, repeated_rec)
            accepted = bool(outcome["accepted"])
            skipped = not accepted
            self._engagement = float(np.clip(self._engagement + outcome["engagement_delta"], 0.0, 1.0))
            abandoned = bool(self.np_random.random() < float(outcome["abandon_prob"]))
            self._last_recommendation = genre_recommended
            self._uncertainty = float(np.clip(self._uncertainty + (0.06 if skipped else -0.12), 0.0, 1.0))

        reward = compute_step_reward(
            self.reward_cfg,
            is_question=question_asked,
            accepted=accepted,
            skipped=skipped,
            abandoned=abandoned,
            engagement=self._engagement,
            over_budget=over_budget,
            repeated_action=repeated_action,
        )
        self._last_action = action

        terminated = abandoned
        truncated = self._step >= self.max_steps
        obs = self._get_obs()
        info = {
            "accepted": accepted,
            "abandoned": abandoned,
            "question_asked": question_asked,
            "recommended": recommended,
            "genre_recommended": GENRES[genre_recommended] if genre_recommended is not None else None,
            "user_type": self._user_type,
            "step_count": self._step,
        }
        return obs, reward, terminated, truncated, info

