from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

GENRES = ["action", "comedy", "drama", "scifi", "documentary"]
QUESTION_TYPES = [
    "familiar",
    "exploratory",
    "serious",
    "light",
    "fast_paced",
    "calm",
]


@dataclass(frozen=True)
class UserProfile:
    name: str
    base_preferences: np.ndarray
    question_tolerance: float
    repetition_sensitivity: float
    abandonment_threshold: float
    engagement_gain_on_good_question: float
    engagement_loss_on_bad_question: float


PROFILE_LIBRARY: Dict[str, UserProfile] = {
    "action_focused": UserProfile(
        name="action_focused",
        base_preferences=np.array([0.5, 0.15, 0.1, 0.2, 0.05], dtype=np.float32),
        question_tolerance=0.55,
        repetition_sensitivity=0.3,
        abandonment_threshold=0.17,
        engagement_gain_on_good_question=0.08,
        engagement_loss_on_bad_question=0.08,
    ),
    "balanced_viewer": UserProfile(
        name="balanced_viewer",
        base_preferences=np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32),
        question_tolerance=0.8,
        repetition_sensitivity=0.2,
        abandonment_threshold=0.11,
        engagement_gain_on_good_question=0.06,
        engagement_loss_on_bad_question=0.05,
    ),
    "novelty_seeking": UserProfile(
        name="novelty_seeking",
        base_preferences=np.array([0.12, 0.15, 0.13, 0.33, 0.27], dtype=np.float32),
        question_tolerance=0.72,
        repetition_sensitivity=0.45,
        abandonment_threshold=0.19,
        engagement_gain_on_good_question=0.09,
        engagement_loss_on_bad_question=0.09,
    ),
    "question_sensitive": UserProfile(
        name="question_sensitive",
        base_preferences=np.array([0.22, 0.24, 0.24, 0.15, 0.15], dtype=np.float32),
        question_tolerance=0.35,
        repetition_sensitivity=0.5,
        abandonment_threshold=0.23,
        engagement_gain_on_good_question=0.04,
        engagement_loss_on_bad_question=0.12,
    ),
}


QUESTION_GENRE_AFFINITY = {
    "familiar": np.array([0.35, 0.25, 0.2, 0.1, 0.1], dtype=np.float32),
    "exploratory": np.array([0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32),
    "serious": np.array([0.08, 0.08, 0.5, 0.12, 0.22], dtype=np.float32),
    "light": np.array([0.15, 0.48, 0.07, 0.22, 0.08], dtype=np.float32),
    "fast_paced": np.array([0.42, 0.17, 0.08, 0.28, 0.05], dtype=np.float32),
    "calm": np.array([0.08, 0.12, 0.35, 0.15, 0.3], dtype=np.float32),
}


class UserSimulator:
    """Stochastic user simulator with hidden preference state."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.profile: UserProfile | None = None
        self.true_preferences = np.ones(len(GENRES), dtype=np.float32) / len(GENRES)
        self.engagement = 0.8

    def reset(self, seed: int | None = None, user_type: str | None = None) -> str:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if user_type is None:
            user_type = self.rng.choice(list(PROFILE_LIBRARY.keys())).item()
        self.profile = PROFILE_LIBRARY[user_type]
        noise = self.rng.normal(0.0, 0.04, size=len(GENRES)).astype(np.float32)
        raw = np.clip(self.profile.base_preferences + noise, 0.01, 1.0)
        self.true_preferences = raw / raw.sum()
        self.engagement = 0.8 + float(self.rng.uniform(-0.1, 0.1))
        return self.profile.name

    def respond_to_question(
        self,
        question_type: str,
        repeated: bool,
    ) -> Dict[str, float | int | bool]:
        assert self.profile is not None
        affinity = QUESTION_GENRE_AFFINITY[question_type]
        alignment = float(np.dot(affinity, self.true_preferences))

        quality = alignment * self.profile.question_tolerance
        if repeated:
            quality -= self.profile.repetition_sensitivity * 0.25

        good_question = quality >= 0.17
        if good_question:
            engagement_delta = self.profile.engagement_gain_on_good_question
            uncertainty_delta = -0.13
        else:
            engagement_delta = -self.profile.engagement_loss_on_bad_question
            uncertainty_delta = 0.05

        friction = max(0.0, 0.5 - quality)
        abandon_prob = self.profile.abandonment_threshold + friction * 0.06
        if repeated:
            abandon_prob += 0.05 * self.profile.repetition_sensitivity

        hinted_genre = int(
            self.rng.choice(np.arange(len(GENRES)), p=affinity / affinity.sum())
        )
        return {
            "engagement_delta": engagement_delta,
            "uncertainty_delta": uncertainty_delta,
            "hinted_genre": hinted_genre,
            "abandon_prob": float(np.clip(abandon_prob, 0.0, 0.95)),
        }

    def respond_to_recommendation(
        self, genre_index: int, repeated: bool
    ) -> Dict[str, float | bool]:
        assert self.profile is not None
        pref = float(self.true_preferences[genre_index])
        repeated_penalty = self.profile.repetition_sensitivity * (0.12 if repeated else 0.0)
        accept_prob = np.clip(0.1 + 0.78 * pref + 0.15 * self.engagement - repeated_penalty, 0.0, 0.98)
        accepted = bool(self.rng.random() < accept_prob)

        if accepted:
            engagement_delta = 0.08
            abandon_prob = max(0.0, self.profile.abandonment_threshold - 0.15)
        else:
            engagement_delta = -0.08
            abandon_prob = self.profile.abandonment_threshold + (0.10 * (1.0 - pref))
            if repeated:
                abandon_prob += 0.05

        return {
            "accepted": accepted,
            "engagement_delta": float(engagement_delta),
            "abandon_prob": float(np.clip(abandon_prob, 0.0, 0.98)),
        }

