from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


@dataclass
class EvalSummary:
    average_cumulative_reward: float
    acceptance_rate: float
    abandonment_rate: float
    average_questions_asked: float
    average_session_length: float
    algorithm: str

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "algorithm": self.algorithm,
            "average_cumulative_reward": self.average_cumulative_reward,
            "acceptance_rate": self.acceptance_rate,
            "abandonment_rate": self.abandonment_rate,
            "average_questions_asked": self.average_questions_asked,
            "average_session_length": self.average_session_length,
        }


def run_policy_evaluation(
    env_factory: Callable[[], object],
    policy_fn: Callable[[object, int, Dict], int],
    episodes: int,
    seed: int,
    algorithm_name: str,
) -> Tuple[EvalSummary, List[Dict]]:
    rows: List[Dict] = []
    total_reward = 0.0
    total_recommendations = 0
    total_acceptances = 0
    abandoned_sessions = 0
    total_questions = 0
    total_steps = 0

    for ep in range(episodes):
        env = env_factory()
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        ep_questions = 0
        ep_recs = 0
        ep_accepts = 0
        ep_abandoned = 0
        ep_steps = 0
        step_idx = 0

        while not done and not truncated:
            action = int(policy_fn(obs, step_idx, {}))
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            step_idx += 1
            ep_questions += int(bool(info["question_asked"]))
            ep_recs += int(bool(info["recommended"]))
            ep_accepts += int(bool(info["accepted"]))
            ep_abandoned = max(ep_abandoned, int(bool(info["abandoned"])))

        rows.append(
            {
                "episode": ep,
                "cumulative_reward": ep_reward,
                "recommendations": ep_recs,
                "acceptances": ep_accepts,
                "abandoned": ep_abandoned,
                "questions_asked": ep_questions,
                "session_length": ep_steps,
            }
        )
        total_reward += ep_reward
        total_recommendations += ep_recs
        total_acceptances += ep_accepts
        abandoned_sessions += ep_abandoned
        total_questions += ep_questions
        total_steps += ep_steps

    summary = EvalSummary(
        algorithm=algorithm_name,
        average_cumulative_reward=total_reward / episodes,
        acceptance_rate=(
            float(total_acceptances) / max(1, float(total_recommendations))
        ),
        abandonment_rate=float(abandoned_sessions) / float(episodes),
        average_questions_asked=float(total_questions) / float(episodes),
        average_session_length=float(total_steps) / float(episodes),
    )
    return summary, rows

