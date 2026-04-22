from __future__ import annotations

from typing import Dict


def compute_step_reward(
    reward_cfg: Dict[str, float],
    *,
    is_question: bool,
    accepted: bool,
    skipped: bool,
    abandoned: bool,
    engagement: float,
    uncertainty_delta: float,
    over_budget: bool,
    repeated_action: bool,
) -> float:
    reward = 0.0
    if accepted:
        reward += reward_cfg["acceptance_reward"]
    if skipped:
        reward += reward_cfg["skip_penalty"]
    if is_question:
        reward += reward_cfg["question_cost"]
        if uncertainty_delta < 0.0:
            reward += reward_cfg["question_information_gain_scale"] * float(-uncertainty_delta)
    if over_budget:
        reward += reward_cfg["over_budget_penalty"]
    if abandoned:
        reward += reward_cfg["abandonment_penalty"]
    if repeated_action:
        reward += reward_cfg["repetition_penalty"]

    reward += reward_cfg["engagement_bonus_scale"] * float(engagement)
    return float(reward)

