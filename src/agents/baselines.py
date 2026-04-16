from __future__ import annotations

import random
from typing import Callable, Dict

from src.env.movie_recommender_env import QUESTION_ACTIONS, RECOMMEND_ACTIONS


def always_recommend_policy() -> Callable:
    action = len(QUESTION_ACTIONS)  # rec_action
    return lambda _obs, _step, _info: action


def always_ask_policy() -> Callable:
    return lambda _obs, step, _info: step % len(QUESTION_ACTIONS)


def ask_once_then_recommend_policy() -> Callable:
    recommend_action = len(QUESTION_ACTIONS)

    def _policy(_obs, step, _info):
        return 0 if step == 0 else recommend_action

    return _policy


def random_policy() -> Callable:
    total_actions = len(QUESTION_ACTIONS) + len(RECOMMEND_ACTIONS)
    return lambda _obs, _step, _info: random.randint(0, total_actions - 1)


def baseline_policies() -> Dict[str, Callable]:
    return {
        "always_recommend": always_recommend_policy(),
        "always_ask": always_ask_policy(),
        "ask_once_then_recommend": ask_once_then_recommend_policy(),
        "random_policy": random_policy(),
    }

