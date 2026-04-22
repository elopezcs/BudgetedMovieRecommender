from copy import deepcopy

from src.env.movie_recommender_env import BudgetedMovieRecommenderEnv
from src.env.reward import compute_step_reward
from src.utils.config import load_config


def test_env_reset_and_step_shape():
    config = load_config("configs/default.yaml")
    env = BudgetedMovieRecommenderEnv(config, seed=123)
    obs, info = env.reset(seed=123)
    assert obs.shape == (12,)
    assert "user_type" in info

    next_obs, reward, terminated, truncated, step_info = env.step(0)
    assert next_obs.shape == (12,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "question_asked" in step_info


def test_env_respects_max_steps_truncation():
    config = load_config("configs/default.yaml")
    config["environment"]["max_steps"] = 3
    env = BudgetedMovieRecommenderEnv(config, seed=1)
    env.reset(seed=1, options={"user_type": "balanced_viewer"})
    terminated = False
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert terminated or truncated


def test_reward_function_penalizes_abandonment():
    config = load_config("configs/default.yaml")
    reward_cfg = config["reward"]
    reward_ok = compute_step_reward(
        reward_cfg,
        is_question=False,
        accepted=True,
        skipped=False,
        abandoned=False,
        engagement=0.8,
        uncertainty_delta=0.0,
        over_budget=False,
        repeated_action=False,
    )
    reward_bad = compute_step_reward(
        reward_cfg,
        is_question=False,
        accepted=False,
        skipped=True,
        abandoned=True,
        engagement=0.2,
        uncertainty_delta=0.06,
        over_budget=False,
        repeated_action=False,
    )
    assert reward_ok > reward_bad


def test_reward_function_rewards_information_gain_on_questions():
    config = load_config("configs/default.yaml")
    reward_cfg = config["reward"]
    helpful_question = compute_step_reward(
        reward_cfg,
        is_question=True,
        accepted=False,
        skipped=False,
        abandoned=False,
        engagement=0.8,
        uncertainty_delta=-0.13,
        over_budget=False,
        repeated_action=False,
    )
    flat_question = compute_step_reward(
        reward_cfg,
        is_question=True,
        accepted=False,
        skipped=False,
        abandoned=False,
        engagement=0.8,
        uncertainty_delta=0.0,
        over_budget=False,
        repeated_action=False,
    )
    assert helpful_question > flat_question


def test_soft_budget_allows_extra_questions_in_auto_mode():
    config = load_config("configs/default.yaml")
    config["environment"]["question_budget"] = 1
    config["environment"]["question_budget_mode"] = "soft"
    env = BudgetedMovieRecommenderEnv(config, seed=7)
    env.reset(seed=7, options={"user_type": "balanced_viewer"})

    calls = {"count": 0}

    def fake_question(_qtype, _repeated):
        calls["count"] += 1
        return {
            "engagement_delta": 0.02,
            "uncertainty_delta": -0.1,
            "hinted_genre": 0,
            "abandon_prob": 0.0,
        }

    env.user.respond_to_question = fake_question

    _, _, _, _, first_info = env.step(0)
    _, _, _, _, second_info = env.step(0)

    assert first_info["question_asked"] is True
    assert first_info["over_budget"] is False
    assert second_info["question_asked"] is True
    assert second_info["over_budget"] is True
    assert env._asked == 2
    assert calls["count"] == 2


def test_hard_budget_blocks_extra_questions_in_auto_mode():
    config = load_config("configs/default.yaml")
    config["environment"]["question_budget"] = 1
    config["environment"]["question_budget_mode"] = "hard"
    env = BudgetedMovieRecommenderEnv(config, seed=7)
    env.reset(seed=7, options={"user_type": "balanced_viewer"})

    calls = {"count": 0}

    def fake_question(_qtype, _repeated):
        calls["count"] += 1
        return {
            "engagement_delta": 0.02,
            "uncertainty_delta": -0.1,
            "hinted_genre": 0,
            "abandon_prob": 0.0,
        }

    env.user.respond_to_question = fake_question

    _, _, _, _, first_info = env.step(0)
    belief_after_first = deepcopy(env._belief)
    engagement_after_first = env._engagement
    uncertainty_after_first = env._uncertainty

    _, _, _, _, second_info = env.step(0)

    assert first_info["question_asked"] is True
    assert first_info["over_budget"] is False
    assert second_info["question_asked"] is False
    assert second_info["over_budget"] is True
    assert env._asked == 1
    assert calls["count"] == 1
    assert env._engagement == engagement_after_first
    assert env._uncertainty == uncertainty_after_first
    assert (env._belief == belief_after_first).all()


def test_hard_budget_blocks_extra_questions_in_interactive_mode():
    config = load_config("configs/default.yaml")
    config["environment"]["question_budget"] = 1
    config["environment"]["question_budget_mode"] = "hard"
    env = BudgetedMovieRecommenderEnv(config, seed=11)
    env.reset(seed=11, options={"user_type": "balanced_viewer"})

    env.apply_manual_response(
        0,
        continuation="continue",
        question_feedback="helpful",
        hinted_genre="scifi",
    )
    belief_after_first = deepcopy(env._belief)
    engagement_after_first = env._engagement
    uncertainty_after_first = env._uncertainty

    _, _, _, _, blocked_info = env.apply_manual_response(
        0,
        continuation="continue",
        question_feedback="helpful",
        hinted_genre="action",
    )

    assert blocked_info["question_asked"] is False
    assert blocked_info["over_budget"] is True
    assert env._asked == 1
    assert env._engagement == engagement_after_first
    assert env._uncertainty == uncertainty_after_first
    assert (env._belief == belief_after_first).all()

