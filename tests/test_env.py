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
        over_budget=False,
        repeated_action=False,
    )
    assert reward_ok > reward_bad

