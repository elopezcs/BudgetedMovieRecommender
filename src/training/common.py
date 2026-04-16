from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.env.movie_recommender_env import ACTIONS, BudgetedMovieRecommenderEnv
from src.utils.io import ensure_dir


def build_env(config: Dict[str, Any], seed: int):
    return BudgetedMovieRecommenderEnv(config=config, seed=seed)


def build_run_dirs(algo: str, run_id: str) -> tuple[Path, Path]:
    model_dir = ensure_dir(Path("models") / algo / run_id)
    result_dir = ensure_dir(Path("results") / algo / run_id)
    return model_dir, result_dir


def action_map() -> Dict[int, str]:
    return {idx: name for idx, name in enumerate(ACTIONS)}

