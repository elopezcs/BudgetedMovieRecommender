from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.agents.baselines import baseline_policies
from src.env.movie_recommender_env import ACTIONS, QUESTION_ACTIONS, BudgetedMovieRecommenderEnv
from src.env.user_simulator import GENRES, PROFILE_LIBRARY, QUESTION_TYPES
from src.inference.inference_adapter import PolicyAdapter
from src.utils.config import load_config

from .schemas import (
    ComparisonMetricModel,
    ComparisonResponse,
    DemoOptionsResponse,
    DemoSessionResponse,
    DemoStartRequest,
    OverviewKpisModel,
    OverviewResponse,
    SessionIdRequest,
    SessionStepModel,
    SessionSummaryModel,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"
RESULTS_DIR = ROOT / "results"
DEBUG_LOG_PATH = ROOT / "debug-5bb150.log"

RL_POLICIES = {"q_learning", "dqn", "ppo"}
ALL_POLICIES = [
    "always_recommend",
    "always_ask",
    "ask_once_then_recommend",
    "random_policy",
    "q_learning",
    "dqn",
    "ppo",
]

QUESTION_LABELS = {
    "familiar": "Do you want something familiar tonight?",
    "exploratory": "Interested in discovering something new?",
    "serious": "Are you in the mood for a more serious story?",
    "light": "Looking for something light and fun?",
    "fast_paced": "Do you want a fast-paced watch?",
    "calm": "Would you prefer a calm, reflective film?",
}


@dataclass
class SessionState:
    session_id: str
    policy: str
    user_profile: str
    question_budget: int
    seed: int
    env: BudgetedMovieRecommenderEnv
    current_observation: np.ndarray
    policy_fn: Callable | None = None
    policy_adapter: PolicyAdapter | None = None
    timeline: List[SessionStepModel] = field(default_factory=list)
    cumulative_reward: float = 0.0
    accepted_recommendations: int = 0
    skipped_recommendations: int = 0
    abandoned: bool = False
    done: bool = False


app = FastAPI(title="Movie Recommender Demo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: Dict[str, SessionState] = {}


def _debug_log(hypothesis_id: str, message: str, data: Dict) -> None:
    payload = {
        "sessionId": "5bb150",
        "runId": "initial",
        "hypothesisId": hypothesis_id,
        "location": "src/demo_api/app.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload) + "\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_subdir(path: Path) -> Path | None:
    if not path.exists():
        return None
    candidates = sorted([entry for entry in path.iterdir() if entry.is_dir()])
    return candidates[-1] if candidates else None


def _load_latest_comparison_metrics() -> tuple[list[ComparisonMetricModel], str | None, str | None]:
    latest = _latest_subdir(RESULTS_DIR / "comparisons")
    if latest is None:
        return [], None, "No comparison outputs found under results/comparisons."

    summary_path = latest / "comparison_summary.json"
    if not summary_path.exists():
        return [], str(summary_path), "Latest comparison run is missing comparison_summary.json."

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = [ComparisonMetricModel(**item) for item in payload]
    return metrics, str(summary_path.relative_to(ROOT)), None


def _load_latest_run_metrics() -> list[ComparisonMetricModel]:
    rows: list[ComparisonMetricModel] = []
    for algo in ("q_learning", "dqn", "ppo"):
        run_dir = _latest_subdir(RESULTS_DIR / algo)
        if run_dir is None:
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(ComparisonMetricModel(**payload))
    return rows


def _initial_summary() -> SessionSummaryModel:
    return SessionSummaryModel(
        total_reward=0.0,
        accepted_recommendations=0,
        skipped_recommendations=0,
        abandoned=False,
        session_length=0,
        questions_used=0,
    )


def _question_usage(observation: np.ndarray, budget: int) -> tuple[int, int]:
    remaining_ratio = float(observation[9])
    estimated_remaining = int(round(remaining_ratio * budget))
    estimated_remaining = max(0, min(budget, estimated_remaining))
    used = max(0, budget - estimated_remaining)
    return used, estimated_remaining


def _build_response(session: SessionState) -> DemoSessionResponse:
    questions_used = session.timeline[-1].questions_used if session.timeline else 0
    summary = SessionSummaryModel(
        total_reward=session.cumulative_reward,
        accepted_recommendations=session.accepted_recommendations,
        skipped_recommendations=session.skipped_recommendations,
        abandoned=session.abandoned,
        session_length=len(session.timeline),
        questions_used=questions_used,
    )
    return DemoSessionResponse(
        session_id=session.session_id,
        policy=session.policy,  # type: ignore[arg-type]
        user_profile=session.user_profile,
        question_budget=session.question_budget,
        done=session.done,
        current_observation=[float(x) for x in session.current_observation.tolist()],
        latest_step=session.timeline[-1] if session.timeline else None,
        timeline=session.timeline,
        summary=summary,
    )


def _policy_action(session: SessionState, step_index: int) -> int:
    if session.policy in RL_POLICIES:
        assert session.policy_adapter is not None
        prediction = session.policy_adapter.predict_action(
            [float(x) for x in session.current_observation.tolist()]
        )
        return int(prediction["action_id"])

    assert session.policy_fn is not None
    return int(session.policy_fn(session.current_observation, step_index, {}))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/results/comparisons/latest", response_model=ComparisonResponse)
def latest_comparison() -> ComparisonResponse:
    metrics, source_path, warning = _load_latest_comparison_metrics()
    if not metrics:
        metrics = _load_latest_run_metrics()
    return ComparisonResponse(
        timestamp=_timestamp(),
        source_path=source_path,
        metrics=metrics,
        warning=warning,
    )


@app.get("/api/results/runs/latest", response_model=ComparisonResponse)
def latest_runs() -> ComparisonResponse:
    metrics = _load_latest_run_metrics()
    warning = None if metrics else "No run summaries found in results/q_learning, results/dqn, or results/ppo."
    return ComparisonResponse(
        timestamp=_timestamp(),
        source_path=None,
        metrics=metrics,
        warning=warning,
    )


@app.get("/api/results/overview", response_model=OverviewResponse)
def overview() -> OverviewResponse:
    metrics, _, warning = _load_latest_comparison_metrics()
    if not metrics:
        metrics = _load_latest_run_metrics()

    if not metrics:
        return OverviewResponse(
            timestamp=_timestamp(),
            top_algorithm=None,
            kpis=OverviewKpisModel(
                best_reward=None,
                best_acceptance_rate=None,
                avg_questions_asked=None,
                best_performing_algorithm=None,
            ),
            metrics=[],
            warning=warning or "No metrics found. Run evaluation and comparison scripts first.",
        )

    sorted_by_reward = sorted(metrics, key=lambda row: row.average_cumulative_reward, reverse=True)
    top = sorted_by_reward[0]
    avg_questions = sum(metric.average_questions_asked for metric in metrics) / len(metrics)

    return OverviewResponse(
        timestamp=_timestamp(),
        top_algorithm=top.algorithm,
        kpis=OverviewKpisModel(
            best_reward=top.average_cumulative_reward,
            best_acceptance_rate=max(metric.acceptance_rate for metric in metrics),
            avg_questions_asked=avg_questions,
            best_performing_algorithm=top.algorithm,
        ),
        metrics=sorted_by_reward,
        warning=warning,
    )


@app.get("/api/demo/options", response_model=DemoOptionsResponse)
def demo_options() -> DemoOptionsResponse:
    # region agent log
    _debug_log("H5", "demo_options_entry", {"config_path": str(DEFAULT_CONFIG_PATH)})
    # endregion
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
        default_budget = int(config["environment"]["question_budget"])
        response = DemoOptionsResponse(
            policies=ALL_POLICIES,
            user_profiles=sorted(list(PROFILE_LIBRARY.keys())),
            budget_range={"min": 1, "max": 10, "default": default_budget},
        )
        # region agent log
        _debug_log(
            "H5",
            "demo_options_success",
            {
                "policies_count": len(response.policies),
                "profiles_count": len(response.user_profiles),
                "default_budget": response.budget_range["default"],
            },
        )
        # endregion
        return response
    except Exception as exc:
        # region agent log
        _debug_log("H5", "demo_options_exception", {"error": str(exc)})
        # endregion
        raise


@app.post("/api/demo/session/start", response_model=DemoSessionResponse)
def start_session(payload: DemoStartRequest) -> DemoSessionResponse:
    config = load_config(DEFAULT_CONFIG_PATH)
    session_id = str(uuid4())
    seed = payload.seed if payload.seed is not None else int(config["seed"])

    config_copy = deepcopy(config)
    config_copy["environment"]["question_budget"] = int(payload.question_budget)
    env = BudgetedMovieRecommenderEnv(config_copy, seed=seed)
    observation, _ = env.reset(seed=seed, options={"user_type": payload.user_profile})

    state = SessionState(
        session_id=session_id,
        policy=payload.policy,
        user_profile=payload.user_profile,
        question_budget=payload.question_budget,
        seed=seed,
        env=env,
        current_observation=np.array(observation, dtype=np.float32),
    )

    if payload.policy in RL_POLICIES:
        state.policy_adapter = PolicyAdapter(payload.policy)
    else:
        state.policy_fn = baseline_policies()[payload.policy]

    SESSIONS[session_id] = state
    return _build_response(state)


@app.post("/api/demo/session/next", response_model=DemoSessionResponse)
def next_session_step(payload: SessionIdRequest) -> DemoSessionResponse:
    session = SESSIONS.get(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.done:
        return _build_response(session)

    step_index = len(session.timeline)
    action_id = _policy_action(session, step_index)
    action_name = ACTIONS[action_id]
    next_observation, reward, terminated, truncated, info = session.env.step(action_id)
    session.current_observation = np.array(next_observation, dtype=np.float32)
    session.cumulative_reward += float(reward)

    action_type = "ask" if action_id < len(QUESTION_ACTIONS) else "recommend"
    question_text = None
    recommendation_genre = None
    if action_type == "ask":
        question_type = QUESTION_TYPES[action_id]
        question_text = QUESTION_LABELS.get(question_type, question_type)
    else:
        recommendation_genre = GENRES[action_id - len(QUESTION_ACTIONS)]
        if info.get("accepted"):
            session.accepted_recommendations += 1
        else:
            session.skipped_recommendations += 1

    if info.get("abandoned"):
        session.abandoned = True

    questions_used, remaining = _question_usage(session.current_observation, session.question_budget)
    belief_values = session.current_observation[: len(GENRES)]
    belief = {genre: float(score) for genre, score in zip(GENRES, belief_values)}

    step = SessionStepModel(
        step_index=step_index,
        action_id=action_id,
        action_name=action_name,
        action_type=action_type,
        question_text=question_text,
        recommendation_genre=recommendation_genre,
        accepted=bool(info.get("accepted")),
        abandoned=bool(info.get("abandoned")),
        reward=float(reward),
        cumulative_reward=float(session.cumulative_reward),
        engagement=float(session.current_observation[10]),
        uncertainty=float(session.current_observation[5]),
        questions_used=questions_used,
        question_budget=session.question_budget,
        question_budget_remaining=remaining,
        belief=belief,
        terminated=bool(terminated),
        truncated=bool(truncated),
        user_type=session.user_profile,
    )

    session.timeline.append(step)
    session.done = bool(terminated or truncated)
    return _build_response(session)


@app.post("/api/demo/session/reset", response_model=DemoSessionResponse)
def reset_session(payload: SessionIdRequest) -> DemoSessionResponse:
    session = SESSIONS.get(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    observation, _ = session.env.reset(seed=session.seed, options={"user_type": session.user_profile})
    session.current_observation = np.array(observation, dtype=np.float32)
    session.timeline = []
    session.cumulative_reward = 0.0
    session.accepted_recommendations = 0
    session.skipped_recommendations = 0
    session.abandoned = False
    session.done = False
    return _build_response(session)
