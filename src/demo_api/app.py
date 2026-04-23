from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.env.movie_recommender_env import ACTIONS, QUESTION_ACTIONS, BudgetedMovieRecommenderEnv
from src.env.user_simulator import GENRES, PROFILE_LIBRARY, QUESTION_TYPES
from src.inference.inference_adapter import PolicyAdapter
from src.utils.config import load_config

from .schemas import (
    ComparisonMetricModel,
    DemoPolicyName,
    ComparisonResponse,
    DemoOptionsResponse,
    DemoSessionResponse,
    DemoStartRequest,
    ManualResponseRequest,
    OverviewKpisModel,
    OverviewResponse,
    PendingActionModel,
    SessionIdRequest,
    SessionStepModel,
    SessionSummaryModel,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"
RESULTS_DIR = ROOT / "results"
DEBUG_LOG_PATH = ROOT / "debug-5bb150.log"

SESSION_MODES = ["auto", "interactive"]
ALL_POLICIES = [
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
    policy: DemoPolicyName
    user_profile: str
    mode: str
    question_budget: int
    seed: int
    env: BudgetedMovieRecommenderEnv
    current_observation: np.ndarray
    policy_adapter: PolicyAdapter | None = None
    timeline: List[SessionStepModel] = field(default_factory=list)
    cumulative_reward: float = 0.0
    accepted_recommendations: int = 0
    skipped_recommendations: int = 0
    abandoned: bool = False
    done: bool = False
    awaiting_manual_response: bool = False
    pending_action: PendingActionModel | None = None


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


def _pretty_label(value: str) -> str:
    return value.replace("_", " ").title()


def _action_details(action_id: int) -> dict[str, str | None]:
    action_name = ACTIONS[action_id]
    action_type = "ask" if action_id < len(QUESTION_ACTIONS) else "recommend"
    question_text = None
    recommendation_genre = None
    if action_type == "ask":
        question_type = QUESTION_TYPES[action_id]
        question_text = QUESTION_LABELS.get(question_type, question_type)
    else:
        recommendation_genre = GENRES[action_id - len(QUESTION_ACTIONS)]
    return {
        "action_name": action_name,
        "action_type": action_type,
        "question_text": question_text,
        "recommendation_genre": recommendation_genre,
    }


def _build_pending_action(
    action_id: int,
    step_index: int,
    *,
    fallback_applied: bool = False,
    fallback_reason: str | None = None,
    original_attempted_action_name: str | None = None,
) -> PendingActionModel:
    details = _action_details(action_id)
    return PendingActionModel(
        step_index=step_index,
        action_id=action_id,
        action_name=str(details["action_name"]),
        action_type=str(details["action_type"]),  # type: ignore[arg-type]
        question_text=details["question_text"],
        recommendation_genre=details["recommendation_genre"],
        fallback_applied=fallback_applied,
        fallback_reason=fallback_reason,
        original_attempted_action_name=original_attempted_action_name,
    )


def _user_response_label(*, accepted: bool, abandoned: bool) -> str:
    if accepted:
        return "Accepted"
    if abandoned:
        return "Abandoned"
    return "Continued"


def _manual_response_summary(
    *,
    action_type: str,
    continuation: str,
    question_feedback: str | None = None,
    hinted_genre: str | None = None,
    recommendation_feedback: str | None = None,
) -> str:
    if action_type == "ask":
        summary = _pretty_label(question_feedback or "neutral")
        if hinted_genre:
            summary = f"{summary}; hinted {_pretty_label(hinted_genre)}"
    else:
        summary = _pretty_label(recommendation_feedback or "skipped")
    end_state = "Abandoned session" if continuation == "abandon" else "Continued session"
    return f"{summary}; {end_state}"


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
        policy=session.policy,
        user_profile=session.user_profile,
        mode=session.mode,  # type: ignore[arg-type]
        question_budget=session.question_budget,
        done=session.done,
        awaiting_manual_response=session.awaiting_manual_response,
        current_observation=[float(x) for x in session.current_observation.tolist()],
        latest_step=session.timeline[-1] if session.timeline else None,
        timeline=session.timeline,
        summary=summary,
        pending_action=session.pending_action,
    )


def _policy_action(session: SessionState, _step_index: int) -> int:
    assert session.policy_adapter is not None
    prediction = session.policy_adapter.predict_action(
        [float(x) for x in session.current_observation.tolist()]
    )
    return int(prediction["action_id"])


def _is_question_action(action_id: int) -> bool:
    return action_id < len(QUESTION_ACTIONS)


def _select_fallback_recommendation(session: SessionState) -> int:
    belief_values = session.current_observation[: len(GENRES)]
    top_genre_idx = int(np.argmax(belief_values))
    return len(QUESTION_ACTIONS) + top_genre_idx


def _resolve_action_for_demo(
    session: SessionState, action_id: int
) -> tuple[int, bool, str | None, str | None]:
    if not _is_question_action(action_id):
        return action_id, False, None, None

    questions_used, _ = _question_usage(session.current_observation, session.question_budget)
    if session.env.question_budget_mode != "hard" or questions_used < session.question_budget:
        return action_id, False, None, None

    original_details = _action_details(action_id)
    fallback_action_id = _select_fallback_recommendation(session)
    return (
        fallback_action_id,
        True,
        "Question budget exhausted; replaced blocked ask with fallback recommendation.",
        str(original_details["action_name"]),
    )


def _commit_step(
    session: SessionState,
    *,
    action_id: int,
    next_observation: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Dict,
    manual_response: str | None = None,
    fallback_applied: bool = False,
    fallback_reason: str | None = None,
    original_attempted_action_name: str | None = None,
) -> DemoSessionResponse:
    details = _action_details(action_id)
    action_type = str(details["action_type"])

    session.current_observation = np.array(next_observation, dtype=np.float32)
    session.cumulative_reward += float(reward)

    if action_type == "recommend":
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
        step_index=len(session.timeline),
        action_id=action_id,
        action_name=str(details["action_name"]),
        action_type=action_type,  # type: ignore[arg-type]
        question_text=details["question_text"],
        recommendation_genre=details["recommendation_genre"],
        accepted=bool(info.get("accepted")),
        abandoned=bool(info.get("abandoned")),
        reward=float(reward),
        cumulative_reward=float(session.cumulative_reward),
        engagement=float(session.current_observation[10]),
        uncertainty=float(session.current_observation[5]),
        questions_used=questions_used,
        question_budget=session.question_budget,
        question_budget_remaining=remaining,
        question_asked=bool(info.get("question_asked")),
        over_budget=bool(info.get("over_budget")),
        belief=belief,
        terminated=bool(terminated),
        truncated=bool(truncated),
        user_type=session.user_profile,
        user_response_label=_user_response_label(
            accepted=bool(info.get("accepted")),
            abandoned=bool(info.get("abandoned")),
        ),
        manual_response=manual_response,
        fallback_applied=fallback_applied,
        fallback_reason=fallback_reason,
        original_attempted_action_name=original_attempted_action_name,
    )

    session.timeline.append(step)
    session.done = bool(terminated or truncated)
    session.awaiting_manual_response = False
    session.pending_action = None
    return _build_response(session)


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
            session_modes=SESSION_MODES,  # type: ignore[arg-type]
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
        mode=payload.mode,
        question_budget=payload.question_budget,
        seed=seed,
        env=env,
        current_observation=np.array(observation, dtype=np.float32),
    )

    state.policy_adapter = PolicyAdapter(payload.policy)

    SESSIONS[session_id] = state
    return _build_response(state)


@app.post("/api/demo/session/next", response_model=DemoSessionResponse)
def next_session_step(payload: SessionIdRequest) -> DemoSessionResponse:
    session = SESSIONS.get(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.done:
        return _build_response(session)
    if session.awaiting_manual_response:
        raise HTTPException(status_code=409, detail="Session is waiting for a manual response.")

    step_index = len(session.timeline)
    attempted_action_id = _policy_action(session, step_index)
    action_id, fallback_applied, fallback_reason, original_attempted_action_name = _resolve_action_for_demo(
        session, attempted_action_id
    )
    if session.mode == "interactive":
        session.pending_action = _build_pending_action(
            action_id,
            step_index,
            fallback_applied=fallback_applied,
            fallback_reason=fallback_reason,
            original_attempted_action_name=original_attempted_action_name,
        )
        session.awaiting_manual_response = True
        return _build_response(session)

    next_observation, reward, terminated, truncated, info = session.env.step(action_id)
    return _commit_step(
        session,
        action_id=action_id,
        next_observation=np.array(next_observation, dtype=np.float32),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=info,
        fallback_applied=fallback_applied,
        fallback_reason=fallback_reason,
        original_attempted_action_name=original_attempted_action_name,
    )


@app.post("/api/demo/session/respond", response_model=DemoSessionResponse)
def respond_to_interactive_step(payload: ManualResponseRequest) -> DemoSessionResponse:
    session = SESSIONS.get(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.mode != "interactive":
        raise HTTPException(status_code=400, detail="Manual responses are only supported in interactive mode.")
    if session.done:
        return _build_response(session)
    if not session.awaiting_manual_response or session.pending_action is None:
        raise HTTPException(status_code=409, detail="No pending action is awaiting a manual response.")

    pending = session.pending_action
    if pending.action_type == "ask":
        if payload.question_feedback is None or payload.hinted_genre is None:
            raise HTTPException(
                status_code=400,
                detail="Interactive question responses require question_feedback and hinted_genre.",
            )
    else:
        if payload.recommendation_feedback is None:
            raise HTTPException(
                status_code=400,
                detail="Interactive recommendation responses require recommendation_feedback.",
            )

    try:
        next_observation, reward, terminated, truncated, info = session.env.apply_manual_response(
            pending.action_id,
            continuation=payload.continuation,
            question_feedback=payload.question_feedback,
            hinted_genre=payload.hinted_genre,
            recommendation_feedback=payload.recommendation_feedback,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _commit_step(
        session,
        action_id=pending.action_id,
        next_observation=np.array(next_observation, dtype=np.float32),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=info,
        manual_response=_manual_response_summary(
            action_type=pending.action_type,
            continuation=payload.continuation,
            question_feedback=payload.question_feedback,
            hinted_genre=payload.hinted_genre,
            recommendation_feedback=payload.recommendation_feedback,
        ),
        fallback_applied=pending.fallback_applied,
        fallback_reason=pending.fallback_reason,
        original_attempted_action_name=pending.original_attempted_action_name,
    )


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
    session.awaiting_manual_response = False
    session.pending_action = None
    return _build_response(session)
