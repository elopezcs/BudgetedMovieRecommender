from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


PolicyName = Literal[
    "always_recommend",
    "always_ask",
    "ask_once_then_recommend",
    "random_policy",
    "q_learning",
    "dqn",
    "ppo",
]

SessionMode = Literal["auto", "interactive"]
QuestionFeedback = Literal["helpful", "neutral", "annoying"]
RecommendationFeedback = Literal[
    "accepted_strong",
    "accepted_weak",
    "skipped",
    "rejected_annoyed",
]
ContinuationChoice = Literal["continue", "abandon"]


class DemoStartRequest(BaseModel):
    policy: PolicyName
    user_profile: str
    question_budget: int = Field(ge=1, le=12)
    seed: Optional[int] = None
    mode: SessionMode = "auto"


class SessionIdRequest(BaseModel):
    session_id: str


class PendingActionModel(BaseModel):
    step_index: int
    action_id: int
    action_name: str
    action_type: Literal["ask", "recommend"]
    question_text: Optional[str]
    recommendation_genre: Optional[str]
    fallback_applied: bool = False
    fallback_reason: Optional[str] = None
    original_attempted_action_name: Optional[str] = None


class ManualResponseRequest(BaseModel):
    session_id: str
    continuation: ContinuationChoice
    question_feedback: Optional[QuestionFeedback] = None
    hinted_genre: Optional[str] = None
    recommendation_feedback: Optional[RecommendationFeedback] = None


class SessionStepModel(BaseModel):
    step_index: int
    action_id: int
    action_name: str
    action_type: Literal["ask", "recommend"]
    question_text: Optional[str]
    recommendation_genre: Optional[str]
    accepted: bool
    abandoned: bool
    reward: float
    cumulative_reward: float
    engagement: float
    uncertainty: float
    questions_used: int
    question_budget: int
    question_budget_remaining: int
    question_asked: bool
    over_budget: bool
    belief: Dict[str, float]
    terminated: bool
    truncated: bool
    user_type: str
    user_response_label: str
    manual_response: Optional[str] = None
    fallback_applied: bool = False
    fallback_reason: Optional[str] = None
    original_attempted_action_name: Optional[str] = None


class SessionSummaryModel(BaseModel):
    total_reward: float
    accepted_recommendations: int
    skipped_recommendations: int
    abandoned: bool
    session_length: int
    questions_used: int


class DemoSessionResponse(BaseModel):
    session_id: str
    policy: PolicyName
    user_profile: str
    mode: SessionMode
    question_budget: int
    done: bool
    awaiting_manual_response: bool
    current_observation: List[float]
    latest_step: Optional[SessionStepModel]
    timeline: List[SessionStepModel]
    summary: SessionSummaryModel
    pending_action: Optional[PendingActionModel]


class DemoOptionsResponse(BaseModel):
    policies: List[PolicyName]
    user_profiles: List[str]
    session_modes: List[SessionMode]
    budget_range: Dict[str, int]


class ComparisonMetricModel(BaseModel):
    algorithm: str
    average_cumulative_reward: float
    acceptance_rate: float
    abandonment_rate: float
    average_questions_asked: float
    average_session_length: float


class ComparisonResponse(BaseModel):
    timestamp: str
    source_path: Optional[str]
    metrics: List[ComparisonMetricModel]
    warning: Optional[str] = None


class OverviewKpisModel(BaseModel):
    best_reward: Optional[float]
    best_acceptance_rate: Optional[float]
    avg_questions_asked: Optional[float]
    best_performing_algorithm: Optional[str]


class OverviewResponse(BaseModel):
    timestamp: str
    top_algorithm: Optional[str]
    kpis: OverviewKpisModel
    metrics: List[ComparisonMetricModel]
    warning: Optional[str] = None
