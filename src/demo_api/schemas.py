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


class DemoStartRequest(BaseModel):
    policy: PolicyName
    user_profile: str
    question_budget: int = Field(ge=1, le=12)
    seed: Optional[int] = None


class SessionIdRequest(BaseModel):
    session_id: str


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
    belief: Dict[str, float]
    terminated: bool
    truncated: bool
    user_type: str


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
    question_budget: int
    done: bool
    current_observation: List[float]
    latest_step: Optional[SessionStepModel]
    timeline: List[SessionStepModel]
    summary: SessionSummaryModel


class DemoOptionsResponse(BaseModel):
    policies: List[PolicyName]
    user_profiles: List[str]
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
