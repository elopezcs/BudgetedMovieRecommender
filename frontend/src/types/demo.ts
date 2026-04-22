import type { PolicyName } from './results'

export type SessionMode = 'auto' | 'interactive'
export type QuestionFeedback = 'helpful' | 'neutral' | 'annoying'
export type RecommendationFeedback = 'accepted_strong' | 'accepted_weak' | 'skipped' | 'rejected_annoyed'
export type ContinuationChoice = 'continue' | 'abandon'

export interface DemoStartRequest {
  policy: PolicyName
  user_profile: string
  question_budget: number
  seed?: number | null
  mode: SessionMode
}

export interface PendingAction {
  step_index: number
  action_id: number
  action_name: string
  action_type: 'ask' | 'recommend'
  question_text: string | null
  recommendation_genre: string | null
  fallback_applied: boolean
  fallback_reason: string | null
  original_attempted_action_name: string | null
}

export interface ManualResponseRequest {
  session_id: string
  continuation: ContinuationChoice
  question_feedback?: QuestionFeedback
  hinted_genre?: string
  recommendation_feedback?: RecommendationFeedback
}

export interface SessionStep {
  step_index: number
  action_id: number
  action_name: string
  action_type: 'ask' | 'recommend'
  question_text: string | null
  recommendation_genre: string | null
  accepted: boolean
  abandoned: boolean
  reward: number
  cumulative_reward: number
  engagement: number
  uncertainty: number
  questions_used: number
  question_budget: number
  question_budget_remaining: number
  question_asked: boolean
  over_budget: boolean
  belief: Record<string, number>
  terminated: boolean
  truncated: boolean
  user_type: string
  user_response_label: string
  manual_response: string | null
  fallback_applied: boolean
  fallback_reason: string | null
  original_attempted_action_name: string | null
}

export interface SessionSummary {
  total_reward: number
  accepted_recommendations: number
  skipped_recommendations: number
  abandoned: boolean
  session_length: number
  questions_used: number
}

export interface DemoSessionResponse {
  session_id: string
  policy: PolicyName
  user_profile: string
  mode: SessionMode
  question_budget: number
  done: boolean
  awaiting_manual_response: boolean
  current_observation: number[]
  latest_step: SessionStep | null
  timeline: SessionStep[]
  summary: SessionSummary
  pending_action: PendingAction | null
}

export interface DemoOptionsResponse {
  policies: PolicyName[]
  user_profiles: string[]
  session_modes: SessionMode[]
  budget_range: {
    min: number
    max: number
    default: number
  }
}
