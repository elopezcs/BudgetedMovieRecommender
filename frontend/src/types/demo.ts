import type { PolicyName } from './results'

export interface DemoStartRequest {
  policy: PolicyName
  user_profile: string
  question_budget: number
  seed?: number | null
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
  belief: Record<string, number>
  terminated: boolean
  truncated: boolean
  user_type: string
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
  question_budget: number
  done: boolean
  current_observation: number[]
  latest_step: SessionStep | null
  timeline: SessionStep[]
  summary: SessionSummary
}

export interface DemoOptionsResponse {
  policies: PolicyName[]
  user_profiles: string[]
  budget_range: {
    min: number
    max: number
    default: number
  }
}
