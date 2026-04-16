export type PolicyName =
  | 'always_recommend'
  | 'always_ask'
  | 'ask_once_then_recommend'
  | 'random_policy'
  | 'q_learning'
  | 'dqn'
  | 'ppo'

export interface ComparisonMetric {
  algorithm: PolicyName
  average_cumulative_reward: number
  acceptance_rate: number
  abandonment_rate: number
  average_questions_asked: number
  average_session_length: number
}

export interface OverviewResponse {
  timestamp: string
  top_algorithm: PolicyName | null
  kpis: {
    best_reward: number | null
    best_acceptance_rate: number | null
    avg_questions_asked: number | null
    best_performing_algorithm: PolicyName | null
  }
  metrics: ComparisonMetric[]
  warning?: string
}

export interface ComparisonResponse {
  timestamp: string
  source_path: string | null
  metrics: ComparisonMetric[]
  warning?: string
}
