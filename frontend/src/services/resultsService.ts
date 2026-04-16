import { apiRequest } from './apiClient'
import type { ComparisonResponse, OverviewResponse } from '../types/results'

export function fetchOverview(): Promise<OverviewResponse> {
  return apiRequest<OverviewResponse>('/api/results/overview')
}

export function fetchLatestComparison(): Promise<ComparisonResponse> {
  return apiRequest<ComparisonResponse>('/api/results/comparisons/latest')
}
