import { apiRequest } from './apiClient'
import type { DemoOptionsResponse, DemoSessionResponse, DemoStartRequest, ManualResponseRequest } from '../types/demo'

export function fetchDemoOptions(): Promise<DemoOptionsResponse> {
  return apiRequest<DemoOptionsResponse>('/api/demo/options')
}

export function startSession(payload: DemoStartRequest): Promise<DemoSessionResponse> {
  return apiRequest<DemoSessionResponse>('/api/demo/session/start', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function nextStep(sessionId: string): Promise<DemoSessionResponse> {
  return apiRequest<DemoSessionResponse>('/api/demo/session/next', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
  })
}

export function resetSession(sessionId: string): Promise<DemoSessionResponse> {
  return apiRequest<DemoSessionResponse>('/api/demo/session/reset', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
  })
}

export function submitManualResponse(payload: ManualResponseRequest): Promise<DemoSessionResponse> {
  return apiRequest<DemoSessionResponse>('/api/demo/session/respond', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}
