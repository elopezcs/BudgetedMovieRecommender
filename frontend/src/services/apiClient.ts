const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

function debugLog(hypothesisId: string, message: string, data: Record<string, unknown>) {
  fetch('http://127.0.0.1:7886/ingest/8d170cbe-0335-444a-b77a-5f8cced98eaa', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Debug-Session-Id': '5bb150',
    },
    body: JSON.stringify({
      sessionId: '5bb150',
      runId: 'initial',
      hypothesisId,
      location: 'frontend/src/services/apiClient.ts',
      message,
      data,
      timestamp: Date.now(),
    }),
  }).catch(() => {})
}

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const requestUrl = `${API_BASE_URL}${path}`
  // #region agent log
  debugLog('H1', 'api_request_start', { path, requestUrl, method: init?.method ?? 'GET' })
  // #endregion

  try {
    const response = await fetch(requestUrl, {
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
      ...init,
    })

    // #region agent log
    debugLog('H2', 'api_response_received', { path, status: response.status, ok: response.ok })
    // #endregion

    if (!response.ok) {
      const details = await response.text()
      // #region agent log
      debugLog('H3', 'api_response_not_ok', {
        path,
        status: response.status,
        detailsPreview: details.slice(0, 250),
      })
      // #endregion
      throw new Error(details || `Request failed (${response.status})`)
    }

    return response.json() as Promise<T>
  } catch (error) {
    // #region agent log
    debugLog('H4', 'api_request_exception', {
      path,
      error: error instanceof Error ? error.message : 'unknown_error',
    })
    // #endregion
    throw error
  }
}
