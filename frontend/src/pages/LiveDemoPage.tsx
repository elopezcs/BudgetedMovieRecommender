import { useEffect, useMemo, useState } from 'react'
import { Card } from '../components/Card'
import { FeedbackState } from '../components/FeedbackState'
import { PageTitle } from '../components/PageTitle'
import { BeliefChart } from '../features/demo/BeliefChart'
import { BudgetBar } from '../features/demo/BudgetBar'
import { FinalSummaryCard } from '../features/demo/FinalSummaryCard'
import { ScenarioControls } from '../features/demo/ScenarioControls'
import { SessionTimeline } from '../features/demo/SessionTimeline'
import { fetchDemoOptions, nextStep, resetSession, startSession } from '../services/demoService'
import type { DemoOptionsResponse, DemoSessionResponse, DemoStartRequest } from '../types/demo'
import { toFixed, toPercent } from '../utils/format'

const defaultRequest: DemoStartRequest = {
  policy: 'q_learning',
  user_profile: 'balanced_viewer',
  question_budget: 4,
  seed: null,
}

export function LiveDemoPage() {
  const [options, setOptions] = useState<DemoOptionsResponse | null>(null)
  const [request, setRequest] = useState<DemoStartRequest>(defaultRequest)
  const [session, setSession] = useState<DemoSessionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [autoplay, setAutoplay] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  useEffect(() => {
    fetchDemoOptions()
      .then((payload) => {
        setOptions(payload)
        setRequest((prev) => ({
          ...prev,
          policy: payload.policies[0] ?? prev.policy,
          user_profile: payload.user_profiles[0] ?? prev.user_profile,
          question_budget: payload.budget_range.default,
        }))
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  useEffect(() => {
    if (!autoplay || !session || session.done) {
      return
    }

    const timer = window.setInterval(() => {
      void handleNextStep()
    }, 1300)

    return () => window.clearInterval(timer)
  })

  const latestStep = session?.latest_step ?? null

  const progress = useMemo(() => {
    if (!latestStep) {
      return { engagement: 0, uncertainty: 0 }
    }
    return {
      engagement: latestStep.engagement,
      uncertainty: latestStep.uncertainty,
    }
  }, [latestStep])

  async function handleStart() {
    setLoading(true)
    setError(null)
    setAutoplay(false)
    try {
      const payload = await startSession(request)
      setSession(payload)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function handleNextStep() {
    if (!session || session.done) {
      return
    }
    setLoading(true)
    try {
      const payload = await nextStep(session.session_id)
      setSession(payload)
      if (payload.done) {
        setAutoplay(false)
      }
    } catch (err) {
      setError((err as Error).message)
      setAutoplay(false)
    } finally {
      setLoading(false)
    }
  }

  async function handleReset() {
    if (!session) {
      return
    }
    setLoading(true)
    setAutoplay(false)
    try {
      const payload = await resetSession(session.session_id)
      setSession(payload)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <PageTitle
        eyebrow="Session Playback"
        title="Live Demo: Ask vs Recommend"
        description="Run a step-by-step simulation to show how different policies trade off exploration, user fatigue, and recommendation outcomes."
      />

      <Card>
        <ScenarioControls
          options={options}
          value={request}
          loading={loading}
          onChange={setRequest}
          onStart={handleStart}
          onNext={() => void handleNextStep()}
          onReset={() => void handleReset()}
          canStep={!session?.done}
          hasSession={Boolean(session)}
        />
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            className="rounded-lg border border-slate-600 px-4 py-2 text-sm text-slate-200 disabled:opacity-50"
            disabled={!session || session.done}
            onClick={() => setAutoplay((prev) => !prev)}
          >
            {autoplay ? 'Pause Playback' : 'Play Session'}
          </button>
          <button
            type="button"
            className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300"
            onClick={() => setShowAdvanced((prev) => !prev)}
          >
            {showAdvanced ? 'Hide Advanced Details' : 'Show Advanced Details'}
          </button>
        </div>
      </Card>

      {error ? <FeedbackState title="Demo service error" message={error} /> : null}

      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <h3 className="text-lg font-semibold text-white">Session Timeline</h3>
          <p className="mt-2 text-sm text-slate-400">
            Each step records the system action, user response, and resulting state update.
          </p>
          <div className="mt-4 max-h-[30rem] overflow-auto pr-2">
            {session?.timeline?.length ? (
              <SessionTimeline timeline={session.timeline} />
            ) : (
              <FeedbackState
                title="No steps yet"
                message="Start a session and click Next Step (or Play Session) to generate timeline events."
              />
            )}
          </div>
        </Card>

        <div className="space-y-6">
          <Card>
            <h3 className="text-lg font-semibold text-white">Belief Over Genres</h3>
            <p className="mt-2 text-sm text-slate-400">Updated after every action to show internal preference belief.</p>
            <div className="mt-4">
              <BeliefChart step={latestStep} />
            </div>
          </Card>
          <Card>
            <h3 className="text-lg font-semibold text-white">Budget and State</h3>
            <div className="mt-4 space-y-4">
              <BudgetBar used={latestStep?.questions_used ?? 0} total={session?.question_budget ?? request.question_budget} />
              <div className="grid grid-cols-2 gap-3 text-sm text-slate-300">
                <p>
                  Engagement: <span className="text-white">{toPercent(progress.engagement)}</span>
                </p>
                <p>
                  Uncertainty: <span className="text-white">{toPercent(progress.uncertainty)}</span>
                </p>
                <p>
                  Step Count: <span className="text-white">{session?.timeline.length ?? 0}</span>
                </p>
                <p>
                  Cumulative Reward:{' '}
                  <span className="text-white">{toFixed(session?.summary.total_reward, 2)}</span>
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>

      <Card>
        <h3 className="text-lg font-semibold text-white">Final Session Summary</h3>
        <p className="mt-2 text-sm text-slate-400">Use this panel during the presentation to conclude policy behavior.</p>
        <div className="mt-4">
          {session ? (
            <FinalSummaryCard summary={session.summary} />
          ) : (
            <FeedbackState title="Session not started" message="Initialize a scenario to generate outcomes." />
          )}
        </div>
      </Card>

      {showAdvanced && latestStep ? (
        <Card>
          <h3 className="text-lg font-semibold text-white">Advanced Step Details</h3>
          <p className="mt-2 text-sm text-slate-400">
            This section intentionally exposes lower-level internals for technical Q&A.
          </p>
          <pre className="mt-4 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
            {JSON.stringify(latestStep, null, 2)}
          </pre>
        </Card>
      ) : null}
    </div>
  )
}
